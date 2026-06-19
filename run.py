"""
MGKMC Main Execution Script
===========================

This script acts as the primary entry point for launching simulations.
It parses `config.yaml` for system, material, physics, boundary conditions,
and output settings, and automatically instantiates and executes the 
`ThermalSimulation` environment.

Usage:
------
    python run.py
"""

import os
import sys
import yaml
import numpy as np

# Add local path to ensure mgkmc is importable if run from source dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mgkmc import ThermalSimulation, KmcSimulation2D
from mgkmc.microstructure import generate_field

def main():
    # ---------------------------------------------------------
    # 1. Load Configuration
    # ---------------------------------------------------------
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find {config_path} in the current directory.")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Apply pyfftw thread limit globally
    try:
        import pyfftw
        num_threads = config.get('system', {}).get('num_threads', 1)
        pyfftw.config.NUM_THREADS = int(num_threads)
    except Exception as e:
        print(f" [run.py] Warning: could not set pyfftw threads: {e}")
        
    # Apply Seed
    seed = config.get('seed', 42)
    np.random.seed(seed)

    def parse_interval(val):
        if val is None: return "none"
        if isinstance(val, str):
            if val.lower() in ["none", "current", "last"]:
                return val.lower()
            try:
                return int(val)
            except ValueError:
                return val.lower()
        return val
    
    simulation_type   = config.get('simulation_type', 'kmc').lower()
    strain_assumption = config.get('system', {}).get('strain_assumption', 'small_strain').lower()
    print(f"Loaded configuration from {config_path}")
    print(f"Simulation Type selected: '{simulation_type}'")
    print(f"Strain Assumption: '{strain_assumption}'")
    
    # ---------------------------------------------------------
    # 2. Extract System & Material Properties
    # ---------------------------------------------------------
    sys_conf = config['system']
    dimensionality = sys_conf.get('dimensionality', '3d').lower()
    plane_mode = sys_conf.get('plane_mode', 'plane_strain').lower()
    hyperelastic_model = sys_conf.get('hyperelastic_model', 'svk').lower()
    nx, ny, nz = sys_conf['nx'], sys_conf['ny'], sys_conf.get('nz', 1)
    use_3d_barriers = sys_conf.get('3d_barriers', False)
    solver = sys_conf.get('solver', 'al').lower()
    
    shape = (nx, ny, nz) if dimensionality == '3d' else (nx, ny)
    
    # Helper to parse material property that could be a dictionary or a float
    def parse_material_property(config_prop, default_val):
        if config_prop is None:
            return "constant", default_val, {}
        if isinstance(config_prop, dict):
            mode = config_prop.get('mode', 'constant')
            val = config_prop.get('value', default_val)
            params = config_prop.get('parameters', {})
            return mode, val, params
        else:
            return "constant", float(config_prop), {}

    # Material Elastic Modulus (E) and Poisson's ratio (nu), or Lamé parameters (mu, lambda)
    mat_conf = config['material']
    if 'mu' in mat_conf and 'lambda' in mat_conf:
        mu_mode, mu_val, mu_params = parse_material_property(mat_conf.get('mu'), 26.92)
        mu_field = generate_field(
            mu_mode,
            shape,
            constant_val=mu_val,
            params=mu_params
        )
        lambda_mode, lambda_val, lambda_params = parse_material_property(mat_conf.get('lambda'), 40.38)
        lambda_field = generate_field(
            lambda_mode,
            shape,
            constant_val=lambda_val,
            params=lambda_params
        )
        # Scale to Pa if supplied in GPa
        if mu_field.mean() < 1e6:
            print(" [run.py] Detected mu in GPa (mean < 1e6). Converting to Pa (*1e9).")
            mu_field = mu_field * 1e9
        if lambda_field.mean() < 1e6:
            print(" [run.py] Detected lambda in GPa (mean < 1e6). Converting to Pa (*1e9).")
            lambda_field = lambda_field * 1e9

        # Calculate E_field and nu_field from mu_field and lambda_field
        denom = lambda_field + mu_field
        denom_safe = np.where(denom == 0.0, 1e-20, denom)
        E_field = mu_field * (3.0 * lambda_field + 2.0 * mu_field) / denom_safe
        nu_field = lambda_field / (2.0 * denom_safe)
        print(" [run.py] Calculated E and nu fields from mu and lambda fields.")
    else:
        E_mode, E_val, E_params = parse_material_property(mat_conf.get('E'), 70.0)
        E_field = generate_field(
            E_mode, 
            shape, 
            constant_val=E_val,
            params=E_params
        )
        nu_mode, nu_val, nu_params = parse_material_property(mat_conf.get('nu'), 0.3)
        nu_field = generate_field(
            nu_mode, 
            shape, 
            constant_val=nu_val,
            params=nu_params
        )
        # Autoconvert E to Pa if supplied in GPa
        if E_field.mean() < 1e6:
            print(" [run.py] Detected E in GPa (mean < 1e6). Converting to Pa (*1e9).")
            E_field = E_field * 1e9

    # Load Murnaghan parameters (defaulting to 0.0)
    A_val = float(mat_conf.get('A', 0.0))
    B_val = float(mat_conf.get('B', 0.0))
    C_val = float(mat_conf.get('C', 0.0))

    # Autoconvert Murnaghan constants to Pa if supplied in GPa (values < 1e6 except 0.0)
    if abs(A_val) < 1e6 and A_val != 0.0:
        A_val *= 1e9
    if abs(B_val) < 1e6 and B_val != 0.0:
        B_val *= 1e9
    if abs(C_val) < 1e6 and C_val != 0.0:
        C_val *= 1e9

    # Load Secant Degradation parameters (d, k)
    d_val = float(mat_conf.get('d', 0.0))
    k_val = float(mat_conf.get('k', 0.0))

    # ---------------------------------------------------------
    # 3. Setup Physics & Dynamics
    # ---------------------------------------------------------
    phys_conf = config.get('physics', {})
    dyn_conf = config.get('dynamics', {})
    out_conf = config.get('output', {})
    det_conf = config.get('detection', {})
    bar_conf = config.get('barriers', {})
    bar_type = bar_conf.get('type', 'gaussian')
    bar_kwargs = bar_conf.get('kwargs', {})
    
    # Parse Thermal config
    therm_conf = config.get('thermal', {})
    enable_thermal = therm_conf.get('enable_thermal', False)
    Cp = therm_conf.get('Cp', 420.0)
    rho = therm_conf.get('rho', 6125.0)
    thermal_diffusivity = therm_conf.get('thermal_diffusivity', 3.0e-6)
    thermal_coords = therm_conf.get('thermal_coords', 'pixel')
    temperature_cap = therm_conf.get('temperature_cap', 1000.0)
    thermostat = therm_conf.get('thermostat', False)
    tau_bath = therm_conf.get('tau_bath', 0.0)

    if simulation_type == "linear_elastic":
        sim = None
        print(f"\n'linear_elastic' {dimensionality.upper()} mode selected. Skipping KMC initialization.")
    elif dimensionality == "2d":
        print("Initializing KmcSimulation2D environment...")
        sim = KmcSimulation2D(
            nx, ny,
            M=sys_conf['M'],
            gamma0=sys_conf['gamma0'],
            E_field=E_field,
            nu_field=nu_field,
            pixel=sys_conf.get('pixel', 1.0),
            plane_mode=plane_mode,
            barrier_generator=bar_type,
            barrier_kwargs=bar_kwargs,
            
            # Physics Parameters
            jp=phys_conf.get('jp', 20),
            jt=phys_conf.get('jt', 20),
            neighbor_softening_fraction=phys_conf.get('neighbor_softening_fraction', 0.0),
            softening_scheme=phys_conf.get('softening_scheme', 'isotropic'),
            softening_cap=phys_conf.get('softening_cap', 2.0),
            q_act_temp=phys_conf.get('q_act_temp', 0.37),
            redraw_directions=phys_conf.get('redraw_directions', True),
            redraw_barriers=phys_conf.get('redraw_barriers', True),
            stz_mode=phys_conf.get('stz_mode', 'simple_shear'),
            # nu0, etc. handled below in Dynamics
            
            # Outputs
            output_dir=out_conf.get('directory', 'output'),
            
            # Dynamics Parameters
            temperature=float(dyn_conf.get('temperature', 0.0)),
            strain_rate=float(dyn_conf.get('physical_strain_rate', 1.0e7)),
            nu0=float(dyn_conf.get('nu0', 1.0e13)),
            stability_threshold=phys_conf.get('stability_threshold', 0.0),
            fast_patching=dyn_conf.get('fast_patching', None),
            instability_mode=dyn_conf.get('instability_mode', 'cascade'),
            cascade_timing=dyn_conf.get('cascade_timing', 'none'),
            scale_rate_by_volume=dyn_conf.get('scale_rate_by_volume', False),
            
            # Thermal Parameters
            enable_thermal=enable_thermal,
            Cp=Cp,
            rho=rho,
            thermal_diffusivity=thermal_diffusivity,
            thermal_coords=thermal_coords,
            temperature_cap=temperature_cap,
            tau_bath=tau_bath,
            strain_assumption=strain_assumption,
            hyperelastic_model=hyperelastic_model,
            A_m=A_val,
            B_m=B_val,
            C_m=C_val,
            solver=solver
        )
    else:
        print("Initializing 3D ThermalSimulation environment...")
        sim = ThermalSimulation(
            nx, ny, nz,
            M=sys_conf['M'],
            gamma0=sys_conf['gamma0'],
            E_field=E_field,
            nu_field=nu_field,
            pixel=sys_conf.get('pixel', 1.0),
            barrier_generator=bar_type,
            barrier_kwargs=bar_kwargs,
            
            # Physics Parameters
            jp=phys_conf.get('jp', 20),
            jt=phys_conf.get('jt', 20),
            neighbor_softening_fraction=phys_conf.get('neighbor_softening_fraction', 0.0),
            softening_scheme=phys_conf.get('softening_scheme', 'isotropic'),
            softening_cap=phys_conf.get('softening_cap', 2.0),
            q_act_temp=phys_conf.get('q_act_temp', 0.37),
            stability_threshold=phys_conf.get('stability_threshold', 0.0),
            redraw_directions=phys_conf.get('redraw_directions', True),
            redraw_barriers=phys_conf.get('redraw_barriers', True),
            
            # Outputs
            output_dir=out_conf.get('directory', 'output'),
            
            # Dynamics Parameters
            temperature=float(dyn_conf.get('temperature', 0.0)),
            strain_rate=float(dyn_conf.get('physical_strain_rate', 1.0e7)),
            nu0=float(dyn_conf.get('nu0', 1.0e13)),
            instability_mode=dyn_conf.get('instability_mode', 'cascade'),
            cascade_timing=dyn_conf.get('cascade_timing', 'none'),
            scale_rate_by_volume=dyn_conf.get('scale_rate_by_volume', True),
            fast_patching=dyn_conf.get('fast_patching', None),
            
            # Thermal Parameters
            enable_thermal=enable_thermal,
            Cp=Cp,
            rho=rho,
            thermal_diffusivity=thermal_diffusivity,
            thermal_coords=thermal_coords,
            temperature_cap=temperature_cap,
            thermostat=thermostat,
            tau_bath=tau_bath,
            strain_assumption=strain_assumption,
            use_3d_barriers=use_3d_barriers,
            hyperelastic_model=hyperelastic_model,
            A_m=A_val,
            B_m=B_val,
            C_m=C_val,
            solver=solver
        )

    # ---------------------------------------------------------
    # 4. Prepare Boundary Conditions & Run
    # ---------------------------------------------------------
    bc_conf = config['boundary_conditions']
    
    if dimensionality == "2d":
        comp_map = {'xx': (0, 0), 'yy': (1, 1), 'xy': (0, 1), 'yx': (1, 0)}
    else:
        comp_map = {
            'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2),
            'xy': (0, 1), 'yx': (1, 0),
            'xz': (0, 2), 'zx': (2, 0),
            'yz': (1, 2), 'zy': (2, 1)
        }
    # Reverse map: tuple -> label string
    comp_label = {v: k for k, v in comp_map.items()}

    def parse_comp(c):
        if isinstance(c, (list, tuple)) and len(c) == 2:
            return tuple(c)
        c_str = str(c).strip().strip('[]()').replace(',', ' ').lower()
        if c_str in comp_map:
            return comp_map[c_str]
        parts = c_str.split()
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
        return None

    driving_raw = bc_conf.get('driving_component', 'xx')
    component = parse_comp(driving_raw)
    
    stress_targets = {}
    for k_str, val in bc_conf.get('mixed_targets', {}).items():
        k_tup = parse_comp(k_str)
        if k_tup is not None:
            stress_targets[k_tup] = float(val) * 1e9 if float(val) < 1e6 else float(val)
            
    print(f"\n--- Starting execution ---")
    print(f"Grid: {nx}x{ny}x{nz if dimensionality=='3d' else ''} ({dimensionality.upper()})")
    if dimensionality == '2d':
        if plane_mode == 'plane_stress':
            print("Mode: Plane Stress (sigma_zz = 0, epsilon_zz free)")
        else:
            print("Mode: Plane Strain (epsilon_zz = 0, sigma_zz free)")

    loading_conf = config.get('loading', {})
    eps_target = float(loading_conf.get('eps_target', dyn_conf.get('eps_target', 0.14)))
    step_size = float(loading_conf.get('step_size', dyn_conf.get('step_size', 1e-4)))
    calculated_n_steps = int(abs(eps_target) / abs(step_size))

    if sim:
        print(f"Temperature: {sim.temperature} K, Strain Rate: {sim.strain_rate:.2e} 1/s")
    _drv_lbl = comp_label.get(tuple(component), str(component))
    _tgt_lbl = {comp_label.get(k, str(k)): v for k, v in stress_targets.items()}
    print(f"Driving Strain Component: {_drv_lbl}")
    print(f"Target Stresses (Relaxed Components): {_tgt_lbl}")
    
    if simulation_type == "linear_elastic":
        if dimensionality == "2d":
            out_dir        = out_conf.get('directory', 'output')
            os.makedirs(out_dir, exist_ok=True)
            tol_macro_pa   = float(bc_conf.get('mixed_tol', 1.0)) * 1e6
            log_path       = os.path.join(out_dir, 'summary_log.txt')
            global_log_path= os.path.join(out_dir, 'global_log.txt')
            enable_console = bool(out_conf.get('enable_console', True))

            # ------------------------------------------------------------------
            # Route by strain_assumption
            # ------------------------------------------------------------------
            if strain_assumption == 'finite_strain':
                from mgkmc.finite_strain_simulator import finite_strain_simulation_2d

                # Convert mixed_targets to dict {(i,j): Pa}
                fs_mixed = {}
                for key, val in stress_targets.items():
                    if key[0] < 2 and key[1] < 2:
                        fs_mixed[key] = float(val)

                # Read interval configurations
                chk_val = parse_interval(out_conf.get('checkpoint_interval', 'none'))
                vtk_val = parse_interval(out_conf.get('vtk_interval', 'none'))

                print(f"\nRunning 2D Finite-Strain Newton-CG Solver ({plane_mode}) "
                      f"for {calculated_n_steps} steps...")

                F_mac_arr, Sig_mac_arr, P_mac_arr, F_list, Sig_list = \
                    finite_strain_simulation_2d(
                        E=E_field,
                        nu=nu_field,
                        driving_component=component,
                        eps_target=eps_target,
                        n_steps=calculated_n_steps,
                        mixed_targets=fs_mixed,
                        plane_mode=plane_mode,
                        pixel=sys_conf.get('pixel', 1.0),
                        tol_macro=tol_macro_pa,
                        store=True,
                        log_path=log_path,
                        global_log_path=global_log_path,
                        enable_console=enable_console,
                        checkpoint_interval=chk_val,
                        checkpoint_path=os.path.join(out_dir, 'checkpoint'),
                        vtk_interval=vtk_val,
                        vtk_path=os.path.join(out_dir, 'step'),
                        model_type=hyperelastic_model,
                        A_m=A_val,
                        B_m=B_val,
                        C_m=C_val,
                        solver=solver
                    )

                print(f"2D Finite-Strain simulation completed. "
                      f"Logs written to {out_dir}.")
                return

            else:
                # ---- Small-strain path ----
                if hyperelastic_model == 'secant_degradation':
                    from mgkmc.linear_elastic_simulator import secant_elastic_simulation_2d

                    # Derive lam/mu from E/nu (or read directly if supplied)
                    if 'mu' in mat_conf and 'lambda' in mat_conf:
                        mu_mode, mu_val, mu_params = parse_material_property(mat_conf.get('mu'), 26.92)
                        lam_mode, lam_val, lam_params  = parse_material_property(mat_conf.get('lambda'), 40.38)
                        mu_field_sec = generate_field(mu_mode, shape, constant_val=mu_val, params=mu_params)
                        lam_field_sec = generate_field(lam_mode, shape, constant_val=lam_val, params=lam_params)
                        if mu_field_sec.mean() < 1e6:
                            mu_field_sec *= 1e9
                        if lam_field_sec.mean() < 1e6:
                            lam_field_sec *= 1e9
                        if plane_mode == 'plane_stress':
                            print(" [run.py] Applying plane stress correction to lambda (lambda* = 2*lambda*mu/(lambda + 2*mu)).")
                            lam_field_sec = 2.0 * lam_field_sec * mu_field_sec / (lam_field_sec + 2.0 * mu_field_sec)
                    else:
                        # Compute from E, nu (plane-strain convention)
                        if plane_mode == 'plane_stress':
                            lam_field_sec = E_field * nu_field / (1.0 - nu_field**2)
                        else:
                            lam_field_sec = E_field * nu_field / ((1.0 + nu_field) * (1.0 - 2.0 * nu_field))
                        mu_field_sec = E_field / (2.0 * (1.0 + nu_field))

                    target_strain_mask = np.ones((2, 2), dtype=bool)
                    target_values      = np.zeros((2, 2))
                    target_strain_mask[component] = True
                    target_values[component]      = eps_target
                    for key, val in stress_targets.items():
                        if key[0] < 2 and key[1] < 2:
                            target_values[key]      = val
                            target_strain_mask[key] = False

                    chk_val = parse_interval(out_conf.get('checkpoint_interval', 'none'))
                    vtk_val = parse_interval(out_conf.get('vtk_interval', 'none'))

                    print(f"\nRunning 2D Secant Elastic Degradation Solver ({plane_mode}) "
                          f"for {calculated_n_steps} steps "
                          f"[d={d_val:.4f}, k={k_val:.2f}]...")
                    eps_mac_list, sig_mac_list, eps_list, sig_list = \
                        secant_elastic_simulation_2d(
                            lam=lam_field_sec, mu=mu_field_sec,
                            d=d_val, k=k_val,
                            target_strain_mask=target_strain_mask,
                            target_values=target_values,
                            n_steps=calculated_n_steps,
                            pixel=sys_conf['pixel'],
                            plane_mode=plane_mode,
                            store=True,
                            tol_macro=tol_macro_pa,
                            log_path=log_path,
                            global_log_path=global_log_path,
                            driving_component=component,
                            enable_console=enable_console,
                            checkpoint_interval=chk_val,
                            checkpoint_path=os.path.join(out_dir, 'checkpoint'),
                            vtk_interval=vtk_val,
                            vtk_path=os.path.join(out_dir, 'step')
                        )
                    print(f"2D Secant Elastic simulation completed. "
                          f"Logs written to {out_dir}.")
                    return

                else:
                    # ---- Original small-strain linear-elastic path (unchanged) ----
                    from mgkmc.linear_elastic_simulator import linear_elastic_simulation_2d

                    target_strain_mask = np.ones((2, 2), dtype=bool)
                    target_values      = np.zeros((2, 2))

                    target_strain_mask[component] = True
                    target_values[component]      = eps_target

                    for key, val in stress_targets.items():
                        if key[0] < 2 and key[1] < 2:
                            target_values[key]      = val
                            target_strain_mask[key] = False

                    chk_val = parse_interval(out_conf.get('checkpoint_interval', 'none'))
                    vtk_val = parse_interval(out_conf.get('vtk_interval', 'none'))

                    print(f"\nRunning 2D Small-Strain Mixed Solver ({plane_mode}) "
                          f"for {calculated_n_steps} steps...")
                    eps_mac_list, sig_mac_list, eps_list, sig_list = \
                        linear_elastic_simulation_2d(
                            E=E_field, nu=nu_field,
                            target_strain_mask=target_strain_mask,
                            target_values=target_values,
                            n_steps=calculated_n_steps,
                            pixel=sys_conf['pixel'],
                            plane_mode=plane_mode,
                            store=True,
                            tol_macro=tol_macro_pa,
                            log_path=log_path,
                            global_log_path=global_log_path,
                            driving_component=component,
                            enable_console=enable_console,
                            checkpoint_interval=chk_val,
                            checkpoint_path=os.path.join(out_dir, 'checkpoint'),
                            vtk_interval=vtk_val,
                            vtk_path=os.path.join(out_dir, 'step')
                        )

                    print(f"2D Elastic simulation completed. "
                          f"Data output to checkpoints in {out_dir}.")
                    return
        elif dimensionality == "3d":
            out_dir = out_conf.get('directory', 'output')
            os.makedirs(out_dir, exist_ok=True)

            chk_val      = parse_interval(out_conf.get('checkpoint_interval', 'none'))
            vtk_val      = parse_interval(out_conf.get('vtk_interval', 'none'))
            tol_macro_pa    = float(bc_conf.get('mixed_tol', 1.0)) * 1e6
            log_path        = os.path.join(out_dir, 'summary_log.txt')
            global_log_path = os.path.join(out_dir, 'global_log.txt')
            enable_console  = bool(out_conf.get('enable_console', True))

            if strain_assumption == 'finite_strain':
                from mgkmc.finite_strain_simulator import finite_strain_simulation_3d

                # Convert mixed targets to dict {(i,j): Pa}
                fs_mixed = {key: float(val) for key, val in stress_targets.items()}

                print(f"\nRunning 3D Finite-Strain Newton-CG Solver for {calculated_n_steps} steps...")
                F_mac_arr, Sig_mac_arr, P_mac_arr, F_list, Sig_list = \
                    finite_strain_simulation_3d(
                        E=E_field, nu=nu_field,
                        driving_component=component,
                        eps_target=eps_target,
                        n_steps=calculated_n_steps,
                        mixed_targets=fs_mixed,
                        pixel=sys_conf.get('pixel', 1.0),
                        tol_macro=tol_macro_pa,
                        store=True,
                        log_path=log_path,
                        global_log_path=global_log_path,
                        enable_console=enable_console,
                        checkpoint_interval=chk_val,
                        checkpoint_path=os.path.join(out_dir, 'checkpoint'),
                        vtk_interval=vtk_val,
                        vtk_path=os.path.join(out_dir, 'step'),
                        model_type=hyperelastic_model,
                        A_m=A_val,
                        B_m=B_val,
                        C_m=C_val,
                        solver=solver
                    )

                print(f"3D Finite-Strain simulation completed. Logs written to {out_dir}.")
                return
            else:
                if hyperelastic_model == 'secant_degradation':
                    from mgkmc.linear_elastic_simulator import secant_elastic_simulation_3d

                    if 'mu' in mat_conf and 'lambda' in mat_conf:
                        mu_mode, mu_val, mu_params = parse_material_property(mat_conf.get('mu'), 26.92)
                        lam_mode, lam_val, lam_params  = parse_material_property(mat_conf.get('lambda'), 40.38)
                        mu_field_sec = generate_field(mu_mode, shape, constant_val=mu_val, params=mu_params)
                        lam_field_sec = generate_field(lam_mode, shape, constant_val=lam_val, params=lam_params)
                        if mu_field_sec.mean() < 1e6:
                            mu_field_sec *= 1e9
                        if lam_field_sec.mean() < 1e6:
                            lam_field_sec *= 1e9
                    else:
                        lam_field_sec = E_field * nu_field / ((1.0 + nu_field) * (1.0 - 2.0 * nu_field))
                        mu_field_sec = E_field / (2.0 * (1.0 + nu_field))

                    target_strain_mask = np.ones((3, 3), dtype=bool)
                    target_values      = np.zeros((3, 3))
                    target_strain_mask[component] = True
                    target_values[component]      = eps_target
                    for key, val in stress_targets.items():
                        target_values[key]      = val
                        target_strain_mask[key] = False

                    print(f"\nRunning 3D Secant Elastic Degradation Solver for {calculated_n_steps} steps "
                          f"[d={d_val:.4f}, k={k_val:.2f}]...")
                    eps_mac_list, sig_mac_list, eps_list, sig_list = \
                        secant_elastic_simulation_3d(
                            lam=lam_field_sec, mu=mu_field_sec,
                            d=d_val, k=k_val,
                            target_strain_mask=target_strain_mask,
                            target_values=target_values,
                            n_steps=calculated_n_steps,
                            pixel=sys_conf['pixel'],
                            store=True,
                            tol_macro=tol_macro_pa,
                            log_path=log_path,
                            global_log_path=global_log_path,
                            driving_component=component,
                            enable_console=enable_console,
                            checkpoint_interval=chk_val,
                            checkpoint_path=os.path.join(out_dir, "checkpoint"),
                            vtk_interval=vtk_val,
                            vtk_path=os.path.join(out_dir, "step")
                        )
                    print(f"3D Secant Elastic simulation completed. Logs written to {out_dir}.")
                    return

                else:
                    from mgkmc.linear_elastic_simulator import linear_elastic_simulation_3d

                    target_strain_mask = np.ones((3, 3), dtype=bool)
                    target_values      = np.zeros((3, 3))

                    target_strain_mask[component] = True
                    target_values[component]      = eps_target

                    for key, val in stress_targets.items():
                        target_values[key]      = val
                        target_strain_mask[key] = False

                    print(f"\nRunning 3D Linear Elastic Solver for {calculated_n_steps} steps...")
                    eps_mac_list, sig_mac_list, eps_list, sig_list = linear_elastic_simulation_3d(
                        E=E_field, nu=nu_field,
                        target_strain_mask=target_strain_mask,
                        target_values=target_values,
                        n_steps=calculated_n_steps,
                        pixel=sys_conf['pixel'],
                        store=True,
                        tol_macro=tol_macro_pa,
                        log_path=log_path,
                        global_log_path=global_log_path,
                        driving_component=component,
                        enable_console=enable_console,
                        checkpoint_interval=chk_val,
                        checkpoint_path=os.path.join(out_dir, "checkpoint"),
                        vtk_interval=vtk_val,
                        vtk_path=os.path.join(out_dir, "step")
                    )

                    print(f"3D Elastic simulation completed. Data output to checkpoints in {out_dir}.")
                    return
        else:
            n_global_eval = 0
    else:
        n_global_eval = calculated_n_steps
        
    # Execute mixed loading for 3D AQS/KMC
    sim.run_simulation(
        n_global_steps=n_global_eval,
        step_size=step_size,
        component=component,
        stress_targets=stress_targets,
        mixed_tol=float(bc_conf.get('mixed_tol', 1e-4)),
        
        # Output interval setups
        vtk_interval=parse_interval(out_conf.get('vtk_interval', 'none')),
        vtk_elastic_only=out_conf.get('vtk_elastic_only', True),
        checkpoint_interval=parse_interval(out_conf.get('checkpoint_interval', 'none')),
        checkpoint_elastic_only=out_conf.get('checkpoint_elastic_only', True),
        enable_save_q=out_conf.get('enable_save_q', False),
        save_q_interval=out_conf.get('save_q_interval', None),
        
        # Logging flags
        enable_console_log=out_conf.get('enable_console', True),
        summary_filename=out_conf.get('summary_filename', 'summary_log.txt'),
        enable_summary_log=out_conf.get('enable_summary_log', True),
        enable_global_log=out_conf.get('enable_global_log', True),
        enable_cascade_log=out_conf.get('enable_cascade_log', True),
        enable_kmc_log=out_conf.get('enable_kmc_log', True),
        track_cascades=out_conf.get('track_cascades', False),
        max_kmc_steps_pct=det_conf.get('max_kmc_steps_pct', 0.3),
        max_cascade_steps_pct=det_conf.get('max_cascade_steps_pct', 0.3)
    )
    
    # If the user requested pure linear_elastic analysis, we halt here.
    if simulation_type == "linear_elastic":
        print("\n'linear_elastic' mode selected. The simulation will not proceed into plastic yielding.")
        print("Initial elastic equilibrium has been established and exported. Exiting.")
        return
    
    print("\nSimulation successfully completed via run.py.")

if __name__ == "__main__":
    main()
