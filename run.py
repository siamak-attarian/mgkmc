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
from mgkmc import ThermalSimulation
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
    
    simulation_type = config.get('simulation_type', 'kmc').lower()
    print(f"Loaded configuration from {config_path}")
    print(f"Simulation Type selected: '{simulation_type}'")
    
    # ---------------------------------------------------------
    # 2. Extract System & Material Properties
    # ---------------------------------------------------------
    sys_conf = config['system']
    dimensionality = sys_conf.get('dimensionality', '3d').lower()
    plane_mode = sys_conf.get('plane_mode', 'plane_strain').lower()
    nx, ny, nz = sys_conf['nx'], sys_conf['ny'], sys_conf.get('nz', 1)
    
    shape = (nx, ny, nz) if dimensionality == '3d' else (nx, ny)
    
    # Material Elastic Modulus (E) and Poisson's ratio (nu)
    mat_conf = config['material']
    E_field = generate_field(
        mat_conf['E']['mode'], 
        shape, 
        constant_val=mat_conf['E'].get('value', 70.0),
        params=mat_conf['E'].get('parameters', {})
    )
    nu_field = generate_field(
        mat_conf['nu']['mode'], 
        shape, 
        constant_val=mat_conf['nu'].get('value', 0.3),
        params=mat_conf['nu'].get('parameters', {})
    )

    # Autoconvert E to Pa if supplied in GPa
    if E_field.mean() < 1e6:
        print(" [run.py] Detected E in GPa (mean < 1e6). Converting to Pa (*1e9).")
        E_field = E_field * 1e9

    # ---------------------------------------------------------
    # 3. Setup Physics & Dynamics
    # ---------------------------------------------------------
    phys_conf = config.get('physics', {})
    dyn_conf = config.get('dynamics', {})
    out_conf = config.get('output', {})
    
    print("Initializing ThermalSimulation environment...")
    
    if simulation_type == "linear_elastic":
        sim = None
        print(f"\n'linear_elastic' {dimensionality.upper()} mode selected. Skipping 3D ThermalSimulation initialization.")
    else:
        sim = ThermalSimulation(
            nx, ny, nz,
            M=sys_conf['M'],
            gamma0=sys_conf['gamma0'],
            E_field=E_field,
            nu_field=nu_field,
            pixel=sys_conf.get('pixel', 1.0),
            
            # Physics Parameters
        jp=phys_conf.get('jp', 20),
        jt=phys_conf.get('jt', 20),
        neighbor_softening_fraction=phys_conf.get('neighbor_softening_fraction', 0.0),
        softening_scheme=phys_conf.get('softening_scheme', 'isotropic'),
        softening_cap=phys_conf.get('softening_cap', 2.0),
        tau=phys_conf.get('tau', np.inf),
        q_act_temp=phys_conf.get('q_act_temp', 0.37),
        stability_threshold=phys_conf.get('stability_threshold', 0.0),
        redraw_directions=phys_conf.get('redraw_directions', True),
        redraw_barriers=phys_conf.get('redraw_barriers', True),
        
        # Outputs
        output_dir=out_conf.get('directory', 'output'),
        
        # Dynamics Parameters
        temperature=float(dyn_conf.get('temperature', 0.0)),
        strain_rate=float(dyn_conf.get('physical_strain_rate', 1.0e7)),
        strain_rate_sensitivity=float(dyn_conf.get('strain_rate_sensitivity', 0.0)),
        nu0=float(dyn_conf.get('nu0', 1.0e13)),
        instability_mode=dyn_conf.get('instability_mode', 'cascade'),
        cascade_timing=dyn_conf.get('cascade_timing', 'none'),
        scale_rate_by_volume=dyn_conf.get('scale_rate_by_volume', True),
        fast_patching=dyn_conf.get('fast_patching', None)
    )

    # ---------------------------------------------------------
    # 4. Prepare Boundary Conditions & Run
    # ---------------------------------------------------------
    bc_conf = config['boundary_conditions']
    
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
        return (0, 0)

    driving_raw = bc_conf.get('driving_component', 'xx')
    component = parse_comp(driving_raw)
    
    stress_targets = {}
    for k_str, val in bc_conf.get('mixed_targets', {}).items():
        k_tup = parse_comp(k_str)
        stress_targets[k_tup] = float(val)
            
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
    calculated_n_steps = int(eps_target / step_size)

    if sim:
        print(f"Temperature: {sim.temperature} K, Strain Rate: {sim.strain_rate:.2e} 1/s")
    _drv_lbl = comp_label.get(tuple(component), str(component))
    _tgt_lbl = {comp_label.get(k, str(k)): v for k, v in stress_targets.items()}
    print(f"Driving Strain Component: {_drv_lbl}")
    print(f"Target Stresses (Relaxed Components): {_tgt_lbl}")
    
    if simulation_type == "linear_elastic":
        if dimensionality == "2d":
            from mgkmc.linear_elastic_simulator import linear_elastic_simulation_2d
            out_dir = out_conf.get('directory', 'output')
            os.makedirs(out_dir, exist_ok=True)
            
            # By default, components not mentioned are fixed rigidly (strain = 0)
            target_strain_mask = np.ones((2,2), dtype=bool)
            target_values = np.zeros((2,2))
            
            target_strain_mask[component] = True
            target_values[component] = eps_target
            
            for key, val in stress_targets.items():
                if key[0] < 2 and key[1] < 2:
                    target_values[key] = val
                    target_strain_mask[key] = False
            
            chk_val      = parse_interval(out_conf.get('checkpoint_interval', 'none'))
            vtk_val      = parse_interval(out_conf.get('vtk_interval', 'none'))
            # mixed_tol is expressed in MPa in the config; solver works in Pa
            tol_macro_pa = float(bc_conf.get('mixed_tol', 1.0)) * 1e6
            log_path        = os.path.join(out_dir, 'summary_log.txt')
            global_log_path = os.path.join(out_dir, 'global_log.txt')
            enable_console  = bool(out_conf.get('enable_console', True))
            print(f"\nRunning 2D Mixed Solver ({plane_mode}) for {calculated_n_steps} steps...")
            eps_mac_list, sig_mac_list, eps_list, sig_list = linear_elastic_simulation_2d(
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
                checkpoint_path=os.path.join(out_dir, "checkpoint"),
                vtk_interval=vtk_val,
                vtk_path=os.path.join(out_dir, "step")
            )
            
            print(f"2D Elastic simulation completed. Data output to checkpoints in {out_dir}.")
            return
        elif dimensionality == "3d":
            from mgkmc.linear_elastic_simulator import linear_elastic_simulation_3d
            out_dir = out_conf.get('directory', 'output')
            os.makedirs(out_dir, exist_ok=True)

            target_strain_mask = np.ones((3, 3), dtype=bool)
            target_values      = np.zeros((3, 3))

            target_strain_mask[component] = True
            target_values[component]      = eps_target

            for key, val in stress_targets.items():
                target_values[key]      = val
                target_strain_mask[key] = False

            chk_val      = parse_interval(out_conf.get('checkpoint_interval', 'none'))
            vtk_val      = parse_interval(out_conf.get('vtk_interval', 'none'))
            tol_macro_pa    = float(bc_conf.get('mixed_tol', 1.0)) * 1e6
            log_path        = os.path.join(out_dir, 'summary_log.txt')
            global_log_path = os.path.join(out_dir, 'global_log.txt')
            enable_console  = bool(out_conf.get('enable_console', True))

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
    sim.run_mixed(
        n_global_steps=n_global_eval,
        strain_rate=float(dyn_conf.get('physical_strain_rate', 1e7)),
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
        track_cascades=out_conf.get('track_cascades', False)
    )
    
    # If the user requested pure linear_elastic analysis, we halt here.
    if simulation_type == "linear_elastic":
        print("\n'linear_elastic' mode selected. The simulation will not proceed into plastic yielding.")
        print("Initial elastic equilibrium has been established and exported. Exiting.")
        return
    
    print("\nSimulation successfully completed via run.py.")

if __name__ == "__main__":
    main()
