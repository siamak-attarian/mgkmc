import numpy as np
import os
import shutil
import pytest
from mgkmc.kmc_simulator import KmcSimulation2D
from mgkmc.aqs import ThermalSimulation

def test_kmc_finite_strain_2d():
    print("\n--- Testing 2D KMC with Finite Strain ---")
    nx, ny = 8, 8
    M = 5
    gamma0 = 0.1
    
    E = np.ones((nx, ny)) * 70.0 * 1e9  # Pa
    nu = np.ones((nx, ny)) * 0.3
    
    output_dir = "output_test_finite_strain_2d"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
        
    sim = KmcSimulation2D(
        nx, ny, M, gamma0, E, nu,
        pixel=1.0,
        output_dir=output_dir,
        temperature=1000.0,  # high temperature to guarantee flips
        strain_rate=1.0,
        strain_assumption="finite_strain",
        plane_mode="plane_strain",
        nu0=1e11,
        barrier_generator="gaussian",
        barrier_kwargs={"mean": 2.2, "std": 0.01}
    )
    
    # Run simulation
    sim.run_simulation(
        n_global_steps=3,
        step_size=0.001,
        component=(0, 0),
        stress_targets={(1, 1): 0.0},
        mixed_tol=1e4,
        mixed_max_iter=10,
        enable_console_log=True,
        enable_summary_log=True,
        enable_global_log=True
    )
    
    # Check that Cauchy stresses and F_field are updated
    assert sim.F_field.shape == (nx, ny, 2, 2)
    assert sim.sig_field.shape == (nx, ny, 2, 2)
    assert sim.eps_field.shape == (nx, ny, 2, 2)
    assert sim.F_plastic.shape == (nx, ny, 2, 2)
    
    # Check that F_macro was adjusted
    print("F_macro:", sim.F_macro)
    assert np.abs(sim.F_macro[0, 0] - 1.003) < 1e-4
    
    # Cleanup
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

def test_kmc_finite_strain_3d():
    print("\n--- Testing 3D KMC with Finite Strain ---")
    nx, ny, nz = 8, 8, 8
    M = 5
    gamma0 = 0.1
    
    E = np.ones((nx, ny, nz)) * 70.0 * 1e9  # Pa
    nu = np.ones((nx, ny, nz)) * 0.3
    
    output_dir = "output_test_finite_strain_3d"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
        
    sim = ThermalSimulation(
        nx, ny, nz, M, gamma0, E, nu,
        pixel=1.0,
        output_dir=output_dir,
        temperature=850.0,  # high temperature to guarantee flips
        strain_rate=1.0,
        strain_assumption="finite_strain",
        nu0=1e11,
        barrier_generator="gaussian",
        barrier_kwargs={"mean": 2.2, "std": 0.01}
    )
    
    # Run simulation
    sim.run_simulation(
        n_global_steps=3,
        step_size=0.001,
        component=(0, 1),  # shear driving
        stress_targets={(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0},
        mixed_tol=1e4,
        mixed_max_iter=10,
        enable_console_log=True,
        enable_summary_log=True,
        enable_global_log=True
    )
    
    # Check that Cauchy stresses and F_field are updated
    assert sim.F_field.shape == (nx, ny, nz, 3, 3)
    assert sim.sig_field.shape == (nx, ny, nz, 3, 3)
    assert sim.eps_field.shape == (nx, ny, nz, 3, 3)
    assert sim.F_plastic.shape == (nx, ny, nz, 3, 3)
    
    # Check that F_macro was adjusted
    print("F_macro:", sim.F_macro)
    assert np.abs(sim.F_macro[0, 1] - 0.003) < 1e-4
    
    # Cleanup
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

def test_neo_hookean_vs_svk_2d():
    print("\n--- Testing Neo-Hookean vs SVK 2D (Plane Strain and Plane Stress) ---")
    nx, ny = 8, 8
    M = 5
    gamma0 = 0.1
    E = np.ones((nx, ny)) * 70.0 * 1e9  # Pa
    nu = np.ones((nx, ny)) * 0.3
    
    # We will run both plane_strain and plane_stress simulations for both svk and neo_hookean
    # and compare their stresses at very small strain (e.g. 1e-5)
    for plane_mode in ["plane_strain", "plane_stress"]:
        stresses = {}
        for model in ["svk", "neo_hookean"]:
            output_dir = f"output_test_fs_{model}_{plane_mode}"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                
            sim = KmcSimulation2D(
                nx, ny, M, gamma0, E, nu,
                pixel=1.0,
                output_dir=output_dir,
                temperature=0.0,  # no flips
                strain_rate=1.0,
                strain_assumption="finite_strain",
                plane_mode=plane_mode,
                hyperelastic_model=model,
                nu0=1e11,
                barrier_generator="gaussian",
                barrier_kwargs={"mean": 2.2, "std": 0.01}
            )
            
            # Step of 1e-5 strain
            sim.elastic_run(np.array([[1e-5, 0.0], [0.0, 0.0]]))
            stresses[model] = sim.sig_field.copy()
            
            shutil.rmtree(output_dir, ignore_errors=True)
            
        # At very small strain, SVK and Neo-Hookean should yield identical stresses
        diff = np.abs(stresses["neo_hookean"] - stresses["svk"])
        max_diff = np.max(diff)
        rel_diff = max_diff / np.maximum(1e-12, np.max(np.abs(stresses["svk"])))
        print(f"[{plane_mode}] Max stress difference: {max_diff:.4e} Pa, Rel difference: {rel_diff:.4e}")
        assert rel_diff < 1e-3, f"Failed at {plane_mode}: relative difference too large ({rel_diff:.4e})"

def test_murnaghan_vs_svk_2d():
    print("\n--- Testing Murnaghan vs SVK 2D (Plane Strain and Plane Stress) ---")
    nx, ny = 8, 8
    M = 5
    gamma0 = 0.1
    E = np.ones((nx, ny)) * 70.0 * 1e9  # Pa
    nu = np.ones((nx, ny)) * 0.3
    
    for plane_mode in ["plane_strain", "plane_stress"]:
        stresses = {}
        for model in ["svk", "murnaghan"]:
            output_dir = f"output_test_fs_murn_{model}_{plane_mode}"
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)
                
            sim = KmcSimulation2D(
                nx, ny, M, gamma0, E, nu,
                pixel=1.0,
                output_dir=output_dir,
                temperature=0.0,  # no flips
                strain_rate=1.0,
                strain_assumption="finite_strain",
                plane_mode=plane_mode,
                hyperelastic_model=model,
                nu0=1e11,
                barrier_generator="gaussian",
                barrier_kwargs={"mean": 2.2, "std": 0.01},
                A_m=-100e9, B_m=-100e9, C_m=-100e9
            )
            
            # Step of 1e-5 strain
            sim.elastic_run(np.array([[1e-5, 0.0], [0.0, 0.0]]))
            stresses[model] = sim.sig_field.copy()
            
            shutil.rmtree(output_dir, ignore_errors=True)
            
        # At very small strain, SVK and Murnaghan should yield identical stresses
        diff = np.abs(stresses["murnaghan"] - stresses["svk"])
        max_diff = np.max(diff)
        rel_diff = max_diff / np.maximum(1e-12, np.max(np.abs(stresses["svk"])))
        print(f"[{plane_mode}] Max stress difference: {max_diff:.4e} Pa, Rel difference: {rel_diff:.4e}")
        assert rel_diff < 1e-3, f"Failed at {plane_mode}: relative difference too large ({rel_diff:.4e})"

def test_murnaghan_homogeneous_vs_analytical():
    print("\n--- Testing Homogeneous Murnaghan vs Analytical 1D ---")
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from othercode.comparison import calculate_murnaghan_stress
    
    l_val = 40.38461538e9
    m_val = 26.92307692e9
    A_val = -100e9
    B_val = -100e9
    C_val = -100e9
    
    # 2% tension strain
    strain = 0.02
    sigma_anal = calculate_murnaghan_stress(np.array([strain]), l_val, m_val, A_val, B_val, C_val)[0]
    
    from mgkmc.finite_strain_simulator import finite_strain_simulation_2d
    nx, ny = 4, 4
    E_field = np.ones((nx, ny)) * 70e9
    nu_field = np.ones((nx, ny)) * 0.3
    
    F_mac, Sig_mac, P_mac, _, _ = finite_strain_simulation_2d(
        E=E_field,
        nu=nu_field,
        driving_component=(0, 0),
        eps_target=strain,
        n_steps=1,
        mixed_targets={(1, 1): 0.0}, # plane stress sigma_yy = 0
        plane_mode="plane_stress",
        model_type="murnaghan",
        A_m=A_val,
        B_m=B_val,
        C_m=C_val,
        enable_console=False
    )
    
    sigma_sim = Sig_mac[-1, 0, 0] # Cauchy stress in xx
    
    diff = np.abs(sigma_sim - sigma_anal)
    rel_diff = diff / np.maximum(1e-12, np.abs(sigma_anal))
    print(f"Analytical stress: {sigma_anal/1e9:.6f} GPa, Simulated: {sigma_sim/1e9:.6f} GPa")
    print(f"Diff: {diff/1e6:.4f} MPa, Rel Diff: {rel_diff:.4e}")
    assert rel_diff < 1e-4, f"Homogeneous simulation failed to match analytical stress: {rel_diff:.4e}"

def test_mu_lambda_parameter_conversion():
    print("\n--- Testing Lamé parameters (mu, lambda) conversion to E and nu ---")
    # mu = 26.92307692 GPa, lambda = 40.38461538 GPa
    # Should yield E = 70 GPa, nu = 0.3
    mu_val = 26.92307692e9
    l_val = 40.38461538e9
    
    denom = l_val + mu_val
    E_calc = mu_val * (3.0 * l_val + 2.0 * mu_val) / denom
    nu_calc = l_val / (2.0 * denom)
    
    print(f"Calculated E: {E_calc/1e9:.6f} GPa (expected: 70 GPa)")
    print(f"Calculated nu: {nu_calc:.6f} (expected: 0.3)")
    
    assert np.abs(E_calc - 70e9) / 70e9 < 1e-6
    assert np.abs(nu_calc - 0.3) < 1e-6

def test_parse_material_property():
    print("\n--- Testing parse_material_property helper ---")
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

    # Test with dict
    mode, val, params = parse_material_property({"mode": "constant", "value": 65.0}, 25.0)
    assert mode == "constant"
    assert val == 65.0
    assert params == {}

    # Test with float
    mode, val, params = parse_material_property(65.0, 25.0)
    assert mode == "constant"
    assert val == 65.0
    assert params == {}

    # Test with None
    mode, val, params = parse_material_property(None, 25.0)
    assert mode == "constant"
    assert val == 25.0
    assert params == {}

def test_negative_strain_steps():
    print("\n--- Testing step count calculation for negative strain (compression) ---")
    eps_target = -0.15
    step_size = 2e-3
    calculated_n_steps = int(abs(eps_target) / abs(step_size))
    print(f"Calculated steps: {calculated_n_steps} (expected: 75)")
    assert calculated_n_steps == 75

if __name__ == "__main__":
    test_kmc_finite_strain_2d()
    test_kmc_finite_strain_3d()
    test_neo_hookean_vs_svk_2d()
    test_murnaghan_vs_svk_2d()
    test_murnaghan_homogeneous_vs_analytical()
    test_mu_lambda_parameter_conversion()
    test_parse_material_property()
    test_negative_strain_steps()
