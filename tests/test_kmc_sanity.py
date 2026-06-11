import numpy as np
import os
import shutil
from mgkmc.aqs import ThermalSimulation
from mgkmc.kmc_simulator import KmcSimulation2D


def test_kmc_t0():
    print("\n--- Testing T=0 (No KMC) ---")
    nx, ny, nz = 8, 8, 8
    M = 5
    gamma0 = 0.1
    
    E = np.ones((nx, ny, nz)) * 70.0 # GPa
    nu = np.ones((nx, ny, nz)) * 0.3
    
    sim = ThermalSimulation(nx, ny, nz, M, gamma0, E, nu, 
                             output_dir="output_test_t0",
                             temperature=0.0,
                             strain_rate=1.0)
    
    # Pure shear
    strain_inc = np.zeros((3,3))
    strain_inc[0,1] = 0.002 # 0.2% per step
    strain_inc[1,0] = 0.002
    
    sim.run(n_global_steps=5, strain_increment_tensor=strain_inc)
    
    # Check logs
    log_path = os.path.join("output_test_t0", "kmc_log.txt")
    if os.path.exists(log_path):
        print("FAIL: KMC log should not exist for T=0")
    else:
        print("PASS: No KMC log for T=0")

def test_kmc_t300():
    print("\n--- Testing T=300 (With KMC) ---")
    nx, ny, nz = 8, 8, 8
    M = 5
    gamma0 = 0.1
    
    E = np.ones((nx, ny, nz)) * 70.0
    nu = np.ones((nx, ny, nz)) * 0.3
    
    # Very slow strain rate to encourage KMC events?
    # Or just normal rate but high T?
    # At T=300, barriers ~0.5 eV might effectively be frozen if attempt freq is 1e13
    # rate ~ 1e13 * exp(-0.5 / (8.6e-5 * 300)) = 1e13 * exp(-0.5 / 0.0258) = 1e13 * exp(-19) ~ 1e13 * 5e-9 ~ 5e4 Hz.
    # So events happen every 2e-5 seconds.
    # If strain rate is 1.0 -> dt_elastic ~ 0.002 s.
    # So we should see MANY KMC events.
    
    sim = ThermalSimulation(nx, ny, nz, M, gamma0, E, nu, 
                             output_dir="output_test_t300",
                             temperature=1000.0, # High T to guarantee events
                             strain_rate=1.0)
    
    strain_inc = np.zeros((3,3))
    strain_inc[0,1] = 0.002
    strain_inc[1,0] = 0.002
    
    sim.run(n_global_steps=2, strain_increment_tensor=strain_inc)
    
    # Check logs
    kmc_log = os.path.join("output_test_t300", "kmc_log.txt")
    if os.path.exists(kmc_log):
        with open(kmc_log, 'r') as f:
            lines = f.readlines()
        print(f"PASS: KMC log exists. Lines: {len(lines)}")
        if len(lines) > 1:
            print("PASS: KMC events recorded.")
        else:
            print("WARN: Log created but empty?")
    else:
        print("FAIL: No KMC log created for T=1000")

def test_kmc_small_strain_2d_mixed():
    print("\n--- Testing 2D KMC with Small Strain Mixed BCs ---")
    nx, ny = 8, 8
    M = 5
    gamma0 = 0.1
    
    E = np.ones((nx, ny)) * 70.0 * 1e9  # Pa
    nu = np.ones((nx, ny)) * 0.3
    
    output_dir = "output_test_small_strain_2d"
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
        strain_assumption="small_strain",
        plane_mode="plane_strain",
        nu0=1e11,
        barrier_generator="gaussian",
        barrier_kwargs={"mean": 2.2, "std": 0.01}
    )
    
    # Run simulation with stress control target in yy
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
    
    # Verify that eps_macro is updated and stresses are computed
    assert sim.sig_field.shape == (nx, ny, 2, 2)
    assert sim.eps_field.shape == (nx, ny, 2, 2)
    
    # Assert lateral relaxation did not diverge and convergence was achieved
    # sig_yy mean should be close to the target of 0.0
    sig_mean = sim.sig_field.mean(axis=(0, 1))
    print("Final mean stress:", sig_mean)
    assert np.abs(sig_mean[1, 1]) < 1e6, f"sig_yy ({sig_mean[1,1]}) not relaxed to 0.0!"
    
    # Cleanup
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass

if __name__ == "__main__":
    test_kmc_t0()
    test_kmc_t300()
    test_kmc_small_strain_2d_mixed()

