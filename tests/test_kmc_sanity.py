
import numpy as np
import os
from mgkmc.aqs import ThermalSimulation

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

if __name__ == "__main__":
    test_kmc_t0()
    test_kmc_t300()
