import numpy as np
import os
import sys

# Ensure the local mgkmc package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mgkmc import KmcSimulation2D

def test_run():
    print("Testing KmcSimulation2D with secant_degradation hyperelastic model...")
    nx, ny = 16, 16
    M = 4
    gamma0 = 0.1
    E = np.full((nx, ny), 70.0e9)
    nu = np.full((nx, ny), 0.3)
    
    # Initialize the simulation with secant degradation parameters
    sim = KmcSimulation2D(
        nx=nx, ny=ny, M=M, gamma0=gamma0,
        E_field=E, nu_field=nu,
        strain_assumption="small_strain",
        hyperelastic_model="secant_degradation",
        d=0.5, k=0.1,
        output_dir="output_test_secant_kmc"
    )
    
    # Run a single step to verify everything solves correctly
    print("Running a single elastic step...")
    eps_macro = np.zeros((2, 2))
    eps_macro[0, 0] = 0.01  # 1% strain along xx
    sig_mean = sim.elastic_run(eps_macro)
    print(f"Elastic run successful. Mean stress (MPa):\n{sig_mean / 1e6}")
    
    # Let's run a short simulation with 5 global steps
    print("Running a short simulation...")
    sim.run_simulation(
        n_global_steps=5,
        step_size=1e-4,
        component=(0, 0),
        enable_console_log=True
    )
    print("Verification completed successfully!")

if __name__ == "__main__":
    test_run()
