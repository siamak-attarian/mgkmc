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

if __name__ == "__main__":
    test_kmc_finite_strain_2d()
    test_kmc_finite_strain_3d()
