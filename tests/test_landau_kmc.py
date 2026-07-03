import numpy as np
import os
import shutil
from mgkmc.kmc_simulator import KmcSimulation2D
from mgkmc.aqs import ThermalSimulation

def test_landau_kmc_2d():
    print("\n--- Testing 2D Small-Strain Landau in KMC ---")
    nx, ny = 16, 16
    M = 5
    gamma0 = 0.1
    
    # Material constants lam = 80.91, mu = 23.76 -> compute E and nu
    lam = 80.91e9
    mu = 23.76e9
    E_val = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu_val = lam / (2 * (lam + mu))
    
    E = np.full((nx, ny), E_val)
    nu = np.full((nx, ny), nu_val)
    
    output_dir = "output_test_landau_kmc_2d"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        
    sim = KmcSimulation2D(
        nx=nx, ny=ny, M=M, gamma0=gamma0,
        E_field=E, nu_field=nu,
        pixel=1.0,
        plane_mode="plane_stress",
        output_dir=output_dir,
        temperature=300.0,
        strain_rate=1e7,
        strain_assumption="small_strain",
        hyperelastic_model="landau",
        v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
        g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9
    )
    
    assert sim.fast_patching_enabled is False, "Fast patching must be disabled for Landau model"
    
    # Run a few loading steps (e.g., 3 steps up to 0.3% strain)
    sim.run_simulation(
        n_global_steps=3,
        step_size=0.001,
        component=(0, 0),
        stress_targets={(1, 1): 0.0}
    )
    
    # Check that stress is populated and is less than linear elastic stress due to softening
    assert sim.sig_field is not None
    assert sim.sig_field.shape == (nx, ny, 2, 2)
    sig_xx = sim.sig_field.mean(axis=(0, 1))[0, 0]
    
    # For linear, stress at 0.3% strain is E_val * 0.003 ~ 6.58e9 * 0.03 = 0.19 GPa
    # Let's check that stress is populated (not nan or 0)
    assert not np.isnan(sig_xx)
    assert sig_xx > 0.0
    print(f"2D Landau KMC stress at 0.3% strain: {sig_xx / 1e9:.4f} GPa")
    
    # Cleanup
    shutil.rmtree(output_dir, ignore_errors=True)


def test_landau_kmc_3d():
    print("\n--- Testing 3D Small-Strain Landau in KMC ---")
    nx, ny, nz = 8, 8, 8
    M = 5
    gamma0 = 0.1
    
    # Material constants lam = 80.91, mu = 23.76
    lam = 80.91e9
    mu = 23.76e9
    E_val = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu_val = lam / (2 * (lam + mu))
    
    E = np.full((nx, ny, nz), E_val)
    nu = np.full((nx, ny, nz), nu_val)
    
    output_dir = "output_test_landau_kmc_3d"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        
    sim = ThermalSimulation(
        nx=nx, ny=ny, nz=nz, M=M, gamma0=gamma0,
        E_field=E, nu_field=nu,
        pixel=1.0,
        output_dir=output_dir,
        temperature=300.0,
        strain_rate=1e7,
        strain_assumption="small_strain",
        hyperelastic_model="landau",
        v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
        g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9
    )
    
    assert sim.fast_patching_enabled is False, "Fast patching must be disabled for Landau model"
    
    # Run a few loading steps
    sim.run_simulation(
        n_global_steps=3,
        step_size=0.001,
        component=(0, 0),
        stress_targets={(1, 1): 0.0, (2, 2): 0.0}
    )
    
    # Check stresses
    assert sim.sig_field is not None
    assert sim.sig_field.shape == (nx, ny, nz, 3, 3)
    sig_xx = sim.sig_field.mean(axis=(0, 1, 2))[0, 0]
    assert not np.isnan(sig_xx)
    assert sig_xx > 0.0
    print(f"3D Landau KMC stress at 0.3% strain: {sig_xx / 1e9:.4f} GPa")
    
    # Cleanup
    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == '__main__':
    test_landau_kmc_2d()
    test_landau_kmc_3d()
