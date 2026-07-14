import numpy as np
import os
import sys

# Ensure the local mgkmc package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mgkmc import KmcSimulation2D
from mgkmc.linear_elastic_simulator import spectral_solver_rose_2d
from mgkmc.elasticity import compute_lame_2d

def test_kmc_run_plane_strain():
    print("--- Test 1: KmcSimulation2D (plane_strain) ---")
    nx, ny = 16, 16
    M = 4
    gamma0 = 0.1
    E = np.full((nx, ny), 70.0e9)
    nu = np.full((nx, ny), 0.3)
    
    sim = KmcSimulation2D(
        nx=nx, ny=ny, M=M, gamma0=gamma0,
        E_field=E, nu_field=nu,
        strain_assumption="small_strain",
        hyperelastic_model="rose",
        eta=5.0, mu_floor_fraction=0.1,
        plane_mode="plane_strain",
        output_dir="output_test_rose_kmc_strain"
    )
    
    print("Running a single elastic step...")
    eps_macro = np.zeros((2, 2))
    eps_macro[0, 0] = 0.01  # 1% strain along xx
    sig_mean = sim.elastic_run(eps_macro)
    print(f"Elastic run successful. Mean stress (MPa):\n{sig_mean / 1e6}")
    
    print("Running short simulation...")
    sim.run_simulation(
        n_global_steps=5,
        step_size=1e-4,
        component=(0, 0),
        enable_console_log=True
    )
    print("Plane strain test complete!\n")

def test_kmc_run_plane_stress():
    print("--- Test 2: KmcSimulation2D (plane_stress) ---")
    nx, ny = 16, 16
    M = 4
    gamma0 = 0.1
    E = np.full((nx, ny), 70.0e9)
    nu = np.full((nx, ny), 0.3)
    
    sim = KmcSimulation2D(
        nx=nx, ny=ny, M=M, gamma0=gamma0,
        E_field=E, nu_field=nu,
        strain_assumption="small_strain",
        hyperelastic_model="rose",
        eta=5.0, mu_floor_fraction=0.1,
        plane_mode="plane_stress",
        output_dir="output_test_rose_kmc_stress"
    )
    
    print("Running a single elastic step...")
    eps_macro = np.zeros((2, 2))
    eps_macro[0, 0] = 0.01  # 1% strain along xx
    sig_mean = sim.elastic_run(eps_macro)
    print(f"Elastic run successful. Mean stress (MPa):\n{sig_mean / 1e6}")
    
    print("Running short simulation...")
    sim.run_simulation(
        n_global_steps=5,
        step_size=1e-4,
        component=(0, 0),
        enable_console_log=True
    )
    print("Plane stress test complete!\n")

def test_analytical_comparison():
    print("--- Test 3: Analytical Uniaxial Tension Plane Stress Comparison ---")
    
    # Constants
    E = 67.9e9  # Pa (Young's modulus from guide section 5)
    nu = 0.285   # Poisson ratio
    eta = 5.0    # Softening parameter
    mu_floor_fraction = 0.1
    
    # Derived parameters
    lam, mu = compute_lame_2d(E, nu, plane_mode="plane_strain") # 3D Lamé parameters
    K = lam + (2.0 / 3.0) * mu
    G0 = mu
    
    print(f"Material Constants: G0 = {G0/1e9:.2f} GPa, K = {K/1e9:.2f} GPa, eta = {eta}")
    
    # Pick a point on the analytical curve
    # eps_eq = 1 / eta = 0.2 (peak of local curve before floor)
    eps_eq = 0.15
    G_sec = G0 * np.exp(-eta * eps_eq)
    
    sigma_xx_analytical = 3.0 * G_sec * eps_eq
    eps_xx_analytical = eps_eq + sigma_xx_analytical / (9.0 * K)
    
    print(f"Analytical Uniaxial tension (plane-stress):")
    print(f"  Chosen eps_eq = {eps_eq:.4f}")
    print(f"  Required eps_xx = {eps_xx_analytical:.6f}")
    print(f"  Expected sigma_xx = {sigma_xx_analytical/1e9:.6f} GPa")
    
    # Run the spectral solver for a homogeneous medium
    # Prescribe eps_xx = eps_xx_analytical, and free transverse stress sig_yy = 0
    # On a homogeneous 2x2 grid, the fields will be perfectly uniform.
    nx, ny = 2, 2
    lam_field = np.full((nx, ny), lam)
    mu_field = np.full((nx, ny), mu)
    
    # Set up solver target
    target_strain_mask = np.array([[True, False],
                                   [False, False]])
    target_values = np.array([[eps_xx_analytical, 0.0],
                              [0.0, 0.0]]) # component (1,1) is stress-free (sig_yy = 0)
    
    from mgkmc.linear_elastic_simulator import rose_elastic_simulation_2d
    
    eps_macro, sig_macro, _, _ = rose_elastic_simulation_2d(
        lam_field, mu_field, eta, mu_floor_fraction=mu_floor_fraction,
        target_strain_mask=target_strain_mask,
        target_values=target_values,
        n_steps=1, # single step is fine
        pixel=1.0,
        plane_mode="plane_stress",
        enable_console=False
    )
    
    # The final step is at index 1
    sig_xx_solver = sig_macro[1, 0, 0]
    eps_xx_solver = eps_macro[1, 0, 0]
    eps_yy_solver = eps_macro[1, 1, 1]
    sig_yy_solver = sig_macro[1, 1, 1]
    
    print(f"Solver Results:")
    print(f"  eps_xx = {eps_xx_solver:.6f} (target: {eps_xx_analytical:.6f})")
    print(f"  eps_yy = {eps_yy_solver:.6f}")
    print(f"  sigma_xx = {sig_xx_solver/1e9:.6f} GPa (expected: {sigma_xx_analytical/1e9:.6f} GPa)")
    print(f"  sigma_yy = {sig_yy_solver/1e9:.6f} GPa (expected: 0.0 GPa)")
    
    diff_sig = np.abs(sig_xx_solver - sigma_xx_analytical) / sigma_xx_analytical
    print(f"Relative difference: {diff_sig:.3e}")
    assert diff_sig < 1e-4, f"Discrepancy too large! Diff: {diff_sig}"
    print("Analytical comparison test successful!")

if __name__ == "__main__":
    test_kmc_run_plane_strain()
    test_kmc_run_plane_stress()
    test_analytical_comparison()
