# examples/plane_stress_z_demo.py
"""
Plane stress in Z, Y fixed demonstration with a homogeneous material.

- Material: E = 70 GPa, ν = 0.30 (uniform fields)
- Geometry: 32 × 32 × 1 voxels
- Loading: 2 % macroscopic strain in the x‑direction, with y fixed (eps_yy=0) 
  and z stress‑free (sigma_zz=0).

The script runs the spectral solver, visualises the fields, and prints a
comparison between the numerical result and the analytical solution.

Run with:
    python examples/plane_stress_z_demo.py
"""

import numpy as np
from mgkmc import (
    generate_correlated_field,
    run_simulation,
    plot_fields,
    get_plane_stress_z_fixed_y, # Use get_plane_stress_z_fixed_y for Case 2
)
import matplotlib.pyplot as plt


def main():
    # ------------------------------------------------------------------
    # 1. Homogeneous material fields
    # ------------------------------------------------------------------
    nx, ny, nz = 32, 32, 1
    pixel = 0.5
    seed = 1
    np.random.seed(seed)

    # Uniform elastic modulus and Poisson ratio fields
    E_val = 70e9
    nu_val = 0.30
    E = E_val * np.ones((nx, ny, nz))
    nu = nu_val * np.ones((nx, ny, nz))

    # ------------------------------------------------------------------
    # 2. Apply 2 % macroscopic strain in x, y fixed, z stress-free
    # ------------------------------------------------------------------
    eps_target = 0.02  # 2 %

    # --- Analytical Solution Calculation ---

    # 1. Calculate the required strain component epsilon_zz (from helper logic)
    eps_zz_ana = -nu_val / (1 - nu_val) * eps_target
    
    # 2. Setup Analytical Strain Tensor
    eps_ana = np.zeros((3, 3))
    eps_ana[0, 0] = eps_target
    eps_ana[1, 1] = 0.0 # Fixed
    eps_ana[2, 2] = eps_zz_ana # Calculated

    # 3. Setup Analytical Stress Tensor (Plane Stress in Z, Plane Strain in Y)
    # This is equivalent to plane stress in the X-Y plane (sigma_z=0), 
    # but with epsilon_y=0 instead of a free edge.
    # The stress is calculated using the 2D plane stress compliance/stiffness:
    # sigma_xx = (E / (1 - nu^2)) * (eps_xx + nu*eps_yy) 
    # For eps_yy=0, the formula simplifies.
    
    E_prime = E_val / (1 - nu_val**2) # Effective Young's Modulus for plane stress
    nu_prime = nu_val * E_prime / E_val # Effective Poisson's ratio for plane stress (nu_val)
    
    sig_ana = np.zeros((3, 3))
    
    # Stress_xx: sigma_xx = E_prime * eps_xx
    sig_ana[0, 0] = E_prime * eps_target
    
    # Stress_yy: sigma_yy = nu_prime * sigma_xx = nu_val * E_prime * eps_xx
    sig_ana[1, 1] = nu_val * E_prime * eps_target
    
    # Stress_zz is zero (Plane Stress condition)
    sig_ana[2, 2] = 0.0 

    # --- Run Simulation ---
    
    # Use the helper for Case 2
    epsM, sigM, eps_list, sig_list = run_simulation(
        E,
        nu,
        loading_func=get_plane_stress_z_fixed_y,
        loading_params={
            "eps_xx": eps_target,
            "E": E_val,
            "nu": nu_val,
        },
        n_steps=1,  # single step is enough for a linear problem
        pixel=pixel,
        max_iter=200,
    )

    # Final fields (only one step, so list length = 1)
    eps = eps_list[-1]
    sig = sig_list[-1]

    # ------------------------------------------------------------------
    # 3. Visualise fields with the generic helper
    # ------------------------------------------------------------------
    plot_fields(E, nu, eps, sig, title="Plane Stress in Z, Y fixed demo")
    

    # ------------------------------------------------------------------
    # 4. Compare macroscopic stress with analytical solution
    # ------------------------------------------------------------------
    # Numerical macroscopic stress/strain at final step
    sig_num = sigM[-1]
    eps_num = epsM[-1]

    print("--- Plane Stress in Z, Y Fixed Comparison ---")
    print(f"Applied ε_x = {eps_target:.4%}, ε_y = 0.0%, σ_z = 0.0 Pa")
    
    components = ["xx", "yy", "zz", "xy", "xz", "yz"]
    indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    
    print("\nStress Comparison (Pa):")
    print(f"{'Comp':<5} {'Analytical':>15} {'Numerical':>15} {'Error':>15}")
    print("-" * 54)
    for comp, (i, j) in zip(components, indices):
        ana = sig_ana[i, j]
        num = sig_num[i, j]
        err = abs(num - ana)
        print(f"σ_{comp:<3} {ana:15,.0f} {num:15,.0f} {err:15,.0f}")

    print("\nStrain Comparison:")
    print(f"{'Comp':<5} {'Analytical':>15} {'Numerical':>15} {'Error':>15}")
    print("-" * 54)
    for comp, (i, j) in zip(components, indices):
        ana = eps_ana[i, j]
        num = eps_num[i, j]
        err = abs(num - ana)
        print(f"ε_{comp:<3} {ana:15.6f} {num:15.6f} {err:15.6e}")

    # ------------------------------------------------------------------
    # 5. Simple stress‑strain curve (single point) for illustration
    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 4))
    plt.plot([0, eps_target], [0, sig_num[0, 0]], "o-", label="Numerical σ_xx")
    plt.plot([0, eps_target], [0, sig_ana[0, 0]], "x--", label="Analytical σ_xx")
    plt.xlabel("Macroscopic strain ε_x")
    plt.ylabel("Stress σ_xx (Pa)")
    plt.title("Stress-strain response (Plane Stress in Z, Y Fixed)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()