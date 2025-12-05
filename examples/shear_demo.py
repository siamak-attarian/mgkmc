# examples/shear_demo.py
"""
Pure shear strain in XY demonstration with a homogeneous material.

- Material: E = 70 GPa, ν = 0.30 (uniform fields)
- Geometry: 32 × 32 × 1 voxels
- Loading: 0.02 macroscopic shear strain in the XY-direction (tensorial shear strain).

The script runs the spectral solver, visualises the fields, and prints a
comparison between the numerical result and the analytical solution.

Run with:
    python examples/shear_demo.py
"""

import numpy as np
from mgkmc import (
    generate_correlated_field,
    run_simulation,
    plot_fields,
    get_pure_shear_xy, # Use get_pure_shear_xy for Case 4
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
    # 2. Apply 0.02 pure shear strain in XY
    # ------------------------------------------------------------------
    # Note: eps_target is the tensorial shear strain (gamma_xy / 2)
    eps_target = 0.02 

    # --- Analytical Solution Calculation ---

    # 1. Calculate Shear Modulus (G)
    G_val = E_val / (2 * (1 + nu_val))
    
    # 2. Setup Analytical Strain Tensor (Given input)
    eps_ana = np.zeros((3, 3))
    eps_ana[0, 1] = eps_ana[1, 0] = eps_target # tensorial shear strain
    
    # 3. Setup Analytical Stress Tensor (Only shear stress is non-zero)
    sig_ana = np.zeros((3, 3))
    # Stress_xy = 2 * G * eps_xy
    sig_ana[0, 1] = sig_ana[1, 0] = 2 * G_val * eps_target

    # --- Run Simulation ---
    
    # Use the helper for Case 4
    epsM, sigM, eps_list, sig_list = run_simulation(
        E,
        nu,
        loading_func=get_pure_shear_xy,
        loading_params={
            "eps_xy": eps_target,
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
    plot_fields(E, nu, eps, sig, title="Pure Shear Strain in XY demo")
    

    # ------------------------------------------------------------------
    # 4. Compare macroscopic stress with analytical solution
    # ------------------------------------------------------------------
    # Numerical macroscopic stress/strain at final step
    sig_num = sigM[-1]
    eps_num = epsM[-1]

    print("--- Pure Shear Strain in XY Comparison ---")
    print(f"Applied ε_xy (tensorial) = {eps_target:.4f}")
    
    components = ["xx", "yy", "zz", "xy", "xz", "yz"]
    indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    
    print("\nStress Comparison (Pa):")
    print(f"{'Comp':<5} {'Analytical':>15} {'Numerical':>15} {'Error':>15}")
    print("-" * 54)
    for comp, (i, j) in zip(components, indices):
        ana = sig_ana[i, j]
        num = sig_num[i, j]
        err = abs(num - ana)
        # Only print the upper-triangular component for shear stress comparison
        if i <= j:
            print(f"σ_{comp:<3} {ana:15,.0f} {num:15,.0f} {err:15,.0f}")


    print("\nStrain Comparison:")
    print(f"{'Comp':<5} {'Analytical':>15} {'Numerical':>15} {'Error':>15}")
    print("-" * 54)
    for comp, (i, j) in zip(components, indices):
        ana = eps_ana[i, j]
        num = eps_num[i, j]
        err = abs(num - ana)
        if i <= j:
            print(f"ε_{comp:<3} {ana:15.6f} {num:15.6f} {err:15.6e}")

    # ------------------------------------------------------------------
    # 5. Simple stress‑strain curve (single point) for illustration
    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 4))
    plt.plot([0, 2*eps_target], [0, sig_num[0, 1]], "o-", label="Numerical τ_xy")
    plt.plot([0, 2*eps_target], [0, sig_ana[0, 1]], "x--", label="Analytical τ_xy")
    plt.xlabel("Macroscopic Engineering Shear Strain γ_xy")
    plt.ylabel("Shear Stress τ_xy (Pa)")
    plt.title("Stress-strain response (Pure Shear Strain)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()