# examples/plane_stress_demo.py
"""
Plane‑stress demonstration with a homogeneous material.

- Material: E = 70 GPa, ν = 0.30 (uniform fields)
- Geometry: 32 × 32 × 1 voxels (effectively 2‑D)
- Loading: 2 % macroscopic strain in the x‑direction, y‑direction fixed (ε_y = 0),
  out‑of‑plane stress σ_z = 0 (plane‑stress condition).

The script runs the spectral solver, visualises the fields, and prints a
comparison between the numerical result and the analytical solution.

Run with:
    python examples/plane_stress_demo.py
"""

import numpy as np
from mgkmc import (
    generate_correlated_field,
    run_simulation,
    plot_fields,
    get_plane_stress_z_fixed_y,
)
import matplotlib.pyplot as plt


def analytical_plane_stress(eps_x: float, E: float, nu: float):
    """Return analytical σ_xx, σ_yy for plane‑stress with ε_x prescribed and ε_y = 0.
    The result follows:
        σ_xx = E * eps_x / (1 - nu**2)
        σ_yy = nu * σ_xx
    """
    sigma_x = E * eps_x / (1 - nu ** 2)
    sigma_y = nu * sigma_x
    return sigma_x, sigma_y


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
    # 2. Apply 2 % macroscopic strain in x, y fixed (plane‑stress)
    # ------------------------------------------------------------------
    eps_target = 0.02  # 2 %

    # Calculate required stresses to achieve eps_x with eps_y=0 in plane stress
    sigma_x_target, sigma_y_target = analytical_plane_stress(eps_target, E_val, nu_val)

    # Use the new helper for Case 2: Plane stress in Z, Y fixed, X input
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
    plot_fields(E, nu, eps, sig, title="Plane‑stress demo")

    # ------------------------------------------------------------------
    # 4. Compare macroscopic stress with analytical solution
    # ------------------------------------------------------------------
    # Numerical macroscopic stress is returned as sigM (array of shape (n_steps+1, 3, 3))
    # We want the values at the final step (-1)
    sigma_x_num = float(sigM[-1, 0, 0])
    sigma_y_num = float(sigM[-1, 1, 1])

    print("--- Plane‑stress comparison (units: Pa) ---")
    print(f"Applied ε_x = {eps_target:.4%}")
    print(f"Analytical σ_xx = {sigma_x_target:,.0f} Pa")
    print(f"Numerical  σ_xx = {sigma_x_num:,.0f} Pa")
    print(f"Analytical σ_yy = {sigma_y_target:,.0f} Pa")
    print(f"Numerical  σ_yy = {sigma_y_num:,.0f} Pa")

    # ------------------------------------------------------------------
    # 5. Simple stress‑strain curve (single point) for illustration
    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 4))
    plt.plot([0, eps_target], [0, sigma_x_num], "o-", label="Numerical σ_xx")
    plt.plot([0, eps_target], [0, sigma_x_target], "x--", label="Analytical σ_xx")
    plt.xlabel("Macroscopic strain ε_x")
    plt.ylabel("Stress σ_xx (Pa)")
    plt.title("Stress‑strain response (plane stress)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
