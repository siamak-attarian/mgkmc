# examples/uniaxial_strain_demo.py
"""
Uniaxial strain demonstration with a homogeneous material.

- Material: E = 70 GPa, ν = 0.30 (uniform fields)
- Geometry: 32 × 32 × 1 voxels
- Loading: 2 % macroscopic strain in the x‑direction, with y and z fixed
  (uniaxial strain condition: eps_yy=eps_zz=0).

The script runs the spectral solver, visualises the fields, and prints a
comparison between the numerical result and the analytical solution.

Run with:
    python examples/uniaxial_strain_demo.py
"""

import numpy as np
from mgkmc import (
    generate_correlated_field,
    run_simulation,
    plot_fields,
    get_strain_tensor,  # Use get_strain_tensor for Case 1
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
    # 2. Apply 2 % macroscopic strain in x, y and z fixed
    # ------------------------------------------------------------------
    eps_target = 0.02  # 2 %

    # Calculate analytical solution for comparison (Uniaxial Strain)
    # Stress-Strain relation for Uniaxial Strain:
    # C_11 = E(1-nu) / ((1+nu)(1-2nu))
    # C_12 = E*nu / ((1+nu)(1-2nu))
    
    # Calculate Lamé constants (optional, but helpful for C_ijkl)
    # Lame_mu = E / (2 * (1 + nu))
    # Lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    # Stiffness matrix component C_11 = Lame_lambda + 2*Lame_mu
    C11 = E_val * (1 - nu_val) / ((1 + nu_val) * (1 - 2 * nu_val))
    # Stiffness matrix component C_12 = Lame_lambda
    C12 = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
    
    sig_ana = np.zeros((3, 3))
    # Analytical stress components
    sig_ana[0, 0] = C11 * eps_target
    sig_ana[1, 1] = sig_ana[2, 2] = C12 * eps_target
    
    # Analytical strain tensor (given input)
    eps_ana = np.zeros((3, 3))
    eps_ana[0, 0] = eps_target
    eps_ana[1, 1] = 0.0
    eps_ana[2, 2] = 0.0

    # Use the helper for Case 1: All strains given directly
    # The input strain tensor is the target macroscopic strain
    input_eps_tensor = np.copy(eps_ana) 

    epsM, sigM, eps_list, sig_list = run_simulation(
        E,
        nu,
        loading_func=get_strain_tensor,
        loading_params={
            "eps_tensor": input_eps_tensor,
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
    plot_fields(E, nu, eps, sig, title="Uniaxial strain demo")

    # ------------------------------------------------------------------
    # 4. Compare macroscopic stress with analytical solution
    # ------------------------------------------------------------------
    # Numerical macroscopic stress/strain at final step
    sig_num = sigM[-1]
    eps_num = epsM[-1]

    print("--- Uniaxial strain comparison ---")
    print(f"Applied ε_x = {eps_target:.4%}, ε_y = 0.0%, ε_z = 0.0%")
    
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
    # 5. Simple stress-strain curve (single point) for illustration
    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 4))
    plt.plot([0, eps_target], [0, sig_num[0, 0]], "o-", label="Numerical σ_xx")
    plt.plot([0, eps_target], [0, sig_ana[0, 0]], "x--", label="Analytical σ_xx")
    plt.xlabel("Macroscopic strain ε_x")
    plt.ylabel("Stress σ_xx (Pa)")
    plt.title("Stress-strain response (uniaxial strain)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()