# examples/plane_strain_inclusion_demo.py
"""
Plane Strain (eps_yy=eps_zz=0) with a circular inclusion demonstration.

- Loading: 2 % macroscopic strain in the x‑direction (eps_xx=0.02, eps_yy=eps_zz=0).
- System: 128 × 128 × 1 voxels.
- Material: 
    - Matrix: E_mat = 70 GPa, ν_mat = 0.30
    - Inclusion: E_inc = 90 GPa, ν_inc = 0.20
    - Inclusion is a circle of radius 10 in the center.

The script runs the spectral solver, visualises the fields, and prints a
comparison between the numerical result and the Mori-Tanaka estimate
for the composite's macroscopic stress.

Run with:
    python examples/plane_strain_inclusion_demo.py
"""

import numpy as np
from mgkmc import (
    generate_correlated_field,
    run_simulation,
    plot_fields,
    get_strain_tensor, # Use get_strain_tensor for Case 1
    export_simulation_vtk,
)
import matplotlib.pyplot as plt


def main():
    # ------------------------------------------------------------------
    # 1. Heterogeneous material fields (Matrix + Inclusion)
    # ------------------------------------------------------------------
    nx, ny, nz = 32,32,1
    pixel = 0.5
    seed = 1
    np.random.seed(seed)

    # Material Properties
    E_mat = 70e9
    nu_mat = 0.30
    E_inc = 90e9
    nu_inc = 0.20

    # Initialize fields with Matrix properties
    E = E_mat * np.ones((nx, ny, nz))
    nu = nu_mat * np.ones((nx, ny, nz))
    
    # Create circular inclusion mask
    center_x, center_y = nx // 2, ny // 2
    radius = 5
    
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    inclusion_mask = R <= radius

    # Assign Inclusion properties
    E[inclusion_mask] = E_inc
    nu[inclusion_mask] = nu_inc
    
    # Calculate Volume Fraction of Inclusion
    Vf = inclusion_mask.sum() / (nx * ny * nz)
    print(f"Inclusion volume fraction (Vf): {Vf:.4f}")

    # ------------------------------------------------------------------
    # 2. Apply 2 % macroscopic Plane Strain (eps_yy=eps_zz=0)
    # ------------------------------------------------------------------
    eps_target = 0.02  # 2 %

    # Analytical (Mori-Tanaka) Solution for Comparison
    # For a heterogeneous material, we use the Mori-Tanaka estimate for 
    # the effective stiffness tensor (C_eff) and then calculate stress:
    # sigma_Mori_Tanaka = C_eff * eps_M
    
    # Due to complexity, we will calculate the simple volume average of 
    # the **Plane Strain** stiffness coefficient C11 for a rough estimate
    # of the macroscopic stress $\sigma_{xx}^M$.
    
    # C11 = E(1-nu) / ((1+nu)(1-2nu)) in 3D
    def C11(E_val, nu_val):
        return E_val * (1 - nu_val) / ((1 + nu_val) * (1 - 2 * nu_val))
    
    C11_mat = C11(E_mat, nu_mat)
    C11_inc = C11(E_inc, nu_inc)
    
    # Simple Volume Average Stiffness (C_Voigt) - The Voigt bound (upper bound)
    C11_vol_avg = Vf * C11_inc + (1 - Vf) * C11_mat
    
    # Analytical/Estimated Stress (using Voigt bound)
    sig_ana = np.zeros((3, 3))
    # We only estimate the $\sigma_{xx}$ component for simplicity
    sig_ana[0, 0] = C11_vol_avg * eps_target
    
    # Analytical Strain Tensor (Given input)
    eps_ana = np.zeros((3, 3))
    eps_ana[0, 0] = eps_target
    eps_ana[1, 1] = 0.0 # Fixed (Plane Strain)
    eps_ana[2, 2] = 0.0 # Fixed (Plane Strain)
    
    # Use the helper for Case 1: All strains given directly
    input_eps_tensor = np.copy(eps_ana) 

    epsM, sigM, eps_list, sig_list = run_simulation(
        E,
        nu,
        loading_func=get_strain_tensor,
        loading_params={
            "eps_tensor": input_eps_tensor,
        },
        n_steps=10,  # Multiple steps needed for heterogeneous fields
        pixel=pixel,
        max_iter=500,
        tol=1e-8 # Stricter tolerance for better results
    )

    # Final fields
    eps = eps_list[-1]
    sig = sig_list[-1]

    # ------------------------------------------------------------------
    # 3. Visualise fields 
    # ------------------------------------------------------------------
    plot_fields(E, nu, eps, sig, title="Plane Strain Inclusion Demo")
    

    # ------------------------------------------------------------------
    # 4. Compare macroscopic stress with estimate
    # ------------------------------------------------------------------
    # Numerical macroscopic stress/strain at final step
    sig_num = sigM[-1]
    eps_num = epsM[-1]

    print("--- Plane Strain Inclusion Comparison ---")
    print(f"Applied ε_x = {eps_target:.4%}, ε_y = 0.0%, ε_z = 0.0%")
    print(f"Matrix: E={E_mat/1e9:.0f} GPa, ν={nu_mat:.2f} | Inclusion: E={E_inc/1e9:.0f} GPa, ν={nu_inc:.2f}")
    
    components = ["xx", "yy", "zz", "xy", "xz", "yz"]
    indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    
    print("\nMacroscopic Stress Comparison (Pa):")
    print(f"{'Comp':<5} {'Voigt Est.':>15} {'Numerical':>15} {'Difference':>15}")
    print("-" * 54)
    # Note: Only Voigt estimate is calculated, no full Mori-Tanaka
    for comp, (i, j) in zip(components, indices):
        ana = sig_ana[i, j] if i==0 and j==0 else 0.0
        num = sig_num[i, j]
        diff = abs(num - ana)
        print(f"σ_{comp:<3} {ana:15,.0f} {num:15,.0f} {diff:15,.0f}")

    print("\nMacroscopic Strain Comparison (Should match input):")
    print(f"{'Comp':<5} {'Input':>15} {'Numerical':>15} {'Error':>15}")
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
    plt.plot([0, eps_target], [0, sig_num[0, 0]], "o-", label="Numerical $\\sigma_{xx}$")
    plt.plot([0, eps_target], [0, sig_ana[0, 0]], "x--", label="Voigt Bound Est. $\\sigma_{xx}$")
    plt.xlabel("Macroscopic strain $\\varepsilon_x$")
    plt.ylabel("Stress $\\sigma_{xx}$ (Pa)")
    plt.title("Stress-strain response (Plane Strain Composite)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------------
    # 6. Extract and Plot ALL Stress/Strain Profiles
    # ------------------------------------------------------------------
    
    print("\n--- Extracting Full Stress and Strain Profiles ---")
    
    nx, ny, nz = E.shape
    center_x = nx // 2
    center_y = ny // 2
    radius = 10 
    eps_target = eps_ana[0, 0] # 0.02
    
    # Define the components and their tensor indices
    components = {
        'xx': (0, 0),
        'yy': (1, 1),
        'zz': (2, 2),
    }

    # --- Data Extraction ---
    
    # Extract data along the X-Line (y=center_y, x-sweep)
    data_x_line = {}
    for comp, (i, j) in components.items():
        # Strain: eps[x, y=center_y, z=0, i, j]
        data_x_line[f'eps_{comp}'] = eps[:, center_y, 0, i, j]
        # Stress: sig[x, y=center_y, z=0, i, j]
        data_x_line[f'sig_{comp}'] = sig[:, center_y, 0, i, j]

    # Extract data along the Y-Line (x=center_x, y-sweep)
    data_y_line = {}
    for comp, (i, j) in components.items():
        # Strain: eps[x=center_x, y, z=0, i, j]
        data_y_line[f'eps_{comp}'] = eps[center_x, :, 0, i, j]
        # Stress: sig[x=center_x, y, z=0, i, j]
        data_y_line[f'sig_{comp}'] = sig[center_x, :, 0, i, j]

    x_coords = np.arange(nx)
    y_coords = np.arange(ny)

    # --- Plotting Functions ---

    def plot_profiles(coords, data, title_prefix, axis_label, axis_len):
        
        # --- Plot 1: Stress Profiles ---
        plt.figure(figsize=(10, 5))
        
        plt.plot(coords, data['sig_xx'], label='$\\sigma_{xx}$', linewidth=2)
        plt.plot(coords, data['sig_yy'], label='$\\sigma_{yy}$', linewidth=2)
        plt.plot(coords, data['sig_zz'], label='$\\sigma_{zz}$', linewidth=2)
        # print(data['sig_yy']- data['sig_zz'])
        
        # Highlight Inclusion Region
        plt.axvspan(axis_len // 2 - radius, axis_len // 2 + radius, color='r', alpha=0.1, label='Inclusion Region')
        plt.axvline(axis_len // 2 - radius, color='r', linestyle='--', linewidth=0.8)
        plt.axvline(axis_len // 2 + radius, color='r', linestyle='--', linewidth=0.8)
        
        plt.axhline(0, color='gray', linestyle=':', linewidth=0.5)
        
        plt.xlabel(axis_label)
        plt.ylabel("Stress (Pa)")
        plt.title(f"{title_prefix} Stress Profiles ($\\sigma_{{ii}}$)")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--')
        plt.show()

        # --- Plot 2: Strain Profiles ---
        plt.figure(figsize=(10, 5))
        
        plt.plot(coords, data['eps_xx'], label='$\\varepsilon_{xx}$', linewidth=2)
        plt.plot(coords, data['eps_yy'], label='$\\varepsilon_{yy}$', linewidth=2)
        plt.plot(coords, data['eps_zz'], label='$\\varepsilon_{zz}$', linewidth=2)
        # print(data['eps_yy']- data['eps_zz'])
        
        # Reference lines for far-field strain
        plt.axhline(eps_target, color='g', linestyle=':', label='$\\varepsilon^M_{xx}$ Target')
        plt.axhline(0.0, color='b', linestyle=':', label='$\\varepsilon^M_{yy}, \\varepsilon^M_{zz}$ Target')
        
        # Highlight Inclusion Region
        plt.axvspan(axis_len // 2 - radius, axis_len // 2 + radius, color='r', alpha=0.1, label='Inclusion Region')
        plt.axvline(axis_len // 2 - radius, color='r', linestyle='--', linewidth=0.8)
        plt.axvline(axis_len // 2 + radius, color='r', linestyle='--', linewidth=0.8)

        plt.xlabel(axis_label)
        plt.ylabel("Strain")
        plt.title(f"{title_prefix} Strain Profiles ($\\varepsilon_{{ii}}$)")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--')
        plt.show()

    # --- Execute Plotting ---
    
    # Plot X-Line Profiles
    plot_profiles(x_coords, data_x_line, "X-Line (y=64)", "X-coordinate (voxel index)", nx)

    # Plot Y-Line Profiles
    plot_profiles(y_coords, data_y_line, "Y-Line (x=64)", "Y-coordinate (voxel index)", ny)
    # ------------------------------------------------------------------
    export_simulation_vtk(eps_list, sig_list, E, nu, pixel, steps="last", prefix="inclusion_")
if __name__ == "__main__":
    main()