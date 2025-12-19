"""
Comparison of Pure Strain Control vs Mixed Boundary Control (Uniaxial Tension).
System: 32x32x1
"""
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mgkmc import AthermalSimulation
from mgkmc.elasticity import get_uniaxial_stress_x

def main():
    print("=" * 60)
    print("AQS DEMO 3: Pure Strain vs Mixed Control")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. Setup Parameters
    # ----------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    
    nx, ny, nz = 32, 32, 1
    pixel = 0.7  # nm
    M = 20
    gamma0 = 0.14
    
    # Material properties
    E_mean = 70.0  # GPa
    nu_mean = 0.3
    
    # # Heterogeneous fields
    # E = np.random.normal(E_mean, E_mean*0.1, (nx, ny, nz)) # GPa
    # nu = np.random.normal(nu_mean, 0.05, (nx, ny, nz))
    # nu = np.clip(nu, 0.0, 0.49)
    
    # Material properties (Homogeneous for this demo)
    # E in GPa, nu is dimensionless
    E = np.full((nx, ny, nz), 70.0)  # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    # Simulation settings
    N_STEPS = 80
    TARGET_STRAIN = 0.08 # 4% strain
    STRAIN_RATE = TARGET_STRAIN / N_STEPS
    
    OUTPUT_DIR_PURE = "aqs_demo3_pure"
    OUTPUT_DIR_MIXED = "aqs_demo3_mixed"
    
    # cleanup
    for d in [OUTPUT_DIR_PURE, OUTPUT_DIR_MIXED]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # ----------------------------------------------------
    # 2. Run Pure Strain Control (Baseline)
    # ----------------------------------------------------
    print("\n[1/2] Running Pure Strain Control...")
    sim_pure = AthermalSimulation(
        nx, ny, nz, M, gamma0, E, nu, pixel,
        output_dir=OUTPUT_DIR_PURE,
        softening_enabled=True,
        softening_params={"jp": 11, "jt": 33},
        softening_scheme="directional"
    )
    
    try:
        sim_pure.run(
            n_global_steps=N_STEPS,
            vtk_mode=None, # Disable VTK for speed
            loading_func=get_uniaxial_stress_x,
            loading_params={
                "eps_xx": TARGET_STRAIN,
                "E": E_mean, # GPa
                "nu": nu_mean
            }
        )
    except RuntimeError as e:
        print(f"Pure simulation stopped early: {e}")

    # ----------------------------------------------------
    # 3. Run Mixed Control (New Method)
    # ----------------------------------------------------
    print("\n[2/2] Running Mixed Control...")
    
    # Re-seed for identical initial conditions
    np.random.seed(SEED)
    # # Heterogeneous fields
    # E = np.random.normal(E_mean, E_mean*0.1, (nx, ny, nz)) # GPa
    # nu = np.random.normal(nu_mean, 0.05, (nx, ny, nz))
    # nu = np.clip(nu, 0.0, 0.49)
    
    # Material properties (Homogeneous for this demo)
    # E in GPa, nu is dimensionless
    E = np.full((nx, ny, nz), 70.0)  # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    sim_mixed = AthermalSimulation(
        nx, ny, nz, M, gamma0, E, nu, pixel,
        output_dir=OUTPUT_DIR_MIXED,
        softening_enabled=True,
        softening_params={"jp": 11, "jt": 33},
        softening_scheme="directional"
    )
    
    try:
        sim_mixed.run_mixed(
            n_global_steps=N_STEPS,
            strain_rate=STRAIN_RATE,
            component=(0,0), # Drive eps_xx
            stress_targets={(1,1): 0.0, (2,2): 0.0}, # Target sig_yy=0, sig_zz=0
            mixed_tol=1e6, # 1 MPa tolerance
            mixed_max_iter=20,
            vtk_mode=None
        )
    except RuntimeError as e:
        print(f"Mixed simulation stopped early: {e}")

    # ----------------------------------------------------
    # 4. Analysis & Plotting
    # ----------------------------------------------------
    print("\nGenerating Comparison Plots...")
    
    hist_pure = np.array(sim_pure.history_global)
    hist_mixed = np.array(sim_mixed.history_global)
    
    # Extract data (history_global stores: eps_xx, sig_xx_GPa)
    # But wait! history_global stores specifically (eps_macro_curr[0,0], sig_macro_curr[0,0]/1e9)
    # We want Transverse Stresses!
    # history_global ONLY accumulated (eps_xx, sig_xx). 
    # We need to extract full history from the logs to see Sigma_YY and Sigma_ZZ.
    
    from mgkmc.analysis import extract_history
    
    data_pure = extract_history(OUTPUT_DIR_PURE)['global']
    data_mixed = extract_history(OUTPUT_DIR_MIXED)['global']
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Helper to safe extract
    def plot_data(ax, data, style, label, col_y):
        if data is not None and data.ndim == 2 and data.shape[0] > 0:
            eps = data[:, 1]
            sig = data[:, col_y] / 1e9
            ax.plot(eps*100, sig, style, label=label)
            return True
        return False

    # 1. Axial Response (Sig_xx is col 7)
    ax = axes[0]
    has_pure = plot_data(ax, data_pure, 'b-', 'Pure Strain Control', 7)
    has_mixed = plot_data(ax, data_mixed, 'r--', 'Mixed Control (Relaxed)', 7)
    
    if not has_pure:
        ax.text(0.5, 0.5, "Pure Sim Failed Early", transform=ax.transAxes, ha='center', color='blue')

    ax.set_xlabel(r'Strain $\epsilon_{xx}$ (%)')
    ax.set_ylabel(r'Stress $\sigma_{xx}$ (GPa)')
    ax.set_title('Axial Stress-Strain Response')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Transverse Stresses (yy=8, zz=9)
    ax = axes[1]
    if data_pure is not None and data_pure.ndim == 2 and data_pure.shape[0] > 0:
        eps = data_pure[:, 1] * 100
        ax.plot(eps, data_pure[:, 8]/1e9, 'b-', label=r'Pure: $\sigma_{yy}$')
        ax.plot(eps, data_pure[:, 9]/1e9, 'b:', label=r'Pure: $\sigma_{zz}$', alpha=0.6)
    
    if data_mixed is not None and data_mixed.ndim == 2 and data_mixed.shape[0] > 0:
        eps = data_mixed[:, 1] * 100
        ax.plot(eps, data_mixed[:, 8]/1e9, 'r-', label=r'Mixed: $\sigma_{yy}$')
        ax.plot(eps, data_mixed[:, 9]/1e9, 'r:', label=r'Mixed: $\sigma_{zz}$', alpha=0.6)
    
    ax.set_xlabel(r'Strain $\epsilon_{xx}$ (%)')
    ax.set_ylabel('Transverse Stress (GPa)')
    ax.set_title('Lateral Stress Accumulation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ax.set_ylim(-0.1, 0.1) # Zoom in around zero if possible
    
    plt.tight_layout()
    plt.savefig("aqs_demo3_comparison.png", dpi=150)
    print("Comparison plot saved to 'aqs_demo3_comparison.png'")

if __name__ == "__main__":
    main()
