import numpy as np
import os
import shutil
from mgkmc import ThermalSimulation

def main():
    # ----------------------------------------------------
    # 1. Setup Parameters
    # ----------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    print(f"Random seed set to {SEED}")

    nx, ny, nz = 32,32, 1
    pixel = 0.7
    M = 20
    gamma0 = 0.14
    
    # Material properties (Homogeneous for this demo)
    # E in GPa, nu is dimensionless
    E = np.full((nx, ny, nz), 70.0)  # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    # ----------------------------------------------------
    # 2. Configuration for Experimentation
    # ----------------------------------------------------
    # Change these values to test different physics!
    
    # Softening Configuration
    ENABLE_SOFTENING = True 
    SOFTENING_SCHEME = "directional"  # Options: "isotropic", "directional"
    SOFTENING_PARAMS = {"jp": 15, "jt": 45}
    SOFTENING_CAP = 0.51 # Limit on g_p. Set to None for unlimited.
    
    # Simulation Control
    DEBUG_FIRST_FLIP = False     # Set True to see details of the first instability
    OUTPUT_DIR = "aqs_demo_output_directional_15_45_cap"

    # ----------------------------------------------------
    # 3. Custom Barrier Generator
    # ----------------------------------------------------
    def my_barrier_generator(n_modes):
        # Example: Normal distribution with mean=2.0, std=0.6
        random_barriers = np.random.normal(loc=2.0, scale=0.6, size=n_modes)
        min_barrier = 0.5
        clipped_barriers = np.clip(random_barriers, a_min=min_barrier, a_max=None)
        return clipped_barriers
    
    # ----------------------------------------------------
    # 4. Initialize Simulation
    # ----------------------------------------------------
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    sim = ThermalSimulation(
        nx, ny, nz,
        M=M, 
        gamma0=gamma0,
        E_field=E, 
        nu_field=nu,
        pixel=pixel,
        barrier_generator=my_barrier_generator,
        # Default random mode generator is used if not specified
        output_dir=OUTPUT_DIR,
        
        # Softening Physics 
        softening_enabled=ENABLE_SOFTENING,
        softening_params=SOFTENING_PARAMS,
        softening_scheme=SOFTENING_SCHEME,
        softening_cap=SOFTENING_CAP,
        
        # Solver & Debugging
        solver_args=None, # Default: {"max_iter": 200, "tol": 1e-6}
        debug_first_flip=DEBUG_FIRST_FLIP
    )

    # ----------------------------------------------------
    # 5. Define Loading Protocol
    # ----------------------------------------------------
    # Uniaxial tension in X: epsilon_xx increment
    strain_inc = np.zeros((3,3))
    strain_inc[0,0] = 1e-4 # 0.01% per step
    
    n_steps = 2000
    
    # ----------------------------------------------------
    # 6. Run
    # ----------------------------------------------------
    # Options for vtk_mode: "global", "detailed", None
    sim.run(n_steps, strain_inc, vtk_mode="global")
    
    print(f"\nDemo complete. Check '{OUTPUT_DIR}' for results.")
    print(f"  - global_log.txt")
    print(f"  - detailed_cascade.txt")
    print(f"  - VTK files")

    # ----------------------------------------------------
    # 7. Plot Stress-Strain Curves
    # ----------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        
        # Data
        hist_global = np.array(sim.history_global)
        hist_detailed = np.array(sim.history_detailed)
        
        # Plot 1: Standard AQS (Combined)
        plt.figure(figsize=(10, 6))
        
        if len(hist_detailed) > 0:
            # Overplot detailed jagged curve first (background)
            plt.plot(hist_detailed[:,0]*100, hist_detailed[:,1], 'r-', label='Detailed Path (Cascades)', alpha=0.4, linewidth=0.8)
            
        if len(hist_global) > 0:
            plt.plot(hist_global[:,0]*100, hist_global[:,1], 'b-o', label='Global Equilibrium', alpha=0.9, markersize=4)
            
        plt.xlabel('Strain $\epsilon_{xx}$ (%)')
        plt.ylabel('Stress $\sigma_{xx}$ (GPa)')
        plt.title(f'AQS Response (Softening: {SOFTENING_SCHEME if ENABLE_SOFTENING else "OFF"})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, "stress_strain.png")
        plt.savefig(plot_path)
        print(f"  - Plot saved to '{plot_path}'")
        
    except ImportError:
        print("\nNote: Matplotlib not installed. Skipping plot generation.")

if __name__ == "__main__":
    main()
