import numpy as np
import os
import shutil
from mgkmc import AthermalSimulation

def main():
    print("=" * 60)
    print("AQS DEMO 4: Large System Mixed Boundary Condition (Uniaxial Tension)")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. Setup Parameters
    # ----------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    print(f"Random seed set to {SEED}")

    # System Size (Matching demo 2)
    nx, ny, nz = 128, 128, 1
    pixel = 0.7
    M = 20
    gamma0 = 0.14
    
    # Material properties (Homogeneous)
    # E in GPa, nu is dimensionless
    E = np.full((nx, ny, nz), 70.0)  # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    # ----------------------------------------------------
    # 2. Configuration for Experimentation
    # ----------------------------------------------------
    
    # Softening Configuration
    ENABLE_SOFTENING = True 
    SOFTENING_SCHEME = "directional" 
    SOFTENING_PARAMS = {"jp": 11, "jt": 33}
    SOFTENING_CAP = -np.log(0.4) 
    
    # Simulation Control
    DEBUG_FIRST_FLIP = False
    OUTPUT_DIR = "aqs_demo_4_mixed_128_new_2000_mixed_try"

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
        
    sim = AthermalSimulation(
        nx, ny, nz,
        M=M, 
        gamma0=gamma0,
        E_field=E, 
        nu_field=nu,
        pixel=pixel,
        barrier_generator=my_barrier_generator,
        output_dir=OUTPUT_DIR,
        
        # Softening Physics 
        softening_enabled=ENABLE_SOFTENING,
        softening_params=SOFTENING_PARAMS,
        softening_scheme=SOFTENING_SCHEME,
        softening_cap=SOFTENING_CAP,
        
        # Solver & Debugging
        solver_args=None,
        debug_first_flip=DEBUG_FIRST_FLIP
    )

    # ----------------------------------------------------
    # 5. Run Mixed Simulation
    # ----------------------------------------------------
    # Uniaxial Tension: Drive X, Relax Y and Z to 0.0 stress
    
    eps_target = 0.14  # 20% strain
    n_steps = 1400
    strain_rate = eps_target / n_steps
    
    # Options for vtk_mode: "global", "detailed", None
    # Using "global" to see evolution
    sim.run_mixed(
        n_global_steps=n_steps,
        strain_rate=strain_rate,
        component=(0,0), # Drive eps_xx
        stress_targets={(1,1): 0.0, (2,2): 0.0}, # Target sig_yy=0, sig_zz=0
        mixed_tol=1e6, # 1 MPa convergence tolerance
        mixed_max_iter=10, 
        vtk_mode="global"
    )
    
    print(f"\nDemo complete. Check '{OUTPUT_DIR}' for results.")

    # ----------------------------------------------------
    # 6. Plot Stress-Strain Curves (Robust)
    # ----------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        
        hist_global = np.array(sim.history_global)
        hist_detailed = np.array(sim.history_detailed)
        
        plt.figure(figsize=(10, 6))
        
        if len(hist_detailed) > 0:
            # Overplot detailed jagged curve first (background)
            plt.plot(hist_detailed[:,0]*100, hist_detailed[:,1], 'r-', label='Detailed Path (Cascades)', alpha=0.4, linewidth=0.8)
            
        if len(hist_global) > 0:
            plt.plot(hist_global[:,0]*100, hist_global[:,1], 'b-o', label='Global Equilibrium', alpha=0.9, markersize=4)
            
        plt.xlabel(r'Strain $\epsilon_{xx}$ (%)')
        plt.ylabel(r'Stress $\sigma_{xx}$ (GPa)')
        plt.title(f'Mixed Uniaxial Tension (Softening: {SOFTENING_SCHEME if ENABLE_SOFTENING else "OFF"})')
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
