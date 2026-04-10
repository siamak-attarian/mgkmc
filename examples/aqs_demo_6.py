import numpy as np
import os
import shutil
from mgkmc import ThermalSimulation

def main():
    print("=" * 60)
    print("AQS DEMO 6: Mixed BC Simulation at T=300 K")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. Setup Parameters
    # ----------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    print(f"Random seed set to {SEED}")

    # System Size (Matching demo 5)
    nx, ny, nz = 128, 128, 1
    pixel = 0.69
    M = 16
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
    SOFTENING_PARAMS = {"jp": 4.18, "jt": 5.11}
    SOFTENING_CAP = 0.78# -np.log(0.4)
    
    # Simulation Control
    DEBUG_FIRST_FLIP = False
    OUTPUT_DIR = "aqs_demo_6_thermal_c_style"

    # Thermal Parameters
    TEMPERATURE = 600.0 # Kelvin
    PHYSICAL_STRAIN_RATE = 1e5 # 1/s (Controls KMC vs Elastic timescale)

    # ----------------------------------------------------
    # 3. Custom Barrier Generator
    # ----------------------------------------------------
    def my_barrier_generator(n_modes):
        # Example: Normal distribution with mean=2.0, std=0.6
        random_barriers = np.random.normal(loc=2.17, scale=0.37, size=n_modes)
        min_barrier = 0.3 # Cutoff to ensure stability at RT
        clipped_barriers = np.clip(random_barriers, a_min=min_barrier, a_max=None)
        return clipped_barriers
    
    # ----------------------------------------------------
    # 4. Initialize Simulation
    # ----------------------------------------------------
    if os.path.exists(OUTPUT_DIR):
        try:
             shutil.rmtree(OUTPUT_DIR)
        except OSError:
             print(f"Warning: Could not remove {OUTPUT_DIR}. Using existing directory.")
        
    sim = ThermalSimulation(
        nx, ny, nz,
        M=M, 
        gamma0=gamma0,
        E_field=E, 
        nu_field=nu,
        pixel=pixel,
        barrier_generator=my_barrier_generator,
        # mode_generator=None, # Use default
        output_dir=OUTPUT_DIR,
        
        # Softening Physics 
        softening_enabled=ENABLE_SOFTENING,
        softening_params=SOFTENING_PARAMS,
        softening_scheme=SOFTENING_SCHEME,
        softening_cap=SOFTENING_CAP,
        
        # Thermal Physics (NEW)
        temperature=TEMPERATURE,
        strain_rate=PHYSICAL_STRAIN_RATE, 
        stability_threshold=0.33, # C-Code behavior: Treat |Q| < 0.33 as unstable
        
        # Solver & Debugging
        solver_args=None,
        debug_first_flip=DEBUG_FIRST_FLIP
    )

    # ----------------------------------------------------
    # 5. Run Mixed Simulation
    # ----------------------------------------------------
    # Uniaxial Tension: Drive X, Relax Y and Z to 0.0 stress
    
    eps_target = 0.14  # 14% strain
    # For on_demand, n_steps is max iterations.
    # eps_target is achieved by accumulating ELASTIC steps.
    # We should set n_steps high enough to encompass many KMC events if needed.
    n_steps = 1400 # 1400 Elastic steps (total events will be higher)
    strain_rate_per_step = eps_target / n_steps
    
    # NEW FEATURE CONFIGURATION
    CHECKPOINT_INTERVAL = 1
    KEEP_CHECKPOINTS = True
    STOP_ON_STRESS_DROP = 0.20 # Stop if stress drops by 20%
    STOP_POST_DROP_STEPS = 10  # Continue for 10 steps after detection
    
    sim.run_mixed(
        n_global_steps=n_steps,
        strain_rate=strain_rate_per_step,
        component=(0,0), # Drive eps_xx
        stress_targets={(1,1): 0.0, (2,2): 0.0}, # Target sig_yy=0, sig_zz=0
        mixed_tol=1e6, # 1 MPa convergence tolerance
        mixed_max_iter=10, 
        vtk_mode=None, # Disable global VTK for speed unless requested
        
        # New Arguments
        checkpoint_interval=CHECKPOINT_INTERVAL,
        keep_checkpoints=KEEP_CHECKPOINTS,
        stop_on_stress_drop=STOP_ON_STRESS_DROP,
        stress_drop_component=(0,0), # Monitor Sigma_xx
        stop_post_drop_steps=STOP_POST_DROP_STEPS,
        kmc_mode="on_demand" # C-Style: One event per step
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
            plt.plot(hist_detailed[:,0]*100, hist_detailed[:,1], 'r-', label='Detailed Path', alpha=0.4, linewidth=0.8)
            
        if len(hist_global) > 0:
            plt.plot(hist_global[:,0]*100, hist_global[:,1], 'b-o', label='Global Equilibrium', alpha=0.9, markersize=4)
            
        plt.xlabel(r'Strain $\epsilon_{xx}$ (%)')
        plt.ylabel(r'Stress $\sigma_{xx}$ (GPa)')
        plt.title(f'T={TEMPERATURE}K Mixed BC (C-Style Event Loop)\nCheckpoints={CHECKPOINT_INTERVAL}, Rate={PHYSICAL_STRAIN_RATE}')
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
