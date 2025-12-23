import numpy as np
import os
import shutil
from mgkmc import AthermalSimulation

def main():
    print("=" * 60)
    print("AQS DEMO 6 (Numba): Mixed BC Simulation at T=600 K")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. Setup Parameters
    # ----------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    print(f"Random seed set to {SEED}")

    # System Size
    nx, ny, nz = 128, 128, 1
    pixel = 0.69
    M = 16
    gamma0 = 0.14
    
    # Material properties
    E = np.full((nx, ny, nz), 70.0)  # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    # ----------------------------------------------------
    # 2. Configuration
    # ----------------------------------------------------
    
    # Softening
    ENABLE_SOFTENING = True
    SOFTENING_SCHEME = "directional" 
    JP = 4.18
    JT = 5.11
    SOFTENING_CAP = 0.78
    
    if not ENABLE_SOFTENING:
        JP = 0.0
        JT = 0.0
        print("Softening DISABLED (JP=0, JT=0)")
    else:
        print(f"Softening ENABLED (JP={JP}, JT={JT}, Cap={SOFTENING_CAP})")
    
    # Simulation Control
    OUTPUT_DIR = "aqs_demo_6_numba"
    TEMPERATURE = 600.0 # Kelvin
    PHYSICAL_STRAIN_RATE = 1e5 # 1/s

    # Barrier Generator matching legacy
    def my_barrier_generator(shape):
        # shape is (nx, ny, nz, M)
        # Normal distribution with mean=2.17, std=0.37
        random_barriers = np.random.normal(loc=2.17, scale=0.37, size=shape)
        # Clip at 0.3 to ensure stability at RT/Low T
        return np.clip(random_barriers, a_min=0.3, a_max=None)
    
    # ----------------------------------------------------
    # 3. Initialize
    # ----------------------------------------------------
    if os.path.exists(OUTPUT_DIR):
        try:
             shutil.rmtree(OUTPUT_DIR)
        except OSError:
             pass
        
    sim = AthermalSimulation(
        nx, ny, nz,
        M=M, 
        gamma0=gamma0,
        E_field=E, 
        nu_field=nu,
        pixel=pixel,
        barrier_generator=my_barrier_generator,
        output_dir=OUTPUT_DIR,
        
        # Softening Physics (Flat API)
        softening_scheme=SOFTENING_SCHEME,
        softening_cap=SOFTENING_CAP,
        jp=JP, 
        jt=JT,
        
        # Thermal
        temperature=TEMPERATURE,
        strain_rate=PHYSICAL_STRAIN_RATE, 
        stability_threshold=0.33 # C-Code behavior
    )

    # ----------------------------------------------------
    # 4. Run Mixed
    # ----------------------------------------------------
    eps_target = 0.14
    # Legacy used 1400 steps -> 0.0001 strain per step
    n_steps = 1400 
    strain_rate_per_step = eps_target / n_steps
    
    # New Features Configuration
    CHECKPOINT_INTERVAL = 1
    KEEP_CHECKPOINTS = True
    STOP_ON_STRESS_DROP = 0.20 # 20% drop
    STOP_POST_DROP_STEPS = 10
    
    sim.run_mixed(
        n_global_steps=n_steps,
        strain_rate=strain_rate_per_step,
        component=(0,0),
        stress_targets={(1,1): 0.0, (2,2): 0.0},
        mixed_tol=1e6, # 1 MPa
        kmc_mode="on_demand",
        
        # Checkpoint & Detection
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoint"),
        keep_checkpoints=KEEP_CHECKPOINTS,
        stop_on_stress_drop=STOP_ON_STRESS_DROP,
        stress_drop_component=(0,0),
        stop_post_drop_steps=STOP_POST_DROP_STEPS,
        ignore_drop_steps=10,
        checkpoint_elastic_only=True
    )
    
    print(f"\nDemo complete. Check '{OUTPUT_DIR}'")

    # ----------------------------------------------------
    # 5. Plot
    # ----------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        hist_global = np.array(sim.history_global)
        plt.figure(figsize=(10, 6))
        if len(hist_global) > 0:
            plt.plot(hist_global[:,0]*100, hist_global[:,1], 'b-o', label='Global', markersize=3)
        plt.xlabel(r'Strain (%)')
        plt.ylabel(r'Stress (GPa)')
        plt.title(f'Numba Replicated T={TEMPERATURE}K')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "stress_strain.png"))
        print(f"Plot saved.")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
