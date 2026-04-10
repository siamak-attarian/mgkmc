"""
Demonstration of checkpoint save/load functionality.

This script shows how to:
1. Run a simulation for N steps
2. Save a checkpoint
3. Load the checkpoint
4. Continue the simulation from the checkpoint
"""

import numpy as np
import os
import shutil
from mgkmc import ThermalSimulation
from mgkmc.elasticity import get_uniaxial_stress_x

def main():
    print("=" * 60)
    print("CHECKPOINT SAVE/LOAD DEMONSTRATION")
    print("=" * 60)
    print()
    
    # Setup
    SEED = 42
    np.random.seed(SEED)
    
    nx, ny, nz = 16,16,1
    pixel = 0.7
    M = 20
    gamma0 = 0.14
    
    E = np.full((nx, ny, nz), 70.0)  # GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    OUTPUT_DIR = "checkpoint_demo_output"
    CHECKPOINT_FILE = "checkpoint_demo.h5"
    
    # Clean up previous runs
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    # ========================================
    # Part 1: Run simulation and save checkpoint
    # ========================================
    print("Part 1: Running simulation for 50 steps...")
    print()
    
    sim1 = ThermalSimulation(
        nx, ny, nz,
        M=M,
        gamma0=gamma0,
        E_field=E,
        nu_field=nu,
        pixel=pixel,
        output_dir=OUTPUT_DIR,
        softening_enabled=True,
        softening_params={"jp": 11, "jt": 33},
        softening_scheme="directional"
    )
    
    # Run for 50 steps
    sim1.run(
        n_global_steps=50,
        vtk_mode="global",
        loading_func=get_uniaxial_stress_x,
        loading_params={
            "eps_xx": 0.10,
            "E": E.mean(),
            "nu": nu.mean(),
        }
    )
    
    print()
    print(f"Simulation completed 50 steps.")
    print(f"Final stress (xx): {sim1.history_global[-1][1]:.2f} GPa")
    print(f"Final strain (xx): {sim1.history_global[-1][0]:.4f}")
    print()
    
    # Save checkpoint
    print(f"Saving checkpoint to '{CHECKPOINT_FILE}'...")
    sim1.save_checkpoint(CHECKPOINT_FILE)
    print("Checkpoint saved successfully!")
    print()
    
    # ========================================
    # Part 2: Load checkpoint and continue
    # ========================================
    print("Part 2: Loading checkpoint and continuing simulation...")
    print()
    
    # Load checkpoint
    sim2 = ThermalSimulation.load_checkpoint(CHECKPOINT_FILE)
    
    print(f"Checkpoint loaded from step {sim2.current_step}")
    print(f"Loaded stress (xx): {sim2.history_global[-1][1]:.2f} GPa")
    print(f"Loaded strain (xx): {sim2.history_global[-1][0]:.4f}")
    print()
    
    # Verify continuity
    print("Verifying checkpoint integrity...")
    stress_before = sim1.history_global[-1][1]
    stress_after = sim2.history_global[-1][1]
    
    if abs(stress_before - stress_after) < 1e-6:
        print("[PASS] Checkpoint verified: stress values match!")
    else:
        print(f"[FAIL] Warning: stress mismatch ({stress_before} vs {stress_after})")
    print()
    
    # Continue simulation for another 50 steps
    print("Continuing simulation for another 50 steps...")
    
    # Update output directory to avoid overwriting
    sim2.output_dir = OUTPUT_DIR + "_continued"
    
    # IMPORTANT: Must manually create directory and update log paths
    if not os.path.exists(sim2.output_dir):
        os.makedirs(sim2.output_dir)
        
    sim2.global_log_path = os.path.join(sim2.output_dir, "global_log.txt")
    sim2.cascade_log_path = os.path.join(sim2.output_dir, "detailed_cascade.txt")
    
    sim2._init_logs()
    
    try:
        sim2.run(
            n_global_steps=50,
            vtk_mode="global",
            loading_func=get_uniaxial_stress_x,
            loading_params={
                "eps_xx": 0.20,  # Continue to 20% strain
                "E": E.mean(),
                "nu": nu.mean(),
            }
        )
    except RuntimeError as e:
        print(f"\n[INFO] Simulation stopped early: {e}")
        print("This is expected for small systems or large avalanches.")
    
    print()
    print(f"Continued simulation completed.")
    print(f"Final stress (xx): {sim2.history_global[-1][1]:.2f} GPa")
    print(f"Final strain (xx): {sim2.history_global[-1][0]:.4f}")
    print()
    
    # ========================================
    # Summary
    # ========================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original simulation: 0 -> 50 steps")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")
    print()
    print(f"Continued simulation: 50 -> 100 steps")
    print(f"  Output: {sim2.output_dir}")
    print()
    print("Checkpoint save/load demonstration complete!")
    print()
    
    # Plot comparison
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Original simulation
        hist1 = np.array(sim1.history_global)
        ax.plot(hist1[:, 0] * 100, hist1[:, 1], 'b-o', 
                label='Original (0-50 steps)', markersize=4)
        
        # Continued simulation
        hist2 = np.array(sim2.history_global)
        ax.plot(hist2[:, 0] * 100, hist2[:, 1], 'r-s', 
                label='Continued (50-100 steps)', markersize=4)
        
        ax.axvline(hist1[-1, 0] * 100, color='k', linestyle='--', 
                   alpha=0.5, label='Checkpoint')
        
        ax.set_xlabel('Strain ε_xx (%)')
        ax.set_ylabel('Stress σ_xx (GPa)')
        ax.set_title('Checkpoint Save/Load Demonstration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_file = "checkpoint_demo_plot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to '{plot_file}'")
        
    except ImportError:
        print("Matplotlib not available, skipping plot")

if __name__ == "__main__":
    main()
