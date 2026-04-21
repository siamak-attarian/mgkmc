"""
Example 04: Checkpoints and Restarts
====================================
Shows how to configure periodic checkpointing safely and how to resume
a simulation seamlessly from an HDF5 checkpoint.
"""
import os
import yaml
import numpy as np
from mgkmc import ThermalSimulation

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Note: the example deletes checkp_dir on consecutive runs to be safe
    output_dir = "output_checkpointing"
    
    # ---------------------------------------------------------
    # PART 1: Run and Save
    # ---------------------------------------------------------
    print("Initializing Simulation (Part 1)...")
    nx, ny, nz = 16, 16, 1
    sim = ThermalSimulation(
        nx, ny, nz,
        M=config['system']['M'],
        gamma0=config['system']['gamma0'],
        E_field=np.full((nx, ny, nz), 70.0),
        nu_field=np.full((nx, ny, nz), 0.3),
        pixel=config['system']['pixel'],
        output_dir=output_dir,
        temperature=0.0
    )

    print("Running initial steps...")
    # Using 'checkpoint_interval' automatically dumps State, History, Random state to .h5
    sim.run_mixed(
        n_global_steps=50,
        strain_rate=1e7,
        component=(0, 1),
        checkpoint_interval=20,
        checkpoint_path=f"{output_dir}/demo_checkpoint",
        checkpoint_mode="periodic",
        enable_console_log=True
    )
    
    # Alternatively, you can always manually save anytime:
    final_cp = f"{output_dir}/manual_checkpoint.h5"
    sim.save_checkpoint(final_cp)
    print(f"Manual checkpoint saved to {final_cp}")
    del sim

    # ---------------------------------------------------------
    # PART 2: Resume Checkpoint
    # ---------------------------------------------------------
    print("\nLoading checkpoint and continuing... (Part 2)")
    sim2 = ThermalSimulation.load_checkpoint(final_cp)
    
    sim2.run_mixed(
        n_global_steps=50, # Will continue running for 50 MORE steps
        strain_rate=1e7,
        component=(0, 1),
        append_logs=True, # Prevent overwriting the previous log files
        enable_console_log=True
    )

    print("Simulation Complete!")

if __name__ == "__main__":
    main()
