"""Quick test of checkpoint functionality"""
import numpy as np
import os
import shutil
from mgkmc import ThermalSimulation
from mgkmc.elasticity import get_uniaxial_stress_x

def test_checkpoint_logic():
    # Setup
    np.random.seed(42)
    nx, ny, nz = 16, 16, 3
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)

    OUTPUT_DIR = "test_checkpoint_output"
    CHECKPOINT_FILE = "test_checkpoint.h5"

    # Clean up
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
        except:
            pass
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
        except:
            pass

    print("Creating simulation...")
    sim1 = ThermalSimulation(
        nx, ny, nz,
        M=10,
        gamma0=0.14,
        E_field=E,
        nu_field=nu,
        pixel=0.7,
        output_dir=OUTPUT_DIR,
        jp=11, 
        jt=33
    )

    print("Running 10 steps...")
    sim1.run_mixed(
        n_global_steps=10,
        strain_rate=1.0e7,
        component=(0, 0),
        stress_targets={(1, 1): 0.0, (2, 2): 0.0},
        vtk_mode="none",
        enable_console_log=False
    )

    print(f"Completed. Final stress: {sim1.sig_field.mean(axis=(0,1,2))[0,0]/1e9:.4f} GPa")

    print(f"\nSaving checkpoint...")
    sim1.save_checkpoint(CHECKPOINT_FILE)

    print(f"Loading checkpoint...")
    sim2 = ThermalSimulation.load_checkpoint(CHECKPOINT_FILE)

    print(f"Loaded. Stress: {sim2.sig_field.mean(axis=(0,1,2))[0,0]/1e9:.4f} GPa")

    # Verify
    stress_diff = abs(sim1.sig_field.mean(axis=(0,1,2))[0,0] - sim2.sig_field.mean(axis=(0,1,2))[0,0])
    
    # Clean up
    try:
        shutil.rmtree(OUTPUT_DIR)
        os.remove(CHECKPOINT_FILE)
    except:
        pass
        
    print("\nCleanup complete.")
    assert stress_diff < 1e-5, f"Stress mismatch ({stress_diff})"
