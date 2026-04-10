"""Quick test of checkpoint functionality"""
import numpy as np
import os
import shutil
from mgkmc import ThermalSimulation
from mgkmc.elasticity import get_uniaxial_stress_x

# Setup
np.random.seed(42)
nx, ny, nz = 16, 16, 3
E = np.full((nx, ny, nz), 70.0)
nu = np.full((nx, ny, nz), 0.3)

OUTPUT_DIR = "test_checkpoint_output"
CHECKPOINT_FILE = "test_checkpoint.h5"

# Clean up
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

print("Creating simulation...")
sim1 = ThermalSimulation(
    nx, ny, nz,
    M=10,
    gamma0=0.14,
    E_field=E,
    nu_field=nu,
    pixel=0.7,
    output_dir=OUTPUT_DIR,
    softening_enabled=True,
    softening_params={"jp": 11, "jt": 33}
)

print("Running 10 steps...")
sim1.run(
    n_global_steps=10,
    vtk_mode=None,
    loading_func=get_uniaxial_stress_x,
    loading_params={"eps_xx": 0.05, "E": E.mean(), "nu": nu.mean()}
)

print(f"Completed. Final stress: {sim1.history_global[-1][1]:.4f} GPa")

print(f"\nSaving checkpoint...")
sim1.save_checkpoint(CHECKPOINT_FILE)

print(f"Loading checkpoint...")
sim2 = ThermalSimulation.load_checkpoint(CHECKPOINT_FILE)

print(f"Loaded. Stress: {sim2.history_global[-1][1]:.4f} GPa")

# Verify
stress_diff = abs(sim1.history_global[-1][1] - sim2.history_global[-1][1])
if stress_diff < 1e-10:
    print(f"\n[PASS] TEST PASSED: Checkpoint save/load works correctly!")
else:
    print(f"\n[FAIL] TEST FAILED: Stress mismatch ({stress_diff})")

# Clean up
shutil.rmtree(OUTPUT_DIR)
os.remove(CHECKPOINT_FILE)
print("\nCleanup complete.")
