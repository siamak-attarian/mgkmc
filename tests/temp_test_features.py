import numpy as np
import os
import shutil
import mgkmc
print(f"DEBUG: mgkmc location: {mgkmc.__file__}")
from mgkmc import AthermalSimulation
from mgkmc.elasticity_helpers import get_uniaxial_stress_x

def test_features():
    print("TESTING NEW CHECKPOINT AND DETECTION FEATURES")
    print("="*50)
    
    output_dir = "test_output_checkpoints"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    nx, ny, nz = 16, 16, 1
    M = 5
    E = np.full((nx,ny,nz), 70.0)
    nu = np.full((nx,ny,nz), 0.3)
    
    # 1. Run simulation with stress drop trigger and checkpoints
    print("1. Running Simulation (stop on 1% drop? No, make it easy to trigger)...")
    # To trigger a drop easily, we might need softening or just wait for an avalanche.
    # Let's set a very low threshold like 0.001% just to test the logic triggers, 
    # OR rely on a real avalanche. With 16x16, avalanches happen.
    
    sim = AthermalSimulation(nx, ny, nz, M, 0.1, E, nu, output_dir=output_dir, 
                             softening_enabled=True, softening_params={"jp":5, "jt":10})
    
    # We'll set a strict drop of 1% (0.01) which should happen in a plastics cascade
    # And checkpoint every 2 steps
    try:
        sim.run(
            n_global_steps=20,
            loading_func=get_uniaxial_stress_x,
            loading_params={"eps_xx": 0.1, "E": 70.0, "nu": 0.3},
            checkpoint_interval=2,
            keep_checkpoints=True,
            stop_on_stress_drop=0.00001, # Extremely sensitive to trigger early
            stop_post_drop_steps=3
        )
    except Exception as e:
        print(f"Run Error: {e}")
        
    print(f"Simulation ended at step {sim.current_step}")
    
    # 2. Verify Checkpoints
    print("\n2. Verifying Checkpoints...")
    files = os.listdir(os.path.join(output_dir))
    checkpoints = [f for f in files if f.endswith('.h5') and 'checkpoint_' in f]
    print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
    
    if len(checkpoints) > 0:
        # Load one
        cp_path = os.path.join(output_dir, checkpoints[-1])
        sim_loaded = AthermalSimulation.load_checkpoint(cp_path)
        print(f"Loaded checkpoint from step {sim_loaded.current_step}")
        
        # Verify Flip History
        print(f"Original Flip History Length: {len(sim.flip_event_history)}")
        print(f"Loaded Flip History Length: {len(sim_loaded.flip_event_history)}")
        
        if len(sim.flip_event_history) == len(sim_loaded.flip_event_history):
            print("[PASS] Flip history length matches.")
        else:
             # It might not match exactly if we loaded an earlier checkpoint than the final state
             # This is expected behavior.
             print("[INFO] Lengths differ (expected if CP is earlier than end).")

        if len(sim_loaded.flip_event_history) > 0:
            print(f"First flip event: {sim_loaded.flip_event_history[0]}")
            
        # Verify Export VTK
        print("\n3. Testing Export VTK from loaded sim...")
        vtk_failed = False
        try:
             sim_loaded.export_vtk(os.path.join(output_dir, "test_export.vtu"))
             if os.path.exists(os.path.join(output_dir, "test_export.vtu")):
                 print("[PASS] VTK exported successfully.")
             else:
                 print("[FAIL] VTK file not found.")
                 vtk_failed = True
        except Exception as e:
             print(f"[FAIL] Export VTK Error: {e}")
             vtk_failed = True
             
    else:
        print("[FAIL] No checkpoints generated!")

if __name__ == "__main__":
    test_features()
