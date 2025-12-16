import numpy as np
import os
import shutil
import argparse
from mgkmc import AthermalSimulation
from mgkmc.stz.cascade import apply_flip
from mgkmc.stz.update_fft import update_stress_fft_full

def visualize_cascade(step=753, data_dir="aqs_demo_5_mixed_checkpoint_shearband", output_base="aqs_demo_5_results"):
    print("==================================================")
    print(f"VISUALIZING CASCADE AT STEP {step}")
    print("==================================================")
    
    # 1. Paths
    prev_step = step - 1
    cp_prev = os.path.join(data_dir, f"checkpoint_{prev_step:06d}.h5")
    cp_curr = os.path.join(data_dir, f"checkpoint_{step:06d}.h5")
    
    if not os.path.exists(cp_prev):
        print(f"Error: Previous checkpoint {cp_prev} not found.")
        return
    if not os.path.exists(cp_curr):
        print(f"Error: Target checkpoint {cp_curr} not found.")
        return
        
    out_dir = os.path.join(output_base, f"cascade_{step}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    # 2. Load Data
    print(f"Loading previous state from Step {prev_step}...")
    sim = AthermalSimulation.load_checkpoint(cp_prev)
    
    print(f"Loading history from Step {step} (just for flip list)...")
    # We load into a temporary sim just to get the history list
    sim_target = AthermalSimulation.load_checkpoint(cp_curr)
    full_history = sim_target.flip_event_history
    
    # Filter flips for the target step
    # History format: (global_step, local_step, x, y, z, m)
    target_flips = [f for f in full_history if f[0] == step]
    
    if not target_flips:
        print(f"No flips found for Step {step}.")
        return
        
    print(f"Found {len(target_flips)} flips in Step {step}.")
    
    # 3. Setup Replay
    # Apply Driving Strain (Approximate for visual purposes)
    # Getting params from loaded state
    if hasattr(sim, 'loading_params') and sim.loading_params:
        strain_rate = sim.loading_params.get('rate', 0.0)
        component = tuple(sim.loading_params.get('comp', (0,0)))
        print(f"Applying driving strain: rate={strain_rate:.2e} on component {component}")
        
        eps_inc = np.zeros((3,3))
        eps_inc[component] = strain_rate
        sim._apply_strain_increment(eps_inc)
    else:
        print("Warning: Could not determine driving strain from checkpoint. Skipping strain increment.")
        
    # Export Frame 0 (Start of step but after elastic load)
    print("Exporting Frame 0...")
    fname = os.path.join(out_dir, "frame_0000.vtu")
    sim.export_vtk(fname)
    
    # 4. Replay Loop
    print("Replaying flips (grouped by cascade step)...")
    
    import itertools
    
    # Group by local_step (index 1 in the tuple)
    # Ensure sorted by local_step just in case
    target_flips.sort(key=lambda x: x[1])
    
    grouped_flips = itertools.groupby(target_flips, key=lambda x: x[1])
    
    frame_idx = 0
    for local_step, group in grouped_flips:
        flips_in_batch = list(group)
        
        # Apply all flips in this batch
        for flip in flips_in_batch:
            g_step, l_step, x, y, z, m = flip
            voxel = sim.grid[x,y,z]
            apply_flip(voxel, m, jp=sim.jp, jt=sim.jt, g_max=sim.softening_cap)
            
        # Update Stress Field ONCE per batch
        sim.eps_field, sim.sig_field, _, _ = update_stress_fft_full(
            sim.grid, sim.eps_macro, sim.E, sim.nu, 
            pixel=sim.pixel, **(sim.solver_args or {})
        )
        
        # Export VTK
        frame_idx += 1
        fname = os.path.join(out_dir, f"frame_{frame_idx:04d}.vtu")
        print(f"  Frame {frame_idx} (Local Step {local_step}, {len(flips_in_batch)} flips)", end='\r')
             
        sim.export_vtk(fname)
        
    print(f"\nDone. {frame_idx} frames saved to {out_dir}")

if __name__ == "__main__":
    visualize_cascade()
