import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import shutil
import re

from mgkmc import ThermalSimulation

def analyze_results():
    print("==================================================")
    print("AQS DEMO 5 : RESULT ANALYSIS")
    print("==================================================")
    
    # 1. Configuration
    data_dir = "aqs_demo_5_mixed_checkpoint_shearband"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    output_dir = "aqs_demo_5_results"
    vtk_dir = os.path.join(output_dir, "vtk")
    if not os.path.exists(vtk_dir):
        os.makedirs(vtk_dir)
        
    # 2. Find Checkpoints
    cp_files = glob.glob(os.path.join(data_dir, "checkpoint_*.h5"))
    # Sort by step number
    def get_step_from_name(fname):
        match = re.search(r'checkpoint_(\d+).h5', fname)
        return int(match.group(1)) if match else 0
        
    cp_files.sort(key=get_step_from_name)
    
    if not cp_files:
        print("No checkpoints found.")
        return
        
    print(f"Found {len(cp_files)} checkpoints.")

    # 3. Processing Loop
    history_strain = []
    history_stress = []
    
    history_step = []
    
    # Elastic backbone stats
    history_elastic_strain_avg = []
    history_never_flipped_count = []
    
    # For cumulative flip map (assuming same grid size for all)
    cumulative_flip_map = None
    
    # Dictionary to store cascade details {step: list of flips}
    cascade_details = {} 

    for cp_file in cp_files:
        step = get_step_from_name(cp_file)
        # print(f"Processing Step {step}...", end='\r')
        
        sim = ThermalSimulation.load_checkpoint(cp_file)
        
        # A. Global Stress-Strain
        # Recalculate mean from fields to be sure
        eps_xx = sim.eps_field.mean(axis=(0,1,2))[0,0]
        sig_xx = sim.sig_field.mean(axis=(0,1,2))[0,0]
        
        history_step.append(step)
        history_strain.append(eps_xx)
        history_stress.append(sig_xx / 1e9) # GPa
        
        # B. Flip Analysis
        # sim.flip_event_history is list of (global_step, local_step, x, y, z, m)
        # It contains ALL history up to this point.
        
        if cumulative_flip_map is None:
            cumulative_flip_map = np.zeros((sim.nx, sim.ny, sim.nz), dtype=int)
            
        # Optimization: We could convert to numpy array once
        # But to be robust, let's just process the list.
        # Ideally, we want the Incremental flips since last step, but reloading from scratch
        # gives us the Cumulative state accurately.
        
        # Re-zero the map if we want to be safe, or just trust we are iterating in order?
        # Actually, sim.flip_event_history grows. So we should re-compute the map from scratch 
        # for *this* checkpoint's state to ensure we handle restarts/loading correctly.
        
        temp_flip_map = np.zeros((sim.nx, sim.ny, sim.nz), dtype=int)
        
        current_step_flips = []
        
        for f in sim.flip_event_history:
            g_step, l_step, x, y, z, m = f
            temp_flip_map[x, y, z] += 1
            if g_step == step:
                current_step_flips.append(f)
        
        cascade_details[step] = current_step_flips
        
        # Update our main map (for latest state)
        cumulative_flip_map = temp_flip_map
        
        # C. Elastic Backbone Analysis
        # Voxels that have NEVER flipped (count == 0)
        never_flipped_mask = (cumulative_flip_map == 0)
        n_never = np.sum(never_flipped_mask)
        
        if n_never > 0:
            avg_eps_elastic = sim.eps_field[never_flipped_mask, 0, 0].mean()
        else:
            avg_eps_elastic = 0.0 # Should not happen unless totally plastic
            
        history_never_flipped_count.append(n_never)
        history_elastic_strain_avg.append(avg_eps_elastic)
        
        # D. VTK Export
        vtk_name = os.path.join(vtk_dir, f"step_{step:06d}.vtu")
        sim.export_vtk(vtk_name)

    print("\nProcessing complete.")
    
    # 4. Plotting
    history_strain = np.array(history_strain)
    history_stress = np.array(history_stress)
    history_elastic_strain_avg = np.array(history_elastic_strain_avg)
    history_never_flipped_count = np.array(history_never_flipped_count)
    
    # Plot 1: Stress-Strain
    plt.figure(figsize=(10,6))
    plt.plot(history_strain*100, history_stress, 'b-o', markersize=4, label="Global Stress")
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (GPa)")
    plt.title("Stress-Strain Response")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "stress_strain.png"))
    plt.close()
    
    # Plot 2: Elastic Backbone
    plt.figure(figsize=(10,6))
    plt.plot(history_strain*100, history_elastic_strain_avg*100, 'g-', label="Avg Strain (Never Flipped)")
    plt.plot(history_strain*100, history_strain*100, 'k--', alpha=0.5, label="System Strain (Reference)")
    plt.xlabel("System Strain (%)")
    plt.ylabel("Elastic Backbone Strain (%)")
    plt.title("Evolution of Elastic Backbone Strain")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "elastic_backbone_strain.png"))
    plt.close()
    
    # Plot 3: Plastic Fraction
    total_voxels = sim.nx * sim.ny * sim.nz
    plastic_fraction = 1.0 - (history_never_flipped_count / total_voxels)
    
    plt.figure(figsize=(10,6))
    plt.plot(history_strain*100, plastic_fraction*100, 'r-')
    plt.xlabel("System Strain (%)")
    plt.ylabel("Volume Fraction Yielded (%)")
    plt.title("Cumulative Plastic Volume")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "plastic_fraction.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    print(f"VTK files saved to {vtk_dir}")
    
    # 5. Helper Function for User Query
    print("\n--- Cascade Retrieval ---")
    
    def get_cascade_info(step_req):
        if step_req in cascade_details:
            flips = cascade_details[step_req]
            print(f"Step {step_req}: {len(flips)} total flips.")
            if len(flips) > 0:
                print("Detailed Flips (GlobalStep, LocalStep, x, y, z, m):")
                # Print first 20 if many
                for i, f in enumerate(flips):
                    if i < 20:
                        print(f"  {f}")
                    else:
                        print(f"  ... and {len(flips)-20} more.")
                        break
        else:
            print(f"Step {step_req} not found in processed data.")
            
    # Example: Show cascade for the step with maximum stress drop
    # Calculate drops
    drops = -np.diff(history_stress)
    if len(drops) > 0:
        max_drop_idx = np.argmax(drops)
        step_drop = history_step[max_drop_idx + 1] # +1 because diff reduces len
        print(f"\nLargest stress drop detected at Step {step_drop} (Drop: {drops[max_drop_idx]:.3f} GPa)")
        get_cascade_info(step_drop)
        
    return get_cascade_info # Return function so user can use it interactively if running in python -i

if __name__ == "__main__":
    analyze_results()
