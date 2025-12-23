
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import h5py
import re


from mgkmc.analysis.vtk import export_to_vtk

def analyze_results():
    print("==================================================")
    print("AQS DEMO 6 (Numba) : RESULT ANALYSIS")
    print("==================================================")
    
    # 1. Configuration
    data_dir = "aqs_demo_6_numba"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    output_dir = "aqs_demo_6_numba_results"
    vtk_dir = os.path.join(output_dir, "vtk")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(vtk_dir):
        os.makedirs(vtk_dir)
        
    # 2. Find Checkpoints
    # Match sequential elastic checkpoints
    cp_files = glob.glob(os.path.join(data_dir, "checkpoint_elastic_*.h5"))
    
    def get_chk_id(fname):
        match = re.search(r'checkpoint(?:_elastic)?_(\d+).h5', fname)
        return int(match.group(1)) if match else 0
        
    cp_files.sort(key=get_chk_id)
    
    if not cp_files:
        print("No elastic checkpoints found. Checking for standard checkpoints...")
        cp_files = glob.glob(os.path.join(data_dir, "checkpoint_*.h5"))
        cp_files.sort()
        
    if not cp_files:
        print("No checkpoints found.")
        return
        
    print(f"Found {len(cp_files)} checkpoints.")

    # 3. Parse Cascade Log (for detailed history)
    cascade_log_path = os.path.join(data_dir, "cascade_log.txt")
    cascade_details = {} 
    
    if os.path.exists(cascade_log_path):
        print("Parsing cascade_log.txt for detailed flip history...")
        try:
            with open(cascade_log_path, 'r') as f:
                next(f) # Skip Header
                for line in f:
                    parts = line.strip().split(maxsplit=3)
                    if len(parts) >= 3:
                        g_step = int(parts[0])
                        # l_step = int(parts[1])
                        # n_unstable = int(parts[2])
                        flips_str = parts[3] if len(parts) > 3 else ""
                        
                        if g_step not in cascade_details:
                            cascade_details[g_step] = []
                        
                        # flips_str is like "(x,y,z,m);(x,y,z,m)"
                        # Just store the raw string or parse tuples?
                        # Let's clean it up a bit
                        flips_clean = flips_str.replace(';', ' ')
                        cascade_details[g_step].append(flips_clean)
        except Exception as e:
            print(f"Warning: Failed to parse cascade log: {e}")
    else:
        print("cascade_log.txt not found. Detailed flip info unavailable.")

    # 4. Processing Loop
    history_strain = []
    history_stress = []
    history_step = []
    
    # Elastic backbone stats
    history_elastic_strain_avg = []
    history_never_flipped_count = []
    
    # Use h5py directly to avoid Class mismatches or import issues
    for i, cp_file in enumerate(cp_files):
        # Print progress every 10 files
        if i % 10 == 0:
            print(f"Processing {i}/{len(cp_files)}: {os.path.basename(cp_file)}", end='\r')
            
        with h5py.File(cp_file, 'r') as f:
            # Metadata
            try:
                step = f['metadata'].attrs['step']
                # Fallback for old checkpoints where step was stored as strain (float)
                if isinstance(step, float) and step < 1.0:
                     # Try to derive from filename
                     step = get_chk_id(cp_file)
            except KeyError:
                step = get_chk_id(cp_file)
            
            # Fields
            eps_field = f['fields']['eps_field'][()]
            sig_field = f['fields']['sig_field'][()]
            E_field = f['fields']['E_field'][()]
            nu_field = f['fields']['nu_field'][()]
            
            # Plasticity
            eps_plastic = f['grid']['eps_plastic'][()] # (nx,ny,nz,3,3)
            soft_prop = f['grid']['soft_prop'][()] # (nx,ny,nz,4)
            
            # Calculations
            # A. Global Means
            eps_curr = eps_field.mean(axis=(0,1,2))
            sig_curr = sig_field.mean(axis=(0,1,2))
            
            history_step.append(step)
            history_strain.append(eps_curr[0,0])
            history_stress.append(sig_curr[0,0]/1e9) # GPa
            
            # B. Yield Map (Plasticity check)
            # Compute norm of plastic strain tensor per voxel
            # norm = sqrt(sum(sq)) using simple double contraction
            ep_sq = np.sum(eps_plastic**2, axis=(3,4))
            yielded_mask = ep_sq > 1e-12 # Threshold for floating point zero
            
            # C. Elastic Backbone Analysis
            # Backbone = Not Yielded
            backbone_mask = ~yielded_mask
            n_backbone = np.sum(backbone_mask)
            
            if n_backbone > 0:
                # Average Eps_xx of backbone
                avg_eps_elastic = eps_field[backbone_mask, 0, 0].mean()
            else:
                avg_eps_elastic = 0.0
                
            history_never_flipped_count.append(n_backbone) # Approximates "never flipped" if plastic strain only grows
            history_elastic_strain_avg.append(avg_eps_elastic)

            # D. VTK Export
            # Use filename step or id? Use the derived step for plotting consistency
            # But ensure filesystem uniqueness
            vtk_name = os.path.join(vtk_dir, f"step_{int(step):06d}.vtu")
            # If step is same for multiple files (e.g. restarts), maybe suffix?
            # Safe to overwrite for now.
            
            export_to_vtk(vtk_name, eps_field, sig_field, E_field, nu_field,
                          eps_plastic_field=eps_plastic, soft_prop_field=soft_prop,
                          match_matplotlib_orientation=True)

    print("\nProcessing complete.")
    
    # 5. Plotting
    history_strain = np.array(history_strain)
    history_stress = np.array(history_stress)
    history_elastic_strain_avg = np.array(history_elastic_strain_avg)
    history_never_flipped_count = np.array(history_never_flipped_count)
    
    # Plot 1: Stress-Strain
    plt.figure(figsize=(10,6))
    plt.plot(history_strain*100, history_stress, 'b-', linewidth=1, label="Global Stress")
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (GPa)")
    plt.title("Stress-Strain Response (Numba)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "stress_strain.png"))
    plt.close()
    
    # Plot 2: Elastic Backbone
    plt.figure(figsize=(10,6))
    plt.plot(history_strain*100, history_elastic_strain_avg*100, 'g-', label="Avg Strain (Elastic Backbone)")
    plt.plot(history_strain*100, history_strain*100, 'k--', alpha=0.5, label="System Strain (Reference)")
    plt.xlabel("System Strain (%)")
    plt.ylabel("Elastic Backbone Strain (%)")
    plt.title("Evolution of Elastic Backbone Strain")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "elastic_backbone_strain.png"))
    plt.close()
    
    # Plot 3: Plastic Fraction
    if len(cp_files) > 0:
        total_voxels = eps_field.shape[0] * eps_field.shape[1] * eps_field.shape[2]
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
    
    # 6. Helper Function for User Query
    print("\n--- Cascade Retrieval ---")
    
    def get_cascade_info(step_req):
        if step_req in cascade_details:
            flips_list = cascade_details[step_req]
            print(f"Step {step_req}: {len(flips_list)} cascade events recorded.")
            print("Events (Truncated):")
            for i, f in enumerate(flips_list):
                if i < 10:
                    print(f"  Batch {i+1}: {f}")
                else:
                    print(f"  ... {len(flips_list)-10} more batches.")
                    break
        else:
            print(f"Step {step_req}: No cascade activity recorded or log missing.")
            
    # Example: Show cascade for the step with maximum stress drop within valid range
    if len(history_stress) > 5:
        drops = -np.diff(history_stress)
        if len(drops) > 0:
            max_drop_idx = np.argmax(drops)
            step_drop = history_step[max_drop_idx + 1] 
            print(f"\nLargest stress drop detected at Step {step_drop} (Drop: {drops[max_drop_idx]:.3f} GPa)")
            get_cascade_info(step_drop)
        
    return get_cascade_info

if __name__ == "__main__":
    analyze_results()
