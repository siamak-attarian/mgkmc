import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Ensure mgkmc is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mgkmc import ThermalSimulation
from mgkmc.stz.barriers import compute_barrier
from mgkmc.stz.kmc import compute_rates

def analyze_checkpoint(cp_path):
    print(f"Loading {cp_path}...")
    try:
        sim = ThermalSimulation.load_checkpoint(cp_path)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Patch for old checkpoints or safety
    if not hasattr(sim, 'stability_threshold'):
        sim.stability_threshold = 0.33
        
    print(f"\n--- Simulation State ---")
    print(f"Step: {sim.current_step}")
    print(f"Time: {sim.time:.4e} s")
    print(f"Temp: {sim.temperature} K (kT = {8.617e-5 * sim.temperature:.4f} eV)")
    print(f"Strain Rate: {sim.strain_rate} /s")
    
    # 1. Compute Full Rate Spectrum
    print("\nComputing Rates...")
    kB = 8.617e-5
    beta = 1.0 / (kB * sim.temperature) if sim.temperature > 0 else 1.0
    
    # We need to manually compute barriers to analyze components
    # Loop over all voxels is expensive? 128x128x1 = 16384 voxels. Manageable.
    
    n_voxels = sim.nx * sim.ny * sim.nz
    all_Q = []
    all_rates = []
    
    
    # Pre-calculate barriers for all voxels (since they are not saved in checkpoint)
    print("Pre-calculating barriers...")
    for x in range(sim.nx):
        for y in range(sim.ny):
            for z in range(sim.nz):
                voxel = sim.grid[x,y,z]
                voxel.Q = compute_barrier(voxel, sim.volume, softening_scheme=sim.softening_scheme)

    # Sample analysis (Top 20 fastest)
    # We'll compute rates using the vectorized function first to find hot spots
    rates, indices, total_rate = compute_rates(
        sim.grid, sim.volume, sim.temperature,
        strain_rate_sensitivity=sim.strain_rate_sensitivity,
        applied_strain_rate=sim.strain_rate,
        current_time=sim.time
    )
    
    print(f"Total System Rate: {total_rate:.4e} Hz")
    print(f"Expected time to next KMC event: {1.0/total_rate:.4e} s")
    
    if len(rates) == 0:
        print("No finite rates found (System is frozen).")
        return

    # Sort rates descending
    sorted_idx = np.argsort(rates)[::-1]
    top_n = 20
    
    print(f"\n--- Top {top_n} Fastest Events ---")
    print(f"{'Rank':<5} {'Loc':<12} {'Mode':<5} {'Q (eV)':<10} {'Rate (Hz)':<12} {'dt (s)':<10}")
    
    sim.stability_threshold = 0.33
    
    for rank, i in enumerate(sorted_idx[:top_n]):
        flat_idx = i # indices in 'rates' array correspond to 'indices' list
        x, y, z, m = indices[flat_idx]
        rate = rates[flat_idx]
        
        # Re-compute detailed barrier info for this voxel
        voxel = sim.grid[x,y,z]
        # Q_eff = Q0 - Bias - Softening
        # We need to peek inside compute_barrier logic or approximate it
        # Actually, let's just inspect the voxel closely
        
        # Calculate Work (Bias)
        # work = 0.5 * volume * sum(eps0 * sigma) * conversion
        # This is hard to replicate exactly without importing internal constants
        # But we can get Q directly
        Q_all = compute_barrier(voxel, sim.volume, softening_scheme=sim.softening_scheme)
        Q_val = Q_all[m]
        
        # Is it unstable?
        status = "STABLE"
        if Q_val < sim.stability_threshold:
            status = "UNSTABLE (Should Cascade!)"
            
        print(f"{rank+1:<5} ({x},{y},{z})  {m:<5} {Q_val:<10.4f} {rate:<12.4e} {1.0/rate:<10.4e} {status}")

    # 2. Global Softening Statistics
    print("\n--- Softening Field Statistics ---")
    gp_vals = []
    gt_vals = []
    for x in range(sim.nx):
        for y in range(sim.ny):
             v = sim.grid[x,y,0]
             gp_vals.append(v.g_p)
             gt_vals.append(v.g_t)
             
    gp_vals = np.array(gp_vals)
    gt_vals = np.array(gt_vals)
    
    print(f"g_p (Plastic): Min={gp_vals.min():.4f}, Max={gp_vals.max():.4f}, Mean={gp_vals.mean():.4f}")
    print(f"g_t (Thermal): Min={gt_vals.min():.4f}, Max={gt_vals.max():.4f}, Mean={gt_vals.mean():.4f}")
    
    # 3. Elastic vs KMC Comparison
    if sim.strain_increment_tensor is not None:
         # Estimate dt_elastic
         d_eps = abs(sim.strain_increment_tensor[0,0])
         if d_eps == 0: d_eps = 1e-5 # Fallback
         dt_elastic = d_eps / sim.strain_rate
         print(f"\n--- Time Step Comparison ---")
         print(f"dt_elastic (approx): {dt_elastic:.4e} s")
         print(f"dt_KMC (mean):       {1.0/total_rate:.4e} s")
         print(f"Ratio (KMC steps per Elastic step): {dt_elastic * total_rate:.1f}")
         
         if dt_elastic * total_rate > 1000:
             print("\nCONCLUSION: The simulation is KMC-dominated because there are low barriers (~0.4-0.5 eV)")
             print("relative to the temperature (250K) and strain rate.")
    
    # Plot Histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Re-compute ALL barriers for histogram (Subsampling if needed)
    # We will sample 1000 voxels to save time if grid is huge
    sample_voxels = []
    for _ in range(1000):
        rx, ry = np.random.randint(0, sim.nx), np.random.randint(0, sim.ny)
        sample_voxels.append(sim.grid[rx, ry, 0])
        
    all_sample_Q = []
    for v in sample_voxels:
        qs = compute_barrier(v, sim.volume, softening_scheme=sim.softening_scheme)
        all_sample_Q.extend(qs)
        
    plt.hist(all_sample_Q, bins=50, alpha=0.7, color='blue', label='Q (Sampled)')
    plt.axvline(x=sim.stability_threshold, color='r', linestyle='--', label='Stability Threshold')
    plt.xlabel('Activation Energy Q (eV)')
    plt.ylabel('Count (Modes)')
    plt.title('Barrier Distribution (Sampled)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(gp_vals, bins=50, alpha=0.7, color='green')
    plt.xlabel('Plastic Softening g_p')
    plt.ylabel('Count (Voxels)')
    plt.title('Softening State')
    
    plt.tight_layout()
    plt.savefig('investigation_plots.png')
    print("\nPlots saved to 'investigation_plots.png'")

if __name__ == "__main__":
    # Find latest checkpoint
    base_dir = r"d:\OneDrive - UW-Madison\7-MetallicGlass\11-Antigravity\mgkmc\examples\aqs_demo_6_thermal_c_style"
    
    checkpoints = [f for f in os.listdir(base_dir) if f.startswith("checkpoint") and f.endswith(".h5")]
    if not checkpoints:
        print("No checkpoints found.")
        sys.exit(1)
        
    # Sort by step number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_cp = checkpoints[-1]
    
    path = os.path.join(base_dir, latest_cp)
    analyze_checkpoint(path)
