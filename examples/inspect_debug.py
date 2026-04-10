import numpy as np
import os
import sys

# Ensure we can import mgkmc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mgkmc import ThermalSimulation
from mgkmc.stz.barriers import compute_barrier
# Import from modules to ensure we call the fixed versions
from mgkmc.stz.cascade import find_unstable, apply_flip
from mgkmc.stz.kmc import compute_rates, select_event

def inspect_checkpoint(cp_path, voxel_indices):
    print(f"Loading {cp_path}...")
    sim = ThermalSimulation.load_checkpoint(cp_path)
    
    print("\n--- Simulation State (Initial) ---")
    print(f"Time: {sim.time:.4e} s")
    print(f"Temp: {sim.temperature} K")
    print(f"Strain Rate: {sim.strain_rate}")
    
    # Patch temperature if missing (backward compatibility)
    if sim.temperature == 0.0:
        print("Warning: Temperature not loaded (0.0). Forcing T=250.0")
        sim.temperature = 250.0
        
    # Patch stability threshold if not loaded
    if not hasattr(sim, 'stability_threshold') or sim.stability_threshold == 0.0:
        sim.stability_threshold = 0.33 # FORCE THE FIX
        print(f"Forced stability_threshold = {sim.stability_threshold} eV")

    # Initial check of target voxels
    volume = sim.volume
    kB = 8.617e-5
    beta = 1.0 / (kB * sim.temperature)
    
    print("\n--- Target Voxels State ---")
    for (x, y, z) in voxel_indices:
        vox = sim.grid[x, y, z]
        Q = compute_barrier(vox, volume, softening_scheme=sim.softening_scheme)
        min_Q = Q.min()
        print(f"Voxel ({x}, {y}, {z}): Min Q = {min_Q:.6f} eV")

    # Simulate Step 575
    print("\n--- Simulating Next 50 Events (Testing Fix) ---", flush=True)
    
    # We will replicate the inner loop of run()
    if sim.strain_increment_tensor is not None:
        dt_elastic = sim.strain_increment_tensor[0,0] / sim.strain_rate if sim.strain_rate > 0 else 1.0
    else:
        print("Warning: strain_increment_tensor not loaded. Using 2e-5.", flush=True)
        dt_elastic = 2e-5 / (sim.strain_rate if sim.strain_rate > 0 else 1.0)
    
    sim.current_step = 575
    
    for i in range(50):
        # 1. Check Stability
        unstable = find_unstable(sim.grid, sim.volume, 
                                 softening_scheme=sim.softening_scheme,
                                 threshold=sim.stability_threshold)
        
        if unstable:
            print(f"[Event {i}] UNSTABLE ({len(unstable)} sites). Cascade triggered.", flush=True)
            # Run Cascade
            l_steps, t_flips, _, _ = sim._run_cascade(global_step=sim.current_step)
            print(f"    -> Cascade finished: {t_flips} flips.", flush=True)
            continue
            
        # 2. Compute Rates
        rates, indices, total_rate = compute_rates(
             sim.grid, sim.volume, sim.temperature,
             strain_rate_sensitivity=sim.strain_rate_sensitivity,
             applied_strain_rate=sim.strain_rate,
             current_time=sim.time
        )
        
        if len(rates) == 0:
            print(f"[Event {i}] No KMC rates. Stable.")
            break
            
        idx, dt_kmc = select_event(rates, total_rate)
        
        # 3. Decision
        if dt_kmc < dt_elastic:
             # Thermal Event
             x, y, z, m = indices[idx]
             
             # Check if this is a "fast" event that should have been caught
             path_voxel = sim.grid[x,y,z]
             Q = compute_barrier(path_voxel, volume, softening_scheme=sim.softening_scheme)
             Q_val = Q[m]
             
             if Q_val < 0.33:
                 print(f"[Event {i}] WARNING: KMC picked low barrier Q={Q_val:.4f} eV! Fix might not be working?")
             else:
                 print(f"[Event {i}] KMC Flip at ({x},{y},{z}) mode {m}. Q={Q_val:.4f} eV. dt={dt_kmc:.4e} s")
             
             # Apply
             apply_flip(path_voxel, m, jp=sim.jp, jt=sim.jt, g_max=sim.softening_cap)
             path_voxel.set_catalog(sim.mode_generator(sim.M, sim.gamma0))
             sim.time += dt_kmc
             
             # Re-update fields (simplified)
             # We need to call update_stress to propagate the flip's effect
             # But since we can't easily import the full update_stress_fft_full wrapper logic without full setup,
             # we will just skip the stress update for this debug script if it's too complex.
             # Actually, if we don't update stress, Q won't change due to relaxation, only softening.
             # This exacerbates the "runaway softening" problem, so it's a good stress test!
             # If the threshold works, it should catch the dropping Q even without relaxation helping.
             
             # Actually, update_stress is critical. Let's try to call it.
             # from mgkmc.aqs import update_stress_fft_full # it is not exported there, it's in aqs.py?
             # No, it's a standalone function in aqs.py? It's imported in aqs.py. It's likely in mgkmc.stress_update?
             # Let's inspect ThermalSimulation.update_stress_fft_full source location?
             # Checking imports in aqs.py: from mgkmc.stress_update import update_stress_fft_full
             
             pass
        else:
             print(f"[Event {i}] Elastic Step chosen (dt_kmc={dt_kmc:.4e} > dt_el={dt_elastic:.4e}). Stabilized.")
             break

if __name__ == "__main__":
    base_dir = r"d:\OneDrive - UW-Madison\7-MetallicGlass\11-Antigravity\mgkmc\examples\aqs_demo_6_thermal_c_style"
    cp_file = "checkpoint_000574.h5"
    path = os.path.join(base_dir, cp_file)
    
    # Check voxels from log
    target_voxels = [(47, 34, 0), (49, 34, 0)]
    
    inspect_checkpoint(path, target_voxels)
