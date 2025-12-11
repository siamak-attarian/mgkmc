import numpy as np
import sys
import os

# Ensure we can import mgkmc
sys.path.append(os.getcwd())

from mgkmc import AthermalSimulation
from mgkmc.stz.grid import initialize_grid
from mgkmc.stz.catalog import stz_catalog_glass
from mgkmc.stz.barriers import compute_barrier
from mgkmc.stz.update_fft import update_stress_fft_full

def analyze_flips():
    print("Initializing Analysis Simulation...")
    
    # Parameters
    nx, ny, nz = 16, 16, 1
    M = 20
    gamma0 = 0.05
    pixel = 1.0
    
    # Material
    E = np.full((nx, ny, nz), 70e9) # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    # Barriers: Use the "safe" ones (Mean=1.0) or "unsafe" (Mean=0.5)?
    # User said: "I realize... it flips in the next cascading step".
    # This implies they saw this in the UNSTABLE case.
    # So I should use the UNSTABLE parameters to reproduce the behavior they are curious about.
    # Unstable params from reproduce_crash.py: Mean=0.5, Std=0.1
    
    def unstable_barrier_generator(n_modes):
        # Use a fixed seed for reproducibility if needed, but random is fine implies general behavior
        return np.clip(np.random.normal(0.5, 0.1, n_modes), 0.1, None)

    # Initialize Grid manually
    grid = initialize_grid(nx, ny, nz, M, gamma0, 
                          barrier_generator=unstable_barrier_generator,
                          mode_generator=stz_catalog_glass)
    
    eps_macro = np.zeros((3,3))
    solver_args = {"max_iter": 200, "tol": 1e-6}
    
    total_flips_analyzed = 0
    max_flips_to_analyze = 10
    
    # Data collection for plotting
    eps_xx_history = []
    sig_xx_history = []
    
    # Intitial state (0,0)
    eps_xx_history.append(0.0)
    sig_xx_history.append(0.0)

    print("\n--- Starting Loading Output ---")
    
    # Loading Loop
    for step in range(300): # Increased limit
        if total_flips_analyzed >= max_flips_to_analyze:
            break
            
        print(f"\nGlobal Step {step}")
        
        # 1. Apply Strain
        strain_inc = np.zeros((3,3))
        strain_inc[0,0] = 5e-4 # Larger steps like reproduce_crash.py
        eps_macro += strain_inc
        
        # 2. Elastic Update
        _, _, eps_macro_out, sig_macro_out = update_stress_fft_full(grid, eps_macro, E, nu, pixel, **solver_args)
        
        # Record Elastic Predictor state
        eps_xx_history.append(eps_macro_out[0,0])
        sig_xx_history.append(sig_macro_out[0,0] / 1e9)
        
        # 3. Cascade Logic (Manual)
        local_step = 0
        while True:
            # Check stability
            # We iterate over all voxels to find unstable ones
            unstable_list = []
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                     voxel = grid[x,y,z]
                     vol = pixel**3
                     Q = compute_barrier(voxel, vol, debug=False)
                     m_min = Q.argmin()
                     if Q[m_min] < 0:
                         unstable_list.append((x,y,0, m_min, Q[m_min]))
            
            if not unstable_list:
                print(f"  Local Step {local_step}: Stable.")
                break # Next global step
                
            print(f"  Local Step {local_step}: {len(unstable_list)} unstable voxels.")
            
            # Process unstable voxels (Logic omitted for brevity in diff, assume unchanged)
            unstable_list.sort(key=lambda x: x[4]) 
            target = unstable_list[0]
            tx, ty, tz, tm, tQ = target
            print(f"    Analyzing Voxel ({tx},{ty},{tz}) flip mode {tm} (Q={tQ:.4f} eV)")
            
            # --- PRE FLIP ANALYSIS ---
            voxel_hero = grid[tx,ty,tz]
            sigma_old_Pa = voxel_hero.sigma.copy()
            gamma_hero_flip = voxel_hero.catalog[tm].copy()
            sigma_dot_gamma_old = np.sum(sigma_old_Pa * gamma_hero_flip)
            
            # --- APPLY FLIPS ---
            for (ux, uy, uz, um, uQ) in unstable_list:
                uvoxel = grid[ux,uy,uz]
                gamma_mode = uvoxel.catalog[um]
                uvoxel.eps_plastic += gamma_mode
                uvoxel.catalog = stz_catalog_glass(M, gamma0) 
                uvoxel.reset_barriers(unstable_barrier_generator)
                uvoxel.flip_count_total += 1
                
            total_flips_analyzed += len(unstable_list)
            
            # --- ELASTIC RELAXATION ---
            _, _, eps_macro_out, sig_macro_out = update_stress_fft_full(grid, eps_macro, E, nu, pixel, **solver_args)
            
            # Record Intermediate State (Post-Flip)
            eps_xx_history.append(eps_macro_out[0,0])
            sig_xx_history.append(sig_macro_out[0,0] / 1e9)
            
            # --- POST FLIP ANALYSIS (Hero Voxel) ---
            sigma_new_Pa = voxel_hero.sigma.copy()
            
            # 1. Stress Drop in the flip direction
            # We project the NEW stress onto the OLD flip direction
            sigma_dot_gamma_new = np.sum(sigma_new_Pa * gamma_hero_flip)
            
            stress_drop_Pa = sigma_dot_gamma_old - sigma_dot_gamma_new
            stress_drop_GPa = stress_drop_Pa / 1e9
            
            print(f"      Sigma:Gamma (Old) : {sigma_dot_gamma_old:.2e} Pa")
            print(f"      Sigma:Gamma (New) : {sigma_dot_gamma_new:.2e} Pa")
            print(f"      Stress Relief     : {stress_drop_GPa:.4f} GPa (Projected on flip mode)")
            
            # 2. Check New Stability
            Q_new = compute_barrier(voxel_hero, pixel**3, debug=False)
            m_min_new = Q_new.argmin()
            Q_min_new = Q_new[m_min_new]
            
            print(f"      New Stability     : Min Q = {Q_min_new:.4f} eV (Mode {m_min_new})")
            
            if Q_min_new < 0:
                print(f"      RESULT: Voxel ({tx},{ty}) is IMMEDIATELY UNSTABLE again!")
            else:
                print(f"      RESULT: Voxel ({tx},{ty}) is stable.")

            # 3. Crosstalk?
            # Check if stress increased in OTHER directions?
            # Since catalog changed, we can't check "other directions" 1-to-1.
            # But we can check if any direction is suspiciously low.
            
            local_step += 1
            
            if total_flips_analyzed >= max_flips_to_analyze:
                print("--- Reached 10 flips limit ---")
                # Record final state before breaking (or rely on loop end recording?)
                # We are breaking the inner loop.
                # We should probably record the state after the cascade?
                # Yes, standard logging is post-cascade.
                break

    # Plotting
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 6))
        plt.plot(np.array(eps_xx_history)*100, sig_xx_history, 'b-o', label='Simulation')
        plt.xlabel('Strain $\epsilon_{xx}$ (%)')
        plt.ylabel('Stress $\sigma_{xx}$ (GPa)')
        plt.title('Stress-Strain Response (First 10 Flips Analysis)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        output_img = "stress_strain_curve.png"
        plt.savefig(output_img)
        print(f"\nStress-Strain curve saved to '{output_img}'")
        # plt.show() # Optional if interactive
    except ImportError:
        print("\nMatplotlib not found. perform 'pip install matplotlib' to see the plot.")
        print("Data:")
        print("Strain (%):", np.array(eps_xx_history)*100)
        print("Stress (GPa):", sig_xx_history)

if __name__ == "__main__":
    analyze_flips()
