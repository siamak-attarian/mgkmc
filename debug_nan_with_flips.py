import numpy as np
import os
import shutil
from mgkmc.kmc_simulator import KmcSimulation2D
from mgkmc.finite_strain_simulator import build_finite_strain_bc, finite_strain_solver_step_2d

def debug_run():
    nx, ny = 8, 8
    M = 5
    gamma0 = 0.1
    
    E = np.ones((nx, ny)) * 70.0 * 1e9  # Pa
    nu = np.ones((nx, ny)) * 0.3
    
    output_dir = "output_debug_flips"
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except Exception:
            pass
        
    sim = KmcSimulation2D(
        nx, ny, M, gamma0, E, nu,
        pixel=1.0,
        output_dir=output_dir,
        temperature=1000.0,
        strain_rate=1.0,
        strain_assumption="finite_strain",
        plane_mode="plane_strain",
        nu0=1e11
    )
    
    # We will run simulation steps manually
    sim.driving_component = (0,0)
    sim.stress_targets = {(1, 1): 0.0}
    
    # Step 0
    sim.elastic_run(sim.eps_macro)
    print("Step 0 Sig_mean:", sim.sig_field.mean(axis=(0,1)))
    
    # Let's run a few KMC steps by calling run_simulation but we can run it step-by-step
    # Actually, we can run KMC loop logic directly here
    strain_unit = np.zeros((2,2))
    strain_unit[(0,0)] = 1.0
    dt_step = 0.001
    remaining_time = 0.0
    
    step = 1
    elastic_steps_done = 0
    
    while elastic_steps_done < 3:
        remaining_time += dt_step
        while remaining_time > 0:
            sim.update_barriers()
            from mgkmc.kmc_simulator_functions import compute_rates_2d, select_event_2d, decode_index_2d
            rates, indices, total_rate = compute_rates_2d(sim.Q, sim.volume, sim.Tlocal, sim.nu0, instability_mode="cascade")
            if total_rate > 0:
                t_wait = -np.log(np.random.rand()) / total_rate
                if t_wait < remaining_time:
                    sim.time += t_wait
                    sim.eps_macro += strain_unit * (sim.strain_rate * t_wait)
                    remaining_time -= t_wait
                    idx_flat = indices[select_event_2d(rates, total_rate)]
                    x, y, m = decode_index_2d(idx_flat, sim.ny, sim.M)
                    
                    # Print max shear of F_plastic
                    max_fp_shear = np.max(np.abs(sim.F_plastic[:, :, 0, 1]))
                    
                    C = sim.catalog[x,y,m].copy()
                    
                    Fp_old = sim.F_plastic[x, y].copy()
                    I_plus_C = np.eye(2) + C
                    det_I_plus_C = I_plus_C[0,0] * I_plus_C[1,1] - I_plus_C[0,1] * I_plus_C[1,0]
                    I_plus_C = I_plus_C / np.sqrt(max(1e-12, det_I_plus_C))
                    Fp_new = np.dot(I_plus_C, Fp_old)
                    
                    sim.F_plastic[x, y] = Fp_new
                    
                    # Update softening
                    e11, e22, e12 = C[0,0], C[1,1], C[0,1]
                    sum_sq = (e12**2) + (e22**2 + e11**2 + (e11 - e22)**2) / 6.0
                    sim.soft_prop[x,y,0] += sim.jp * sum_sq
                    sim.last_event_time[x,y] = sim.time
                    sim.prev_strain_dir[x,y] = C
                    
                    sig_prev = sim.sig_field.mean(axis=(0,1))
                    try:
                        sim.elastic_run(sim.eps_macro)
                    except Exception as ex:
                        print(f"Exception during elastic run at event {step}: {ex}")
                        
                    sig_new = sim.sig_field.mean(axis=(0,1))
                    print(f"KMC event {step}: Voxel ({x},{y}) mode {m}, Fp_shear_max: {max_fp_shear:.4f}, Sig_mean:", sig_new)
                    
                    if np.isnan(sig_new).any():
                        print("\n=== NaN DETECTED ===")
                        print(f"Event: {step}, Voxel: ({x},{y}), Mode: {m}")
                        print("C mode matrix:\n", C)
                        print("F_plastic before flip:\n", Fp_old)
                        print("F_plastic after flip:\n", Fp_new)
                        print("det of F_plastic old:", np.linalg.det(Fp_old))
                        print("det of F_plastic new:", np.linalg.det(Fp_new))
                        print("Sig_mean before this flip:\n", sig_prev)
                        
                        # Let's check K4 and Fe at that voxel
                        Fe_in = np.einsum('xyij->ijxy', sim.F_field)
                        Fp_in = np.einsum('xyij->ijxy', sim.F_plastic)
                        from mgkmc.finite_strain_simulator import _invert_Fp_2d, _dot22
                        Fe_field = _dot22(Fe_in, _invert_Fp_2d(Fp_in))
                        print("Fe at voxel:\n", Fe_field[:,:,x,y])
                        print("det of Fe at voxel:", np.linalg.det(Fe_field[:,:,x,y]))
                        break
                        
                    step += 1
                    continue
            sim.eps_macro += strain_unit * (sim.strain_rate * remaining_time)
            sim.time += remaining_time
            remaining_time = 0
            
        print(f"\n--- Running Elastic Step {elastic_steps_done+1} with eps_macro = {sim.eps_macro[0,0]} ---")
        sigM = sim.elastic_run(sim.eps_macro)
        print("Sig_mean after Elastic Step:", sigM)
        
        # Check if Sig_mean has nan
        if np.isnan(sigM).any():
            print("NaN detected! Checking F_field, F_plastic, K4...")
            # Let's inspect F_plastic determinants
            dets = [np.linalg.det(sim.F_plastic[x,y]) for x in range(nx) for y in range(ny)]
            print("F_plastic det range:", min(dets), "to", max(dets))
            
            # Let's inspect Fe determinants
            Fe = np.einsum('xyij->ijxy', sim.F_field)
            Fp = np.einsum('xyij->ijxy', sim.F_plastic)
            from mgkmc.finite_strain_simulator import _invert_Fp_2d, _dot22
            Fe_field = _dot22(Fe, _invert_Fp_2d(Fp))
            Fe_dets = [np.linalg.det(Fe_field[:,:,x,y]) for x in range(nx) for y in range(ny)]
            print("Fe det range:", min(Fe_dets), "to", max(Fe_dets))
            break
            
        elastic_steps_done += 1

if __name__ == "__main__":
    debug_run()
