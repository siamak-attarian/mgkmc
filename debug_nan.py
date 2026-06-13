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
    
    output_dir = "output_debug"
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
        plane_mode="plane_strain"
    )
    
    # Run the simulation steps manually
    sim.driving_component = (0,0)
    sim.stress_targets = {(1, 1): 0.0}
    
    sim.elastic_run(sim.eps_macro)
    
    # Run 40 KMC steps (flips)
    print("Running 40 manual KMC flips...")
    for i in range(40):
        sim.update_barriers()
        from mgkmc.kmc_simulator_functions import find_unstable_2d, decode_index_2d
        unstable = find_unstable_2d(sim.Q, sim.stability_threshold)
        if len(unstable) > 0:
            x, y, m = unstable[0]
            C = sim.catalog[x,y,m].copy()
            I_plus_C = np.eye(2) + C
            sim.F_plastic[x, y] = np.dot(I_plus_C, sim.F_plastic[x, y])
            
            e11, e22, e12 = C[0,0], C[1,1], C[0,1]
            sum_sq = (e12**2) + (e22**2 + e11**2 + (e11 - e22)**2) / 6.0
            gp_new = sim.soft_prop[x,y,0] + sim.jp * sum_sq
            sim.soft_prop[x,y,0] = gp_new
            sim.soft_prop[x,y,1] = sim.jt * sum_sq
            sim.last_event_time[x,y] = sim.time
            sim.prev_strain_dir[x,y] = C
            
        sim.elastic_run(sim.eps_macro)
        
    print("Done 40 KMC steps. Sig_mean:", sim.sig_field.mean(axis=(0,1)))
    
    # Now run step 41 (the first ELAST step with eps_macro = 0.001)
    print("\n--- Running Step 41 ---")
    sim.eps_macro[0, 0] += 0.001
    
    # We will run the solver step manually with prints
    eps_s = sim.eps_macro[0, 0]
    F_bar, F_mask, P_tgt, P_mask = build_finite_strain_bc(
        (0,0), eps_s, sim.stress_targets, sim.plane_mode
    )
    
    F_in = np.einsum('xyij->ijxy', sim.F_field)
    Fp_in = np.einsum('xyij->ijxy', sim.F_plastic)
    
    print("F_in mean:", F_in.mean(axis=(2,3)))
    print("Fp_in mean:", Fp_in.mean(axis=(2,3)))
    print("F_bar initially:", F_bar)
    
    # Call solver step with print inside
    F_start = F_in.copy()
    F_bar_current = F_bar.copy()
    
    for it_mac in range(10):
        DbarF = F_bar_current - F_start.mean(axis=(2, 3))
        DbarF_grid = np.einsum('ij,xy->ijxy', DbarF, np.ones((nx, ny)))
        
        from mgkmc.finite_strain_simulator import constitutive_hyperelastic_2d, cauchy_from_P, _project
        import scipy.sparse.linalg as sp
        
        P, K4 = constitutive_hyperelastic_2d(F_start, sim.C4_fs, sim.I2_fs, sim.I4_fs, sim.I4rt_fs, Fp=Fp_in)
        
        print(f"\nMacro Iter {it_mac}:")
        print("P mean:", P.mean(axis=(2,3)))
        print("K4 finite values:", np.isfinite(K4).all())
        
        def G_op(A2):
            return _project(A2, sim.Ghat4_fs)
            
        def K_dF_op(dFm_flat):
            dF = dFm_flat.reshape(2, 2, nx, ny)
            from mgkmc.finite_strain_simulator import _trans2, _ddot42
            return _trans2(_ddot42(K4, _trans2(dF)))
            
        def G_K_dF(dFm_flat):
            return G_op(K_dF_op(dFm_flat)).reshape(-1)
            
        A_op = sp.LinearOperator(
            shape=(F_start.size, F_start.size),
            matvec=G_K_dF,
            dtype='float64'
        )
        
        rhs = -G_op(K_dF_op(DbarF_grid.reshape(-1))).reshape(-1)
        print("rhs norm:", np.linalg.norm(rhs))
        
        dFm, info = sp.cg(A_op, rhs, tol=1e-8)
        print("CG info:", info, "dFm norm:", np.linalg.norm(dFm))
        
        dF = dFm.reshape(2, 2, nx, ny)
        F_curr = F_start + DbarF_grid + dF
        
        P, K4 = constitutive_hyperelastic_2d(F_curr, sim.C4_fs, sim.I2_fs, sim.I4_fs, sim.I4rt_fs, Fp=Fp_in)
        Sig_field = cauchy_from_P(P, F_curr)
        Sig_mac = Sig_field.mean(axis=(2, 3))
        
        print("Sig_mac:", Sig_mac)
        
        stress_err = np.zeros((2, 2))
        stress_err[P_mask] = P_tgt[P_mask] - Sig_mac[P_mask]
        print("stress_err:", stress_err)
        
        E_avg = sim.E_field.mean()
        nu_avg = sim.nu_field.mean()
        d_F_mat = (stress_err - nu_avg * np.trace(stress_err) * np.eye(2)) / E_avg
        print("d_F_mat:", d_F_mat)
        
        F_bar_current += d_F_mat
        print("F_bar_current updated:", F_bar_current)

if __name__ == "__main__":
    debug_run()
