import os
import sys
import time
import yaml
import numpy as np
import scipy.sparse.linalg as sp
import inspect

# Ensure mgkmc is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mgkmc.finite_strain_simulator
from mgkmc.finite_strain_simulator import (
    _project, _trans2, _ddot42, cauchy_from_P,
    _fft2_tensor, _ifft2_tensor
)
from mgkmc.kmc_simulator import KmcSimulation2D
from mgkmc.microstructure import generate_field

# Statistics containers
stats_db = []
fft_times = []
ifft_times = []
constitutive_times = []
project_times = []
gkdf_times = []

# Step type tracker
current_step_type = 'init'

# Instrument FFT/IFFT
original_fft2_tensor = mgkmc.finite_strain_simulator._fft2_tensor
original_ifft2_tensor = mgkmc.finite_strain_simulator._ifft2_tensor

def instrumented_fft2_tensor(A):
    t0 = time.perf_counter()
    res = original_fft2_tensor(A)
    t1 = time.perf_counter()
    fft_times.append(t1 - t0)
    return res

def instrumented_ifft2_tensor(A_hat):
    t0 = time.perf_counter()
    res = original_ifft2_tensor(A_hat)
    t1 = time.perf_counter()
    ifft_times.append(t1 - t0)
    return res

mgkmc.finite_strain_simulator._fft2_tensor = instrumented_fft2_tensor
mgkmc.finite_strain_simulator._ifft2_tensor = instrumented_ifft2_tensor

# OPTIMIZED constitutive model
def optimized_constitutive_hyperelastic_2d(F, C4, I2, I4, I4rt, Fp=None):
    ndim = 2
    t0 = time.perf_counter()
    
    if Fp is not None:
        # Inlined Fp inversion for speed
        det = Fp[0, 0] * Fp[1, 1] - Fp[0, 1] * Fp[1, 0]
        det_safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
        Fp_inv = np.zeros_like(Fp)
        Fp_inv[0, 0] =  Fp[1, 1] / det_safe
        Fp_inv[1, 1] =  Fp[0, 0] / det_safe
        Fp_inv[0, 1] = -Fp[0, 1] / det_safe
        Fp_inv[1, 0] = -Fp[1, 0] / det_safe
        
        Fe = np.einsum('ijxy,jkxy->ikxy', F, Fp_inv, optimize=True)
        E_GL = 0.5 * (np.einsum('jixy,jkxy->ikxy', Fe, Fe, optimize=True) - I2)
        S = np.einsum('ijklxy,lkxy->ijxy', C4, E_GL, optimize=True)
        
        P = np.einsum('ikxy,klxy,jlxy->ijxy', Fe, S, Fp_inv, optimize=True)
        S_ref = np.einsum('mkxy,klxy,jlxy->mjxy', Fp_inv, S, Fp_inv, optimize=True)
        
        # Decomposed K4 contraction:
        A = np.einsum('klmnxy,jlxy->kjmnxy', C4, Fp_inv, optimize=True)
        B = np.einsum('kjmnxy,bmxy->kjbnxy', A, Fp_inv, optimize=True)
        C = np.einsum('kjbnxy,ikxy->ijbnxy', B, Fe, optimize=True)
        term2 = np.einsum('ijbnxy,anxy->ijabxy', C, Fe, optimize=True)
        
        term1 = np.einsum('bjxy,ia->ijabxy', S_ref, np.eye(ndim), optimize=True)
        K4 = term1 + term2
    else:
        E_GL = 0.5 * (np.einsum('jixy,jkxy->ikxy', F, F, optimize=True) - I2)
        S = np.einsum('ijklxy,lkxy->ijxy', C4, E_GL, optimize=True)
        
        term1 = np.einsum('ijxy,jkmnxy->ikmnxy', S, I4, optimize=True)
        
        FC4 = np.einsum('ijxy,jkmnxy->ikmnxy', F, C4, optimize=True)
        Ft = np.einsum('ijxy->jixy', F)
        FC4Ft = np.einsum('ijklxy,lmxy->ijkmxy', FC4, Ft, optimize=True)
        term2_part1 = np.einsum('ijklxy,lkmnxy->ijmnxy', I4rt, FC4Ft, optimize=True)
        term2 = np.einsum('ijklxy,lkmnxy->ijmnxy', term2_part1, I4rt, optimize=True)
        
        K4 = term1 + term2

    t1 = time.perf_counter()
    constitutive_times.append(t1 - t0)
    return P, K4

mgkmc.finite_strain_simulator.constitutive_hyperelastic_2d = optimized_constitutive_hyperelastic_2d

# Instrument project
original_project = mgkmc.finite_strain_simulator._project
def instrumented_project(*args, **kwargs):
    t0 = time.perf_counter()
    res = original_project(*args, **kwargs)
    t1 = time.perf_counter()
    project_times.append(t1 - t0)
    return res
mgkmc.finite_strain_simulator._project = instrumented_project

# Instrument finite_strain_solver_step_2d
def instrumented_finite_strain_solver_step_2d(
    F, F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
    driving_component=(0, 0), P_target=None, P_mask=None,
    E_avg=100e9, nu_avg=0.3,
    tol_NW=1e-5, tol_CG=1e-6, max_NW=20,  # MODIFIED tol_CG from 1e-8 to 1e-6
    tol_macro=1e6, max_iter_macro=20,
    enable_console=True
):
    ndim = 2
    nx, ny = F.shape[2], F.shape[3]
    if P_target is None:
        P_target = np.zeros((ndim, ndim))
    if P_mask is None:
        P_mask = np.zeros((ndim, ndim), dtype=bool)

    # Detect current phase from call stack
    stack = inspect.stack()
    if len(stack) > 2:
        caller_frame = stack[2]
        caller_name = caller_frame.function
        lineno = caller_frame.lineno
    else:
        caller_name = "unknown"
        lineno = 0
    
    if caller_name == '_run_cascade':
        step_type = 'instability'
    elif caller_name == 'run_simulation':
        if lineno < 465:
            step_type = 'init'
        elif lineno >= 580:
            step_type = 'elastic'
        else:
            step_type = 'stz'
    else:
        step_type = f'unknown_caller_{caller_name}'

    F_start = F.copy()
    F_bar_current = F_bar.copy()
    
    F_final = F_start.copy()
    P_final = None
    Sig_final = None
    K4_final = None
    max_err = 0.0

    solver_t0 = time.perf_counter()
    
    # Store detailed stats for this solver call
    call_stats = {
        'step_type': step_type,
        'strain_xx': F_bar_current[0, 0] - 1.0,
        'macro_iterations': []
    }

    actual_macro_iters = 0
    for it_mac in range(max_iter_macro):
        actual_macro_iters += 1
        mac_stats = {
            'macro_iter': it_mac,
            'newton_iterations': []
        }
        
        DbarF = F_bar_current - F_start.mean(axis=(2, 3))
        DbarF_grid = np.einsum('ij,xy->ijxy', DbarF, np.ones((nx, ny)))
        
        P, K4 = optimized_constitutive_hyperelastic_2d(F_start, C4, I2, I4, I4rt, Fp=Fp)
        
        def G_op(A2):
            return instrumented_project(A2, Ghat4)
            
        def K_dF_op(dFm_flat):
            dF = dFm_flat.reshape(ndim, ndim, nx, ny)
            return _trans2(_ddot42(K4, _trans2(dF)))
            
        def G_K_dF(dFm_flat):
            t0 = time.perf_counter()
            res = G_op(K_dF_op(dFm_flat)).reshape(-1)
            t1 = time.perf_counter()
            gkdf_times.append(t1 - t0)
            return res
            
        A_op = sp.LinearOperator(
            shape=(F_start.size, F_start.size),
            matvec=G_K_dF,
            dtype='float64'
        )
        
        F_curr = F_start.copy()
        
        for i_NW in range(max_NW):
            if i_NW == 0:
                rhs = -G_op(K_dF_op(DbarF_grid.reshape(-1))).reshape(-1)
            else:
                rhs = -G_op(P).reshape(-1)
                
            # Run BiCGSTAB and time it, counting iterations
            bicg_iters = [0]
            def bicg_callback(xk):
                bicg_iters[0] += 1
                
            t_bicg0 = time.perf_counter()
            try:
                dFm, _ = sp.bicgstab(A_op, rhs, rtol=tol_CG, maxiter=150, callback=bicg_callback)
            except TypeError:
                dFm, _ = sp.bicgstab(A_op, rhs, tol=tol_CG, maxiter=150, callback=bicg_callback)
            t_bicg1 = time.perf_counter()
            
            dF = dFm.reshape(ndim, ndim, nx, ny)
            
            if i_NW == 0:
                F_curr = F_curr + DbarF_grid + dF
            else:
                F_curr = F_curr + dF
                
            P, K4 = optimized_constitutive_hyperelastic_2d(F_curr, C4, I2, I4, I4rt, Fp=Fp)
            
            res_norm = np.linalg.norm(dFm) / (np.linalg.norm(F_curr) + 1e-20)
            
            mac_stats['newton_iterations'].append({
                'newton_step': i_NW,
                'bicgstab_iters': bicg_iters[0],
                'bicgstab_time': t_bicg1 - t_bicg0,
                'res_norm': res_norm
            })
            
            if res_norm < tol_NW and i_NW > 0:
                break
        
        F_mac = F_curr.mean(axis=(2, 3))
        P_mac = P.mean(axis=(2, 3))
        Sig_field = cauchy_from_P(P, F_curr)
        Sig_mac = Sig_field.mean(axis=(2, 3))
        
        F_final = F_curr
        P_final = P
        Sig_final = Sig_field
        K4_final = K4

        call_stats['macro_iterations'].append(mac_stats)

        if not np.any(P_mask):
            break
            
        stress_err = np.zeros((ndim, ndim))
        stress_err[P_mask] = P_target[P_mask] - Sig_mac[P_mask]
        max_err = np.max(np.abs(stress_err[P_mask]))
        
        mac_stats['stress_err'] = max_err
        
        if max_err < tol_macro:
            break
            
        i_drv, j_drv = driving_component
        d_F_mat = (stress_err - nu_avg * np.trace(stress_err) * np.eye(ndim)) / E_avg
        for ii in range(ndim):
            for jj in range(ndim):
                if not (P_mask[ii, jj] and ii == jj):
                    continue
                if (ii, jj) == (i_drv, j_drv):
                    continue
                F_bar_current[ii, jj] += d_F_mat[ii, jj]
    else:
        if enable_console and np.any(P_mask):
            print(f"  Warning: outer BC loop did not converge (max_err={max_err:.2e} Pa)")
            
    solver_t1 = time.perf_counter()
    call_stats['total_time'] = solver_t1 - solver_t0
    call_stats['actual_macro_iters'] = actual_macro_iters
    stats_db.append(call_stats)
    
    return F_final, P_final, Sig_final, K4_final, F_bar_current

# Apply monkey patch to both modules to be safe
mgkmc.finite_strain_simulator.finite_strain_solver_step_2d = instrumented_finite_strain_solver_step_2d
mgkmc.kmc_simulator.finite_strain_solver_step_2d = instrumented_finite_strain_solver_step_2d

def run_profile():
    config_path = r"D:\GoogleDrive\2-MGKMC\mgkmc\test_both\antigravity_test\config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Apply pyfftw thread limit globally
    import pyfftw
    num_threads = config.get('system', {}).get('num_threads', 1)
    pyfftw.config.NUM_THREADS = int(num_threads)
        
    seed = config.get('seed', 42)
    np.random.seed(seed)
    
    sys_conf = config['system']
    dimensionality = sys_conf.get('dimensionality', '2d').lower()
    plane_mode = sys_conf.get('plane_mode', 'plane_strain').lower()
    nx, ny = sys_conf['nx'], sys_conf['ny']
    
    mat_conf = config['material']
    E_field = generate_field(
        mat_conf['E']['mode'], 
        (nx, ny), 
        constant_val=mat_conf['E'].get('value', 70.0),
        params=mat_conf['E'].get('parameters', {})
    )
    nu_field = generate_field(
        mat_conf['nu']['mode'], 
        (nx, ny), 
        constant_val=mat_conf['nu'].get('value', 0.3),
        params=mat_conf['nu'].get('parameters', {})
    )

    if E_field.mean() < 1e6:
        E_field = E_field * 1e9

    phys_conf = config.get('physics', {})
    dyn_conf = config.get('dynamics', {})
    out_conf = config.get('output', {})
    det_conf = config.get('detection', {})
    bar_conf = config.get('barriers', {})
    bar_type = bar_conf.get('type', 'gaussian')
    bar_kwargs = bar_conf.get('kwargs', {})
    
    print("Initializing instrumented KmcSimulation2D...")
    sim = KmcSimulation2D(
        nx, ny,
        M=sys_conf['M'],
        gamma0=sys_conf['gamma0'],
        E_field=E_field,
        nu_field=nu_field,
        pixel=sys_conf.get('pixel', 1.0),
        plane_mode=plane_mode,
        barrier_generator=bar_type,
        barrier_kwargs=bar_kwargs,
        jp=phys_conf.get('jp', 20),
        jt=phys_conf.get('jt', 20),
        neighbor_softening_fraction=phys_conf.get('neighbor_softening_fraction', 0.0),
        softening_scheme=phys_conf.get('softening_scheme', 'isotropic'),
        softening_cap=phys_conf.get('softening_cap', 2.0),
        q_act_temp=phys_conf.get('q_act_temp', 0.37),
        redraw_directions=phys_conf.get('redraw_directions', True),
        redraw_barriers=phys_conf.get('redraw_barriers', True),
        output_dir=out_conf.get('directory', 'output'),
        temperature=float(dyn_conf.get('temperature', 0.0)),
        strain_rate=float(dyn_conf.get('physical_strain_rate', 1.0e7)),
        nu0=float(dyn_conf.get('nu0', 1.0e13)),
        stability_threshold=phys_conf.get('stability_threshold', 0.0),
        fast_patching=dyn_conf.get('fast_patching', None),
        instability_mode=dyn_conf.get('instability_mode', 'cascade'),
        cascade_timing=dyn_conf.get('cascade_timing', 'none'),
        scale_rate_by_volume=dyn_conf.get('scale_rate_by_volume', False),
        strain_assumption=sys_conf.get('strain_assumption', 'small_strain')
    )

    bc_conf = config['boundary_conditions']
    driving_raw = bc_conf.get('driving_component', 'xx')
    comp_map = {'xx': (0, 0), 'yy': (1, 1), 'xy': (0, 1), 'yx': (1, 0)}
    component = comp_map[driving_raw]
    
    stress_targets = {}
    for k_str, val in bc_conf.get('mixed_targets', {}).items():
        stress_targets[comp_map[k_str]] = float(val) * 1e9 if float(val) < 1e6 else float(val)

    loading_conf = config.get('loading', {})
    eps_target = float(loading_conf.get('eps_target', 0.001))
    step_size = float(loading_conf.get('step_size', 0.0001))
    calculated_n_steps = int(eps_target / step_size)

    print(f"Starting simulation run for {calculated_n_steps} steps...")
    t_start = time.perf_counter()
    sim.run_simulation(
        n_global_steps=calculated_n_steps,
        step_size=step_size,
        component=component,
        stress_targets=stress_targets,
        mixed_tol=float(bc_conf.get('mixed_tol', 1e-4)),
        enable_console_log=False, # Suppress normal logging to keep stdout clean
        enable_summary_log=True,
        enable_global_log=True
    )
    t_end = time.perf_counter()
    total_run_time = t_end - t_start
    print(f"Simulation finished in {total_run_time:.2f} seconds.")
    
    # Process and present stats
    print("\n" + "="*80)
    print("PROFILING REPORT")
    print("="*80)
    
    print(f"Total physical runtime: {total_run_time:.2f} seconds")
    print(f"Total FFT calls: {len(fft_times)}")
    print(f"Total IFFT calls: {len(ifft_times)}")
    if fft_times:
        print(f"Average FFT time: {np.mean(fft_times)*1000:.3f} ms (Total: {np.sum(fft_times):.3f} s)")
    if ifft_times:
        print(f"Average IFFT time: {np.mean(ifft_times)*1000:.3f} ms (Total: {np.sum(ifft_times):.3f} s)")
    if constitutive_times:
        print(f"Average Constitutive time: {np.mean(constitutive_times)*1000:.3f} ms (Total: {np.sum(constitutive_times):.3f} s, Calls: {len(constitutive_times)})")
    if project_times:
        print(f"Average Projection (G_hat : A) time: {np.mean(project_times)*1000:.3f} ms (Total: {np.sum(project_times):.3f} s, Calls: {len(project_times)})")
    if gkdf_times:
        print(f"Average G_K_dF (matvec) time: {np.mean(gkdf_times)*1000:.3f} ms (Total: {np.sum(gkdf_times):.3f} s, Calls: {len(gkdf_times)})")
        
    print("\n" + "-"*60)
    print("SOLVER STEPS SUMMARY")
    print("-"*60)
    
    by_type = {}
    for call in stats_db:
        stype = call['step_type']
        if stype not in by_type:
            by_type[stype] = []
        by_type[stype].append(call)
        
    for stype, calls in by_type.items():
        n_calls = len(calls)
        total_time = sum(c['total_time'] for c in calls)
        avg_time = total_time / n_calls
        
        # Macro iterations (lateral contraction steps)
        macro_iters = [c['actual_macro_iters'] for c in calls]
        avg_macro_iters = np.mean(macro_iters)
        max_macro_iters = np.max(macro_iters)
        
        # Newton iterations per macro iteration
        newton_iters = []
        bicg_iters = []
        bicg_times = []
        for c in calls:
            for m in c['macro_iterations']:
                newton_iters.append(len(m['newton_iterations']))
                for nw in m['newton_iterations']:
                    bicg_iters.append(nw['bicgstab_iters'])
                    bicg_times.append(nw['bicgstab_time'])
                    
        avg_nw = np.mean(newton_iters) if newton_iters else 0
        avg_bicg = np.mean(bicg_iters) if bicg_iters else 0
        avg_bicg_time = np.mean(bicg_times) if bicg_times else 0
        
        print(f"Step Type: {stype.upper()}")
        print(f"  Total Calls: {n_calls}")
        print(f"  Total Time: {total_time:.3f} s (Avg: {avg_time:.3f} s per call)")
        print(f"  Lateral Contraction (Macro) Iterations:")
        print(f"    Avg: {avg_macro_iters:.2f} iterations (Max: {max_macro_iters})")
        print(f"  Newton-Raphson iterations per Macro step:")
        print(f"    Avg: {avg_nw:.2f} iterations")
        print(f"  BiCGSTAB iterations per Newton step:")
        print(f"    Avg: {avg_bicg:.2f} iterations")
        print(f"  Average BiCGSTAB time: {avg_bicg_time*1000:.2f} ms")
        print()
        
    print("-"*60)
    print("DETAILED RUNTIME PER STEP VS STRAIN")
    print("-"*60)
    print(f"{'Index':<5} {'Type':<12} {'Strain (%)':<12} {'Total Time (s)':<15} {'Macro Iters':<12} {'Avg Newton/Mac':<15} {'Avg BiCG/Newton':<15}")
    for idx, c in enumerate(stats_db):
        stype = c['step_type']
        strain_pct = c['strain_xx'] * 100.0
        tot_time = c['total_time']
        mac_it = c['actual_macro_iters']
        
        nw_counts = [len(m['newton_iterations']) for m in c['macro_iterations']]
        avg_nw = np.mean(nw_counts) if nw_counts else 0
        
        bicg_counts = []
        for m in c['macro_iterations']:
            for nw in m['newton_iterations']:
                bicg_counts.append(nw['bicgstab_iters'])
        avg_bicg = np.mean(bicg_counts) if bicg_counts else 0
        
        print(f"{idx:<5} {stype:<12} {strain_pct:<12.4f} {tot_time:<15.3f} {mac_it:<12d} {avg_nw:<15.1f} {avg_bicg:<15.1f}")
        
    print("="*80)

if __name__ == "__main__":
    run_profile()
