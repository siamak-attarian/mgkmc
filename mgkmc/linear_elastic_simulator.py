import numpy as np
from .elasticity import (compute_lame_2d, stress_from_strain_2d, green_operator_2d,
                         compute_lame_3d, stress_from_strain_3d, green_operator_3d,
                         stress_from_strain_secant_2d, stress_from_strain_secant_3d,
                         secant_shear_field, von_mises_strain_2d, von_mises_strain_3d,
                         stress_from_strain_landau_2d, stress_from_strain_landau_3d)
from .fft import compute_wave_vectors_2d, compute_wave_vectors_3d, fft_field, ifft_field
from .analysis.vtk import export_to_vtk
import pyfftw.interfaces.numpy_fft as fft

def spectral_solver_2d(E, nu, eps_bar, eps_plastic=None,
                       max_iter=200, tol=1e-6,
                       verbose=False, pixel=1.0, plane_mode="plane_strain",
                       Gamma=None, eps_init=None):
    """
    Single-point 2D Lippmann-Schwinger solver.
    Gamma    : pre-computed Green operator (nx,ny,2,2,2,2). Computed here if None.
    eps_init : optional warm-start field (nx,ny,2,2). Starts from uniform if None.
    """
    nx, ny = E.shape
    E_avg  = E.mean()
    nu_avg = nu.mean()

    if Gamma is None:
        lam0, mu0 = compute_lame_2d(E_avg, nu_avg, plane_mode)
        Lx, Ly = nx * pixel, ny * pixel
        kx, ky = compute_wave_vectors_2d(nx, ny, Lx, Ly)
        Gamma  = green_operator_2d(kx, ky, lam0, mu0)

    # Initialise field
    if eps_init is not None:
        eps = eps_init.copy()
        for i in range(2):
            for j in range(2):
                eps[:, :, i, j] += eps_bar[i, j] - eps_init[:, :, i, j].mean()
    else:
        eps = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps[:, :, i, j] = eps_bar[i, j]

    # Lippmann-Schwinger iteration
    for it in range(max_iter):
        if eps_plastic is not None:
            sig = stress_from_strain_2d(eps - eps_plastic, E, nu, plane_mode)
        else:
            sig = stress_from_strain_2d(eps, E, nu, plane_mode)

        sig0 = stress_from_strain_2d(eps, E * 0 + E_avg, nu * 0 + nu_avg, plane_mode)
        tau  = sig - sig0

        tau_hat = np.zeros((nx, ny, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                tau_hat[:, :, i, j] = fft_field(tau[:, :, i, j])

        eps_tilde_hat = -np.einsum("xykhij,xyij->xykh", Gamma, tau_hat)

        eps_tilde = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps_tilde[:, :, i, j] = ifft_field(eps_tilde_hat[:, :, i, j])

        for i in range(2):
            for j in range(2):
                eps_tilde[:, :, i, j] -= eps_tilde[:, :, i, j].mean()

        eps_new = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps_new[:, :, i, j] = eps_bar[i, j] + eps_tilde[:, :, i, j]

        diff = np.linalg.norm(eps_new - eps) / (np.linalg.norm(eps) + 1e-20)
        eps  = eps_new

        if verbose and it % 10 == 0:
            print(f"Iter {it:03d}: diff = {diff:.3e}")

        if diff < tol:
            break

    if eps_plastic is not None:
        sig_out = stress_from_strain_2d(eps - eps_plastic, E, nu, plane_mode)
    else:
        sig_out = stress_from_strain_2d(eps, E, nu, plane_mode)

    return eps, sig_out, eps.mean(axis=(0, 1)), sig_out.mean(axis=(0, 1))

def linear_elastic_simulation_2d(
    E, nu,
    target_strain_mask,
    target_values,
    n_steps=20,
    pixel=1.0,
    tol_ls=1e-6,
    max_iter_ls=200,
    tol_macro=1e-4,
    max_iter_macro=20,
    store=True,
    plane_mode="plane_strain",
    log_path=None,
    global_log_path=None,
    driving_component=(0, 0),
    enable_console=True,
    vtk_path=None,
    vtk_interval="none",
    **solver_kw
):
    """
    Mixed stress/strain control for raw 2D -- fast version.
    Green operator is pre-computed once; spectral_solver_2d is warm-started
    between load steps (Gamma and eps_init passed explicitly).
    log_path          : optional path to a summary log text file.
    driving_component : (i,j) tuple used to label the log columns.
    """
    import os, time as _time
    from datetime import datetime as _dt
    nx, ny = E.shape

    _comp_labels = {(0,0):"xx",(1,1):"yy",(0,1):"xy",(1,0):"yx",(0,2):"xz",(1,2):"yz"}
    _comp_lbl = _comp_labels.get(tuple(driving_component), "xx")

    _log_f = None
    if log_path:
        _log_f = open(log_path, "w", buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'Eps_'+_comp_lbl:<14} {'Sig_'+_comp_lbl+'(GPa)':<16}\n")
        _log_f.write(_hdr)
        _log_f.write("-" * len(_hdr.rstrip()) + "\n")
    # global_log: all 2D strain+stress components
    _glog_f = None
    if global_log_path:
        _glog_f = open(global_log_path, "w", buffering=1)
        _ghdr = (f"{'GlobalStep':<12} {'Eps_xx':<14} {'Eps_yy':<14} {'Eps_xy':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_xy(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write("-" * len(_ghdr.rstrip()) + "\n")
    _t0 = _time.time()

    # Pre-compute Gamma ONCE
    lam0, mu0 = compute_lame_2d(E.mean(), nu.mean(), plane_mode)
    Lx, Ly    = nx * pixel, ny * pixel
    kx, ky    = compute_wave_vectors_2d(nx, ny, Lx, Ly)
    Gamma     = green_operator_2d(kx, ky, lam0, mu0)

    E_avg  = E.mean()
    nu_avg = nu.mean()

    def get_strain_correction_2d(sigma_err):
        tr_sig = np.trace(sigma_err)
        return (sigma_err - nu_avg * tr_sig * np.eye(2)) / E_avg

    chk_ival = solver_kw.pop("checkpoint_interval", None)
    chk_path = solver_kw.pop("checkpoint_path", "checkpoint")
    if chk_path and chk_ival not in [None, "none"]:
        os.makedirs(os.path.dirname(chk_path) or ".", exist_ok=True)

    targets_path = np.zeros((n_steps + 1, 2, 2))
    for s in range(n_steps + 1):
        targets_path[s] = target_values * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list       = []
    sig_list       = []

    current_eps_bar = np.zeros((2, 2))
    eps_warm        = None          # warm-start field carried forward
    stress_mask     = ~target_strain_mask

    for s in range(n_steps + 1):
        target_s = targets_path[s]
        max_err  = 0.0

        for it_macro in range(max_iter_macro):
            current_eps_bar[target_strain_mask] = target_s[target_strain_mask]

            # spectral_solver_2d now owns the LS loop
            eps, sig, epsM, sigM = spectral_solver_2d(
                E, nu, current_eps_bar,
                max_iter=max_iter_ls, tol=tol_ls,
                pixel=pixel, plane_mode=plane_mode,
                Gamma=Gamma, eps_init=eps_warm
            )

            stress_err = np.zeros((2, 2))
            stress_err[stress_mask] = target_s[stress_mask] - sigM[stress_mask]
            max_err = np.max(np.abs(stress_err[stress_mask])) if np.any(stress_mask) else 0.0

            if max_err < tol_macro:
                break

            d_eps = get_strain_correction_2d(stress_err)
            current_eps_bar[stress_mask] += d_eps[stress_mask]
        else:
            print(f"Warning: Macro loop did not converge at step {s} (err={max_err:.2e})")

        eps_warm = eps  # pass forward as warm-start

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)
        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        _i, _j = driving_component
        _eps_drv = epsM[_i, _j]
        _sig_drv = sigM[_i, _j]
        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"eps_{_comp_lbl}={_eps_drv:.4f}, "
                  f"sig_{_comp_lbl}={_sig_drv/1e9:.4f} GPa")
        if _log_f:
            _now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{_eps_drv:<14.6f} {_sig_drv/1e9:<16.6f}\n"
            )
        if _glog_f:
            _glog_f.write(
                f"{s:<12} "
                f"{epsM[0,0]:<14.6f} {epsM[1,1]:<14.6f} {epsM[0,1]:<14.6f} "
                f"{sigM[0,0]/1e9:<16.6f} {sigM[1,1]/1e9:<16.6f} {sigM[0,1]/1e9:<16.6f}\n"
            )

        if chk_ival is not None and chk_ival not in ["none", "last"]:
            save_chk, cp_name = False, None
            if chk_ival == "current":
                save_chk, cp_name = True, f"{chk_path}.h5"
            elif isinstance(chk_ival, int) and s % chk_ival == 0:
                save_chk, cp_name = True, f"{chk_path}_{s:06d}.h5"
            if save_chk and cp_name:
                save_checkpoint_2d(cp_name, s, E, nu, eps, sig, epsM, sigM, pixel)

        if vtk_interval is not None and vtk_interval not in ["none", "last"]:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                export_to_vtk(vt_name, eps, sig, E, nu, pixel, match_matplotlib_orientation=True)

    if chk_ival == "last":
        save_checkpoint_2d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)
    elif chk_ival not in [None, "none", "last"] and isinstance(chk_ival, int) and n_steps % chk_ival != 0:
        save_checkpoint_2d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)

    if vtk_interval == "last":
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E, nu, pixel, match_matplotlib_orientation=True)
    elif vtk_interval not in [None, "none", "last"] and isinstance(vtk_interval, int) and n_steps % vtk_interval != 0:
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E, nu, pixel, match_matplotlib_orientation=True)

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()
    if enable_console:
        print(_duration_str)

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)

def spectral_solver_3d(E, nu, eps_bar, eps_plastic=None,
                       max_iter=200, tol=1e-6,
                       verbose=False, pixel=1.0,
                       Gamma=None, eps_init=None):
    """
    3D Lippmann-Schwinger solver.
    Gamma    : pre-computed Green operator. Computed here if None.
    eps_init : optional warm-start field. Starts from uniform if None.
    """
    nx, ny, nz = E.shape
    E_avg  = E.mean()
    nu_avg = nu.mean()

    if Gamma is None:
        lam0, mu0 = compute_lame_3d(E_avg, nu_avg)
        Lx, Ly, Lz = nx*pixel, ny*pixel, nz*pixel
        kx, ky, kz = compute_wave_vectors_3d(nx, ny, nz, Lx, Ly, Lz)
        Gamma = green_operator_3d(kx, ky, kz, lam0, mu0)

    if eps_init is not None:
        eps = eps_init.copy()
        for i in range(3):
            for j in range(3):
                eps[:, :, :, i, j] += eps_bar[i, j] - eps_init[:, :, :, i, j].mean()
    else:
        eps = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps[:, :, :, i, j] = eps_bar[i, j]
    
    # ----------------------------------------------------------------------
    # MAIN Lippmann-Schwinger iteration
    # ----------------------------------------------------------------------
    for it in range(max_iter):

        # Stress from heterogeneous moduli: C : (eps - eps_plastic)
        if eps_plastic is not None:
             sig = stress_from_strain_3d(eps - eps_plastic, E, nu)
        else:
             sig = stress_from_strain_3d(eps, E, nu)

        # Reference stress (homogeneous medium): C0 : eps
        sig0 = stress_from_strain_3d(
            eps,
            E * 0 + E_avg,
            nu * 0 + nu_avg
        )
        
        # Polarization stress
        tau = sig - sig0  # [nx,ny,nz,3,3]
        
        # FFT of tau
        tau_hat = np.zeros((nx, ny, nz, 3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                tau_hat[:, :, :, i, j] = fft_field(tau[:, :, :, i, j])
        
        # Apply Green operator
        eps_tilde_hat = -np.einsum("xyzkhij,xyzij->xyzkh", Gamma, tau_hat)
        
        # Inverse FFT
        eps_tilde = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps_tilde[:, :, :, i, j] = ifft_field(eps_tilde_hat[:, :, :, i, j])
        
        # Zero mean value of each component
        for i in range(3):
            for j in range(3):
                eps_tilde[:, :, :, i, j] -= eps_tilde[:, :, :, i, j].mean()
        
        # ------------------------------------------------------------------
        # UPDATE: eps_total = eps_bar + eps_tilde
        # ------------------------------------------------------------------
        eps_new = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps_new[:, :, :, i, j] = (
                    eps_bar[i, j]
                    + eps_tilde[:, :, :, i, j]
                )
        
        # Convergence check
        diff = np.linalg.norm(eps_new - eps) / (np.linalg.norm(eps) + 1e-20)
        eps = eps_new
        
        if verbose and it % 10 == 0:
            print(f"Iter {it:03d}: Δε/ε = {diff:.3e}")
        
        if diff < tol:
            break
    
    # ----------------------------------------------------------------------
    # Final output
    # ----------------------------------------------------------------------
    if eps_plastic is not None:
        sig_out = stress_from_strain_3d(eps - eps_plastic, E, nu)
    else:
        sig_out = stress_from_strain_3d(eps, E, nu)
    eps_macro = eps.mean(axis=(0, 1, 2))
    sig_macro = sig_out.mean(axis=(0, 1, 2))
    
    return eps, sig_out, eps_macro, sig_macro

def linear_elastic_simulation_3d(
    E, nu,
    target_strain_mask,
    target_values,
    n_steps=20,
    pixel=1.0,
    tol_ls=1e-6,
    max_iter_ls=200,
    tol_macro=1e-4,
    max_iter_macro=20,
    store=True,
    log_path=None,
    global_log_path=None,
    driving_component=(0, 0),
    enable_console=True,
    vtk_path=None,
    vtk_interval="none",
    **solver_kw
):
    """
    3D linear elastic mixed stress/strain simulation.
    Gamma pre-computed once; spectral_solver_3d warm-started between steps.
    log_path          : summary_log.txt path (Timestamp/Elapsed/Step/driving comp)
    global_log_path   : global_log.txt path (all 6 independent components)
    driving_component : (i,j) tuple
    enable_console    : print per-step info to console
    """
    import os, time as _time
    from datetime import datetime as _dt
    nx, ny, nz = E.shape

    _comp_labels = {(0,0):"xx",(1,1):"yy",(2,2):"zz",(0,1):"xy",(0,2):"xz",(1,2):"yz",(1,0):"yx",(2,0):"zx",(2,1):"zy"}
    _comp_lbl = _comp_labels.get(tuple(driving_component), "xx")

    # Pre-compute Gamma ONCE
    lam0, mu0 = compute_lame_3d(E.mean(), nu.mean())
    Lx, Ly, Lz = nx * pixel, ny * pixel, nz * pixel
    kx, ky, kz = compute_wave_vectors_3d(nx, ny, nz, Lx, Ly, Lz)
    Gamma = green_operator_3d(kx, ky, kz, lam0, mu0)

    E_avg  = E.mean()
    nu_avg = nu.mean()

    def get_strain_correction_3d(sigma_err):
        tr_sig = np.trace(sigma_err)
        return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg

    # Summary log
    _log_f = None
    if log_path:
        _log_f = open(log_path, "w", buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'Eps_'+_comp_lbl:<14} {'Sig_'+_comp_lbl+'(GPa)':<16}\n")
        _log_f.write(_hdr)
        _log_f.write("-" * len(_hdr.rstrip()) + "\n")

    # Global log (all 6 independent components)
    _glog_f = None
    if global_log_path:
        _glog_f = open(global_log_path, "w", buffering=1)
        _ghdr = (f"{'GlobalStep':<12} "
                 f"{'Eps_xx':<14} {'Eps_yy':<14} {'Eps_zz':<14} {'Eps_xy':<14} {'Eps_xz':<14} {'Eps_yz':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_zz(GPa)':<16} "
                 f"{'Sig_xy(GPa)':<16} {'Sig_xz(GPa)':<16} {'Sig_yz(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write("-" * len(_ghdr.rstrip()) + "\n")
    _t0 = _time.time()

    chk_ival = solver_kw.pop("checkpoint_interval", None)
    chk_path = solver_kw.pop("checkpoint_path", "checkpoint")
    if chk_path and chk_ival not in [None, "none"]:
        os.makedirs(os.path.dirname(chk_path) or ".", exist_ok=True)

    targets_path = np.zeros((n_steps + 1, 3, 3))
    for s in range(n_steps + 1):
        targets_path[s] = target_values * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list       = []
    sig_list       = []

    current_eps_bar = np.zeros((3, 3))
    eps_warm        = None
    stress_mask     = ~target_strain_mask

    for s in range(n_steps + 1):
        target_s = targets_path[s]
        max_err  = 0.0

        for it_macro in range(max_iter_macro):
            current_eps_bar[target_strain_mask] = target_s[target_strain_mask]

            eps, sig, epsM, sigM = spectral_solver_3d(
                E, nu, current_eps_bar,
                max_iter=max_iter_ls, tol=tol_ls,
                pixel=pixel,
                Gamma=Gamma, eps_init=eps_warm
            )

            stress_err = np.zeros((3, 3))
            stress_err[stress_mask] = target_s[stress_mask] - sigM[stress_mask]
            max_err = np.max(np.abs(stress_err[stress_mask])) if np.any(stress_mask) else 0.0

            if max_err < tol_macro:
                break

            d_eps = get_strain_correction_3d(stress_err)
            current_eps_bar[stress_mask] += d_eps[stress_mask]
        else:
            print(f"Warning: Macro loop did not converge at step {s} (err={max_err:.2e})")

        eps_warm = eps

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)
        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        _i, _j = driving_component
        _eps_drv = epsM[_i, _j]
        _sig_drv = sigM[_i, _j]
        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"eps_{_comp_lbl}={_eps_drv:.4f}, "
                  f"sig_{_comp_lbl}={_sig_drv/1e9:.4f} GPa")

        if _log_f:
            _now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{_eps_drv:<14.6f} {_sig_drv/1e9:<16.6f}\n"
            )
        if _glog_f:
            _glog_f.write(
                f"{s:<12} "
                f"{epsM[0,0]:<14.6f} {epsM[1,1]:<14.6f} {epsM[2,2]:<14.6f} "
                f"{epsM[0,1]:<14.6f} {epsM[0,2]:<14.6f} {epsM[1,2]:<14.6f} "
                f"{sigM[0,0]/1e9:<16.6f} {sigM[1,1]/1e9:<16.6f} {sigM[2,2]/1e9:<16.6f} "
                f"{sigM[0,1]/1e9:<16.6f} {sigM[0,2]/1e9:<16.6f} {sigM[1,2]/1e9:<16.6f}\n"
            )

        if chk_ival is not None and chk_ival not in ["none", "last"]:
            save_chk, cp_name = False, None
            if chk_ival == "current":
                save_chk, cp_name = True, f"{chk_path}.h5"
            elif isinstance(chk_ival, int) and s % chk_ival == 0:
                save_chk, cp_name = True, f"{chk_path}_{s:06d}.h5"
            if save_chk and cp_name:
                save_checkpoint_3d(cp_name, s, E, nu, eps, sig, epsM, sigM, pixel)

        if vtk_interval is not None and vtk_interval not in ["none", "last"]:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                export_to_vtk(vt_name, eps, sig, E, nu, pixel, match_matplotlib_orientation=True)

    if chk_ival == "last":
        save_checkpoint_3d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)
    elif chk_ival not in [None, "none", "last"] and isinstance(chk_ival, int) and n_steps % chk_ival != 0:
        save_checkpoint_3d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)

    if vtk_interval == "last":
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E, nu, pixel, match_matplotlib_orientation=True)
    elif vtk_interval not in [None, "none", "last"] and isinstance(vtk_interval, int) and n_steps % vtk_interval != 0:
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E, nu, pixel, match_matplotlib_orientation=True)

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()
    if enable_console:
        print(_duration_str)

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)

def save_checkpoint_2d(path, step, E, nu, eps_field, sig_field, eps_mac, sig_mac, pixel):
    """Saves pure 2D elastic state to an HDF5 checkpoint."""
    import h5py
    import os
    if not path.endswith('.h5'): path += '.h5'
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(path, "w") as f:
        meta = f.create_group('metadata')
        meta.attrs['nx'], meta.attrs['ny'] = E.shape
        meta.attrs['pixel'] = pixel
        meta.attrs['step'] = step
        
        fields = f.create_group('fields')
        fields.create_dataset('eps_field', data=eps_field, compression='gzip')
        fields.create_dataset('sig_field', data=sig_field, compression='gzip')
        fields.create_dataset('E_field', data=E, compression='gzip')
        fields.create_dataset('nu_field', data=nu, compression='gzip')
        
        macro = f.create_group('macro')
        macro.create_dataset('eps_macro', data=eps_mac)
        macro.create_dataset('sig_macro', data=sig_mac)

def save_checkpoint_3d(path, step, E, nu, eps_field, sig_field, eps_mac, sig_mac, pixel):
    """Saves pure 3D elastic state to an HDF5 checkpoint."""
    import h5py, os
    if not path.endswith('.h5'): path += '.h5'
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(path, "w") as f:
        meta = f.create_group('metadata')
        meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz'] = E.shape
        meta.attrs['pixel'] = pixel
        meta.attrs['step'] = step
        fields = f.create_group('fields')
        fields.create_dataset('eps_field', data=eps_field, compression='gzip')
        fields.create_dataset('sig_field', data=sig_field, compression='gzip')
        fields.create_dataset('E_field',   data=E,         compression='gzip')
        fields.create_dataset('nu_field',  data=nu,        compression='gzip')
        macro = f.create_group('macro')
        macro.create_dataset('eps_macro', data=eps_mac)
        macro.create_dataset('sig_macro', data=sig_mac)


# ============================================================================
# Secant Elastic Degradation — nonlinear Lippmann-Schwinger solvers
# ============================================================================

def spectral_solver_secant_2d(lam, mu, d, k,
                              eps_bar, eps_plastic=None,
                              max_iter=400, tol=1e-6,
                              verbose=False, pixel=1.0,
                              plane_mode="plane_strain",
                              Gamma=None, eps_init=None):
    """
    Nonlinear Lippmann-Schwinger solver for the 2-D Secant Elastic
    Degradation model.

    Constitutive law
    ----------------
    sigma = lam * tr(eps - eps_plastic) * I + 2 * mu_sec(eps_eq) * (eps - eps_plastic)
    mu_sec(eps_eq) = mu * [1 - d * (1 - exp(-k * eps_eq))]
    eps_eq = von_mises_strain_2d(eps - eps_plastic)

    Algorithm (nonlinear Moulinec-Suquet fixed-point)
    -------------------------------------------------
    Reference stiffness C^0 = (lam_avg, mu_avg) is held *fixed* (undegraded).
    Each iteration:
        1. Compute sigma_secant(eps - eps_plastic).
        2. Compute reference stress  sigma^0 = lam_avg*tr(eps - eps_plastic)*I + 2*mu_avg*(eps - eps_plastic).
        3. Polarisation stress  tau = sigma_secant - sigma^0.
        4. Apply Green operator:  eps_tilde = -Gamma^0 * tau.
        5. Update:  eps = eps_bar + eps_tilde  (mean-corrected).

    Parameters
    ----------
    lam : float or ndarray, shape (nx, ny)
        First Lame parameter lambda (Pa).
        For plane stress pass the effective lambda* = E*nu/(1-nu^2).
    mu : float or ndarray, shape (nx, ny)
        Undegraded shear modulus (Pa).
    d : float
        Degradation magnitude (0 <= d <= 1).
    k : float
        Degradation rate (dimensionless).
    eps_bar : ndarray, shape (2, 2)
        Prescribed macro-average strain.
    eps_plastic : ndarray or None, shape (nx, ny, 2, 2)
        Plastic strain tensor field.
    max_iter : int
        Maximum fixed-point iterations (default 400).
    tol : float
        Relative convergence tolerance on eps field.
    verbose : bool
        Print convergence info every 10 iterations.
    pixel : float
        Voxel size (nm); only affects the Green operator.
    plane_mode : str
        "plane_strain" or "plane_stress".
    Gamma : ndarray or None
        Pre-computed Green operator (nx,ny,2,2,2,2).
        Computed here if None.
    eps_init : ndarray or None
        Warm-start field (nx,ny,2,2). Starts from uniform if None.

    Returns
    -------
    eps : ndarray, shape (nx, ny, 2, 2)
    sig : ndarray, shape (nx, ny, 2, 2)
    eps_macro : ndarray, shape (2, 2)
    sig_macro : ndarray, shape (2, 2)
    """
    lam  = np.asarray(lam, dtype=float)
    mu   = np.asarray(mu, dtype=float)
    nx, ny = lam.shape if lam.ndim == 2 else mu.shape

    # Reference (undegraded) Lame averages for C^0
    lam_avg = float(lam.mean()) if lam.ndim > 0 else float(lam)
    mu_avg  = float(mu.mean()) if mu.ndim > 0 else float(mu)

    # Pre-compute Green operator if not provided
    if Gamma is None:
        lam0_ref, mu_ref = lam_avg, mu_avg
        Lx, Ly = nx * pixel, ny * pixel
        from .fft import compute_wave_vectors_2d
        from .elasticity import green_operator_2d
        kx, ky = compute_wave_vectors_2d(nx, ny, Lx, Ly)
        Gamma = green_operator_2d(kx, ky, lam0_ref, mu_ref)

    # Initialise strain field
    if eps_init is not None:
        eps = eps_init.copy()
        for i in range(2):
            for j in range(2):
                eps[:, :, i, j] += eps_bar[i, j] - eps_init[:, :, i, j].mean()
    else:
        eps = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps[:, :, i, j] = eps_bar[i, j]

    # Lambda broadcast for reference stress computation
    lam_bc  = lam[..., None, None]   # (nx,ny,1,1) or scalar
    mu_bc   = mu[..., None, None]
    I2      = np.eye(2)[None, None, :, :]

    for it in range(max_iter):
        eps_el = eps - eps_plastic if eps_plastic is not None else eps
        # 1. Secant stress
        sig = stress_from_strain_secant_2d(
            eps_el, lam, mu, d, k, plane_mode=plane_mode
        )

        # 2. Reference stress  sigma^0 = C^0 : eps  (linear, mu = mu_avg)
        tr_eps = np.trace(eps, axis1=2, axis2=3)[..., None, None]
        sig0   = lam_avg * tr_eps * I2 + 2.0 * mu_avg * eps

        # 3. Polarisation
        tau = sig - sig0   # (nx, ny, 2, 2)

        # 4. FFT of tau
        tau_hat = np.zeros((nx, ny, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                tau_hat[:, :, i, j] = fft_field(tau[:, :, i, j])

        # 5. Apply Green operator
        eps_tilde_hat = -np.einsum("xykhij,xyij->xykh", Gamma, tau_hat)

        eps_tilde = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps_tilde[:, :, i, j] = ifft_field(eps_tilde_hat[:, :, i, j])

        # 6. Zero-mean correction
        for i in range(2):
            for j in range(2):
                eps_tilde[:, :, i, j] -= eps_tilde[:, :, i, j].mean()

        # 7. Update eps
        eps_new = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps_new[:, :, i, j] = eps_bar[i, j] + eps_tilde[:, :, i, j]

        diff = np.linalg.norm(eps_new - eps) / (np.linalg.norm(eps) + 1e-20)
        eps  = eps_new

        if verbose and it % 10 == 0:
            print(f"  [secant-LS] iter {it:04d}: rel_diff = {diff:.3e}")

        if diff < tol:
            break

    # Final secant stress
    eps_el = eps - eps_plastic if eps_plastic is not None else eps
    sig_out  = stress_from_strain_secant_2d(eps_el, lam, mu, d, k, plane_mode=plane_mode)
    eps_macro = eps.mean(axis=(0, 1))
    sig_macro = sig_out.mean(axis=(0, 1))
    return eps, sig_out, eps_macro, sig_macro


def spectral_solver_secant_3d(lam, mu, d, k,
                              eps_bar, eps_plastic=None,
                              max_iter=400, tol=1e-6,
                              verbose=False, pixel=1.0,
                              Gamma=None, eps_init=None):
    """
    Nonlinear Lippmann-Schwinger solver for the 3-D Secant Elastic
    Degradation model.

    See ``spectral_solver_secant_2d`` for full documentation.
    """
    lam  = np.asarray(lam, dtype=float)
    mu   = np.asarray(mu, dtype=float)
    nx, ny, nz = (lam.shape if lam.ndim == 3 else mu.shape)

    lam_avg = float(lam.mean())
    mu_avg  = float(mu.mean())

    if Gamma is None:
        Lx, Ly, Lz = nx * pixel, ny * pixel, nz * pixel
        from .fft import compute_wave_vectors_3d
        from .elasticity import green_operator_3d
        kx, ky, kz = compute_wave_vectors_3d(nx, ny, nz, Lx, Ly, Lz)
        Gamma = green_operator_3d(kx, ky, kz, lam_avg, mu_avg)

    if eps_init is not None:
        eps = eps_init.copy()
        for i in range(3):
            for j in range(3):
                eps[:, :, :, i, j] += eps_bar[i, j] - eps_init[:, :, :, i, j].mean()
    else:
        eps = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps[:, :, :, i, j] = eps_bar[i, j]

    I3 = np.eye(3)[None, None, None, :, :]

    for it in range(max_iter):
        eps_el = eps - eps_plastic if eps_plastic is not None else eps
        sig = stress_from_strain_secant_3d(eps_el, lam, mu, d, k)

        tr_eps = np.trace(eps, axis1=3, axis2=4)[..., None, None]
        sig0   = lam_avg * tr_eps * I3 + 2.0 * mu_avg * eps

        tau = sig - sig0

        tau_hat = np.zeros((nx, ny, nz, 3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                tau_hat[:, :, :, i, j] = fft_field(tau[:, :, :, i, j])

        eps_tilde_hat = -np.einsum("xyzkhij,xyzij->xyzkh", Gamma, tau_hat)

        eps_tilde = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps_tilde[:, :, :, i, j] = ifft_field(eps_tilde_hat[:, :, :, i, j])

        for i in range(3):
            for j in range(3):
                eps_tilde[:, :, :, i, j] -= eps_tilde[:, :, :, i, j].mean()

        eps_new = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps_new[:, :, :, i, j] = eps_bar[i, j] + eps_tilde[:, :, :, i, j]

        diff = np.linalg.norm(eps_new - eps) / (np.linalg.norm(eps) + 1e-20)
        eps  = eps_new

        if verbose and it % 10 == 0:
            print(f"  [secant-LS 3D] iter {it:04d}: rel_diff = {diff:.3e}")

        if diff < tol:
            break

    eps_el = eps - eps_plastic if eps_plastic is not None else eps
    sig_out   = stress_from_strain_secant_3d(eps_el, lam, mu, d, k)
    eps_macro = eps.mean(axis=(0, 1, 2))
    sig_macro = sig_out.mean(axis=(0, 1, 2))
    return eps, sig_out, eps_macro, sig_macro


def spectral_solver_landau_2d(lam, mu, v1, v2, v3, g1, g2, g3, g4,
                              eps_bar, eps_plastic=None,
                              max_iter=400, tol=1e-6,
                              verbose=False, pixel=1.0,
                              plane_mode="plane_strain",
                              Gamma=None, eps_init=None):
    """
    Nonlinear Lippmann-Schwinger solver for the 2-D Landau small-strain model.
    """
    lam  = np.asarray(lam, dtype=float)
    mu   = np.asarray(mu, dtype=float)
    nx, ny = lam.shape if lam.ndim == 2 else mu.shape

    # Reference Lame averages for C^0
    lam_avg = float(lam.mean()) if lam.ndim > 0 else float(lam)
    mu_avg  = float(mu.mean()) if mu.ndim > 0 else float(mu)

    # Pre-compute Green operator if not provided
    if Gamma is None:
        lam0_ref, mu_ref = lam_avg, mu_avg
        Lx, Ly = nx * pixel, ny * pixel
        from .fft import compute_wave_vectors_2d
        from .elasticity import green_operator_2d
        kx, ky = compute_wave_vectors_2d(nx, ny, Lx, Ly)
        Gamma = green_operator_2d(kx, ky, lam0_ref, mu_ref)

    # Initialise strain field
    if eps_init is not None:
        eps = eps_init.copy()
        for i in range(2):
            for j in range(2):
                eps[:, :, i, j] += eps_bar[i, j] - eps_init[:, :, i, j].mean()
    else:
        eps = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps[:, :, i, j] = eps_bar[i, j]

    I2 = np.eye(2)[None, None, :, :]

    for it in range(max_iter):
        eps_el = eps - eps_plastic if eps_plastic is not None else eps
        # 1. Landau stress
        sig = stress_from_strain_landau_2d(
            eps_el, lam, mu, v1, v2, v3, g1, g2, g3, g4, plane_mode=plane_mode
        )

        # 2. Reference stress sigma^0 = C^0 : eps
        tr_eps = np.trace(eps, axis1=2, axis2=3)[..., None, None]
        sig0   = lam_avg * tr_eps * I2 + 2.0 * mu_avg * eps

        # 3. Polarisation
        tau = sig - sig0

        # 4. FFT of tau
        tau_hat = np.zeros((nx, ny, 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                tau_hat[:, :, i, j] = fft_field(tau[:, :, i, j])

        # 5. Apply Green operator
        eps_tilde_hat = -np.einsum("xykhij,xyij->xykh", Gamma, tau_hat)

        eps_tilde = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps_tilde[:, :, i, j] = ifft_field(eps_tilde_hat[:, :, i, j])

        # 6. Zero-mean correction
        for i in range(2):
            for j in range(2):
                eps_tilde[:, :, i, j] -= eps_tilde[:, :, i, j].mean()

        # 7. Update eps
        eps_new = np.zeros((nx, ny, 2, 2))
        for i in range(2):
            for j in range(2):
                eps_new[:, :, i, j] = eps_bar[i, j] + eps_tilde[:, :, i, j]

        diff = np.linalg.norm(eps_new - eps) / (np.linalg.norm(eps) + 1e-20)
        eps  = eps_new

        if verbose and it % 10 == 0:
            print(f"  [landau-LS] iter {it:04d}: rel_diff = {diff:.3e}")

        if diff < tol:
            break

    # Final Landau stress
    eps_el = eps - eps_plastic if eps_plastic is not None else eps
    sig_out  = stress_from_strain_landau_2d(eps_el, lam, mu, v1, v2, v3, g1, g2, g3, g4, plane_mode=plane_mode)
    eps_macro = eps.mean(axis=(0, 1))
    sig_macro = sig_out.mean(axis=(0, 1))
    return eps, sig_out, eps_macro, sig_macro


def spectral_solver_landau_3d(lam, mu, v1, v2, v3, g1, g2, g3, g4,
                              eps_bar, eps_plastic=None,
                              max_iter=400, tol=1e-6,
                              verbose=False, pixel=1.0,
                              Gamma=None, eps_init=None):
    """
    Nonlinear Lippmann-Schwinger solver for the 3-D Landau small-strain model.
    """
    lam  = np.asarray(lam, dtype=float)
    mu   = np.asarray(mu, dtype=float)
    nx, ny, nz = lam.shape if lam.ndim == 3 else mu.shape

    # Reference Lame averages for C^0
    lam_avg = float(lam.mean()) if lam.ndim > 0 else float(lam)
    mu_avg  = float(mu.mean()) if mu.ndim > 0 else float(mu)

    # Pre-compute Green operator if not provided
    if Gamma is None:
        lam0_ref, mu_ref = lam_avg, mu_avg
        Lx, Ly, Lz = nx * pixel, ny * pixel, nz * pixel
        from .fft import compute_wave_vectors_3d
        from .elasticity import green_operator_3d
        kx, ky, kz = compute_wave_vectors_3d(nx, ny, nz, Lx, Ly, Lz)
        Gamma = green_operator_3d(kx, ky, kz, lam0_ref, mu_ref)

    # Initialise strain field
    if eps_init is not None:
        eps = eps_init.copy()
        for i in range(3):
            for j in range(3):
                eps[:, :, :, i, j] += eps_bar[i, j] - eps_init[:, :, :, i, j].mean()
    else:
        eps = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps[:, :, :, i, j] = eps_bar[i, j]

    I3 = np.eye(3)[None, None, None, :, :]

    for it in range(max_iter):
        eps_el = eps - eps_plastic if eps_plastic is not None else eps
        # 1. Landau stress
        sig = stress_from_strain_landau_3d(
            eps_el, lam, mu, v1, v2, v3, g1, g2, g3, g4
        )

        # 2. Reference stress sigma^0 = C^0 : eps
        tr_eps = np.trace(eps, axis1=3, axis2=4)[..., None, None]
        sig0   = lam_avg * tr_eps * I3 + 2.0 * mu_avg * eps

        # 3. Polarisation
        tau = sig - sig0

        # 4. FFT of tau
        tau_hat = np.zeros((nx, ny, nz, 3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                tau_hat[:, :, :, i, j] = fft_field(tau[:, :, :, i, j])

        # 5. Apply Green operator
        eps_tilde_hat = -np.einsum("xyzkhij,xyzij->xyzkh", Gamma, tau_hat)

        eps_tilde = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps_tilde[:, :, :, i, j] = ifft_field(eps_tilde_hat[:, :, :, i, j])

        # 6. Zero-mean correction
        for i in range(3):
            for j in range(3):
                eps_tilde[:, :, :, i, j] -= eps_tilde[:, :, :, i, j].mean()

        # 7. Update eps
        eps_new = np.zeros((nx, ny, nz, 3, 3))
        for i in range(3):
            for j in range(3):
                eps_new[:, :, :, i, j] = eps_bar[i, j] + eps_tilde[:, :, :, i, j]

        diff = np.linalg.norm(eps_new - eps) / (np.linalg.norm(eps) + 1e-20)
        eps  = eps_new

        if verbose and it % 10 == 0:
            print(f"  [landau-LS 3D] iter {it:04d}: rel_diff = {diff:.3e}")

        if diff < tol:
            break

    # Final Landau stress
    eps_el = eps - eps_plastic if eps_plastic is not None else eps
    sig_out  = stress_from_strain_landau_3d(eps_el, lam, mu, v1, v2, v3, g1, g2, g3, g4)
    eps_macro = eps.mean(axis=(0, 1, 2))
    sig_macro = sig_out.mean(axis=(0, 1, 2))
    return eps, sig_out, eps_macro, sig_macro


def secant_elastic_simulation_2d(
    lam, mu, d, k,
    target_strain_mask,
    target_values,
    n_steps=20,
    pixel=1.0,
    tol_ls=1e-6,
    max_iter_ls=400,
    tol_macro=1e-4,
    max_iter_macro=20,
    store=True,
    plane_mode="plane_strain",
    log_path=None,
    global_log_path=None,
    driving_component=(0, 0),
    enable_console=True,
    vtk_path=None,
    vtk_interval="none",
    **solver_kw
):
    """
    Mixed stress/strain-controlled 2-D simulation using the Secant Elastic
    Degradation constitutive law.

    Interface mirrors ``linear_elastic_simulation_2d`` exactly, except that
    instead of (E, nu) the material is defined by (lam, mu0, d, k).

    Parameters
    ----------
    lam : ndarray, shape (nx, ny)
        First Lame parameter lambda (Pa).
    mu0 : ndarray, shape (nx, ny)
        Undegraded shear modulus (Pa).
    d : float
        Degradation magnitude (0 <= d <= 1).
    k : float
        Degradation rate (dimensionless).
    target_strain_mask : ndarray bool, shape (2, 2)
        True  -> component is strain-controlled.
        False -> component is stress-controlled.
    target_values : ndarray, shape (2, 2)
        Target strain (Pa) or stress (Pa) for each component.
    n_steps : int
        Number of loading increments.
    pixel : float
        Voxel size (nm).
    tol_ls : float
        Convergence tolerance for the inner LS iteration.
    max_iter_ls : int
        Max iterations for the inner LS solver.
    tol_macro : float
        Tolerance (Pa) for the outer stress-control loop.
    max_iter_macro : int
        Max outer iterations.
    store : bool
        If True, store full per-step field arrays.
    plane_mode : str
        "plane_strain" or "plane_stress".
    log_path : str or None
        Path to summary log file.
    global_log_path : str or None
        Path to global (all components) log file.
    driving_component : tuple (i, j)
        Strain component used for log column labels.
    enable_console : bool
        Print per-step info to stdout.
    vtk_path : str or None
        Base path for VTK output files.
    vtk_interval : str or int
        "none", "last", "current", or integer step interval.

    Returns
    -------
    eps_macro_arr : ndarray, shape (n_steps+1, 2, 2)
    sig_macro_arr : ndarray, shape (n_steps+1, 2, 2)
    eps_list      : list of ndarray (nx, ny, 2, 2)  [only if store=True]
    sig_list      : list of ndarray (nx, ny, 2, 2)  [only if store=True]
    """
    import os, time as _time
    from datetime import datetime as _dt

    lam = np.asarray(lam, dtype=float)
    mu  = np.asarray(mu, dtype=float)
    nx, ny = lam.shape

    _comp_labels = {(0,0):"xx",(1,1):"yy",(0,1):"xy",(1,0):"yx"}
    _comp_lbl = _comp_labels.get(tuple(driving_component), "xx")

    # Pre-compute Green operator once (using undegraded averages as reference)
    lam_avg = float(lam.mean())
    mu_avg  = float(mu.mean())
    Lx, Ly  = nx * pixel, ny * pixel
    kx, ky  = compute_wave_vectors_2d(nx, ny, Lx, Ly)
    Gamma   = green_operator_2d(kx, ky, lam_avg, mu_avg)

    # Strain-correction helper for stress-controlled components
    # Approximate inverse using undegraded isotropic elastic moduli
    E_avg  = mu_avg * (3.0 * lam_avg + 2.0 * mu_avg) / (lam_avg + mu_avg)
    nu_avg = lam_avg / (2.0 * (lam_avg + mu_avg))

    def get_strain_correction_2d(sigma_err):
        tr_sig = np.trace(sigma_err)
        return (sigma_err - nu_avg * tr_sig * np.eye(2)) / E_avg

    # Logging
    _log_f = None
    if log_path:
        _log_f = open(log_path, "w", buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'Eps_'+_comp_lbl:<14} {'Sig_'+_comp_lbl+'(GPa)':<16}\n")
        _log_f.write(_hdr)
        _log_f.write("-" * len(_hdr.rstrip()) + "\n")
    _glog_f = None
    if global_log_path:
        _glog_f = open(global_log_path, "w", buffering=1)
        _ghdr = (f"{'GlobalStep':<12} {'Eps_xx':<14} {'Eps_yy':<14} {'Eps_xy':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_xy(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write("-" * len(_ghdr.rstrip()) + "\n")
    _t0 = _time.time()

    # Checkpoint / VTK settings from solver_kw
    chk_ival = solver_kw.pop("checkpoint_interval", None)
    chk_path = solver_kw.pop("checkpoint_path", "checkpoint")
    if chk_path and chk_ival not in [None, "none"]:
        os.makedirs(os.path.dirname(chk_path) or ".", exist_ok=True)

    targets_path = np.zeros((n_steps + 1, 2, 2))
    for s in range(n_steps + 1):
        targets_path[s] = target_values * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list       = []
    sig_list       = []

    current_eps_bar = np.zeros((2, 2))
    eps_warm        = None
    stress_mask     = ~target_strain_mask

    for s in range(n_steps + 1):
        target_s = targets_path[s]
        max_err  = 0.0

        for it_macro in range(max_iter_macro):
            current_eps_bar[target_strain_mask] = target_s[target_strain_mask]

            eps, sig, epsM, sigM = spectral_solver_secant_2d(
                lam, mu, d, k,
                current_eps_bar,
                max_iter=max_iter_ls, tol=tol_ls,
                pixel=pixel, plane_mode=plane_mode,
                Gamma=Gamma, eps_init=eps_warm
            )

            stress_err = np.zeros((2, 2))
            stress_err[stress_mask] = target_s[stress_mask] - sigM[stress_mask]
            max_err = (np.max(np.abs(stress_err[stress_mask]))
                       if np.any(stress_mask) else 0.0)

            if max_err < tol_macro:
                break

            d_eps = get_strain_correction_2d(stress_err)
            current_eps_bar[stress_mask] += d_eps[stress_mask]
        else:
            print(f"Warning: Macro loop did not converge at step {s} (err={max_err:.2e})")

        eps_warm = eps

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)
        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        _i, _j = driving_component
        _eps_drv = epsM[_i, _j]
        _sig_drv = sigM[_i, _j]
        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"eps_{_comp_lbl}={_eps_drv:.4f}, "
                  f"sig_{_comp_lbl}={_sig_drv/1e9:.4f} GPa")
        if _log_f:
            _now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{_eps_drv:<14.6f} {_sig_drv/1e9:<16.6f}\n"
            )
        if _glog_f:
            _glog_f.write(
                f"{s:<12} "
                f"{epsM[0,0]:<14.6f} {epsM[1,1]:<14.6f} {epsM[0,1]:<14.6f} "
                f"{sigM[0,0]/1e9:<16.6f} {sigM[1,1]/1e9:<16.6f} {sigM[0,1]/1e9:<16.6f}\n"
            )

        if chk_ival is not None and chk_ival not in ["none", "last"]:
            save_chk, cp_name = False, None
            if chk_ival == "current":
                save_chk, cp_name = True, f"{chk_path}.h5"
            elif isinstance(chk_ival, int) and s % chk_ival == 0:
                save_chk, cp_name = True, f"{chk_path}_{s:06d}.h5"
            if save_chk and cp_name:
                save_checkpoint_2d(cp_name, s, None, None, eps, sig, epsM, sigM, pixel)

        if vtk_interval is not None and vtk_interval not in ["none", "last"]:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                # Derive pseudo-E/nu for VTK export from secant moduli at final step
                mu_sec = secant_shear_field(von_mises_strain_2d(eps), mu, d, k)
                E_vtk  = mu_sec * (3.0 * lam + 2.0 * mu_sec) / (lam + mu_sec + 1e-30)
                nu_vtk = lam / (2.0 * (lam + mu_sec) + 1e-30)
                export_to_vtk(vt_name, eps, sig, E_vtk, nu_vtk, pixel,
                              match_matplotlib_orientation=True)

    if chk_ival == "last":
        save_checkpoint_2d(f"{chk_path}_final.h5", n_steps,
                           None, None, eps, sig, epsM, sigM, pixel)

    if vtk_interval == "last" and vtk_path:
        mu_sec = secant_shear_field(von_mises_strain_2d(eps), mu, d, k)
        E_vtk  = mu_sec * (3.0 * lam + 2.0 * mu_sec) / (lam + mu_sec + 1e-30)
        nu_vtk = lam / (2.0 * (lam + mu_sec) + 1e-30)
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E_vtk, nu_vtk, pixel,
                      match_matplotlib_orientation=True)

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()
    if enable_console:
        print(_duration_str)

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)


def secant_elastic_simulation_3d(
    lam, mu, d, k,
    target_strain_mask,
    target_values,
    n_steps=20,
    pixel=1.0,
    tol_ls=1e-6,
    max_iter_ls=400,
    tol_macro=1e-4,
    max_iter_macro=20,
    store=True,
    log_path=None,
    global_log_path=None,
    driving_component=(0, 0),
    enable_console=True,
    vtk_path=None,
    vtk_interval="none",
    **solver_kw
):
    """
    Mixed stress/strain-controlled 3-D simulation using the Secant Elastic
    Degradation constitutive law.

    See ``secant_elastic_simulation_2d`` for full parameter documentation.
    """
    import os, time as _time
    from datetime import datetime as _dt

    lam = np.asarray(lam, dtype=float)
    mu  = np.asarray(mu, dtype=float)
    nx, ny, nz = lam.shape

    _comp_labels = {(0,0):"xx",(1,1):"yy",(2,2):"zz",
                    (0,1):"xy",(0,2):"xz",(1,2):"yz",
                    (1,0):"yx",(2,0):"zx",(2,1):"zy"}
    _comp_lbl = _comp_labels.get(tuple(driving_component), "xx")

    lam_avg = float(lam.mean())
    mu_avg  = float(mu.mean())
    Lx, Ly, Lz = nx * pixel, ny * pixel, nz * pixel
    kx, ky, kz  = compute_wave_vectors_3d(nx, ny, nz, Lx, Ly, Lz)
    Gamma       = green_operator_3d(kx, ky, kz, lam_avg, mu_avg)

    E_avg  = mu_avg * (3.0 * lam_avg + 2.0 * mu_avg) / (lam_avg + mu_avg)
    nu_avg = lam_avg / (2.0 * (lam_avg + mu_avg))

    def get_strain_correction_3d(sigma_err):
        tr_sig = np.trace(sigma_err)
        return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg

    _log_f = None
    if log_path:
        _log_f = open(log_path, "w", buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'Eps_'+_comp_lbl:<14} {'Sig_'+_comp_lbl+'(GPa)':<16}\n")
        _log_f.write(_hdr)
        _log_f.write("-" * len(_hdr.rstrip()) + "\n")
    _glog_f = None
    if global_log_path:
        _glog_f = open(global_log_path, "w", buffering=1)
        _ghdr = (f"{'GlobalStep':<12} "
                 f"{'Eps_xx':<14} {'Eps_yy':<14} {'Eps_zz':<14} "
                 f"{'Eps_xy':<14} {'Eps_xz':<14} {'Eps_yz':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_zz(GPa)':<16} "
                 f"{'Sig_xy(GPa)':<16} {'Sig_xz(GPa)':<16} {'Sig_yz(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write("-" * len(_ghdr.rstrip()) + "\n")
    _t0 = _time.time()

    chk_ival = solver_kw.pop("checkpoint_interval", None)
    chk_path = solver_kw.pop("checkpoint_path", "checkpoint")
    if chk_path and chk_ival not in [None, "none"]:
        os.makedirs(os.path.dirname(chk_path) or ".", exist_ok=True)

    targets_path = np.zeros((n_steps + 1, 3, 3))
    for s in range(n_steps + 1):
        targets_path[s] = target_values * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list       = []
    sig_list       = []

    current_eps_bar = np.zeros((3, 3))
    eps_warm        = None
    stress_mask     = ~target_strain_mask

    for s in range(n_steps + 1):
        target_s = targets_path[s]
        max_err  = 0.0

        for it_macro in range(max_iter_macro):
            current_eps_bar[target_strain_mask] = target_s[target_strain_mask]

            eps, sig, epsM, sigM = spectral_solver_secant_3d(
                lam, mu, d, k,
                current_eps_bar,
                max_iter=max_iter_ls, tol=tol_ls,
                pixel=pixel,
                Gamma=Gamma, eps_init=eps_warm
            )

            stress_err = np.zeros((3, 3))
            stress_err[stress_mask] = target_s[stress_mask] - sigM[stress_mask]
            max_err = (np.max(np.abs(stress_err[stress_mask]))
                       if np.any(stress_mask) else 0.0)

            if max_err < tol_macro:
                break

            d_eps = get_strain_correction_3d(stress_err)
            current_eps_bar[stress_mask] += d_eps[stress_mask]
        else:
            print(f"Warning: Macro loop did not converge at step {s} (err={max_err:.2e})")

        eps_warm = eps

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)
        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        _i, _j = driving_component
        _eps_drv = epsM[_i, _j]
        _sig_drv = sigM[_i, _j]
        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"eps_{_comp_lbl}={_eps_drv:.4f}, "
                  f"sig_{_comp_lbl}={_sig_drv/1e9:.4f} GPa")
        if _log_f:
            _now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{_eps_drv:<14.6f} {_sig_drv/1e9:<16.6f}\n"
            )
        if _glog_f:
            _glog_f.write(
                f"{s:<12} "
                f"{epsM[0,0]:<14.6f} {epsM[1,1]:<14.6f} {epsM[2,2]:<14.6f} "
                f"{epsM[0,1]:<14.6f} {epsM[0,2]:<14.6f} {epsM[1,2]:<14.6f} "
                f"{sigM[0,0]/1e9:<16.6f} {sigM[1,1]/1e9:<16.6f} {sigM[2,2]/1e9:<16.6f} "
                f"{sigM[0,1]/1e9:<16.6f} {sigM[0,2]/1e9:<16.6f} {sigM[1,2]/1e9:<16.6f}\n"
            )

        if chk_ival is not None and chk_ival not in ["none", "last"]:
            save_chk, cp_name = False, None
            if chk_ival == "current":
                save_chk, cp_name = True, f"{chk_path}.h5"
            elif isinstance(chk_ival, int) and s % chk_ival == 0:
                save_chk, cp_name = True, f"{chk_path}_{s:06d}.h5"
            if save_chk and cp_name:
                save_checkpoint_3d(cp_name, s, None, None, eps, sig, epsM, sigM, pixel)

        if vtk_interval is not None and vtk_interval not in ["none", "last"]:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                mu_sec = secant_shear_field(von_mises_strain_3d(eps), mu, d, k)
                E_vtk  = mu_sec * (3.0*lam + 2.0*mu_sec) / (lam + mu_sec + 1e-30)
                nu_vtk = lam / (2.0*(lam + mu_sec) + 1e-30)
                export_to_vtk(vt_name, eps, sig, E_vtk, nu_vtk, pixel,
                              match_matplotlib_orientation=True)

    if chk_ival == "last":
        save_checkpoint_3d(f"{chk_path}_final.h5", n_steps,
                           None, None, eps, sig, epsM, sigM, pixel)

    if vtk_interval == "last" and vtk_path:
        mu_sec = secant_shear_field(von_mises_strain_3d(eps), mu, d, k)
        E_vtk  = mu_sec * (3.0*lam + 2.0*mu_sec) / (lam + mu_sec + 1e-30)
        nu_vtk = lam / (2.0*(lam + mu_sec) + 1e-30)
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E_vtk, nu_vtk, pixel,
                      match_matplotlib_orientation=True)

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()
    if enable_console:
        print(_duration_str)

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)


def landau_elastic_simulation_2d(
    lam, mu, v1, v2, v3, g1, g2, g3, g4,
    target_strain_mask,
    target_values,
    n_steps=20,
    pixel=1.0,
    tol_ls=1e-6,
    max_iter_ls=400,
    tol_macro=1e-4,
    max_iter_macro=20,
    store=True,
    plane_mode="plane_strain",
    log_path=None,
    global_log_path=None,
    driving_component=(0, 0),
    enable_console=True,
    vtk_path=None,
    vtk_interval="none",
    **solver_kw
):
    """
    Mixed stress/strain-controlled 2-D simulation using the Landau small-strain constitutive law.
    """
    import os, time as _time
    from datetime import datetime as _dt

    lam = np.asarray(lam, dtype=float)
    mu  = np.asarray(mu, dtype=float)
    nx, ny = lam.shape

    _comp_labels = {(0,0):"xx",(1,1):"yy",(0,1):"xy",(1,0):"yx"}
    _comp_lbl = _comp_labels.get(tuple(driving_component), "xx")

    # Pre-compute Green operator once
    lam_avg = float(lam.mean())
    mu_avg  = float(mu.mean())
    Lx, Ly  = nx * pixel, ny * pixel
    kx, ky  = compute_wave_vectors_2d(nx, ny, Lx, Ly)
    Gamma   = green_operator_2d(kx, ky, lam_avg, mu_avg)

    # Strain-correction helper for stress-controlled components
    E_avg  = mu_avg * (3.0 * lam_avg + 2.0 * mu_avg) / (lam_avg + mu_avg)
    nu_avg = lam_avg / (2.0 * (lam_avg + mu_avg))

    def get_strain_correction_2d(sigma_err):
        tr_sig = np.trace(sigma_err)
        return (sigma_err - nu_avg * tr_sig * np.eye(2)) / E_avg

    # Logging
    _log_f = None
    if log_path:
        _log_f = open(log_path, "w", buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'Eps_'+_comp_lbl:<14} {'Sig_'+_comp_lbl+'(GPa)':<16}\n")
        _log_f.write(_hdr)
        _log_f.write("-" * len(_hdr.rstrip()) + "\n")
    _glog_f = None
    if global_log_path:
        _glog_f = open(global_log_path, "w", buffering=1)
        _ghdr = (f"{'GlobalStep':<12} {'Eps_xx':<14} {'Eps_yy':<14} {'Eps_xy':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_xy(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write("-" * len(_ghdr.rstrip()) + "\n")
    _t0 = _time.time()

    # Checkpoint / VTK settings from solver_kw
    chk_ival = solver_kw.pop("checkpoint_interval", None)
    chk_path = solver_kw.pop("checkpoint_path", "checkpoint")
    if chk_path and chk_ival not in [None, "none"]:
        os.makedirs(os.path.dirname(chk_path) or ".", exist_ok=True)

    targets_path = np.zeros((n_steps + 1, 2, 2))
    for s in range(n_steps + 1):
        targets_path[s] = target_values * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list       = []
    sig_list       = []

    current_eps_bar = np.zeros((2, 2))
    eps_warm        = None
    stress_mask     = ~target_strain_mask

    for s in range(n_steps + 1):
        target_s = targets_path[s]
        max_err  = 0.0

        for it_macro in range(max_iter_macro):
            current_eps_bar[target_strain_mask] = target_s[target_strain_mask]

            eps, sig, epsM, sigM = spectral_solver_landau_2d(
                lam, mu, v1, v2, v3, g1, g2, g3, g4,
                current_eps_bar,
                max_iter=max_iter_ls, tol=tol_ls,
                pixel=pixel, plane_mode=plane_mode,
                Gamma=Gamma, eps_init=eps_warm
            )

            stress_err = np.zeros((2, 2))
            stress_err[stress_mask] = target_s[stress_mask] - sigM[stress_mask]
            max_err = (np.max(np.abs(stress_err[stress_mask]))
                       if np.any(stress_mask) else 0.0)

            if max_err < tol_macro:
                break

            d_eps = get_strain_correction_2d(stress_err)
            current_eps_bar[stress_mask] += d_eps[stress_mask]
        else:
            print(f"Warning: Macro loop did not converge at step {s} (err={max_err:.2e})")

        eps_warm = eps

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)
        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        _i, _j = driving_component
        _eps_drv = epsM[_i, _j]
        _sig_drv = sigM[_i, _j]
        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"eps_{_comp_lbl}={_eps_drv:.4f}, "
                  f"sig_{_comp_lbl}={_sig_drv/1e9:.4f} GPa")
        if _log_f:
            _now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{_eps_drv:<14.6f} {_sig_drv/1e9:<16.6f}\n"
            )
        if _glog_f:
            _glog_f.write(
                f"{s:<12} "
                f"{epsM[0,0]:<14.6f} {epsM[1,1]:<14.6f} {epsM[0,1]:<14.6f} "
                f"{sigM[0,0]/1e9:<16.6f} {sigM[1,1]/1e9:<16.6f} {sigM[0,1]/1e9:<16.6f}\n"
            )

        if chk_ival is not None and chk_ival not in ["none", "last"]:
            save_chk, cp_name = False, None
            if chk_ival == "current":
                save_chk, cp_name = True, f"{chk_path}.h5"
            elif isinstance(chk_ival, int) and s % chk_ival == 0:
                save_chk, cp_name = True, f"{chk_path}_{s:06d}.h5"
            if save_chk and cp_name:
                save_checkpoint_2d(cp_name, s, None, None, eps, sig, epsM, sigM, pixel)

        if vtk_interval is not None and vtk_interval not in ["none", "last"]:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                E_vtk  = mu * (3.0*lam + 2.0*mu) / (lam + mu + 1e-30)
                nu_vtk = lam / (2.0*(lam + mu) + 1e-30)
                export_to_vtk(vt_name, eps, sig, E_vtk, nu_vtk, pixel,
                              match_matplotlib_orientation=True)

    if chk_ival == "last":
        save_checkpoint_2d(f"{chk_path}_final.h5", n_steps,
                           None, None, eps, sig, epsM, sigM, pixel)

    if vtk_interval == "last" and vtk_path:
        E_vtk  = mu * (3.0*lam + 2.0*mu) / (lam + mu + 1e-30)
        nu_vtk = lam / (2.0*(lam + mu) + 1e-30)
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E_vtk, nu_vtk, pixel,
                      match_matplotlib_orientation=True)

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()
    if enable_console:
        print(_duration_str)

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)


def landau_elastic_simulation_3d(
    lam, mu, v1, v2, v3, g1, g2, g3, g4,
    target_strain_mask,
    target_values,
    n_steps=20,
    pixel=1.0,
    tol_ls=1e-6,
    max_iter_ls=400,
    tol_macro=1e-4,
    max_iter_macro=20,
    store=True,
    log_path=None,
    global_log_path=None,
    driving_component=(0, 0),
    enable_console=True,
    vtk_path=None,
    vtk_interval="none",
    **solver_kw
):
    """
    Mixed stress/strain-controlled 3-D simulation using the Landau small-strain constitutive law.
    """
    import os, time as _time
    from datetime import datetime as _dt

    lam = np.asarray(lam, dtype=float)
    mu  = np.asarray(mu, dtype=float)
    nx, ny, nz = lam.shape

    _comp_labels = {(0,0):"xx",(1,1):"yy",(2,2):"zz",
                    (0,1):"xy",(0,2):"xz",(1,2):"yz",
                    (1,0):"yx",(2,0):"zx",(2,1):"zy"}
    _comp_lbl = _comp_labels.get(tuple(driving_component), "xx")

    # Pre-compute Green operator once
    lam_avg = float(lam.mean())
    mu_avg  = float(mu.mean())
    Lx, Ly, Lz = nx * pixel, ny * pixel, nz * pixel
    kx, ky, kz  = compute_wave_vectors_3d(nx, ny, nz, Lx, Ly, Lz)
    Gamma       = green_operator_3d(kx, ky, kz, lam_avg, mu_avg)

    # Strain-correction helper for stress-controlled components
    E_avg  = mu_avg * (3.0 * lam_avg + 2.0 * mu_avg) / (lam_avg + mu_avg)
    nu_avg = lam_avg / (2.0 * (lam_avg + mu_avg))

    def get_strain_correction_3d(sigma_err):
        tr_sig = np.trace(sigma_err)
        return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg

    # Logging
    _log_f = None
    if log_path:
        _log_f = open(log_path, "w", buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'Eps_'+_comp_lbl:<14} {'Sig_'+_comp_lbl+'(GPa)':<16}\n")
        _log_f.write(_hdr)
        _log_f.write("-" * len(_hdr.rstrip()) + "\n")
    _glog_f = None
    if global_log_path:
        _glog_f = open(global_log_path, "w", buffering=1)
        _ghdr = (f"{'GlobalStep':<12} "
                 f"{'Eps_xx':<14} {'Eps_yy':<14} {'Eps_zz':<14} "
                 f"{'Eps_xy':<14} {'Eps_xz':<14} {'Eps_yz':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_zz(GPa)':<16} "
                 f"{'Sig_xy(GPa)':<16} {'Sig_xz(GPa)':<16} {'Sig_yz(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write("-" * len(_ghdr.rstrip()) + "\n")
    _t0 = _time.time()

    # Checkpoint / VTK settings from solver_kw
    chk_ival = solver_kw.pop("checkpoint_interval", None)
    chk_path = solver_kw.pop("checkpoint_path", "checkpoint")
    if chk_path and chk_ival not in [None, "none"]:
        os.makedirs(os.path.dirname(chk_path) or ".", exist_ok=True)

    targets_path = np.zeros((n_steps + 1, 3, 3))
    for s in range(n_steps + 1):
        targets_path[s] = target_values * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list       = []
    sig_list       = []

    current_eps_bar = np.zeros((3, 3))
    eps_warm        = None
    stress_mask     = ~target_strain_mask

    for s in range(n_steps + 1):
        target_s = targets_path[s]
        max_err  = 0.0

        for it_macro in range(max_iter_macro):
            current_eps_bar[target_strain_mask] = target_s[target_strain_mask]

            eps, sig, epsM, sigM = spectral_solver_landau_3d(
                lam, mu, v1, v2, v3, g1, g2, g3, g4,
                current_eps_bar,
                max_iter=max_iter_ls, tol=tol_ls,
                pixel=pixel,
                Gamma=Gamma, eps_init=eps_warm
            )

            stress_err = np.zeros((3, 3))
            stress_err[stress_mask] = target_s[stress_mask] - sigM[stress_mask]
            max_err = (np.max(np.abs(stress_err[stress_mask]))
                       if np.any(stress_mask) else 0.0)

            if max_err < tol_macro:
                break

            d_eps = get_strain_correction_3d(stress_err)
            current_eps_bar[stress_mask] += d_eps[stress_mask]
        else:
            print(f"Warning: Macro loop did not converge at step {s} (err={max_err:.2e})")

        eps_warm = eps

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)
        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        _i, _j = driving_component
        _eps_drv = epsM[_i, _j]
        _sig_drv = sigM[_i, _j]
        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"eps_{_comp_lbl}={_eps_drv:.4f}, "
                  f"sig_{_comp_lbl}={_sig_drv/1e9:.4f} GPa")
        if _log_f:
            _now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{_eps_drv:<14.6f} {_sig_drv/1e9:<16.6f}\n"
            )
        if _glog_f:
            _glog_f.write(
                f"{s:<12} "
                f"{epsM[0,0]:<14.6f} {epsM[1,1]:<14.6f} {epsM[0,2]:<14.6f} "
                f"{sigM[0,0]/1e9:<16.6f} {sigM[1,1]/1e9:<16.6f} {sigM[0,2]/1e9:<16.6f}\n"
            )

        if chk_ival is not None and chk_ival not in ["none", "last"]:
            save_chk, cp_name = False, None
            if chk_ival == "current":
                save_chk, cp_name = True, f"{chk_path}.h5"
            elif isinstance(chk_ival, int) and s % chk_ival == 0:
                save_chk, cp_name = True, f"{chk_path}_{s:06d}.h5"
            if save_chk and cp_name:
                save_checkpoint_3d(cp_name, s, None, None, eps, sig, epsM, sigM, pixel)

        if vtk_interval is not None and vtk_interval not in ["none", "last"]:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                E_vtk  = mu * (3.0*lam + 2.0*mu) / (lam + mu + 1e-30)
                nu_vtk = lam / (2.0*(lam + mu) + 1e-30)
                export_to_vtk(vt_name, eps, sig, E_vtk, nu_vtk, pixel,
                              match_matplotlib_orientation=True)

    if chk_ival == "last":
        save_checkpoint_3d(f"{chk_path}_final.h5", n_steps,
                           None, None, eps, sig, epsM, sigM, pixel)

    if vtk_interval == "last" and vtk_path:
        E_vtk  = mu * (3.0*lam + 2.0*mu) / (lam + mu + 1e-30)
        nu_vtk = lam / (2.0*(lam + mu) + 1e-30)
        export_to_vtk(f"{vtk_path}_final.vtu", eps, sig, E_vtk, nu_vtk, pixel,
                      match_matplotlib_orientation=True)

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()
    if enable_console:
        print(_duration_str)

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)

