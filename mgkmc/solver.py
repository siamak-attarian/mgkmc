import numpy as np
from .elasticity import compute_lame_2d, stress_from_strain_2d, green_operator_2d, compute_lame_3d, stress_from_strain_3d, green_operator_3d
from .fft import compute_wave_vectors_2d, compute_wave_vectors_3d, fft_field, ifft_field
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
                  f"sig_{_comp_lbl}={_sig_drv/1e6:.2f} MPa")
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

    if chk_ival == "last":
        save_checkpoint_2d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)
    elif chk_ival not in [None, "none", "last"] and isinstance(chk_ival, int) and n_steps % chk_ival != 0:
        save_checkpoint_2d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)

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
                  f"sig_{_comp_lbl}={_sig_drv/1e6:.2f} MPa")

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

    if chk_ival == "last":
        save_checkpoint_3d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)
    elif chk_ival not in [None, "none", "last"] and isinstance(chk_ival, int) and n_steps % chk_ival != 0:
        save_checkpoint_3d(f"{chk_path}_final.h5", n_steps, E, nu, eps, sig, epsM, sigM, pixel)

    if _log_f:  _log_f.close()
    if _glog_f: _glog_f.close()

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
