import numpy as np
from .elasticity import compute_lame, stress_from_strain, green_operator
from .fft import compute_wave_vectors, fft_field, ifft_field
import pyfftw.interfaces.numpy_fft as fft
def spectral_solver_3d(E, nu, eps_bar,
                       max_iter=200, tol=1e-6,
                       verbose=False, pixel=1.0):
    nx, ny, nz = E.shape
    lam, mu = compute_lame(E, nu)
    lam0, mu0 = compute_lame(E.mean(), nu.mean())
    
    Lx, Ly, Lz = nx*pixel, ny*pixel, nz*pixel
    kx, ky, kz = compute_wave_vectors(nx, ny, nz, Lx, Ly, Lz)
    Gamma = green_operator(kx, ky, kz, lam0, mu0)  # Shape: [3,3,3,3,nx,ny,nz]

    # Reshape eps and sig to match reference: [3,3,nx,ny,nz]
    eps = np.zeros((3, 3, nx, ny, nz))
    for i in range(3):
        for j in range(3):
            eps[i, j, :, :, :] = eps_bar[i, j]

    for it in range(max_iter):
        # Compute stress - reshape to match
        sig = np.zeros((3, 3, nx, ny, nz))
        for i in range(3):
            for j in range(3):
                sig[i, j] = stress_from_strain(
                    eps.transpose(2,3,4,0,1), E, nu
                )[..., i, j]
        
        # Reference stress
        sig0 = np.zeros((3, 3, nx, ny, nz))
        for i in range(3):
            for j in range(3):
                sig0[i, j] = stress_from_strain(
                    eps.transpose(2,3,4,0,1),
                    E*0 + E.mean(),
                    nu*0 + nu.mean()
                )[..., i, j]
        
        tau = sig - sig0
        tau_hat = np.zeros((3, 3, nx, ny, nz), dtype=complex)
        for i in range(3):
            for j in range(3):
                tau_hat[i, j] = fft_field(tau[i, j])
        
        # Apply Green operator: Gamma[k,h,i,j] * tau[i,j] -> eps_tilde[k,h]
        eps_tilde_hat = -np.einsum("khijxyz,ijxyz->khxyz", Gamma, tau_hat)
        
        eps_tilde = np.zeros((3, 3, nx, ny, nz))
        for i in range(3):
            for j in range(3):
                eps_tilde[i, j] = ifft_field(eps_tilde_hat[i, j])
        
        # Zero mean
        for i in range(3):
            for j in range(3):
                eps_tilde[i, j] -= eps_tilde[i, j].mean()
        
        eps_new = eps_bar[:, :, None, None, None] + eps_tilde
        
        diff = np.linalg.norm(eps_new - eps) / (np.linalg.norm(eps) + 1e-20)
        eps = eps_new
        
        if verbose and it % 10 == 0:
            print(f"Iter {it:03d}: Δε/ε = {diff:.3e}")
        
        if diff < tol:
            break
    
    # Convert back to original format
    eps_out = eps.transpose(2, 3, 4, 0, 1)
    sig_out = stress_from_strain(eps_out, E, nu)
    eps_macro = eps_out.mean(axis=(0,1,2))
    sig_macro = sig_out.mean(axis=(0,1,2))
    
    return eps_out, sig_out, eps_macro, sig_macro

def run_simulation(
    E, nu,
    loading_func,
    loading_params,
    n_steps=20,
    pixel=1.0,
    store=True,
    **solver_kw
):
    """
    Run a simulation by applying a loading path defined by `loading_func`.

    Parameters
    ----------
    E, nu : np.ndarray
        3D material property fields.
    loading_func : callable
        Function that returns a 3x3 strain tensor (e.g. from mgkmc.elasticity_helpers).
    loading_params : dict
        Arguments to pass to `loading_func`.
    n_steps : int
        Number of load steps to reach the target strain.
    pixel : float
        Voxel size.
    store : bool
        Whether to store full fields at each step.
    **solver_kw : dict
        Additional arguments for `spectral_solver_3d`.

    Returns
    -------
    eps_macro_list : np.ndarray
        Macroscopic strain at each step.
    sig_macro_list : np.ndarray
        Macroscopic stress at each step.
    eps_list : list of np.ndarray
        Full strain fields (if store=True).
    sig_list : list of np.ndarray
        Full stress fields (if store=True).
    """
    # Calculate the final target strain tensor
    target_eps = loading_func(**loading_params)

    # Create linear path from 0 to target_eps
    # shape: (n_steps+1, 3, 3)
    eps_path = np.zeros((n_steps + 1, 3, 3))
    for s in range(n_steps + 1):
        eps_path[s] = target_eps * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list = []
    sig_list = []

    for s in range(n_steps + 1):
        eps_bar = eps_path[s]

        # Run solver
        eps, sig, epsM, sigM = spectral_solver_3d(
            E, nu, eps_bar, pixel=pixel, **solver_kw
        )

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)

        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        print(f"step {s}/{n_steps}: "
              f"eps_xx={epsM[0,0]:.4f}, "
              f"sig_xx={sigM[0,0]/1e6:.2f} MPa, "
              f"sig_yy={sigM[1,1]/1e6:.2f} MPa, "
              f"sig_zz={sigM[2,2]/1e6:.2f} MPa")

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)


def run_mixed_simulation(
    E, nu,
    target_strain_mask,
    target_values,
    n_steps=20,
    pixel=1.0,
    tol_macro=1e-4,
    max_iter_macro=20,
    store=True,
    **solver_kw
):
    """
    Run a simulation with mixed stress/strain control.
    Iteratively adjusts the macroscopic strain to satisfy stress targets.

    Parameters
    ----------
    E, nu : np.ndarray
        3D material property fields.
    target_strain_mask : np.ndarray (3,3) of bool
        True where strain is prescribed, False where stress is prescribed.
    target_values : np.ndarray (3,3)
        Target values for the prescribed component (strain or stress).
    n_steps : int
        Number of load steps.
    tol_macro : float
        Tolerance for macroscopic stress convergence.
    max_iter_macro : int
        Maximum iterations for macroscopic adjustment per step.
    """
    # Initial guess for macroscopic strain: use average properties
    # to convert stress targets to strain targets.
    E_avg = E.mean()
    nu_avg = nu.mean()
    
    # Compliance matrix for isotropic material (Voigt notation approx)
    # We'll just use a simple relaxation update:
    # delta_eps = C_inv * delta_sig
    # C_inv is roughly 1/E * [1 -nu -nu; -nu 1 -nu; ...]
    
    def get_strain_correction(sigma_err):
        # Simple isotropic compliance update
        tr_sig = np.trace(sigma_err)
        return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg

    # Linear path for targets
    targets_path = np.zeros((n_steps + 1, 3, 3))
    for s in range(n_steps + 1):
        targets_path[s] = target_values * (s / n_steps)

    eps_macro_list = []
    sig_macro_list = []
    eps_list = []
    sig_list = []
    
    # Start with zero strain
    current_eps_bar = np.zeros((3, 3))

    for s in range(n_steps + 1):
        target_s = targets_path[s]
        
        # Iteratively adjust current_eps_bar
        for it_macro in range(max_iter_macro):
            # 1. Enforce prescribed strains
            current_eps_bar[target_strain_mask] = target_s[target_strain_mask]
            
            # 2. Run solver
            eps, sig, epsM, sigM = spectral_solver_3d(
                E, nu, current_eps_bar, pixel=pixel, verbose=False, **solver_kw
            )
            
            # 3. Check stress error for prescribed stress components
            # Where mask is False, we compare sigM with target_s
            stress_err = np.zeros((3,3))
            stress_mask = ~target_strain_mask
            stress_err[stress_mask] = target_s[stress_mask] - sigM[stress_mask]
            
            max_err = np.max(np.abs(stress_err[stress_mask])) if np.any(stress_mask) else 0.0
            
            if max_err < tol_macro:
                break
                
            # 4. Update guess for unknown strains
            # We want to change eps_bar such that sigM moves towards target
            # d_eps = S * d_sig
            d_eps = get_strain_correction(stress_err)
            
            # Only update the components where strain is NOT prescribed
            # (The prescribed ones are reset at start of loop anyway)
            current_eps_bar[stress_mask] += d_eps[stress_mask]
            
        else:
            print(f"Warning: Macroscopic loop did not converge at step {s}. Max err: {max_err:.2e}")

        eps_macro_list.append(epsM)
        sig_macro_list.append(sigM)
        
        if store:
            eps_list.append(eps)
            sig_list.append(sig)

        print(f"step {s}/{n_steps}: "
              f"eps_xx={epsM[0,0]:.4f}, "
              f"sig_xx={sigM[0,0]/1e6:.2f} MPa, "
              f"sig_yy={sigM[1,1]/1e6:.2f} MPa, "
              f"sig_zz={sigM[2,2]/1e6:.2f} MPa")

    return (np.array(eps_macro_list),
            np.array(sig_macro_list),
            eps_list, sig_list)
