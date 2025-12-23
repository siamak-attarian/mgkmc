import numpy as np
from ..solver import spectral_solver_3d

def update_stress_fft_full(eps_plastic_field, eps_macro, E, nu, pixel=1.0,
                      max_iter=200, tol=1e-6, verbose=False):
    """
    Run spectral solver using direct arrays (SoA).
    Input:
       eps_plastic_field: (Nx, Ny, Nz, 3, 3)
       eps_macro: (3,3)
    Output:
       Full fields: eps_total, sigma_total, eps_macro_out, sig_macro_out
    """
    
    eps_total, sigma_total, eps_macro_out, sigma_macro_out = spectral_solver_3d(
        E, nu, eps_macro,
        eps_plastic=eps_plastic_field,
        max_iter=max_iter, tol=tol,
        verbose=verbose, pixel=pixel
    )

    return eps_total, sigma_total, eps_macro_out, sigma_macro_out
