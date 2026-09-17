import numpy as np
from ..linear_elastic_simulator import spectral_solver_3d

def update_stress_fft_full(eps_plastic_field, eps_macro, E, nu, pixel=1.0,
                      max_iter=200, tol=1e-6, verbose=False):
    """
    Run the spectral solver using direct arrays (struct-of-arrays layout).

    Parameters
    ----------
    eps_plastic_field : numpy.ndarray
        Plastic eigenstrain field, shape ``(Nx, Ny, Nz, 3, 3)``.
    eps_macro : numpy.ndarray
        Imposed macroscopic strain, shape ``(3, 3)``.
    E, nu : numpy.ndarray
        Young's modulus and Poisson ratio fields.
    pixel : float, optional
        Voxel edge length.
    max_iter : int, optional
        Maximum solver iterations.
    tol : float, optional
        Convergence tolerance.
    verbose : bool, optional
        Print solver progress.

    Returns
    -------
    eps_total : numpy.ndarray
        Total strain field.
    sigma_total : numpy.ndarray
        Total stress field.
    eps_macro_out : numpy.ndarray
        Resulting macroscopic strain.
    sigma_macro_out : numpy.ndarray
        Resulting macroscopic stress.
    """
    
    eps_total, sigma_total, eps_macro_out, sigma_macro_out = spectral_solver_3d(
        E, nu, eps_macro,
        eps_plastic=eps_plastic_field,
        max_iter=max_iter, tol=tol,
        verbose=verbose, pixel=pixel
    )

    return eps_total, sigma_total, eps_macro_out, sigma_macro_out
