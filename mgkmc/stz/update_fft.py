import numpy as np
from ..solver import spectral_solver_3d

def extract_eps_plastic(grid):
    Nx, Ny, Nz = grid.shape
    eps_pl = np.zeros((Nx,Ny,Nz,3,3))
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                eps_pl[x,y,z] = grid[x,y,z].eps_plastic
    return eps_pl

def push_solver_results(grid, eps_total, sigma_total):
    Nx, Ny, Nz = grid.shape
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                voxel = grid[x,y,z]
                voxel.eps_total = eps_total[x,y,z]
                voxel.sigma     = sigma_total[x,y,z]

def update_stress_fft(grid, eps_macro, E, nu, pixel=1.0,
                      max_iter=200, tol=1e-6, verbose=False):

    eps_pl = extract_eps_plastic(grid)

    eps_total, sigma_total, eps_macro_out, sigma_macro_out = spectral_solver_3d(
        E, nu, eps_macro,
        eps_plastic=eps_pl,
        max_iter=max_iter, tol=tol,
        verbose=verbose, pixel=pixel
    )

    push_solver_results(grid, eps_total, sigma_total)

    return eps_macro_out, sigma_macro_out

def update_stress_fft_full(grid, eps_macro, E, nu, pixel=1.0,
                      max_iter=200, tol=1e-6, verbose=False):

    eps_pl = extract_eps_plastic(grid)

    eps_total, sigma_total, eps_macro_out, sigma_macro_out = spectral_solver_3d(
        E, nu, eps_macro,
        eps_plastic=eps_pl,
        max_iter=max_iter, tol=tol,
        verbose=verbose, pixel=pixel
    )

    push_solver_results(grid, eps_total, sigma_total)

    # Return full fields for VTK
    return eps_total, sigma_total, eps_macro_out, sigma_macro_out
