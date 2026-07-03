import numpy as np
from mgkmc import (
    spectral_solver_landau_2d,
    spectral_solver_landau_3d,
    landau_elastic_simulation_2d,
    landau_elastic_simulation_3d,
)
from mgkmc.elasticity import compute_lame_2d, compute_lame_3d

def test_landau_2d_plane_strain():
    # Grid size
    nx, ny = 16, 16
    pixel = 1.0

    # Linear Lame constants
    E = 70.0e9
    nu = 0.3
    lam_val, mu_val = compute_lame_2d(E, nu, plane_mode="plane_strain")
    lam = np.full((nx, ny), lam_val)
    mu = np.full((nx, ny), mu_val)

    # 3rd and 4th order Landau parameters (set to small non-zero values to verify calculation)
    # v1, v2, v3, g1, g2, g3, g4
    v1 = -100.0e9
    v2 = -50.0e9
    v3 = -10.0e9
    g1 = -1000.0e9
    g2 = 500.0e9
    g3 = -100.0e9
    g4 = -200.0e9

    # Prescribed macro strain
    eps_bar = np.array([[0.01, 0.002],
                        [0.002, -0.005]])

    # Run solver
    eps, sig, epsM, sigM = spectral_solver_landau_2d(
        lam=lam, mu=mu, v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4,
        eps_bar=eps_bar, eps_plastic=None,
        max_iter=100, tol=1e-8,
        verbose=True, pixel=pixel,
        plane_mode="plane_strain"
    )

    # Assert shape and convergence
    assert eps.shape == (nx, ny, 2, 2)
    assert sig.shape == (nx, ny, 2, 2)
    # Because Lam and Mu are homogeneous, the local strains should exactly equal the macroscopic ones
    np.testing.assert_allclose(epsM, eps_bar, atol=1e-7)
    np.testing.assert_allclose(eps[0, 0], eps_bar, atol=1e-7)


def test_landau_2d_plane_stress():
    # Grid size
    nx, ny = 16, 16
    pixel = 1.0

    # Linear Lame constants (3D Lame parameters)
    E = 70.0e9
    nu = 0.3
    lam_val, mu_val = compute_lame_3d(E, nu)  # Pass 3D Lamé parameters to Landau stress solver
    lam = np.full((nx, ny), lam_val)
    mu = np.full((nx, ny), mu_val)

    # 3rd and 4th order Landau parameters
    v1 = -100.0e9
    v2 = -50.0e9
    v3 = -10.0e9
    g1 = -1000.0e9
    g2 = 500.0e9
    g3 = -100.0e9
    g4 = -200.0e9

    # Prescribed macro strain
    eps_bar = np.array([[0.01, 0.0],
                        [0.0, -0.003]])

    # Run solver
    eps, sig, epsM, sigM = spectral_solver_landau_2d(
        lam=lam, mu=mu, v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4,
        eps_bar=eps_bar, eps_plastic=None,
        max_iter=100, tol=1e-8,
        verbose=True, pixel=pixel,
        plane_mode="plane_stress"
    )

    assert eps.shape == (nx, ny, 2, 2)
    assert sig.shape == (nx, ny, 2, 2)
    np.testing.assert_allclose(epsM, eps_bar, atol=1e-7)


def test_landau_3d():
    # Grid size
    nx, ny, nz = 8, 8, 8
    pixel = 1.0

    # Linear Lame constants
    E = 70.0e9
    nu = 0.3
    lam_val, mu_val = compute_lame_3d(E, nu)
    lam = np.full((nx, ny, nz), lam_val)
    mu = np.full((nx, ny, nz), mu_val)

    # 3rd and 4th order Landau parameters
    v1 = -100.0e9
    v2 = -50.0e9
    v3 = -10.0e9
    g1 = -1000.0e9
    g2 = 500.0e9
    g3 = -100.0e9
    g4 = -200.0e9

    # Prescribed macro strain
    eps_bar = np.array([[0.01, 0.002, 0.0],
                        [0.002, -0.005, 0.001],
                        [0.0, 0.001, 0.003]])

    # Run solver
    eps, sig, epsM, sigM = spectral_solver_landau_3d(
        lam=lam, mu=mu, v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4,
        eps_bar=eps_bar, eps_plastic=None,
        max_iter=100, tol=1e-8,
        verbose=True, pixel=pixel
    )

    assert eps.shape == (nx, ny, nz, 3, 3)
    assert sig.shape == (nx, ny, nz, 3, 3)
    np.testing.assert_allclose(epsM, eps_bar, atol=1e-7)


def test_landau_simulation_2d():
    nx, ny = 8, 8
    E = 70.0e9
    nu = 0.3
    lam_val, mu_val = compute_lame_2d(E, nu, plane_mode="plane_strain")
    lam = np.full((nx, ny), lam_val)
    mu = np.full((nx, ny), mu_val)

    v1, v2, v3 = -100.0e9, -50.0e9, -10.0e9
    g1, g2, g3, g4 = -1000.0e9, 500.0e9, -100.0e9, -200.0e9

    target_strain_mask = np.ones((2, 2), dtype=bool)
    target_values = np.array([[0.02, 0.0], [0.0, -0.006]])

    # Run simulation
    eps_mac, sig_mac, eps_list, sig_list = landau_elastic_simulation_2d(
        lam=lam, mu=mu, v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4,
        target_strain_mask=target_strain_mask,
        target_values=target_values,
        n_steps=5,
        pixel=1.0,
        plane_mode="plane_strain",
        store=True,
        enable_console=False
    )

    assert eps_mac.shape == (6, 2, 2)
    assert sig_mac.shape == (6, 2, 2)
    assert len(eps_list) == 6
    np.testing.assert_allclose(eps_mac[-1], target_values, atol=1e-7)
