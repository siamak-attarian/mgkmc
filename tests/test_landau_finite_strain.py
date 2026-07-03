import numpy as np
from mgkmc.finite_strain_simulator import finite_strain_simulation_2d, finite_strain_simulation_3d

def test_landau_fs_2d_plane_strain():
    nx, ny = 8, 8
    E = np.full((nx, ny), 70.0e9)
    nu = np.full((nx, ny), 0.3)
    
    # Landau parameters
    v1 = -100.0e9
    v2 = -50.0e9
    v3 = -10.0e9
    g1 = -1000.0e9
    g2 = 500.0e9
    g3 = -100.0e9
    g4 = -200.0e9
    
    # Run simulation under plane strain, tension xx
    F_mac, Sig_mac, P_mac, F_list, Sig_list = finite_strain_simulation_2d(
        E=E, nu=nu,
        driving_component=(0, 0),
        eps_target=0.01,
        n_steps=5,
        plane_mode="plane_strain",
        model_type="landau",
        solver="al",
        v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4,
        enable_console=False
    )
    
    # Check results
    assert F_mac.shape == (6, 2, 2)
    assert Sig_mac.shape == (6, 2, 2)
    np.testing.assert_allclose(F_mac[-1, 0, 0], 1.01, atol=1e-5)

def test_landau_fs_2d_plane_stress():
    nx, ny = 8, 8
    E = np.full((nx, ny), 70.0e9)
    nu = np.full((nx, ny), 0.3)
    
    v1 = -100.0e9
    v2 = -50.0e9
    v3 = -10.0e9
    g1 = -1000.0e9
    g2 = 500.0e9
    g3 = -100.0e9
    g4 = -200.0e9
    
    # Run simulation under plane stress (with mixed targets yy = 0.0)
    F_mac, Sig_mac, P_mac, F_list, Sig_list = finite_strain_simulation_2d(
        E=E, nu=nu,
        driving_component=(0, 0),
        eps_target=0.01,
        n_steps=5,
        mixed_targets={(1, 1): 0.0},
        plane_mode="plane_stress",
        model_type="landau",
        solver="al",
        v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4,
        tol_macro=1e5,
        enable_console=False
    )
    
    assert F_mac.shape == (6, 2, 2)
    assert Sig_mac.shape == (6, 2, 2)
    np.testing.assert_allclose(F_mac[-1, 0, 0], 1.01, atol=1e-5)
    np.testing.assert_allclose(Sig_mac[-1, 1, 1], 0.0, atol=2e5) # yy stress should be zero within tol_macro

def test_landau_fs_3d():
    nx, ny, nz = 4, 4, 4
    E = np.full((nx, ny, nz), 70.0e9)
    nu = np.full((nx, ny, nz), 0.3)
    
    v1 = -100.0e9
    v2 = -50.0e9
    v3 = -10.0e9
    g1 = -1000.0e9
    g2 = 500.0e9
    g3 = -100.0e9
    g4 = -200.0e9
    
    # Run simulation under 3D mixed BCs (tension in xx, yy=zz=0 stress)
    F_mac, Sig_mac, P_mac, F_list, Sig_list = finite_strain_simulation_3d(
        E=E, nu=nu,
        driving_component=(0, 0),
        eps_target=0.01,
        n_steps=5,
        mixed_targets={(1, 1): 0.0, (2, 2): 0.0},
        model_type="landau",
        solver="al",
        v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4,
        tol_macro=1e5,
        enable_console=False
    )
    
    assert F_mac.shape == (6, 3, 3)
    assert Sig_mac.shape == (6, 3, 3)
    np.testing.assert_allclose(F_mac[-1, 0, 0], 1.01, atol=1e-5)
    np.testing.assert_allclose(Sig_mac[-1, 1, 1], 0.0, atol=2e5)
    np.testing.assert_allclose(Sig_mac[-1, 2, 2], 0.0, atol=2e5)
