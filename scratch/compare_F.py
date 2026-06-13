import numpy as np
from mgkmc.finite_strain_simulator import (
    _make_identity_tensors_2d, build_ghat4_2d, build_C4_2d,
    finite_strain_solver_step_2d
)

nx, ny = 16, 16
pixel = 1.0
Lx, Ly = nx * pixel, ny * pixel

# Small contrast: 70 GPa vs 60 GPa
np.random.seed(42)
E = np.ones((nx, ny)) * 70.0 * 1e9  # Pa
E[4:12, 4:12] = 60.0 * 1e9
nu = np.ones((nx, ny)) * 0.3

I2, I4, I4rt, I4s, II = _make_identity_tensors_2d(nx, ny)
Ghat4 = build_ghat4_2d(nx, ny, Lx, Ly)
C4 = build_C4_2d(E, nu, I4s, II, plane_mode="plane_strain")

# Strain step: 0.1%
F_bar = np.array([[1.001, 0.0],
                  [0.0, 0.999]])
P_target = np.zeros((2, 2))
P_mask = np.zeros((2, 2), dtype=bool)

F_init = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))

# Solver 1: Newton-CG
F_ncg, P_ncg, Sig_ncg, K4_ncg, _ = finite_strain_solver_step_2d(
    F_init.copy(), F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
    driving_component=(0, 0), P_target=P_target, P_mask=P_mask,
    E_avg=E.mean(), nu_avg=nu.mean(),
    tol_NW=1e-9, tol_CG=1e-10, max_NW=30,
    solver="newton_cg", pixel=pixel
)

# Solver 2: DBFFT
F_dbfft, P_dbfft, Sig_dbfft, K4_dbfft, _ = finite_strain_solver_step_2d(
    F_init.copy(), F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
    driving_component=(0, 0), P_target=P_target, P_mask=P_mask,
    E_avg=E.mean(), nu_avg=nu.mean(),
    tol_NW=1e-9, tol_CG=1e-10, max_NW=30,
    solver="dbfft", pixel=pixel
)

diff_F = np.linalg.norm(F_dbfft - F_ncg) / np.linalg.norm(F_ncg)
diff_P = np.linalg.norm(P_dbfft - P_ncg) / np.linalg.norm(P_ncg)
print(f"F relative diff: {diff_F:.4e}")
print(f"P relative diff: {diff_P:.4e}")
