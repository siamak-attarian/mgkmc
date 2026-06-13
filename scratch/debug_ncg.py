import numpy as np
import scipy.sparse.linalg as sp
from mgkmc.finite_strain_simulator import (
    _make_identity_tensors_2d, build_ghat4_2d, build_C4_2d,
    constitutive_hyperelastic_2d, _project, _trans2, _ddot42
)

nx, ny = 16, 16
pixel = 1.0
Lx, Ly = nx * pixel, ny * pixel

np.random.seed(42)
E = np.ones((nx, ny)) * 70.0 * 1e9  # Pa
E[4:12, 4:12] = 7.0 * 1e9
nu = np.ones((nx, ny)) * 0.3

I2, I4, I4rt, I4s, II = _make_identity_tensors_2d(nx, ny)
Ghat4 = build_ghat4_2d(nx, ny, Lx, Ly)
C4 = build_C4_2d(E, nu, I4s, II, plane_mode="plane_strain")

F_bar = np.array([[1.001, 0.0],
                  [0.0, 0.999]])
F_init = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))

DbarF = F_bar - F_init.mean(axis=(2, 3))
DbarF_grid = np.einsum('ij,xy->ijxy', DbarF, np.ones((nx, ny)))

P, K4, _ = constitutive_hyperelastic_2d(F_init, C4, I2, I4, I4rt, model_type="svk", plane_mode="plane_strain")

def G_op(A2):
    return _project(A2, Ghat4)

def K_dF_op(dFm_flat):
    dF  = dFm_flat.reshape(2, 2, nx, ny)
    return _trans2(_ddot42(K4, _trans2(dF)))

def G_K_dF(dFm_flat):
    return G_op(K_dF_op(dFm_flat)).reshape(-1)

A_op = sp.LinearOperator(
    shape=(F_init.size, F_init.size),
    matvec=G_K_dF,
    dtype='float64'
)

F = F_init.copy()
print("Starting Newton-CG debug loop...")
for i_NW in range(15):
    if i_NW == 0:
        rhs = -G_op(K_dF_op(DbarF_grid.reshape(-1))).reshape(-1)
    else:
        rhs = -G_op(P).reshape(-1)
        
    res_norm = np.linalg.norm(G_op(P))
    P_norm = np.linalg.norm(P)
    rel_res = res_norm / (P_norm + 1e-20)
    print(f"NW Iter {i_NW}: rel_res = {rel_res:.6e}, res_norm = {res_norm:.6e}, P_norm = {P_norm:.6e}")
    
    dFm, info = sp.bicgstab(A_op, rhs, tol=1e-10, maxiter=150)
    dF = dFm.reshape(2, 2, nx, ny)
    print(f"  dF_norm = {np.linalg.norm(dF):.6e}, CG info = {info}")
    
    if i_NW == 0:
        F = F + DbarF_grid + dF
    else:
        F = F + dF
        
    P, K4, _ = constitutive_hyperelastic_2d(F, C4, I2, I4, I4rt, model_type="svk", plane_mode="plane_strain")
