import numpy as np
import scipy.sparse.linalg as sp
from mgkmc.finite_strain_simulator import (
    _make_identity_tensors_2d, build_ghat4_2d, build_C4_2d,
    constitutive_hyperelastic_2d, _get_frequencies_2d,
    _reconstruct_u_from_F_2d, get_grad_u_2d,
    _fft2_tensor, _ifft2_tensor, _ddot42,
    _invert_matrix_field_2d, solve_dbfft_linear_system_2d
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

F_bar = np.array([[1.05, 0.0],
                  [0.0, 0.95]])
F_init = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))

# Let's run a custom version of _dbfft_step_2d with print statements
F_bar_grid = np.einsum('ij,xy->ijxy', F_bar, np.ones((nx, ny)))
Xi = _get_frequencies_2d(nx, ny, Lx, Ly)
u = _reconstruct_u_from_F_2d(F_init, F_bar, Xi)

print("Starting custom DBFFT debug loop...")
for i_NW in range(15):
    grad_u = get_grad_u_2d(u, Xi)
    F_curr = F_bar_grid + grad_u
    
    P, K4, F33 = constitutive_hyperelastic_2d(
        F_curr, C4, I2, I4, I4rt, Fp=None,
        model_type="svk", plane_mode="plane_strain")
        
    P_hat = _fft2_tensor(P)
    b_hat = -1j * np.einsum('jxy,ijxy->ixy', Xi, P_hat)
    
    res_norm = np.linalg.norm(b_hat)
    P_norm = np.linalg.norm(P)
    rel_res = res_norm / (P_norm + 1e-20)
    
    print(f"NW Iter {i_NW}: rel_res = {rel_res:.6e}, res_norm = {res_norm:.6e}, P_norm = {P_norm:.6e}")
    
    K_avg = K4.mean(axis=(-2, -1))
    A_mat = np.einsum('jxy,ijlk,lxy->ikxy', Xi, K_avg, Xi)
    M_inv = _invert_matrix_field_2d(A_mat)
    
    du_hat, info = solve_dbfft_linear_system_2d(Xi, K4, M_inv, b_hat, tol_CG=1e-8)
    du = _ifft2_tensor(du_hat)
    
    du_norm = np.linalg.norm(du)
    print(f"  du_norm = {du_norm:.6e}, CG info = {info}")
    
    u = u + du

print("Done.")
