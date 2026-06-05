"""
finite_strain_simulator.py
==========================

2D finite-strain FFT-based solver following the variational approach of:

    de Geus, T.W.J. et al. (2017)
    "Finite strain FFT-based non-linear solvers made simple"
    Computer Methods in Applied Mechanics and Engineering
    https://doi.org/10.1016/j.cma.2016.12.032

Primary unknown : deformation gradient F  [ndim, ndim, nx, ny]
Stress measure  : 1st Piola-Kirchhoff P   [ndim, ndim, nx, ny]
Algorithm       : Outer Newton-Raphson  +  inner Conjugate Gradient
Projection op.  : de Geus Ĝ  (purely geometric, no reference stiffness)

Nothing in this file touches the existing small-strain solver.
"""

import numpy as np
import scipy.sparse.linalg as sp
import os
import time as _time
from datetime import datetime as _dt
from .analysis.vtk import export_to_vtk
from .linear_elastic_simulator import save_checkpoint_2d, save_checkpoint_3d

# ---------------------------------------------------------------------------
# Tensor operations  (layout: [ndim, ndim, nx, ny] — tensor indices FIRST)
# ---------------------------------------------------------------------------

def _trans2(A2):
    """Transpose of a 2nd-order tensor field. (ij -> ji)"""
    return np.einsum('ijxy->jixy', A2)

def _ddot42(A4, B2):
    """Double contraction A4:B2  ->  C2_{ij} = A4_{ijlk} B2_{lk}"""
    return np.einsum('ijklxy,lkxy->ijxy', A4, B2)

def _ddot44(A4, B4):
    """Double contraction A4:B4  ->  C4_{ijmn} = A4_{ijlk} B4_{lkmn}"""
    return np.einsum('ijklxy,lkmnxy->ijmnxy', A4, B4)

def _dot22(A2, B2):
    """Single contraction A2·B2  ->  C2_{ik} = A2_{ij} B2_{jk}"""
    return np.einsum('ijxy,jkxy->ikxy', A2, B2)

def _dot24(A2, B4):
    """Single contraction A2·B4  ->  C4_{ikmn} = A2_{ij} B4_{jkmn}"""
    return np.einsum('ijxy,jkmnxy->ikmnxy', A2, B4)

def _dot42(A4, B2):
    """Single contraction A4·B2  ->  C4_{ijkm} = A4_{ijkl} B2_{lm}"""
    return np.einsum('ijklxy,lmxy->ijkmxy', A4, B2)

def _dyad22(A2, B2):
    """Outer (dyadic) product  ->  C4_{ijkl} = A2_{ij} B2_{kl}"""
    return np.einsum('ijxy,klxy->ijklxy', A2, B2)

# ---------------------------------------------------------------------------
# Tensor operations for 3D  (layout: [ndim, ndim, nx, ny, nz] — indices FIRST)
# ---------------------------------------------------------------------------

def _trans2_3d(A2):
    """Transpose of a 2nd-order tensor field. (ij -> ji)"""
    return np.einsum('ijxyz->jixyz', A2)

def _ddot42_3d(A4, B2):
    """Double contraction A4:B2  ->  C2_{ij} = A4_{ijlk} B2_{lk}"""
    return np.einsum('ijklxyz,lkxyz->ijxyz', A4, B2)

def _ddot44_3d(A4, B4):
    """Double contraction A4:B4  ->  C4_{ijmn} = A4_{ijlk} B4_{lkmn}"""
    return np.einsum('ijklxyz,lkmnxyz->ijmnxyz', A4, B4)

def _dot22_3d(A2, B2):
    """Single contraction A2·B2  ->  C2_{ik} = A2_{ij} B2_{jk}"""
    return np.einsum('ijxyz,jkxyz->ikxyz', A2, B2)

def _dot24_3d(A2, B4):
    """Single contraction A2·B4  ->  C4_{ikmn} = A2_{ij} B4_{jkmn}"""
    return np.einsum('ijxyz,jkmnxyz->ikmnxyz', A2, B4)

def _dot42_3d(A4, B2):
    """Single contraction A4·B2  ->  C4_{ijkm} = A4_{ijkl} B2_{lm}"""
    return np.einsum('ijklxyz,lmxyz->ijkmxyz', A4, B2)

def _dyad22_3d(A2, B2):
    """Outer (dyadic) product  ->  C4_{ijkl} = A2_{ij} B2_{kl}"""
    return np.einsum('ijxyz,klxyz->ijklxyz', A2, B2)


# ---------------------------------------------------------------------------
# Identity tensors broadcast over a 3D grid  (nx, ny, nz)
# ---------------------------------------------------------------------------

def _make_identity_tensors_3d(nx, ny, nz):
    """
    Returns (I2, I4, I4rt, I4s, II) broadcast over an (nx, ny, nz) grid.
    Layout: spatial dimensions last → [3, 3, nx, ny, nz] and [3, 3, 3, 3, nx, ny, nz].
    """
    ndim = 3
    i    = np.eye(ndim)
    ones = np.ones((nx, ny, nz))

    I2   = np.einsum('ij,xyz->ijxyz', i, ones)
    I4   = np.einsum('ijkl,xyz->ijklxyz', np.einsum('il,jk', i, i), ones)
    I4rt = np.einsum('ijkl,xyz->ijklxyz', np.einsum('ik,jl', i, i), ones)
    I4s  = 0.5 * (I4 + I4rt)
    II   = _dyad22_3d(I2, I2)

    return I2, I4, I4rt, I4s, II

# ---------------------------------------------------------------------------
# Identity tensors broadcast over a 2D grid  (nx, ny)
# ---------------------------------------------------------------------------

def _make_identity_tensors_2d(nx, ny):
    """
    Returns (I2, I4, I4rt, I4s, II) broadcast over an (nx, ny) grid.
    Layout: spatial dimensions last → [2, 2, nx, ny] and [2, 2, 2, 2, nx, ny].
    """
    ndim = 2
    i    = np.eye(ndim)
    ones = np.ones((nx, ny))

    I2   = np.einsum('ij,xy->ijxy', i, ones)
    I4   = np.einsum('ijkl,xy->ijklxy', np.einsum('il,jk', i, i), ones)
    I4rt = np.einsum('ijkl,xy->ijklxy', np.einsum('ik,jl', i, i), ones)
    I4s  = 0.5 * (I4 + I4rt)
    II   = _dyad22(I2, I2)

    return I2, I4, I4rt, I4s, II


# ---------------------------------------------------------------------------
# De Geus projection operator  Ĝ  (Eq. 19 of the paper)
# ---------------------------------------------------------------------------

def build_ghat4_2d(nx, ny, Lx, Ly, even_grid=False):
    """
    Build the 2D de Geus projection operator in Fourier space.

    Shape: [2, 2, 2, 2, nx, ny]

    Ĝ_{ijlm}(q) = δ_{im} ξ_j(q) ξ_l(q) / |ξ|²
    where ξ_i = q_i / L_i  (scaled frequency)

    Zero at q = 0  (ensures zero mean of δF → prescribed F̄ is preserved).

    Parameters
    ----------
    nx, ny     : grid dimensions
    Lx, Ly     : physical cell size (same units as pixel size)
    even_grid  : ignored, parity is determined automatically per axis
    """
    ndim = 2

    # Centered frequency axes (cycles, not angular)
    # Determine parity for each axis independently
    fx = np.arange(-nx / 2., nx / 2.) if nx % 2 == 0 else np.arange(-(nx - 1) / 2., (nx + 1) / 2.)
    fy = np.arange(-ny / 2., ny / 2.) if ny % 2 == 0 else np.arange(-(ny - 1) / 2., (ny + 1) / 2.)

    # Scaled frequencies ξ_i = q_i / L_i  → shapes (nx,) and (ny,)
    xi_x = fx / Lx
    xi_y = fy / Ly

    # Broadcast to 2D grid
    Xi_x, Xi_y = np.meshgrid(xi_x, xi_y, indexing='ij')   # (nx, ny)
    xi  = np.stack([Xi_x, Xi_y], axis=0)                    # (2, nx, ny)
    xi2 = Xi_x**2 + Xi_y**2                                 # (nx, ny)

    # Nyquist mask for even axes: zero out where frequency is -N/2 (index 0)
    nyquist_mask = np.zeros((nx, ny), dtype=bool)
    if nx % 2 == 0:
        nyquist_mask |= (Xi_x == -nx / (2. * Lx))
    if ny % 2 == 0:
        nyquist_mask |= (Xi_y == -ny / (2. * Ly))

    safe_xi2 = xi2.copy()
    safe_xi2[xi2 == 0] = 1.0  # avoid divide-by-zero at q=0

    Ghat4 = np.zeros((ndim, ndim, ndim, ndim, nx, ny))
    delta  = np.eye(ndim)

    for i in range(ndim):
        for j in range(ndim):
            for l in range(ndim):
                for m in range(ndim):
                    val = delta[i, m] * xi[j] * xi[l] / safe_xi2
                    # Zero at DC frequency  (q = 0)
                    val[xi2 == 0] = 0.0
                    # Zero at Nyquist frequencies
                    val[nyquist_mask] = 0.0
                    Ghat4[i, j, l, m] = val

    return Ghat4


# ---------------------------------------------------------------------------
# FFT helpers — use fftshift/ifftshift convention matching de Geus et al.
# ---------------------------------------------------------------------------

def _fft2_tensor(A):
    """
    Forward FFT of every component of a [ndim, ndim, nx, ny] field.
    Uses fftshift/ifftshift so DC is centered — matches de Geus indexing.
    """
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(A, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )


def _ifft2_tensor(A_hat):
    """
    Inverse FFT of every component of a [ndim, ndim, nx, ny] field.
    Returns real part (imaginary part is numerical noise).
    """
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(A_hat, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    ).real


def _project(A2, Ghat4):
    """Apply the de Geus projection: Ĝ : A2  (all in Fourier space)."""
    A2_hat  = _fft2_tensor(A2)
    res_hat = np.einsum('ijklxy,lkxy->ijxy', Ghat4, A2_hat)
    return _ifft2_tensor(res_hat)


# ---------------------------------------------------------------------------
# Elastic stiffness tensor  C4  (isotropic, plane-strain 2D)
# ---------------------------------------------------------------------------

def build_C4_2d(E_field, nu_field, I4s, II, plane_mode='plane_strain'):
    """
    Build the 4th-order isotropic stiffness tensor on a 2D grid.

    For plane strain:  C = λ I⊗I + 2μ I^s   (3D Lamé constants)
    For plane stress:  effective λ is used    (λ* = 2λμ/(λ+2μ))

    Shape: [2, 2, 2, 2, nx, ny]
    """
    mu  = E_field / (2.0 * (1.0 + nu_field))          # shear modulus

    if plane_mode == 'plane_stress':
        # Effective λ for plane stress
        lam = E_field * nu_field / (1.0 - nu_field**2)
    else:
        # Plane strain: 3D Lamé λ
        lam = E_field * nu_field / ((1.0 + nu_field) * (1.0 - 2.0 * nu_field))

    # Broadcast to [1,1,nx,ny] for einsum with tensors
    lam_ = lam[np.newaxis, np.newaxis, :, :]  # shape trick not needed; II already broadcast
    mu_  = mu[np.newaxis, np.newaxis, :, :]

    # C4 = λ (I⊗I) + 2μ I^s        [2,2,2,2,nx,ny]
    C4 = lam[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * II \
       + 2.0 * mu[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * I4s

    return C4


# ---------------------------------------------------------------------------
# Constitutive model: hyper-elasticity in reference configuration
# ---------------------------------------------------------------------------

def constitutive_hyperelastic_2d(F, C4, I2, I4, I4rt):
    """
    Compute 1st Piola-Kirchhoff stress P and consistent tangent K4 from F.

    Physics (de Geus et al., Section 4):
        E = ½(Fᵀ·F − I)             Green-Lagrange strain
        S = C4 : E                   2nd Piola-Kirchhoff stress
        P = F · S                    1st Piola-Kirchhoff stress
        K4 = S⊗I + Iᴿᵀ:(F·C4·Fᵀ):Iᴿᵀ   consistent tangent

    Parameters
    ----------
    F    : [2, 2, nx, ny]  deformation gradient field
    C4   : [2, 2, 2, 2, nx, ny]  elastic stiffness tensor
    I2   : [2, 2, nx, ny]  identity
    I4   : [2, 2, 2, 2, nx, ny]  4th-order identity
    I4rt : [2, 2, 2, 2, nx, ny]  right-transposed 4th-order identity

    Returns
    -------
    P  : [2, 2, nx, ny]
    K4 : [2, 2, 2, 2, nx, ny]
    """
    E_GL = 0.5 * (_dot22(_trans2(F), F) - I2)          # Green-Lagrange strain
    S    = _ddot42(C4, E_GL)                            # 2nd PK stress
    P    = _dot22(F, S)                                 # 1st PK stress

    # Consistent tangent: K4 = S⊗I + Iᴿᵀ:(F·C4·Fᵀ):Iᴿᵀ
    K4 = _dot24(S, I4) + _ddot44(
             _ddot44(I4rt, _dot42(_dot24(F, C4), _trans2(F))),
             I4rt
         )

    return P, K4


# ---------------------------------------------------------------------------
# Cauchy stress from P and F
# ---------------------------------------------------------------------------

def cauchy_from_P(P, F):
    """
    Convert 1st Piola-Kirchhoff stress P to Cauchy stress σ.

        σ = (1/J) · P · Fᵀ       J = det(F)

    Operates pointwise on [2, 2, nx, ny] fields.
    Returns σ as [2, 2, nx, ny].
    """
    ndim, _, nx, ny = P.shape
    # det(F) per pixel
    J = (F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])  # [nx, ny]
    J_safe = np.where(np.abs(J) < 1e-14, 1e-14, J)

    # P · Fᵀ  pointwise
    PFt = _dot22(P, _trans2(F))                     # [2, 2, nx, ny]

    sigma = PFt / J_safe[np.newaxis, np.newaxis, :, :]
    return sigma


# ---------------------------------------------------------------------------
# 3D de Geus projection operator Ĝ (Eq. 19 of the paper)
# ---------------------------------------------------------------------------

def build_ghat4_3d(nx, ny, nz, Lx, Ly, Lz, even_grid=False):
    """
    Build the 3D de Geus projection operator in Fourier space.

    Shape: [3, 3, 3, 3, nx, ny, nz]

    Ĝ_{ijlm}(q) = δ_{im} ξ_j(q) ξ_l(q) / |ξ|²
    where ξ_i = q_i / L_i

    Zero at q = 0 (enforces zero mean).
    """
    ndim = 3

    # Centered frequency axes (cycles, not angular)
    # Determine parity for each axis independently
    fx = np.arange(-nx / 2., nx / 2.) if nx % 2 == 0 else np.arange(-(nx - 1) / 2., (nx + 1) / 2.)
    fy = np.arange(-ny / 2., ny / 2.) if ny % 2 == 0 else np.arange(-(ny - 1) / 2., (ny + 1) / 2.)
    fz = np.arange(-nz / 2., nz / 2.) if nz % 2 == 0 else np.arange(-(nz - 1) / 2., (nz + 1) / 2.)

    xi_x = fx / Lx
    xi_y = fy / Ly
    xi_z = fz / Lz

    Xi_x, Xi_y, Xi_z = np.meshgrid(xi_x, xi_y, xi_z, indexing='ij')
    xi  = np.stack([Xi_x, Xi_y, Xi_z], axis=0)
    xi2 = Xi_x**2 + Xi_y**2 + Xi_z**2

    # Nyquist mask for even axes: zero out where frequency is -N/2 (index 0)
    nyquist_mask = np.zeros((nx, ny, nz), dtype=bool)
    if nx % 2 == 0:
        nyquist_mask |= (Xi_x == -nx / (2. * Lx))
    if ny % 2 == 0:
        nyquist_mask |= (Xi_y == -ny / (2. * Ly))
    if nz % 2 == 0:
        nyquist_mask |= (Xi_z == -nz / (2. * Lz))

    safe_xi2 = xi2.copy()
    safe_xi2[xi2 == 0] = 1.0

    Ghat4 = np.zeros((ndim, ndim, ndim, ndim, nx, ny, nz))
    delta = np.eye(ndim)

    for i in range(ndim):
        for j in range(ndim):
            for l in range(ndim):
                for m in range(ndim):
                    val = delta[i, m] * xi[j] * xi[l] / safe_xi2
                    val[xi2 == 0] = 0.0
                    val[nyquist_mask] = 0.0
                    Ghat4[i, j, l, m] = val

    return Ghat4


# ---------------------------------------------------------------------------
# FFT helpers for 3D
# ---------------------------------------------------------------------------

def _fft3_tensor(A):
    """Forward FFT of every component of a [ndim, ndim, nx, ny, nz] field."""
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(A, axes=(-3, -2, -1)),
            axes=(-3, -2, -1)
        ),
        axes=(-3, -2, -1)
    )


def _ifft3_tensor(A_hat):
    """Inverse FFT of every component of a [ndim, ndim, nx, ny, nz] field."""
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(A_hat, axes=(-3, -2, -1)),
            axes=(-3, -2, -1)
        ),
        axes=(-3, -2, -1)
    ).real


def _project_3d(A2, Ghat4):
    """Apply the de Geus projection in 3D: Ĝ : A2"""
    A2_hat  = _fft3_tensor(A2)
    res_hat = np.einsum('ijklxyz,lkxyz->ijxyz', Ghat4, A2_hat)
    return _ifft3_tensor(res_hat)


# ---------------------------------------------------------------------------
# Elastic stiffness tensor C4 in 3D
# ---------------------------------------------------------------------------

def build_C4_3d(E_field, nu_field, I4s, II):
    """
    Build the 4th-order isotropic stiffness tensor on a 3D grid.
    C = λ I⊗I + 2μ I^s
    """
    mu  = E_field / (2.0 * (1.0 + nu_field))
    lam = E_field * nu_field / ((1.0 + nu_field) * (1.0 - 2.0 * nu_field))

    # C4 = λ (I⊗I) + 2μ I^s        [3,3,3,3,nx,ny,nz]
    C4 = lam[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * II \
       + 2.0 * mu[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * I4s

    return C4


# ---------------------------------------------------------------------------
# Constitutive model in 3D
# ---------------------------------------------------------------------------

def constitutive_hyperelastic_3d(F, C4, I2, I4, I4rt):
    """Compute 1st Piola-Kirchhoff stress P and consistent tangent K4 in 3D."""
    E_GL = 0.5 * (_dot22_3d(_trans2_3d(F), F) - I2)
    S    = _ddot42_3d(C4, E_GL)
    P    = _dot22_3d(F, S)

    K4 = _dot24_3d(S, I4) + _ddot44_3d(
             _ddot44_3d(I4rt, _dot42_3d(_dot24_3d(F, C4), _trans2_3d(F))),
             I4rt
         )

    return P, K4


# ---------------------------------------------------------------------------
# Cauchy stress from P and F in 3D
# ---------------------------------------------------------------------------

def cauchy_from_P_3d(P, F):
    """Convert 1st Piola-Kirchhoff stress P to Cauchy stress σ in 3D."""
    # det(F) per pixel (3x3 determinant)
    J = (
        F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]) -
        F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0]) +
        F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0])
    )
    J_safe = np.where(np.abs(J) < 1e-14, 1e-14, J)

    PFt = _dot22_3d(P, _trans2_3d(F))
    sigma = PFt / J_safe[np.newaxis, np.newaxis, :, :, :]
    return sigma



# ---------------------------------------------------------------------------
# Boundary condition builder
# ---------------------------------------------------------------------------

def build_finite_strain_bc(driving_component, eps_target_step,
                           mixed_targets, plane_mode, ndim=2):
    """
    Translate the standard MGKMC config keys into finite-strain BC masks.

    Parameters
    ----------
    driving_component : (i, j) tuple  — which F component is driven
    eps_target_step   : float         — current eps increment value
    mixed_targets     : dict {(i,j): stress_value_Pa}
    plane_mode        : 'plane_strain' or 'plane_stress'
    ndim              : 2

    Returns
    -------
    F_bar    : ndarray (ndim, ndim) — prescribed F̄ (components for driven + constraints)
    F_mask   : bool ndarray (ndim, ndim) — True where F̄ is prescribed
    P_target : ndarray (ndim, ndim) — stress targets (Pa), 0 where free
    P_mask   : bool ndarray (ndim, ndim) — True where avg stress is prescribed
    """
    F_bar    = np.eye(ndim)          # start from identity
    F_mask   = np.zeros((ndim, ndim), dtype=bool)
    P_target = np.zeros((ndim, ndim))
    P_mask   = np.zeros((ndim, ndim), dtype=bool)

    i_drv, j_drv = driving_component

    # --- Driven F component ---
    if i_drv == j_drv:
        # Normal component: F_ii = 1 + eps
        F_bar[i_drv, j_drv]   = 1.0 + eps_target_step
    else:
        # Shear component: F_ij = eps (diagonal stays at 1)
        F_bar[i_drv, j_drv]   = eps_target_step
    F_mask[i_drv, j_drv] = True

    # --- Off-diagonal F: prescribed = 0 (no spurious shear) ---
    for ii in range(ndim):
        for jj in range(ndim):
            if ii == jj:
                continue   # diagonal handled below
            if (ii, jj) == (i_drv, j_drv):
                continue   # already set as driving
            # Off-diagonal shear: prescribe F = 0
            F_bar[ii, jj]  = 0.0
            F_mask[ii, jj] = True

    # --- Diagonal components not driven: stress-free or constrained ---
    for k in range(ndim):
        if k == i_drv and i_drv == j_drv:
            continue  # already handled as driving

        comp = (k, k)
        if comp in mixed_targets:
            # Stress-free (or prescribed stress): F_kk is FREE, stress is driven
            # F_bar[k,k] stays at 1.0 initially; outer loop will adjust it
            P_target[k, k] = mixed_targets[comp]
            P_mask[k, k]   = True
        else:
            # No mention in mixed_targets  →  plane_strain default: F_kk = 1.0
            F_bar[k, k]    = 1.0
            F_mask[k, k]   = True

    return F_bar, F_mask, P_target, P_mask


# ---------------------------------------------------------------------------
# Core Newton-CG finite-strain solver (single load step)
# ---------------------------------------------------------------------------

def _newton_cg_step(F, F_bar, Ghat4, C4, I2, I4, I4rt,
                    tol_NW=1e-5, tol_CG=1e-8, max_NW=20):
    """
    Run Newton-CG iterations to enforce  G : P(F) = 0
    with prescribed macroscopic deformation gradient F̄.

    Follows Algorithm 1 of de Geus et al. (2017).

    F_bar is the *absolute* target mean deformation gradient for this step.
    The function computes ΔF̄ = F_bar - mean(F) and applies it as the BC
    on the first Newton iteration.

    Returns
    -------
    F      : converged deformation gradient field [2, 2, nx, ny]
    P      : converged 1st PK stress field
    K4     : converged tangent stiffness
    n_iter : number of Newton iterations taken
    """
    ndim = 2
    nx, ny = F.shape[2], F.shape[3]

    # Increment in macroscopic F to apply this step
    DbarF      = F_bar - F.mean(axis=(2, 3))          # (2,2)
    DbarF_grid = np.einsum('ij,xy->ijxy', DbarF, np.ones((nx, ny)))

    # Pre-compute constitutive response at current F
    P, K4 = constitutive_hyperelastic_2d(F, C4, I2, I4, I4rt)
    Fn    = np.linalg.norm(F)

    def G_op(A2):
        """Apply projection G to a 2nd-order tensor field."""
        return _project(A2, Ghat4)

    def K_dF_op(dFm_flat):
        """Apply  K^LT : δFᵀ  pointwise, return as grid."""
        dF  = dFm_flat.reshape(ndim, ndim, nx, ny)
        return _trans2(_ddot42(K4, _trans2(dF)))

    def G_K_dF(dFm_flat):
        """Linear operator seen by CG:  G : K^LT : δFᵀ"""
        return G_op(K_dF_op(dFm_flat)).reshape(-1)

    A_op = sp.LinearOperator(
        shape=(F.size, F.size),
        matvec=G_K_dF,
        dtype='float64'
    )

    for i_NW in range(max_NW):
        if i_NW == 0:
            # First Newton step: distribute ΔF̄ over the microstructure
            # using the current tangent K4  (de Geus Eq. 27)
            rhs = -G_op(K_dF_op(DbarF_grid.reshape(-1))).reshape(-1)
        else:
            # Subsequent steps: equilibrium residual  (de Geus Eq. 23)
            rhs = -G_op(P).reshape(-1)

        # Solve linear system with CG (backward-compatible tolerance argument)
        try:
            dFm, _ = sp.cg(A_op, rhs, rtol=tol_CG)
        except TypeError:
            dFm, _ = sp.cg(A_op, rhs, tol=tol_CG)
        dF     = dFm.reshape(ndim, ndim, nx, ny)

        # Update F
        if i_NW == 0:
            F = F + DbarF_grid + dF    # apply macro jump + micro fluctuation
        else:
            F = F + dF                 # pure Newton update

        # Recompute constitutive response at updated F
        P, K4 = constitutive_hyperelastic_2d(F, C4, I2, I4, I4rt)

        # Convergence check (skip iteration 0 as in de Geus code)
        res_norm = np.linalg.norm(dFm) / (np.linalg.norm(F) + 1e-20)
        if res_norm < tol_NW and i_NW > 0:
            return F, P, K4, i_NW + 1

    return F, P, K4, max_NW


# ---------------------------------------------------------------------------
# Top-level incremental simulation (public API)
# ---------------------------------------------------------------------------

def finite_strain_simulation_2d(
    E, nu,
    driving_component,
    eps_target,
    n_steps,
    mixed_targets=None,
    plane_mode='plane_strain',
    pixel=1.0,
    tol_NW=1e-5,
    tol_CG=1e-8,
    max_NW=20,
    tol_macro=1e6,      # Pa — outer mixed-BC convergence tolerance
    max_iter_macro=20,
    store=True,
    log_path=None,
    global_log_path=None,
    enable_console=True,
    checkpoint_interval="none",
    checkpoint_path=None,
    vtk_interval="none",
    vtk_path=None,
):
    """
    Incremental 2D finite-strain simulation using the de Geus FFT Newton-CG method.

    The config-facing interface is identical to linear_elastic_simulation_2d:
    driving_component, eps_target, mixed_targets carry the same meaning.
    Internally they are converted to a DAMASK-style F/P BC pair.

    Parameters
    ----------
    E, nu            : (nx, ny) arrays of elastic modulus (Pa) and Poisson's ratio
    driving_component: (i, j) tuple  e.g. (0,0) for xx
    eps_target       : float  — final engineering strain in driving component
    n_steps          : int    — number of load increments
    mixed_targets    : dict {(i,j): stress_Pa}  — stress-controlled components
    plane_mode       : 'plane_strain' (default) or 'plane_stress'
    pixel            : float  — voxel/pixel size (nm or consistent unit)
    tol_NW           : Newton convergence tolerance (||δF||/||F||)
    tol_CG           : CG convergence tolerance
    tol_macro        : outer mixed-BC stress convergence (Pa)
    store            : if True, return full field histories

    Returns
    -------
    F_macro_arr  : (n_steps+1, 2, 2)  macroscopic F
    Sig_macro_arr: (n_steps+1, 2, 2)  macroscopic Cauchy stress
    P_macro_arr  : (n_steps+1, 2, 2)  macroscopic 1st PK stress
    F_list       : list of [2,2,nx,ny] fields (if store=True, else [])
    Sig_list     : list of [2,2,nx,ny] Cauchy stress fields (if store)
    """
    if mixed_targets is None:
        mixed_targets = {}

    if checkpoint_path and checkpoint_interval not in [None, "none"]:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if vtk_path and vtk_interval not in [None, "none"]:
        os.makedirs(os.path.dirname(vtk_path) or ".", exist_ok=True)

    nx, ny = E.shape
    Lx, Ly = nx * pixel, ny * pixel
    ndim   = 2

    # Detect even grid
    even_grid = (nx % 2 == 0) or (ny % 2 == 0)

    # Pre-build projection operator and identity tensors
    Ghat4 = build_ghat4_2d(nx, ny, Lx, Ly, even_grid=even_grid)
    I2, I4, I4rt, I4s, II = _make_identity_tensors_2d(nx, ny)

    # Build material stiffness C4 on the grid
    C4 = build_C4_2d(E, nu, I4s, II, plane_mode=plane_mode)

    # Component labels for logging
    _comp_labels = {(0,0):'xx', (1,1):'yy', (0,1):'xy', (1,0):'yx'}
    _drv_lbl = _comp_labels.get(tuple(driving_component), 'xx')

    # Logging setup
    _log_f  = None
    _glog_f = None

    if log_path:
        _log_f = open(log_path, 'w', buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'F_'+_drv_lbl:<14} {'Eng_strain_'+_drv_lbl:<20} "
                f"{'Sig_'+_drv_lbl+'(GPa)':<18}\n")
        _log_f.write(_hdr)
        _log_f.write('-' * len(_hdr.rstrip()) + '\n')

    if global_log_path:
        _glog_f = open(global_log_path, 'w', buffering=1)
        _ghdr = (f"{'Step':<8} "
                 f"{'F_xx':<12} {'F_yy':<12} {'F_xy':<12} "
                 f"{'EngStr_xx':<14} {'EngStr_yy':<14} "
                 f"{'P_xx(GPa)':<14} {'P_yy(GPa)':<14} {'P_xy(GPa)':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_xy(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write('-' * len(_ghdr.rstrip()) + '\n')

    _t0 = _time.time()

    # Storage
    F_macro_list   = []
    Sig_macro_list = []
    P_macro_list   = []
    F_list         = []
    Sig_list       = []

    # Initialise F = I  (undeformed)
    F = np.einsum('ij,xy->ijxy', np.eye(ndim), np.ones((nx, ny)))

    # Incremental loading
    eps_steps = np.linspace(0.0, eps_target, n_steps + 1)

    for s in range(n_steps + 1):
        eps_s = eps_steps[s]

        # Build target F̄ and stress masks for this step
        F_bar, F_mask, P_tgt, P_mask = build_finite_strain_bc(
            driving_component, eps_s, mixed_targets, plane_mode
        )

        # Save the field at the beginning of this load step.
        # The outer mixed-BC loop always starts Newton from this same state
        # so that DbarF = F_bar - mean(F_start) is computed consistently.
        F_start = F.copy()

        # ---- Outer loop: iterate free F̄ components to satisfy stress BCs ----
        for it_mac in range(max_iter_macro):
            F, P_field, K4, _n_iter = _newton_cg_step(
                F_start.copy(), F_bar, Ghat4, C4, I2, I4, I4rt,
                tol_NW=tol_NW, tol_CG=tol_CG, max_NW=max_NW
            )

            # Macroscopic averages
            F_mac   = F.mean(axis=(2, 3))           # (2,2)
            P_mac   = P_field.mean(axis=(2, 3))     # (2,2)

            # Compute Cauchy stress field and macro average
            Sig_field = cauchy_from_P(P_field, F)
            Sig_mac   = Sig_field.mean(axis=(2, 3)) # (2,2)

            # Check outer mixed-BC convergence
            if not np.any(P_mask):
                break   # no stress BCs → single inner solve suffices

            stress_err = np.zeros((ndim, ndim))
            stress_err[P_mask] = P_tgt[P_mask] - Sig_mac[P_mask]
            max_err = np.max(np.abs(stress_err[P_mask]))

            if max_err < tol_macro:
                break

            # Update free F̄ components using Poisson-coupled elastic correction.
            # Only touch components that are stress-controlled (P_mask=True).
            # The driven F component must remain fixed at F_bar[i_drv, j_drv].
            i_drv, j_drv = driving_component
            E_avg = E.mean()
            nu_avg = nu.mean()
            d_F_mat = (stress_err - nu_avg * np.trace(stress_err) * np.eye(ndim)) / E_avg
            for ii in range(ndim):
                for jj in range(ndim):
                    if not (P_mask[ii, jj] and ii == jj):
                        continue
                    if (ii, jj) == (i_drv, j_drv):
                        continue   # never perturb the driven component
                    F_bar[ii, jj] += d_F_mat[ii, jj]
        else:
            if enable_console:
                print(f"  Warning: outer BC loop did not converge at step {s} "
                      f"(max_err={max_err:.2e} Pa)")

        # Store results
        F_macro_list.append(F_mac.copy())
        Sig_macro_list.append(Sig_mac.copy())
        P_macro_list.append(P_mac.copy())

        if store:
            F_list.append(F.copy())
            Sig_list.append(Sig_field.copy())

        # --- Console output ---
        i_drv, j_drv = driving_component
        F_drv      = F_mac[i_drv, j_drv]
        eng_strain = F_drv - (1.0 if i_drv == j_drv else 0.0)
        Sig_drv    = Sig_mac[i_drv, j_drv]

        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"F_{_drv_lbl}={F_drv:.5f}, "
                  f"eng_strain={eng_strain:.5f}, "
                  f"Sig_{_drv_lbl}={Sig_drv/1e9:.4f} GPa")

        if _log_f:
            _now     = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{F_drv:<14.6f} {eng_strain:<20.6f} "
                f"{Sig_drv/1e9:<18.6f}\n"
            )

        if _glog_f:
            _glog_f.write(
                f"{s:<8} "
                f"{F_mac[0,0]:<12.6f} {F_mac[1,1]:<12.6f} {F_mac[0,1]:<12.6f} "
                f"{F_mac[0,0]-1:<14.6f} {F_mac[1,1]-1:<14.6f} "
                f"{P_mac[0,0]/1e9:<14.6f} {P_mac[1,1]/1e9:<14.6f} {P_mac[0,1]/1e9:<14.6f} "
                f"{Sig_mac[0,0]/1e9:<16.6f} {Sig_mac[1,1]/1e9:<16.6f} {Sig_mac[0,1]/1e9:<16.6f}\n"
            )

        # Checkpoint export
        if checkpoint_interval is not None and checkpoint_interval not in ["none", "last"] and checkpoint_path is not None:
            save_chk, cp_name = False, None
            if checkpoint_interval == "current":
                save_chk, cp_name = True, f"{checkpoint_path}.h5"
            elif isinstance(checkpoint_interval, int) and s % checkpoint_interval == 0:
                save_chk, cp_name = True, f"{checkpoint_path}_{s:06d}.h5"
            if save_chk and cp_name:
                E_GL = 0.5 * (_dot22(_trans2(F), F) - I2)
                eps_vtk = np.einsum('ijxy->xyij', E_GL)
                sig_vtk = np.einsum('ijxy->xyij', Sig_field)
                save_checkpoint_2d(cp_name, s, E, nu, eps_vtk, sig_vtk, F_mac, Sig_mac, pixel)

        # VTK export
        if vtk_interval is not None and vtk_interval not in ["none", "last"] and vtk_path is not None:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                E_GL = 0.5 * (_dot22(_trans2(F), F) - I2)
                eps_vtk = np.einsum('ijxy->xyij', E_GL)
                sig_vtk = np.einsum('ijxy->xyij', Sig_field)
                export_to_vtk(vt_name, eps_vtk, sig_vtk, E, nu, pixel, match_matplotlib_orientation=True)

    # Final step exports
    if checkpoint_path is not None:
        if checkpoint_interval == "last":
            E_GL = 0.5 * (_dot22(_trans2(F), F) - I2)
            eps_vtk = np.einsum('ijxy->xyij', E_GL)
            sig_vtk = np.einsum('ijxy->xyij', Sig_field)
            save_checkpoint_2d(f"{checkpoint_path}_final.h5", n_steps, E, nu, eps_vtk, sig_vtk, F_macro_list[-1], Sig_macro_list[-1], pixel)
        elif checkpoint_interval not in [None, "none", "last"] and isinstance(checkpoint_interval, int) and n_steps % checkpoint_interval != 0:
            E_GL = 0.5 * (_dot22(_trans2(F), F) - I2)
            eps_vtk = np.einsum('ijxy->xyij', E_GL)
            sig_vtk = np.einsum('ijxy->xyij', Sig_field)
            save_checkpoint_2d(f"{checkpoint_path}_final.h5", n_steps, E, nu, eps_vtk, sig_vtk, F_macro_list[-1], Sig_macro_list[-1], pixel)

    if vtk_path is not None:
        if vtk_interval == "last":
            E_GL = 0.5 * (_dot22(_trans2(F), F) - I2)
            eps_vtk = np.einsum('ijxy->xyij', E_GL)
            sig_vtk = np.einsum('ijxy->xyij', Sig_field)
            export_to_vtk(f"{vtk_path}_final.vtu", eps_vtk, sig_vtk, E, nu, pixel, match_matplotlib_orientation=True)
        elif vtk_interval not in [None, "none", "last"] and isinstance(vtk_interval, int) and n_steps % vtk_interval != 0:
            E_GL = 0.5 * (_dot22(_trans2(F), F) - I2)
            eps_vtk = np.einsum('ijxy->xyij', E_GL)
            sig_vtk = np.einsum('ijxy->xyij', Sig_field)
            export_to_vtk(f"{vtk_path}_final.vtu", eps_vtk, sig_vtk, E, nu, pixel, match_matplotlib_orientation=True)

    if _log_f:  _log_f.close()
    if _glog_f: _glog_f.close()

    return (
        np.array(F_macro_list),
        np.array(Sig_macro_list),
        np.array(P_macro_list),
        F_list,
        Sig_list,
    )


# ---------------------------------------------------------------------------
# 3D Boundary condition builder
# ---------------------------------------------------------------------------

def build_finite_strain_bc_3d(driving_component, eps_target_step,
                              mixed_targets, ndim=3):
    """
    Translate the standard MGKMC config keys into 3D finite-strain BC masks.
    """
    F_bar    = np.eye(ndim)          # start from identity
    F_mask   = np.zeros((ndim, ndim), dtype=bool)
    P_target = np.zeros((ndim, ndim))
    P_mask   = np.zeros((ndim, ndim), dtype=bool)

    i_drv, j_drv = driving_component

    # --- Driven F component ---
    if i_drv == j_drv:
        F_bar[i_drv, j_drv]   = 1.0 + eps_target_step
    else:
        F_bar[i_drv, j_drv]   = eps_target_step
    F_mask[i_drv, j_drv] = True

    # --- Off-diagonal F: prescribed = 0 (no shear unless driven) ---
    for ii in range(ndim):
        for jj in range(ndim):
            if ii == jj:
                continue
            if (ii, jj) == (i_drv, j_drv):
                continue
            F_bar[ii, jj]  = 0.0
            F_mask[ii, jj] = True

    # --- Diagonal components not driven: stress-free or constrained ---
    for k in range(ndim):
        if k == i_drv and i_drv == j_drv:
            continue

        comp = (k, k)
        if comp in mixed_targets:
            P_target[k, k] = mixed_targets[comp]
            P_mask[k, k]   = True
        else:
            # default plane strain / fixed diagonal: F_kk = 1.0
            F_bar[k, k]    = 1.0
            F_mask[k, k]   = True

    return F_bar, F_mask, P_target, P_mask


# ---------------------------------------------------------------------------
# Core 3D Newton-CG finite-strain solver step
# ---------------------------------------------------------------------------

def _newton_cg_step_3d(F, F_bar, Ghat4, C4, I2, I4, I4rt,
                       tol_NW=1e-5, tol_CG=1e-8, max_NW=20):
    """
    Run 3D Newton-CG iterations to enforce G : P(F) = 0.
    """
    ndim = 3
    nx, ny, nz = F.shape[2], F.shape[3], F.shape[4]

    DbarF      = F_bar - F.mean(axis=(2, 3, 4))
    DbarF_grid = np.einsum('ij,xyz->ijxyz', DbarF, np.ones((nx, ny, nz)))

    P, K4 = constitutive_hyperelastic_3d(F, C4, I2, I4, I4rt)

    def G_op(A2):
        return _project_3d(A2, Ghat4)

    def K_dF_op(dFm_flat):
        dF  = dFm_flat.reshape(ndim, ndim, nx, ny, nz)
        return _trans2_3d(_ddot42_3d(K4, _trans2_3d(dF)))

    def G_K_dF(dFm_flat):
        return G_op(K_dF_op(dFm_flat)).reshape(-1)

    A_op = sp.LinearOperator(
        shape=(F.size, F.size),
        matvec=G_K_dF,
        dtype='float64'
    )

    for i_NW in range(max_NW):
        if i_NW == 0:
            rhs = -G_op(K_dF_op(DbarF_grid.reshape(-1))).reshape(-1)
        else:
            rhs = -G_op(P).reshape(-1)

        try:
            dFm, _ = sp.cg(A_op, rhs, rtol=tol_CG)
        except TypeError:
            dFm, _ = sp.cg(A_op, rhs, tol=tol_CG)
        dF     = dFm.reshape(ndim, ndim, nx, ny, nz)

        if i_NW == 0:
            F = F + DbarF_grid + dF
        else:
            F = F + dF

        P, K4 = constitutive_hyperelastic_3d(F, C4, I2, I4, I4rt)

        res_norm = np.linalg.norm(dFm) / (np.linalg.norm(F) + 1e-20)
        if res_norm < tol_NW and i_NW > 0:
            return F, P, K4, i_NW + 1

    return F, P, K4, max_NW


# ---------------------------------------------------------------------------
# Top-level 3D finite strain simulation driver (public API)
# ---------------------------------------------------------------------------

def finite_strain_simulation_3d(
    E, nu,
    driving_component,
    eps_target,
    n_steps,
    mixed_targets=None,
    pixel=1.0,
    tol_NW=1e-5,
    tol_CG=1e-8,
    max_NW=20,
    tol_macro=1e6,      # Pa — stress tolerance
    max_iter_macro=20,
    store=True,
    log_path=None,
    global_log_path=None,
    enable_console=True,
    checkpoint_interval="none",
    checkpoint_path=None,
    vtk_interval="none",
    vtk_path=None,
):
    """
    Incremental 3D finite-strain simulation using Newton-CG.
    """
    if mixed_targets is None:
        mixed_targets = {}

    if checkpoint_path and checkpoint_interval not in [None, "none"]:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    if vtk_path and vtk_interval not in [None, "none"]:
        os.makedirs(os.path.dirname(vtk_path) or ".", exist_ok=True)

    nx, ny, nz = E.shape
    Lx, Ly, Lz = nx * pixel, ny * pixel, nz * pixel
    ndim = 3

    # Detect even grid
    even_grid = (nx % 2 == 0) or (ny % 2 == 0) or (nz % 2 == 0)

    # Build 3D operators
    Ghat4 = build_ghat4_3d(nx, ny, nz, Lx, Ly, Lz, even_grid=even_grid)
    I2, I4, I4rt, I4s, II = _make_identity_tensors_3d(nx, ny, nz)
    C4 = build_C4_3d(E, nu, I4s, II)

    # Component labels
    _comp_labels = {
        (0,0):'xx', (1,1):'yy', (2,2):'zz',
        (0,1):'xy', (0,2):'xz', (1,2):'yz',
        (1,0):'yx', (2,0):'zx', (2,1):'zy'
    }
    _drv_lbl = _comp_labels.get(tuple(driving_component), 'xx')

    _log_f = None
    _glog_f = None

    if log_path:
        _log_f = open(log_path, 'w', buffering=1)
        _hdr = (f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} "
                f"{'F_'+_drv_lbl:<14} {'Eng_strain_'+_drv_lbl:<20} "
                f"{'Sig_'+_drv_lbl+'(GPa)':<18}\n")
        _log_f.write(_hdr)
        _log_f.write('-' * len(_hdr.rstrip()) + '\n')

    if global_log_path:
        _glog_f = open(global_log_path, 'w', buffering=1)
        _ghdr = (f"{'Step':<8} "
                 f"{'F_xx':<12} {'F_yy':<12} {'F_zz':<12} {'F_xy':<12} {'F_xz':<12} {'F_yz':<12} "
                 f"{'EngStr_xx':<14} {'EngStr_yy':<14} {'EngStr_zz':<14} "
                 f"{'P_xx(GPa)':<14} {'P_yy(GPa)':<14} {'P_zz(GPa)':<14} "
                 f"{'P_xy(GPa)':<14} {'P_xz(GPa)':<14} {'P_yz(GPa)':<14} "
                 f"{'Sig_xx(GPa)':<16} {'Sig_yy(GPa)':<16} {'Sig_zz(GPa)':<16} "
                 f"{'Sig_xy(GPa)':<16} {'Sig_xz(GPa)':<16} {'Sig_yz(GPa)':<16}\n")
        _glog_f.write(_ghdr)
        _glog_f.write('-' * len(_ghdr.rstrip()) + '\n')

    _t0 = _time.time()

    F_macro_list = []
    Sig_macro_list = []
    P_macro_list = []
    F_list = []
    Sig_list = []

    # Initialize F = I
    F = np.einsum('ij,xyz->ijxyz', np.eye(ndim), np.ones((nx, ny, nz)))

    eps_steps = np.linspace(0.0, eps_target, n_steps + 1)

    for s in range(n_steps + 1):
        eps_s = eps_steps[s]

        F_bar, F_mask, P_tgt, P_mask = build_finite_strain_bc_3d(
            driving_component, eps_s, mixed_targets, ndim=ndim
        )

        F_start = F.copy()

        for it_mac in range(max_iter_macro):
            F, P_field, K4, _n_iter = _newton_cg_step_3d(
                F_start.copy(), F_bar, Ghat4, C4, I2, I4, I4rt,
                tol_NW=tol_NW, tol_CG=tol_CG, max_NW=max_NW
            )

            F_mac = F.mean(axis=(2, 3, 4))
            P_mac = P_field.mean(axis=(2, 3, 4))

            Sig_field = cauchy_from_P_3d(P_field, F)
            Sig_mac = Sig_field.mean(axis=(2, 3, 4))

            if not np.any(P_mask):
                break

            stress_err = np.zeros((ndim, ndim))
            stress_err[P_mask] = P_tgt[P_mask] - Sig_mac[P_mask]
            max_err = np.max(np.abs(stress_err[P_mask]))

            if max_err < tol_macro:
                break

            i_drv, j_drv = driving_component
            E_avg = E.mean()
            nu_avg = nu.mean()
            d_F_mat = (stress_err - nu_avg * np.trace(stress_err) * np.eye(ndim)) / E_avg
            for ii in range(ndim):
                for jj in range(ndim):
                    if not (P_mask[ii, jj] and ii == jj):
                        continue
                    if (ii, jj) == (i_drv, j_drv):
                        continue
                    F_bar[ii, jj] += d_F_mat[ii, jj]
        else:
            if enable_console:
                print(f"  Warning: outer BC loop did not converge at step {s} "
                      f"(max_err={max_err:.2e} Pa)")

        F_macro_list.append(F_mac.copy())
        Sig_macro_list.append(Sig_mac.copy())
        P_macro_list.append(P_mac.copy())

        if store:
            F_list.append(F.copy())
            Sig_list.append(Sig_field.copy())

        i_drv, j_drv = driving_component
        F_drv = F_mac[i_drv, j_drv]
        eng_strain = F_drv - (1.0 if i_drv == j_drv else 0.0)
        Sig_drv = Sig_mac[i_drv, j_drv]

        if enable_console:
            print(f"step {s}/{n_steps}: "
                  f"F_{_drv_lbl}={F_drv:.5f}, "
                  f"eng_strain={eng_strain:.5f}, "
                  f"Sig_{_drv_lbl}={Sig_drv/1e9:.4f} GPa")

        if _log_f:
            _now     = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
            _elapsed = _time.time() - _t0
            _log_f.write(
                f"{_now:<20} {_elapsed:<12.3f} {s:<8} "
                f"{F_drv:<14.6f} {eng_strain:<20.6f} "
                f"{Sig_drv/1e9:<18.6f}\n"
            )

        if _glog_f:
            _glog_f.write(
                f"{s:<8} "
                f"{F_mac[0,0]:<12.6f} {F_mac[1,1]:<12.6f} {F_mac[2,2]:<12.6f} "
                f"{F_mac[0,1]:<12.6f} {F_mac[0,2]:<12.6f} {F_mac[1,2]:<12.6f} "
                f"{F_mac[0,0]-1:<14.6f} {F_mac[1,1]-1:<14.6f} {F_mac[2,2]-1:<14.6f} "
                f"{P_mac[0,0]/1e9:<14.6f} {P_mac[1,1]/1e9:<14.6f} {P_mac[2,2]/1e9:<14.6f} "
                f"{P_mac[0,1]/1e9:<14.6f} {P_mac[0,2]/1e9:<14.6f} {P_mac[1,2]/1e9:<14.6f} "
                f"{Sig_mac[0,0]/1e9:<16.6f} {Sig_mac[1,1]/1e9:<16.6f} {Sig_mac[2,2]/1e9:<16.6f} "
                f"{Sig_mac[0,1]/1e9:<16.6f} {Sig_mac[0,2]/1e9:<16.6f} {Sig_mac[1,2]/1e9:<16.6f}\n"
            )

        # Checkpoint export
        if checkpoint_interval is not None and checkpoint_interval not in ["none", "last"] and checkpoint_path is not None:
            save_chk, cp_name = False, None
            if checkpoint_interval == "current":
                save_chk, cp_name = True, f"{checkpoint_path}.h5"
            elif isinstance(checkpoint_interval, int) and s % checkpoint_interval == 0:
                save_chk, cp_name = True, f"{checkpoint_path}_{s:06d}.h5"
            if save_chk and cp_name:
                E_GL = 0.5 * (_dot22_3d(_trans2_3d(F), F) - I2)
                eps_vtk = np.einsum('ijxyz->xyzij', E_GL)
                sig_vtk = np.einsum('ijxyz->xyzij', Sig_field)
                save_checkpoint_3d(cp_name, s, E, nu, eps_vtk, sig_vtk, F_mac, Sig_mac, pixel)

        # VTK export
        if vtk_interval is not None and vtk_interval not in ["none", "last"] and vtk_path is not None:
            save_vtk, vt_name = False, None
            if vtk_interval == "current":
                vt_name = f"{vtk_path}.vtu"
                save_vtk = True
            elif isinstance(vtk_interval, int) and s % vtk_interval == 0:
                vt_name = f"{vtk_path}_{s:06d}.vtu"
                save_vtk = True
            if save_vtk and vt_name:
                E_GL = 0.5 * (_dot22_3d(_trans2_3d(F), F) - I2)
                eps_vtk = np.einsum('ijxyz->xyzij', E_GL)
                sig_vtk = np.einsum('ijxyz->xyzij', Sig_field)
                export_to_vtk(vt_name, eps_vtk, sig_vtk, E, nu, pixel, match_matplotlib_orientation=True)

    # Final step exports
    if checkpoint_path is not None:
        if checkpoint_interval == "last":
            E_GL = 0.5 * (_dot22_3d(_trans2_3d(F), F) - I2)
            eps_vtk = np.einsum('ijxyz->xyzij', E_GL)
            sig_vtk = np.einsum('ijxyz->xyzij', Sig_field)
            save_checkpoint_3d(f"{checkpoint_path}_final.h5", n_steps, E, nu, eps_vtk, sig_vtk, F_macro_list[-1], Sig_macro_list[-1], pixel)
        elif checkpoint_interval not in [None, "none", "last"] and isinstance(checkpoint_interval, int) and n_steps % checkpoint_interval != 0:
            E_GL = 0.5 * (_dot22_3d(_trans2_3d(F), F) - I2)
            eps_vtk = np.einsum('ijxyz->xyzij', E_GL)
            sig_vtk = np.einsum('ijxyz->xyzij', Sig_field)
            save_checkpoint_3d(f"{checkpoint_path}_final.h5", n_steps, E, nu, eps_vtk, sig_vtk, F_macro_list[-1], Sig_macro_list[-1], pixel)

    if vtk_path is not None:
        if vtk_interval == "last":
            E_GL = 0.5 * (_dot22_3d(_trans2_3d(F), F) - I2)
            eps_vtk = np.einsum('ijxyz->xyzij', E_GL)
            sig_vtk = np.einsum('ijxyz->xyzij', Sig_field)
            export_to_vtk(f"{vtk_path}_final.vtu", eps_vtk, sig_vtk, E, nu, pixel, match_matplotlib_orientation=True)
        elif vtk_interval not in [None, "none", "last"] and isinstance(vtk_interval, int) and n_steps % vtk_interval != 0:
            E_GL = 0.5 * (_dot22_3d(_trans2_3d(F), F) - I2)
            eps_vtk = np.einsum('ijxyz->xyzij', E_GL)
            sig_vtk = np.einsum('ijxyz->xyzij', Sig_field)
            export_to_vtk(f"{vtk_path}_final.vtu", eps_vtk, sig_vtk, E, nu, pixel, match_matplotlib_orientation=True)

    if _log_f:  _log_f.close()
    if _glog_f: _glog_f.close()

    return (
        np.array(F_macro_list),
        np.array(Sig_macro_list),
        np.array(P_macro_list),
        F_list,
        Sig_list,
    )

