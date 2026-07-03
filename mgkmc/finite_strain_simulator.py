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
import pyfftw
pyfftw.interfaces.cache.enable()
import pyfftw.interfaces.numpy_fft as pyfft
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

    # Standard (unshifted) frequencies
    xi_x = np.fft.fftfreq(nx, d=Lx/nx)
    xi_y = np.fft.fftfreq(ny, d=Ly/ny)

    # Broadcast to 2D grid
    Xi_x, Xi_y = np.meshgrid(xi_x, xi_y, indexing='ij')   # (nx, ny)
    xi  = np.stack([Xi_x, Xi_y], axis=0)                    # (2, nx, ny)
    xi2 = Xi_x**2 + Xi_y**2                                 # (nx, ny)

    # Nyquist mask for even axes: zero out where frequency is -N/2
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
# FFT helpers — use unshifted pyfftw
# ---------------------------------------------------------------------------

def _fft2_tensor(A):
    """
    Forward FFT of every component of a [ndim, ndim, nx, ny] field.
    No shifts are needed because Ghat4 is constructed in unshifted frequency order.
    """
    return pyfft.fftn(A, axes=(-2, -1), threads=pyfftw.config.NUM_THREADS)


def _ifft2_tensor(A_hat):
    """
    Inverse FFT of every component of a [ndim, ndim, nx, ny] field.
    Returns real part (imaginary part is numerical noise).
    """
    return pyfft.ifftn(A_hat, axes=(-2, -1), threads=pyfftw.config.NUM_THREADS).real


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

def _invert_Fp_2d(Fp):
    """Pointwise matrix inversion of a 2D deformation gradient field of shape (2, 2, nx, ny)."""
    det = Fp[0, 0] * Fp[1, 1] - Fp[0, 1] * Fp[1, 0]
    det_safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
    
    Fp_inv = np.zeros_like(Fp)
    Fp_inv[0, 0] =  Fp[1, 1] / det_safe
    Fp_inv[1, 1] =  Fp[0, 0] / det_safe
    Fp_inv[0, 1] = -Fp[0, 1] / det_safe
    Fp_inv[1, 0] = -Fp[1, 0] / det_safe
    return Fp_inv


def _invert_Fp_3d(Fp):
    """Pointwise matrix inversion of a 3D deformation gradient field of shape (3, 3, nx, ny, nz)."""
    det = (
        Fp[0, 0] * (Fp[1, 1] * Fp[2, 2] - Fp[1, 2] * Fp[2, 1]) -
        Fp[0, 1] * (Fp[1, 0] * Fp[2, 2] - Fp[1, 2] * Fp[2, 0]) +
        Fp[0, 2] * (Fp[1, 0] * Fp[2, 1] - Fp[1, 1] * Fp[2, 0])
    )
    det_safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
    
    Fp_inv = np.zeros_like(Fp)
    Fp_inv[0, 0] =  (Fp[1, 1] * Fp[2, 2] - Fp[1, 2] * Fp[2, 1]) / det_safe
    Fp_inv[0, 1] = -(Fp[0, 1] * Fp[2, 2] - Fp[0, 2] * Fp[2, 1]) / det_safe
    Fp_inv[0, 2] =  (Fp[0, 1] * Fp[1, 2] - Fp[0, 2] * Fp[1, 1]) / det_safe
    
    Fp_inv[1, 0] = -(Fp[1, 0] * Fp[2, 2] - Fp[1, 2] * Fp[2, 0]) / det_safe
    Fp_inv[1, 1] =  (Fp[0, 0] * Fp[2, 2] - Fp[0, 2] * Fp[2, 0]) / det_safe
    Fp_inv[1, 2] = -(Fp[0, 0] * Fp[1, 2] - Fp[0, 2] * Fp[1, 0]) / det_safe
    
    Fp_inv[2, 0] =  (Fp[1, 0] * Fp[2, 1] - Fp[1, 1] * Fp[2, 0]) / det_safe
    Fp_inv[2, 1] = -(Fp[0, 0] * Fp[2, 1] - Fp[0, 1] * Fp[2, 0]) / det_safe
    Fp_inv[2, 2] =  (Fp[0, 0] * Fp[1, 1] - Fp[0, 1] * Fp[1, 0]) / det_safe
    return Fp_inv


def constitutive_hyperelastic_2d(F, C4, I2, I4, I4rt, Fp=None, model_type="svk", plane_mode="plane_strain",
                                 A_m=0.0, B_m=0.0, C_m=0.0,
                                 v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
    """
    Compute 1st Piola-Kirchhoff stress P and consistent tangent K4 from F,
    supporting an optional plastic deformation gradient field Fp (eigenstrain).
    Supports 'svk', 'neo_hookean', and 'landau' models.
    """
    ndim = 2
    
    if Fp is not None:
        # Inlined for speed
        det = Fp[0, 0] * Fp[1, 1] - Fp[0, 1] * Fp[1, 0]
        det_safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
        Fp_inv = np.zeros_like(Fp)
        Fp_inv[0, 0] =  Fp[1, 1] / det_safe
        Fp_inv[1, 1] =  Fp[0, 0] / det_safe
        Fp_inv[0, 1] = -Fp[0, 1] / det_safe
        Fp_inv[1, 0] = -Fp[1, 0] / det_safe
        
        Fe = np.einsum('ijxy,jkxy->ikxy', F, Fp_inv, optimize=True)
    else:
        Fe = F
        
    if model_type == "neo_hookean":
        if plane_mode == "plane_stress":
            # Extract effective plane stress lam_2d and shear modulus mu from C4
            lam_2d = C4[0, 0, 1, 1]
            mu = C4[0, 1, 0, 1]
            
            # Calculate 3D lambda
            lam_3d = (2.0 * mu * lam_2d) / np.maximum(1e-14, 2.0 * mu - lam_2d)
            
            # Ce = Fe^T * Fe
            Ce = np.einsum('jixy,jkxy->ikxy', Fe, Fe, optimize=True)
            
            # det(Ce) = Je^2
            Je = Fe[0, 0] * Fe[1, 1] - Fe[0, 1] * Fe[1, 0]
            Je_safe = np.maximum(1e-14, Je)
            
            # Newton-Raphson to solve transcendental equation for out-of-plane stretch:
            # x = F33^2
            # mu * x + 0.5 * lam_3d * ln(x) = mu - lam_3d * ln(Je_safe)
            x = np.ones_like(Je_safe)
            target = mu - lam_3d * np.log(Je_safe)
            for _ in range(10):
                f = mu * x + 0.5 * lam_3d * np.log(x) - target
                df = mu + 0.5 * lam_3d / x
                dx = - f / df
                x = np.maximum(1e-10, x + dx)
                if np.max(np.abs(dx)) < 1e-12:
                    break
            F33 = np.sqrt(np.maximum(1e-12, x))
            
            # Inverse of Ce
            detCe = Ce[0, 0] * Ce[1, 1] - Ce[0, 1] * Ce[1, 0]
            detCe_safe = np.where(np.abs(detCe) < 1e-14, 1e-14, detCe)
            
            invC = np.zeros_like(Ce)
            invC[0, 0] =  Ce[1, 1] / detCe_safe
            invC[1, 1] =  Ce[0, 0] / detCe_safe
            invC[0, 1] = -Ce[0, 1] / detCe_safe
            invC[1, 0] = -Ce[1, 0] / detCe_safe
            
            coeff = -mu * x
            S = mu[np.newaxis, np.newaxis, :, :] * I2 + coeff[np.newaxis, np.newaxis, :, :] * invC
            
            coeff_tan = mu * x
            lam_star = (2.0 * lam_3d * coeff_tan) / np.maximum(1e-14, lam_3d + 2.0 * coeff_tan)
            
            term1_tan = lam_star[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * \
                        np.einsum('ijxy,klxy->ijklxy', invC, invC)
            term2_tan = coeff_tan[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * (
                        np.einsum('ikxy,jlxy->ijklxy', invC, invC) +
                        np.einsum('ilxy,jkxy->ijklxy', invC, invC)
            )
            C4_eff = term1_tan + term2_tan
        else:
            # plane_strain: F33 = 1.0, and lam from C4 is the 3D lambda
            lam = C4[0, 0, 1, 1]
            mu = C4[0, 1, 0, 1]
            F33 = np.ones_like(Fe[0, 0])
            
            Ce = np.einsum('jixy,jkxy->ikxy', Fe, Fe, optimize=True)
            
            # det(Ce) = Je^2
            Je = Fe[0, 0] * Fe[1, 1] - Fe[0, 1] * Fe[1, 0]
            Je_safe = np.maximum(1e-14, Je)
            logJe = np.log(Je_safe)
            
            detCe = Ce[0, 0] * Ce[1, 1] - Ce[0, 1] * Ce[1, 0]
            detCe_safe = np.where(np.abs(detCe) < 1e-14, 1e-14, detCe)
            
            invC = np.zeros_like(Ce)
            invC[0, 0] =  Ce[1, 1] / detCe_safe
            invC[1, 1] =  Ce[0, 0] / detCe_safe
            invC[0, 1] = -Ce[0, 1] / detCe_safe
            invC[1, 0] = -Ce[1, 0] / detCe_safe
            
            coeff = -mu + lam * logJe
            S = mu[np.newaxis, np.newaxis, :, :] * I2 + coeff[np.newaxis, np.newaxis, :, :] * invC
            
            coeff_tan = mu - lam * logJe
            term1_tan = lam[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * \
                        np.einsum('ijxy,klxy->ijklxy', invC, invC)
            term2_tan = coeff_tan[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * (
                        np.einsum('ikxy,jlxy->ijklxy', invC, invC) +
                        np.einsum('ilxy,jkxy->ijklxy', invC, invC)
            )
            C4_eff = term1_tan + term2_tan
    elif model_type == "landau":
        nx, ny = Fe.shape[2], Fe.shape[3]
        v1_arr = np.full((nx, ny), float(v1)) if isinstance(v1, (int, float, np.number)) else np.array(v1)
        v2_arr = np.full((nx, ny), float(v2)) if isinstance(v2, (int, float, np.number)) else np.array(v2)
        v3_arr = np.full((nx, ny), float(v3)) if isinstance(v3, (int, float, np.number)) else np.array(v3)
        g1_arr = np.full((nx, ny), float(g1)) if isinstance(g1, (int, float, np.number)) else np.array(g1)
        g2_arr = np.full((nx, ny), float(g2)) if isinstance(g2, (int, float, np.number)) else np.array(g2)
        g3_arr = np.full((nx, ny), float(g3)) if isinstance(g3, (int, float, np.number)) else np.array(g3)
        g4_arr = np.full((nx, ny), float(g4)) if isinstance(g4, (int, float, np.number)) else np.array(g4)

        # Autoconvert constants to Pa if supplied in GPa (values < 1e6 except 0.0)
        for arr in [v1_arr, v2_arr, v3_arr, g1_arr, g2_arr, g3_arr, g4_arr]:
            mask = (np.abs(arr) < 1e6) & (arr != 0.0)
            arr[mask] *= 1e9

        if plane_mode == "plane_stress":
            lam_2d = C4[0, 0, 1, 1]
            mu = C4[0, 1, 0, 1]
            lam = (2.0 * mu * lam_2d) / np.maximum(1e-14, 2.0 * mu - lam_2d)
        else:
            lam = C4[0, 0, 1, 1]
            mu = C4[0, 1, 0, 1]

        Ce = np.einsum('jixy,jkxy->ikxy', Fe, Fe, optimize=True)
        E_GL_2d = 0.5 * (Ce - I2)

        # Build 3D identity tensors broadcast to (nx, ny)
        ones_grid = np.ones((nx, ny))
        i3 = np.eye(3)
        I2_3d = np.einsum('ij,xy->ijxy', i3, ones_grid)
        II_3d = np.einsum('ij,kl,xy->ijklxy', i3, i3, ones_grid)

        # 4th-order symmetric identity in 3D
        I4s_3d = np.zeros((3, 3, 3, 3, nx, ny))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        I4s_3d[i, j, k, l] = 0.5 * (i3[i, k] * i3[j, l] + i3[i, l] * i3[j, k])

        if plane_mode == "plane_stress":
            # Solve for E33 locally and element-wise such that S33(E33) = 0
            # S33 = A + B * E33 + C * E33**2 = 0
            trE_2d = E_GL_2d[0, 0] + E_GL_2d[1, 1]
            trE2_2d = E_GL_2d[0, 0]**2 + E_GL_2d[1, 1]**2 + 2.0 * E_GL_2d[0, 1] * E_GL_2d[1, 0]
            trE3_2d = np.einsum('ijxy,jkxy,kixy->xy', E_GL_2d, E_GL_2d, E_GL_2d)

            # Initial guess: linear plane stress value
            E33 = - (lam / (lam + 2.0 * mu)) * trE_2d

            for iteration in range(20):
                I1 = trE_2d + E33
                I2_inv = trE2_2d + E33**2
                I3_inv = trE3_2d + E33**3

                A_coef = lam * I1 + 0.5 * v1_arr * (I1**2) + v2_arr * I2_inv + (1.0/6.0) * g1_arr * (I1**3) + g2_arr * I1 * I2_inv + (4.0/3.0) * g3_arr * I3_inv
                B_coef = 2.0 * (mu + v2_arr * I1 + 0.5 * g2_arr * (I1**2) + g4_arr * I2_inv)
                C_coef = 4.0 * (v3_arr + g3_arr * I1)

                S33 = A_coef + B_coef * E33 + C_coef * (E33**2)

                # Derivatives for C3333
                dAdE33 = lam + v1_arr * I1 + 0.5 * g1_arr * (I1**2) + g2_arr * I2_inv + 2.0 * (v2_arr + g2_arr * I1) * E33 + 4.0 * g3_arr * (E33**2)
                dBdE33 = 2.0 * (v2_arr + g2_arr * I1) + 4.0 * g4_arr * E33
                dCdE33 = 4.0 * g3_arr

                C3333 = dAdE33 + dBdE33 * E33 + dCdE33 * (E33**2) + B_coef + 2.0 * C_coef * E33
                C3333_safe = np.where(np.abs(C3333) < 1e-12, 1e-12, C3333)

                dE33 = - S33 / C3333_safe
                E33 = E33 + dE33
                if np.max(np.abs(dE33)) < 1e-12:
                    break

            F33 = np.sqrt(np.maximum(1e-12, 1.0 + 2.0 * E33))

            E_GL = np.zeros((3, 3, nx, ny))
            E_GL[0:2, 0:2] = E_GL_2d
            E_GL[2, 2] = E33
        else:
            E_GL = np.zeros((3, 3, nx, ny))
            E_GL[0:2, 0:2] = E_GL_2d
            F33 = np.ones((nx, ny))

        trE = E_GL[0, 0] + E_GL[1, 1] + E_GL[2, 2]
        trE2 = np.einsum('ijxy,jixy->xy', E_GL, E_GL)
        trE3 = np.einsum('ijxy,jkxy,kixy->xy', E_GL, E_GL, E_GL)
        E2 = np.einsum('ijxy,jkxy->ikxy', E_GL, E_GL)

        A_coef = lam * trE + 0.5 * v1_arr * (trE**2) + v2_arr * trE2 + (1.0/6.0) * g1_arr * (trE**3) + g2_arr * trE * trE2 + (4.0/3.0) * g3_arr * trE3
        B_coef = 2.0 * (mu + v2_arr * trE + 0.5 * g2_arr * (trE**2) + g4_arr * trE2)
        C_coef = 4.0 * (v3_arr + g3_arr * trE)

        S_3d = A_coef[np.newaxis, np.newaxis, :, :] * I2_3d \
             + B_coef[np.newaxis, np.newaxis, :, :] * E_GL \
             + C_coef[np.newaxis, np.newaxis, :, :] * E2

        S = S_3d[0:2, 0:2]

        # Derivatives for C4_3d
        dAdE = (lam + v1_arr * trE + 0.5 * g1_arr * (trE**2) + g2_arr * trE2)[np.newaxis, np.newaxis, :, :] * I2_3d \
             + 2.0 * (v2_arr + g2_arr * trE)[np.newaxis, np.newaxis, :, :] * E_GL \
             + 4.0 * g3_arr[np.newaxis, np.newaxis, :, :] * E2

        dBdE = 2.0 * (v2_arr + g2_arr * trE)[np.newaxis, np.newaxis, :, :] * I2_3d \
             + 4.0 * g4_arr[np.newaxis, np.newaxis, :, :] * E_GL

        dCdE = 4.0 * g3_arr[np.newaxis, np.newaxis, :, :] * I2_3d

        # 4th-order derivative term building
        term1 = np.einsum('ijxy,klxy->ijklxy', I2_3d, dAdE)
        term2 = np.einsum('ijxy,klxy->ijklxy', E_GL, dBdE)
        term3 = np.einsum('ijxy,klxy->ijklxy', E2, dCdE)
        term4 = B_coef[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * I4s_3d

        term5_part1 = np.einsum('ikxy,jlxy->ijklxy', I2_3d, E_GL) + np.einsum('ikxy,jlxy->ijklxy', E_GL, I2_3d)
        term5_part2 = np.einsum('ilxy,jkxy->ijklxy', I2_3d, E_GL) + np.einsum('ilxy,jkxy->ijklxy', E_GL, I2_3d)
        K_E = 0.5 * (term5_part1 + term5_part2)
        term5 = C_coef[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * K_E

        C4_3d = term1 + term2 + term3 + term4 + term5

        if plane_mode == "plane_stress":
            C2222 = np.maximum(1e-14, C4_3d[2, 2, 2, 2])
            C22 = C4_3d[:, :, 2, 2] # (3, 3, nx, ny)
            C4_eff = C4_3d[0:2, 0:2, 0:2, 0:2] - \
                     C22[0:2, 0:2, np.newaxis, np.newaxis, :, :] * \
                     C22[np.newaxis, np.newaxis, 0:2, 0:2, :, :] / C2222[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
        else:
            C4_eff = C4_3d[0:2, 0:2, 0:2, 0:2]
    elif model_type == "murnaghan":
        nx, ny = Fe.shape[2], Fe.shape[3]
        A_arr = np.full((nx, ny), float(A_m)) if isinstance(A_m, (int, float, np.number)) else np.array(A_m)
        B_arr = np.full((nx, ny), float(B_m)) if isinstance(B_m, (int, float, np.number)) else np.array(B_m)
        C_arr = np.full((nx, ny), float(C_m)) if isinstance(C_m, (int, float, np.number)) else np.array(C_m)

        if plane_mode == "plane_stress":
            lam_2d = C4[0, 0, 1, 1]
            mu = C4[0, 1, 0, 1]
            lam = (2.0 * mu * lam_2d) / np.maximum(1e-14, 2.0 * mu - lam_2d)
        else:
            lam = C4[0, 0, 1, 1]
            mu = C4[0, 1, 0, 1]

        Ce = np.einsum('jixy,jkxy->ikxy', Fe, Fe, optimize=True)
        E_GL_2d = 0.5 * (Ce - I2)

        if plane_mode == "plane_stress":
            trE_2d = E_GL_2d[0, 0] + E_GL_2d[1, 1]
            trE2_2d = E_GL_2d[0, 0]**2 + E_GL_2d[1, 1]**2 + 2.0 * E_GL_2d[0, 1] * E_GL_2d[1, 0]
            
            a = A_arr + 3.0 * B_arr + C_arr
            b = lam + 2.0 * mu + 2.0 * (A_arr + B_arr) * trE_2d
            c = lam * trE_2d + A_arr * (trE_2d**2) + B_arr * trE2_2d
            
            E33 = np.zeros((nx, ny))
            small_a = np.abs(a) < 1e-12
            
            if np.any(small_a):
                E33[small_a] = -c[small_a] / np.maximum(1e-14, b[small_a])
            if np.any(~small_a):
                disc = b[~small_a]**2 - 4.0 * a[~small_a] * c[~small_a]
                disc = np.maximum(0.0, disc)
                sgn_b = np.sign(b[~small_a])
                sgn_b[sgn_b == 0] = 1.0
                E33[~small_a] = (-b[~small_a] + sgn_b * np.sqrt(disc)) / (2.0 * a[~small_a])
                
            F33 = np.sqrt(np.maximum(1e-12, 1.0 + 2.0 * E33))
            
            E_GL = np.zeros((3, 3, nx, ny))
            E_GL[0:2, 0:2] = E_GL_2d
            E_GL[2, 2] = E33
        else:
            E_GL = np.zeros((3, 3, nx, ny))
            E_GL[0:2, 0:2] = E_GL_2d
            F33 = np.ones((nx, ny))
            
        trE = E_GL[0, 0] + E_GL[1, 1] + E_GL[2, 2]
        trE2 = np.einsum('ijxy,jixy->xy', E_GL, E_GL)
        E2 = np.einsum('ijxy,jkxy->ikxy', E_GL, E_GL)
        
        I3 = np.einsum('ij,xy->ijxy', np.eye(3), np.ones((nx, ny)))
        S_3d = (lam * trE + A_arr * (trE**2) + B_arr * trE2)[np.newaxis, np.newaxis, :, :] * I3 \
             + 2.0 * (mu + B_arr * trE)[np.newaxis, np.newaxis, :, :] * E_GL \
             + C_arr[np.newaxis, np.newaxis, :, :] * E2
             
        S = S_3d[0:2, 0:2]
        
        # Build 3D identity tensors broadcast to (nx, ny)
        ones_grid = np.ones((nx, ny))
        i3 = np.eye(3)
        I2_3d = np.einsum('ij,xy->ijxy', i3, ones_grid)
        II_3d = np.einsum('ij,kl,xy->ijklxy', i3, i3, ones_grid)
        
        # 4th-order symmetric identity in 3D
        I4s_3d = np.zeros((3, 3, 3, 3, nx, ny))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        I4s_3d[i, j, k, l] = 0.5 * (i3[i, k] * i3[j, l] + i3[i, l] * i3[j, k])
                        
        term1 = (lam + 2.0 * A_arr * trE)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * II_3d
        
        I_dyad_E = np.einsum('ijxy,klxy->ijklxy', I2_3d, E_GL)
        E_dyad_I = np.einsum('ijxy,klxy->ijklxy', E_GL, I2_3d)
        term2 = (2.0 * B_arr)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * (I_dyad_E + E_dyad_I)
        
        term3 = (2.0 * (mu + B_arr * trE))[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * I4s_3d
        
        term4_part1 = np.einsum('ikxy,jlxy->ijklxy', I2_3d, E_GL) + np.einsum('ikxy,jlxy->ijklxy', E_GL, I2_3d)
        term4_part2 = np.einsum('ilxy,jkxy->ijklxy', I2_3d, E_GL) + np.einsum('ilxy,jkxy->ijklxy', E_GL, I2_3d)
        K_E = 0.5 * (term4_part1 + term4_part2)
        term4 = C_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :] * K_E
        
        C4_3d = term1 + term2 + term3 + term4
        
        if plane_mode == "plane_stress":
            C2222 = np.maximum(1e-14, C4_3d[2, 2, 2, 2])
            C22 = C4_3d[:, :, 2, 2] # (3, 3, nx, ny)
            C4_eff = C4_3d[0:2, 0:2, 0:2, 0:2] - \
                     C22[0:2, 0:2, np.newaxis, np.newaxis, :, :] * \
                     C22[np.newaxis, np.newaxis, 0:2, 0:2, :, :] / C2222[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]
        else:
            C4_eff = C4_3d[0:2, 0:2, 0:2, 0:2]
    else:
        # Default: St. Venant-Kirchhoff (SVK)
        E_GL = 0.5 * (np.einsum('jixy,jkxy->ikxy', Fe, Fe, optimize=True) - I2)
        S = np.einsum('ijklxy,lkxy->ijxy', C4, E_GL, optimize=True)
        C4_eff = C4
        if plane_mode == "plane_stress":
            lam_2d = C4[0, 0, 1, 1]
            mu = C4[0, 1, 0, 1]
            tr_E = E_GL[0, 0] + E_GL[1, 1]
            E_GL_33 = - (lam_2d / (2.0 * mu)) * tr_E
            F33 = np.sqrt(np.maximum(1e-12, 1.0 + 2.0 * E_GL_33))
        else:
            F33 = np.ones_like(Fe[0, 0])

    if Fp is not None:
        P = np.einsum('ikxy,klxy,jlxy->ijxy', Fe, S, Fp_inv, optimize=True)
        S_ref = np.einsum('mkxy,klxy,jlxy->mjxy', Fp_inv, S, Fp_inv, optimize=True)
        
        A = np.einsum('klmnxy,jlxy->kjmnxy', C4_eff, Fp_inv, optimize=True)
        B = np.einsum('kjmnxy,bmxy->kjbnxy', A, Fp_inv, optimize=True)
        C = np.einsum('kjbnxy,ikxy->ijbnxy', B, Fe, optimize=True)
        term2 = np.einsum('ijbnxy,anxy->ijabxy', C, Fe, optimize=True)
        
        term1 = np.einsum('bjxy,ia->ijabxy', S_ref, np.eye(ndim), optimize=True)
        K4 = term1 + term2
    else:
        P = np.einsum('ijxy,jkxy->ikxy', Fe, S, optimize=True)
        
        term1 = np.einsum('ijxy,jkmnxy->ikmnxy', S, I4, optimize=True)
        
        FC4 = np.einsum('ijxy,jkmnxy->ikmnxy', Fe, C4_eff, optimize=True)
        Ft = np.einsum('ijxy->jixy', Fe)
        FC4Ft = np.einsum('ijklxy,lmxy->ijkmxy', FC4, Ft, optimize=True)
        term2_part1 = np.einsum('ijklxy,lkmnxy->ijmnxy', I4rt, FC4Ft, optimize=True)
        term2 = np.einsum('ijklxy,lkmnxy->ijmnxy', term2_part1, I4rt, optimize=True)
        
        K4 = term1 + term2

    return P, K4, F33


# ---------------------------------------------------------------------------
# Cauchy stress from P and F
# ---------------------------------------------------------------------------

def cauchy_from_P(P, F, F33=None):
    """
    Convert 1st Piola-Kirchhoff stress P to Cauchy stress σ.

        σ = (1/J) · P · Fᵀ       J = det(F) * F33

    Operates pointwise on [2, 2, nx, ny] fields.
    Returns σ as [2, 2, nx, ny].
    """
    ndim, _, nx, ny = P.shape
    # det(F) per pixel
    J = (F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])  # [nx, ny]
    if F33 is not None:
        J = J * F33
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

    # Standard (unshifted) frequencies
    xi_x = np.fft.fftfreq(nx, d=Lx/nx)
    xi_y = np.fft.fftfreq(ny, d=Ly/ny)
    xi_z = np.fft.fftfreq(nz, d=Lz/nz)

    Xi_x, Xi_y, Xi_z = np.meshgrid(xi_x, xi_y, xi_z, indexing='ij')
    xi  = np.stack([Xi_x, Xi_y, Xi_z], axis=0)
    xi2 = Xi_x**2 + Xi_y**2 + Xi_z**2

    # Nyquist mask for even axes: zero out where frequency is -N/2
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
    return pyfft.fftn(A, axes=(-3, -2, -1), threads=pyfftw.config.NUM_THREADS)


def _ifft3_tensor(A_hat):
    """Inverse FFT of every component of a [ndim, ndim, nx, ny, nz] field."""
    return pyfft.ifftn(A_hat, axes=(-3, -2, -1), threads=pyfftw.config.NUM_THREADS).real


def _project_3d(A2, Ghat4):
    """Apply the de Geus projection in 3D: Ĝ : A2"""
    A2_hat  = _fft3_tensor(A2)
    res_hat = np.einsum('ijklxyz,lkxyz->ijxyz', Ghat4, A2_hat)
    return _ifft3_tensor(res_hat)


# ---------------------------------------------------------------------------
# Displacement-Based FFT (DBFFT) Helpers & Solvers
# ---------------------------------------------------------------------------

def _get_frequencies_2d(nx, ny, Lx, Ly):
    """
    Get 2D angular frequency vectors scaled by 2*pi, shape (2, nx, ny).
    Zero out the Nyquist frequency for even grid sizes.
    """
    xi_x = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    xi_y = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    Xi_x, Xi_y = np.meshgrid(xi_x, xi_y, indexing='ij')
    
    if nx % 2 == 0:
        Xi_x[Xi_x == -np.pi * nx / Lx] = 0.0
    if ny % 2 == 0:
        Xi_y[Xi_y == -np.pi * ny / Ly] = 0.0
        
    return np.stack([Xi_x, Xi_y], axis=0)


def _get_frequencies_3d(nx, ny, nz, Lx, Ly, Lz):
    """
    Get 3D angular frequency vectors scaled by 2*pi, shape (3, nx, ny, nz).
    Zero out the Nyquist frequency for even grid sizes.
    """
    xi_x = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx/nx)
    xi_y = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly/ny)
    xi_z = 2.0 * np.pi * np.fft.fftfreq(nz, d=Lz/nz)
    Xi_x, Xi_y, Xi_z = np.meshgrid(xi_x, xi_y, xi_z, indexing='ij')
    
    if nx % 2 == 0:
        Xi_x[Xi_x == -np.pi * nx / Lx] = 0.0
    if ny % 2 == 0:
        Xi_y[Xi_y == -np.pi * ny / Ly] = 0.0
    if nz % 2 == 0:
        Xi_z[Xi_z == -np.pi * nz / Lz] = 0.0
        
    return np.stack([Xi_x, Xi_y, Xi_z], axis=0)


def _invert_matrix_field_2d(A):
    """
    Invert a 2x2 matrix field pointwise in Fourier space.
    A shape: (2, 2, nx, ny)
    """
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    det_safe = np.where(np.abs(det) < 1e-14, 1.0, det)
    
    M_inv = np.zeros_like(A)
    M_inv[0, 0] =  A[1, 1] / det_safe
    M_inv[1, 1] =  A[0, 0] / det_safe
    M_inv[0, 1] = -A[0, 1] / det_safe
    M_inv[1, 0] = -A[1, 0] / det_safe
    
    M_inv[:, :, 0, 0] = 0.0
    return M_inv


def _invert_matrix_field_3d(A):
    """
    Invert a 3x3 matrix field pointwise in Fourier space.
    A shape: (3, 3, nx, ny, nz)
    """
    det = (
        A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) -
        A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) +
        A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0])
    )
    det_safe = np.where(np.abs(det) < 1e-14, 1.0, det)
    
    M_inv = np.zeros_like(A)
    M_inv[0, 0] =  (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) / det_safe
    M_inv[0, 1] = -(A[0, 1] * A[2, 2] - A[0, 2] * A[2, 1]) / det_safe
    M_inv[0, 2] =  (A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]) / det_safe
    
    M_inv[1, 0] = -(A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) / det_safe
    M_inv[1, 1] =  (A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]) / det_safe
    M_inv[1, 2] = -(A[0, 0] * A[1, 2] - A[0, 2] * A[1, 0]) / det_safe
    
    M_inv[2, 0] =  (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]) / det_safe
    M_inv[2, 1] = -(A[0, 0] * A[2, 1] - A[0, 1] * A[2, 0]) / det_safe
    M_inv[2, 2] =  (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / det_safe
    
    M_inv[:, :, 0, 0, 0] = 0.0
    return M_inv


def _reconstruct_u_from_F_2d(F, F_bar, Xi):
    nx, ny = F.shape[2], F.shape[3]
    F_bar_grid = np.einsum('ij,xy->ijxy', F_bar, np.ones((nx, ny)))
    grad_u = F - F_bar_grid
    grad_u_hat = _fft2_tensor(grad_u)
    
    xi2 = np.sum(Xi**2, axis=0)
    xi2_safe = np.where(xi2 < 1e-14, 1.0, xi2)
    
    u_hat = -1j * np.einsum('jxy,ijxy->ixy', Xi, grad_u_hat) / xi2_safe
    u_hat[:, 0, 0] = 0.0
    
    return _ifft2_tensor(u_hat)


def _reconstruct_u_from_F_3d(F, F_bar, Xi):
    nx, ny, nz = F.shape[2], F.shape[3], F.shape[4]
    F_bar_grid = np.einsum('ij,xyz->ijxyz', F_bar, np.ones((nx, ny, nz)))
    grad_u = F - F_bar_grid
    grad_u_hat = _fft3_tensor(grad_u)
    
    xi2 = np.sum(Xi**2, axis=0)
    xi2_safe = np.where(xi2 < 1e-14, 1.0, xi2)
    
    u_hat = -1j * np.einsum('jxyz,ijxyz->ixyz', Xi, grad_u_hat) / xi2_safe
    u_hat[:, 0, 0, 0] = 0.0
    
    return _ifft3_tensor(u_hat)


def get_grad_u_2d(u, Xi):
    u_hat = _fft2_tensor(u)
    grad_u_hat = 1j * Xi[np.newaxis, :, :, :] * u_hat[:, np.newaxis, :, :]
    return _ifft2_tensor(grad_u_hat)


def get_grad_u_3d(u, Xi):
    u_hat = _fft3_tensor(u)
    grad_u_hat = 1j * Xi[np.newaxis, :, :, :, :] * u_hat[:, np.newaxis, :, :, :]
    return _ifft3_tensor(grad_u_hat)


def solve_dbfft_linear_system_2d(Xi, K4, M_inv, b_hat, tol_CG=1e-6, max_iter=150):
    ndim, nx, ny = b_hat.shape
    
    def A_op_func(u_flat):
        u_hat = u_flat.reshape(ndim, nx, ny)
        grad_u_hat = 1j * Xi[np.newaxis, :, :, :] * u_hat[:, np.newaxis, :, :]
        grad_u = _ifft2_tensor(grad_u_hat)
        dP = _ddot42(K4, grad_u)
        dP_hat = _fft2_tensor(dP)
        div_P_hat = 1j * np.einsum('jxy,ijxy->ixy', Xi, dP_hat)
        div_P_hat[:, 0, 0] = 0.0
        return div_P_hat.reshape(-1)

    def M_op_func(r_flat):
        r_hat = r_flat.reshape(ndim, nx, ny)
        z_hat = np.einsum('ijxy,jxy->ixy', M_inv, r_hat)
        return z_hat.reshape(-1)

    A_op = sp.LinearOperator(shape=(b_hat.size, b_hat.size), matvec=A_op_func, dtype='complex128')
    M_op = sp.LinearOperator(shape=(b_hat.size, b_hat.size), matvec=M_op_func, dtype='complex128')
    
    b_flat = b_hat.reshape(-1)
    try:
        sol_flat, info = sp.bicgstab(A_op, b_flat, M=M_op, rtol=tol_CG, maxiter=max_iter)
    except TypeError:
        sol_flat, info = sp.bicgstab(A_op, b_flat, M=M_op, tol=tol_CG, maxiter=max_iter)
        
    return sol_flat.reshape(ndim, nx, ny), info


def solve_dbfft_linear_system_3d(Xi, K4, M_inv, b_hat, tol_CG=1e-6, max_iter=150):
    ndim, nx, ny, nz = b_hat.shape
    
    def A_op_func(u_flat):
        u_hat = u_flat.reshape(ndim, nx, ny, nz)
        grad_u_hat = 1j * Xi[np.newaxis, :, :, :, :] * u_hat[:, np.newaxis, :, :, :]
        grad_u = _ifft3_tensor(grad_u_hat)
        dP = _ddot42_3d(K4, grad_u)
        dP_hat = _fft3_tensor(dP)
        div_P_hat = 1j * np.einsum('jxyz,ijxyz->ixyz', Xi, dP_hat)
        div_P_hat[:, 0, 0, 0] = 0.0
        return div_P_hat.reshape(-1)

    def M_op_func(r_flat):
        r_hat = r_flat.reshape(ndim, nx, ny, nz)
        z_hat = np.einsum('ijxyz,jxyz->ixyz', M_inv, r_hat)
        return z_hat.reshape(-1)

    A_op = sp.LinearOperator(shape=(b_hat.size, b_hat.size), matvec=A_op_func, dtype='complex128')
    M_op = sp.LinearOperator(shape=(b_hat.size, b_hat.size), matvec=M_op_func, dtype='complex128')
    
    b_flat = b_hat.reshape(-1)
    try:
        sol_flat, info = sp.bicgstab(A_op, b_flat, M=M_op, rtol=tol_CG, maxiter=max_iter)
    except TypeError:
        sol_flat, info = sp.bicgstab(A_op, b_flat, M=M_op, tol=tol_CG, maxiter=max_iter)
        
    return sol_flat.reshape(ndim, nx, ny, nz), info


def _dbfft_step_2d(F, F_bar, Xi, C4, I2, I4, I4rt, Fp=None,
                   tol=1e-5, tol_CG=1e-6, max_iter=20,
                   model_type="svk", plane_mode="plane_strain",
                   A_m=0.0, B_m=0.0, C_m=0.0,
                   v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
    nx, ny = F.shape[2], F.shape[3]
    F_bar_grid = np.einsum('ij,xy->ijxy', F_bar, np.ones((nx, ny)))
    u = _reconstruct_u_from_F_2d(F, F_bar, Xi)
    
    if Fp is not None:
        det = Fp[0,0]*Fp[1,1] - Fp[0,1]*Fp[1,0]
        det_s = np.where(np.abs(det) < 1e-14, 1e-14, det)
        Fp_inv = np.zeros_like(Fp)
        Fp_inv[0,0] =  Fp[1,1] / det_s
        Fp_inv[1,1] =  Fp[0,0] / det_s
        Fp_inv[0,1] = -Fp[0,1] / det_s
        Fp_inv[1,0] = -Fp[1,0] / det_s
    else:
        Fp_inv = None
        
    for i_NW in range(max_iter):
        grad_u = get_grad_u_2d(u, Xi)
        F_curr = F_bar_grid + grad_u
        
        P, K4, F33 = constitutive_hyperelastic_2d(
            F_curr, C4, I2, I4, I4rt, Fp=Fp,
            model_type=model_type, plane_mode=plane_mode,
            A_m=A_m, B_m=B_m, C_m=C_m,
            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
            
        P_hat = _fft2_tensor(P)
        b_hat = -1j * np.einsum('jxy,ijxy->ixy', Xi, P_hat)
        
        res_norm = np.linalg.norm(b_hat)
        P_norm = np.linalg.norm(P)
        rel_res = res_norm / (P_norm + 1e-20)
        
        if rel_res < tol and i_NW > 0:
            return F_curr, P, K4, F33, i_NW + 1
            
        K_avg = K4.mean(axis=(-2, -1))
        A_mat = np.einsum('jxy,ijlk,lxy->ikxy', Xi, K_avg, Xi)
        M_inv = _invert_matrix_field_2d(A_mat)
        
        du_hat, _ = solve_dbfft_linear_system_2d(Xi, K4, M_inv, b_hat, tol_CG=tol_CG)
        if np.any(np.isnan(du_hat)) or np.any(np.isinf(du_hat)):
            raise ValueError("Linear solver returned NaN or Inf (BiCGSTAB breakdown).")
        du = _ifft2_tensor(du_hat)
        
        alpha = 1.0
        converged_constitutive = False
        for _ in range(16):
            u_trial = u + alpha * du
            grad_u_trial = get_grad_u_2d(u_trial, Xi)
            F_trial = F_bar_grid + grad_u_trial
            if Fp_inv is not None:
                Fe_trial = np.einsum('ijxy,jkxy->ikxy', F_trial, Fp_inv, optimize=True)
            else:
                Fe_trial = F_trial
            Je = Fe_trial[0,0]*Fe_trial[1,1] - Fe_trial[0,1]*Fe_trial[1,0]
            if np.any(Je <= 1e-4) or np.any(np.isnan(Je)):
                alpha *= 0.5
                continue
            try:
                P_t, K4_t, F33_t = constitutive_hyperelastic_2d(
                    F_trial, C4, I2, I4, I4rt, Fp=Fp,
                    model_type=model_type, plane_mode=plane_mode,
                    A_m=A_m, B_m=B_m, C_m=C_m)
                if np.any(np.isnan(P_t)) or np.any(np.isnan(K4_t)):
                    alpha *= 0.5
                    continue
                u = u_trial
                converged_constitutive = True
                break
            except Exception:
                alpha *= 0.5
        if not converged_constitutive:
            raise ValueError("Constitutive solver failed to converge in line search (potential negative Jacobian/NaN).")
            
    # Newton loop finished without returning -> did not converge
    raise ValueError(f"DBFFT solver did not converge within {max_iter} iterations. Final relative residual: {rel_res:.2e}")


def _dbfft_step_3d(F, F_bar, Xi, C4, I2, I4, I4rt, Fp=None,
                   tol=1e-5, tol_CG=1e-6, max_iter=20,
                   model_type="svk", A_m=0.0, B_m=0.0, C_m=0.0,
                   v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
    nx, ny, nz = F.shape[2], F.shape[3], F.shape[4]
    F_bar_grid = np.einsum('ij,xyz->ijxyz', F_bar, np.ones((nx, ny, nz)))
    u = _reconstruct_u_from_F_3d(F, F_bar, Xi)
    
    if Fp is not None:
        Fp_inv = _invert_Fp_3d(Fp)
    else:
        Fp_inv = None
        
    for i_NW in range(max_iter):
        grad_u = get_grad_u_3d(u, Xi)
        F_curr = F_bar_grid + grad_u
        
        P, K4 = constitutive_hyperelastic_3d(
            F_curr, C4, I2, I4, I4rt, Fp=Fp,
            model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
            
        P_hat = _fft3_tensor(P)
        b_hat = -1j * np.einsum('jxyz,ijxyz->ixyz', Xi, P_hat)
        
        res_norm = np.linalg.norm(b_hat)
        P_norm = np.linalg.norm(P)
        rel_res = res_norm / (P_norm + 1e-20)
        
        if rel_res < tol and i_NW > 0:
            return F_curr, P, K4, i_NW + 1
            
        K_avg = K4.mean(axis=(-3, -2, -1))
        A_mat = np.einsum('jxyz,ijlk,lxyz->ikxyz', Xi, K_avg, Xi)
        M_inv = _invert_matrix_field_3d(A_mat)
        
        du_hat, _ = solve_dbfft_linear_system_3d(Xi, K4, M_inv, b_hat, tol_CG=tol_CG)
        if np.any(np.isnan(du_hat)) or np.any(np.isinf(du_hat)):
            raise ValueError("Linear solver returned NaN or Inf (BiCGSTAB breakdown).")
        du = _ifft3_tensor(du_hat)
        
        alpha = 1.0
        converged_constitutive = False
        for _ in range(16):
            u_trial = u + alpha * du
            grad_u_trial = get_grad_u_3d(u_trial, Xi)
            F_trial = F_bar_grid + grad_u_trial
            if Fp_inv is not None:
                Fe_trial = np.einsum('ijxyz,jkxyz->ikxyz', F_trial, Fp_inv, optimize=True)
            else:
                Fe_trial = F_trial
            Je = (
                Fe_trial[0, 0] * (Fe_trial[1, 1] * Fe_trial[2, 2] - Fe_trial[1, 2] * Fe_trial[2, 1]) -
                Fe_trial[0, 1] * (Fe_trial[1, 0] * Fe_trial[2, 2] - Fe_trial[1, 2] * Fe_trial[2, 0]) +
                Fe_trial[0, 2] * (Fe_trial[1, 0] * Fe_trial[2, 1] - Fe_trial[1, 1] * Fe_trial[2, 0])
            )
            if np.any(Je <= 1e-4) or np.any(np.isnan(Je)):
                alpha *= 0.5
                continue
            try:
                P_t, K4_t = constitutive_hyperelastic_3d(
                    F_trial, C4, I2, I4, I4rt, Fp=Fp,
                    model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
                    v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
                if np.any(np.isnan(P_t)) or np.any(np.isnan(K4_t)):
                    alpha *= 0.5
                    continue
                u = u_trial
                converged_constitutive = True
                break
            except Exception:
                alpha *= 0.5
        if not converged_constitutive:
            raise ValueError("Constitutive solver failed to converge in line search (potential negative Jacobian/NaN).")
            
    # Newton loop finished without returning -> did not converge
    raise ValueError(f"DBFFT solver did not converge within {max_iter} iterations. Final relative residual: {rel_res:.2e}")


# ---------------------------------------------------------------------------
# Elastic stiffness tensor C4 in 3D
# ---------------------------------------------------------------------------
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

def constitutive_hyperelastic_3d(F, C4, I2, I4, I4rt, Fp=None, model_type="svk", A_m=0.0, B_m=0.0, C_m=0.0,
                                 v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
    """
    Compute 1st Piola-Kirchhoff stress P and consistent tangent K4 in 3D,
    supporting an optional plastic deformation gradient field Fp.
    Supports 'svk', 'neo_hookean', and 'landau' models.
    """
    ndim = 3

    if Fp is not None:
        Fp_inv = _invert_Fp_3d(Fp)
        Fe = np.einsum('ijxyz,jkxyz->ikxyz', F, Fp_inv, optimize=True)
    else:
        Fe = F

    if model_type == "landau":
        nx, ny, nz = Fe.shape[2], Fe.shape[3], Fe.shape[4]
        v1_arr = np.full((nx, ny, nz), float(v1)) if isinstance(v1, (int, float, np.number)) else np.array(v1)
        v2_arr = np.full((nx, ny, nz), float(v2)) if isinstance(v2, (int, float, np.number)) else np.array(v2)
        v3_arr = np.full((nx, ny, nz), float(v3)) if isinstance(v3, (int, float, np.number)) else np.array(v3)
        g1_arr = np.full((nx, ny, nz), float(g1)) if isinstance(g1, (int, float, np.number)) else np.array(g1)
        g2_arr = np.full((nx, ny, nz), float(g2)) if isinstance(g2, (int, float, np.number)) else np.array(g2)
        g3_arr = np.full((nx, ny, nz), float(g3)) if isinstance(g3, (int, float, np.number)) else np.array(g3)
        g4_arr = np.full((nx, ny, nz), float(g4)) if isinstance(g4, (int, float, np.number)) else np.array(g4)

        # Autoconvert constants to Pa if supplied in GPa (values < 1e6 except 0.0)
        for arr in [v1_arr, v2_arr, v3_arr, g1_arr, g2_arr, g3_arr, g4_arr]:
            mask = (np.abs(arr) < 1e6) & (arr != 0.0)
            arr[mask] *= 1e9

        lam = C4[0, 0, 1, 1]
        mu = C4[0, 1, 0, 1]

        Ce = np.einsum('jixyz,jkxyz->ikxyz', Fe, Fe, optimize=True)
        E_GL = 0.5 * (Ce - I2)

        trE = E_GL[0, 0] + E_GL[1, 1] + E_GL[2, 2]
        trE2 = np.einsum('ijxyz,jixyz->xyz', E_GL, E_GL)
        trE3 = np.einsum('ijxyz,jkxyz,kixyz->xyz', E_GL, E_GL, E_GL)
        E2 = np.einsum('ijxyz,jkxyz->ikxyz', E_GL, E_GL)

        A_coef = lam * trE + 0.5 * v1_arr * (trE**2) + v2_arr * trE2 + (1.0/6.0) * g1_arr * (trE**3) + g2_arr * trE * trE2 + (4.0/3.0) * g3_arr * trE3
        B_coef = 2.0 * (mu + v2_arr * trE + 0.5 * g2_arr * (trE**2) + g4_arr * trE2)
        C_coef = 4.0 * (v3_arr + g3_arr * trE)

        S = A_coef[np.newaxis, np.newaxis, :, :, :] * I2 \
          + B_coef[np.newaxis, np.newaxis, :, :, :] * E_GL \
          + C_coef[np.newaxis, np.newaxis, :, :, :] * E2

        # Derivatives for C4_eff
        dAdE = (lam + v1_arr * trE + 0.5 * g1_arr * (trE**2) + g2_arr * trE2)[np.newaxis, np.newaxis, :, :, :] * I2 \
             + 2.0 * (v2_arr + g2_arr * trE)[np.newaxis, np.newaxis, :, :, :] * E_GL \
             + 4.0 * g3_arr[np.newaxis, np.newaxis, :, :, :] * E2

        dBdE = 2.0 * (v2_arr + g2_arr * trE)[np.newaxis, np.newaxis, :, :, :] * I2 \
             + 4.0 * g4_arr[np.newaxis, np.newaxis, :, :, :] * E_GL

        dCdE = 4.0 * g3_arr[np.newaxis, np.newaxis, :, :, :] * I2

        I4s = 0.5 * (I4 + I4rt)
        II = _dyad22_3d(I2, I2)

        term1 = np.einsum('ijxyz,klxyz->ijklxyz', I2, dAdE)
        term2 = np.einsum('ijxyz,klxyz->ijklxyz', E_GL, dBdE)
        term3 = np.einsum('ijxyz,klxyz->ijklxyz', E2, dCdE)
        term4 = B_coef[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * I4s

        term5_part1 = np.einsum('ikxyz,jlxyz->ijklxyz', I2, E_GL) + np.einsum('ikxyz,jlxyz->ijklxyz', E_GL, I2)
        term5_part2 = np.einsum('ilxyz,jkxyz->ijklxyz', I2, E_GL) + np.einsum('ilxyz,jkxyz->ijklxyz', E_GL, I2)
        K_E = 0.5 * (term5_part1 + term5_part2)
        term5 = C_coef[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * K_E

        C4_eff = term1 + term2 + term3 + term4 + term5
    elif model_type == "murnaghan":
        nx, ny, nz = Fe.shape[2], Fe.shape[3], Fe.shape[4]
        A_arr = np.full((nx, ny, nz), float(A_m)) if isinstance(A_m, (int, float, np.number)) else np.array(A_m)
        B_arr = np.full((nx, ny, nz), float(B_m)) if isinstance(B_m, (int, float, np.number)) else np.array(B_m)
        C_arr = np.full((nx, ny, nz), float(C_m)) if isinstance(C_m, (int, float, np.number)) else np.array(C_m)

        lam = C4[0, 0, 1, 1]
        mu = C4[0, 1, 0, 1]

        Ce = np.einsum('jixyz,jkxyz->ikxyz', Fe, Fe, optimize=True)
        E_GL = 0.5 * (Ce - I2)

        trE = E_GL[0, 0] + E_GL[1, 1] + E_GL[2, 2]
        trE2 = np.einsum('ijxyz,jixyz->xyz', E_GL, E_GL)
        E2 = np.einsum('ijxyz,jkxyz->ikxyz', E_GL, E_GL)

        S = (lam * trE + A_arr * (trE**2) + B_arr * trE2)[np.newaxis, np.newaxis, :, :, :] * I2 \
          + 2.0 * (mu + B_arr * trE)[np.newaxis, np.newaxis, :, :, :] * E_GL \
          + C_arr[np.newaxis, np.newaxis, :, :, :] * E2

        # Tangent stiffness C4_eff
        I4s = 0.5 * (I4 + I4rt)
        II = _dyad22_3d(I2, I2)
        
        term1 = (lam + 2.0 * A_arr * trE)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * II

        I_dyad_E = np.einsum('ijxyz,klxyz->ijklxyz', I2, E_GL)
        E_dyad_I = np.einsum('ijxyz,klxyz->ijklxyz', E_GL, I2)
        term2 = (2.0 * B_arr)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * (I_dyad_E + E_dyad_I)

        term3 = (2.0 * (mu + B_arr * trE))[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * I4s

        term4_part1 = np.einsum('ikxyz,jlxyz->ijklxyz', I2, E_GL) + np.einsum('ikxyz,jlxyz->ijklxyz', E_GL, I2)
        term4_part2 = np.einsum('ilxyz,jkxyz->ijklxyz', I2, E_GL) + np.einsum('ilxyz,jkxyz->ijklxyz', E_GL, I2)
        K_E = 0.5 * (term4_part1 + term4_part2)
        term4 = C_arr[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :, :] * K_E

        C4_eff = term1 + term2 + term3 + term4
    else:
        E_GL = 0.5 * (np.einsum('jixyz,jkxyz->ikxyz', Fe, Fe, optimize=True) - I2)
        S    = np.einsum('ijklxyz,lkxyz->ijxyz', C4, E_GL, optimize=True)
        C4_eff = C4

    if Fp is not None:
        P = np.einsum('ikxyz,klxyz,jlxyz->ijxyz', Fe, S, Fp_inv, optimize=True)
        S_ref = np.einsum('mkxyz,klxyz,jlxyz->mjxyz', Fp_inv, S, Fp_inv, optimize=True)
        
        A = np.einsum('klmnxyz,jlxyz->kjmnxyz', C4_eff, Fp_inv, optimize=True)
        B = np.einsum('kjmnxyz,bmxyz->kjbnxyz', A, Fp_inv, optimize=True)
        C = np.einsum('kjbnxyz,ikxyz->ijbnxyz', B, Fe, optimize=True)
        term2 = np.einsum('ijbnxyz,anxyz->ijabxyz', C, Fe, optimize=True)
        
        term1 = np.einsum('bjxyz,ia->ijabxyz', S_ref, np.eye(ndim), optimize=True)
        K4 = term1 + term2
    else:
        P = np.einsum('ijxyz,jkxyz->ikxyz', Fe, S, optimize=True)
        
        term1 = np.einsum('ijxyz,jkmnxyz->ikmnxyz', S, I4, optimize=True)
        
        FC4 = np.einsum('ijxyz,jkmnxyz->ikmnxyz', Fe, C4_eff, optimize=True)
        Ft = np.einsum('ijxyz->jixyz', Fe)
        FC4Ft = np.einsum('ijklxyz,lmxyz->ijkmxyz', FC4, Ft, optimize=True)
        term2_part1 = np.einsum('ijklxyz,lkmnxyz->ijmnxyz', I4rt, FC4Ft, optimize=True)
        term2 = np.einsum('ijklxyz,lkmnxyz->ijmnxyz', term2_part1, I4rt, optimize=True)
        
        K4 = term1 + term2

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
                           mixed_targets, plane_mode, ndim=2, F_bar_initial=None):
    """
    Translate the standard MGKMC config keys into finite-strain BC masks.

    Parameters
    ----------
    driving_component : (i, j) tuple  — which F component is driven
    eps_target_step   : float         — current eps increment value
    mixed_targets     : dict {(i,j): stress_value_Pa}
    plane_mode        : 'plane_strain' or 'plane_stress'
    ndim              : 2
    F_bar_initial     : ndarray (ndim, ndim), optional — previously converged F_bar

    Returns
    -------
    F_bar    : ndarray (ndim, ndim) — prescribed F̄ (components for driven + constraints)
    F_mask   : bool ndarray (ndim, ndim) — True where F̄ is prescribed
    P_target : ndarray (ndim, ndim) — stress targets (Pa), 0 where free
    P_mask   : bool ndarray (ndim, ndim) — True where avg stress is prescribed
    """
    if F_bar_initial is None:
        F_bar = np.eye(ndim)          # start from identity
    else:
        F_bar = F_bar_initial.copy()
        
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
            # Keep F_bar[k,k] as is (from F_bar_initial if provided, otherwise 1.0)
            P_target[k, k] = mixed_targets[comp]
            P_mask[k, k]   = True
        else:
            # No mention in mixed_targets  →  plane_strain default: F_kk = 1.0
            F_bar[k, k]    = 1.0
            F_mask[k, k]   = True

    return F_bar, F_mask, P_target, P_mask


# ---------------------------------------------------------------------------
# Augmented Lagrangian (AL) finite-strain solver  (2D)
# ---------------------------------------------------------------------------
# Reference:
#   Zecevic, Lebensohn & Capolungo (2022) — LS-EVPFFT, Table 1
#   Michel, Moulinec & Suquet (2001) — Augmented Lagrangian for composites
#
# Algorithm (adapted for hyperelastic reference-configuration formulation):
#
#   Given  F (initial guess), F_bar (macro BC), L0 (reference stiffness)
#   Initialise  λ = P(F)  (auxiliary stress, starts at constitutive stress)
#
#   Loop until convergence:
#     1. Polarisation:  φ = λ - L0 : F          (fluctuation polarisation)
#     2. Green's step:  δF = -G : φ              (enforce equilibrium of φ)
#     3. Apply BC:      F ← F + δF,  then set mean(F) = F_bar
#     4. Constitutive:  P = constitutive(F, Fp)  (full nonlinear evaluation)
#     5. Local update:  λ ← P                   (simplified AL update, β→∞)
#     6. Convergence:   ||G : P||_F / ||P||_F < tol
#
# The reference stiffness L0 is the volume-average of C4.  It is fixed for
# the entire solve and is always positive-definite → no degenerate tangents.
# ---------------------------------------------------------------------------

def _al_step_2d(F, F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
                tol=1e-5, max_iter=200,
                model_type="svk", plane_mode="plane_strain",
                A_m=0.0, B_m=0.0, C_m=0.0,
                v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
    """
    Augmented Lagrangian solver for a single 2D load step.

    Solves  G : P(F) = 0  subject to  mean(F) = F_bar.

    Parameters
    ----------
    F        : (2, 2, nx, ny)  deformation gradient (initial guess)
    F_bar    : (2, 2)          prescribed macro deformation gradient
    Ghat4    : projection operator
    C4       : (2, 2, 2, 2, nx, ny)  elastic stiffness field
    I2, I4, I4rt : identity tensors
    Fp       : (2, 2, nx, ny) or None — plastic DG field
    tol      : convergence tolerance on  ||G:P|| / ||P||
    max_iter : maximum AL iterations
    model_type, plane_mode, A_m, B_m, C_m : passed to constitutive model

    Returns
    -------
    F, P, K4, F33, n_iter
    """
    ndim = 2
    nx, ny = F.shape[2], F.shape[3]

    # Reference (homogeneous) stiffness: spatial average of C4
    # Shape: (2, 2, 2, 2) — same stiffness at every point
    L0 = C4.mean(axis=(-2, -1))                     # (2, 2, 2, 2)
    L0_field = np.einsum('ijkl,xy->ijklxy', L0, np.ones((nx, ny)))

    # Invert Fp once if provided
    if Fp is not None:
        det = Fp[0,0]*Fp[1,1] - Fp[0,1]*Fp[1,0]
        det_s = np.where(np.abs(det) < 1e-14, 1e-14, det)
        Fp_inv = np.zeros_like(Fp)
        Fp_inv[0,0] =  Fp[1,1] / det_s
        Fp_inv[1,1] =  Fp[0,0] / det_s
        Fp_inv[0,1] = -Fp[0,1] / det_s
        Fp_inv[1,0] = -Fp[1,0] / det_s
    else:
        Fp_inv = None

    # Enforce macro BC on initial F
    DbarF = F_bar - F.mean(axis=(2, 3))
    F = F + np.einsum('ij,xy->ijxy', DbarF, np.ones((nx, ny)))

    # Initial constitutive evaluation
    P, K4, F33 = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, Fp=Fp,
        model_type=model_type, plane_mode=plane_mode,
        A_m=A_m, B_m=B_m, C_m=C_m,
        v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)

    # Initialise auxiliary stress
    lam = P.copy()

    def G_op(A2):
        return _project(A2, Ghat4)

    for i in range(max_iter):
        # Step 1: polarisation field  φ = λ - L0:F
        phi = lam - _ddot42(L0_field, F)

        # Step 2: Green's operator step  δF = -G:φ
        dF = -G_op(phi)

        # Step 3: update F and re-enforce BC with backtracking
        success_step = False
        for beta in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            F_try = F + beta * dF
            DbarF = F_bar - F_try.mean(axis=(2, 3))
            F_try = F_try + np.einsum('ij,xy->ijxy', DbarF, np.ones((nx, ny)))
            
            if Fp_inv is not None:
                Fe_try = np.einsum('ijxy,jkxy->ikxy', F_try, Fp_inv, optimize=True)
            else:
                Fe_try = F_try
            Je_try = Fe_try[0,0]*Fe_try[1,1] - Fe_try[0,1]*Fe_try[1,0]
            
            try:
                P_new, K4_new, F33_new = constitutive_hyperelastic_2d(
                    F_try, C4, I2, I4, I4rt, Fp=Fp,
                    model_type=model_type, plane_mode=plane_mode,
                    A_m=A_m, B_m=B_m, C_m=C_m,
                    v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
                if not np.any(np.isnan(P_new)) and not np.any(np.isnan(K4_new)) and np.all(Je_try > 1e-4):
                    F = F_try
                    P, K4, F33 = P_new, K4_new, F33_new
                    success_step = True
                    break
            except Exception:
                pass
        
        if not success_step:
            break

        # Step 5: update auxiliary stress  λ ← P
        lam = P.copy()

        # Step 6: convergence check
        GP = G_op(P)
        res = np.linalg.norm(GP) / (np.sqrt(GP.size) * (np.mean(np.abs(P)) + 1e-20))
        if res < tol and i > 0:
            return F, P, K4, F33, i + 1

    # If the loop finishes without returning, it means it didn't converge or broke
    GP = G_op(P)
    res = np.linalg.norm(GP) / (np.sqrt(GP.size) * (np.mean(np.abs(P)) + 1e-20))
    raise ValueError(f"AL solver did not converge or exploded. Final relative residual: {res:.2e}")


# ---------------------------------------------------------------------------
# Augmented Lagrangian (AL) finite-strain solver  (3D)
# ---------------------------------------------------------------------------

def _al_step_3d(F, F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
                tol=1e-5, max_iter=200,
                model_type="svk", A_m=0.0, B_m=0.0, C_m=0.0,
                v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
    """
    Augmented Lagrangian solver for a single 3D load step.
    Same algorithm as _al_step_2d but for 3D fields.
    """
    ndim = 3
    nx, ny, nz = F.shape[2], F.shape[3], F.shape[4]

    L0 = C4.mean(axis=(-3, -2, -1))                  # (3, 3, 3, 3)
    L0_field = np.einsum('ijkl,xyz->ijklxyz', L0, np.ones((nx, ny, nz)))

    if Fp is not None:
        Fp_inv = _invert_Fp_3d(Fp)
    else:
        Fp_inv = None

    # Enforce macro BC on initial F
    DbarF = F_bar - F.mean(axis=(2, 3, 4))
    F = F + np.einsum('ij,xyz->ijxyz', DbarF, np.ones((nx, ny, nz)))

    P, K4 = constitutive_hyperelastic_3d(
        F, C4, I2, I4, I4rt, Fp=Fp,
        model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
        v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)

    lam = P.copy()

    def G_op(A2):
        return _project_3d(A2, Ghat4)

    for i in range(max_iter):
        phi = lam - _ddot42_3d(L0_field, F)
        dF  = -G_op(phi)

        success_step = False
        for beta in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            F_try = F + beta * dF
            DbarF = F_bar - F_try.mean(axis=(2, 3, 4))
            F_try = F_try + np.einsum('ij,xyz->ijxyz', DbarF, np.ones((nx, ny, nz)))
            
            if Fp_inv is not None:
                Fe_try = np.einsum('ijxyz,jkxyz->ikxyz', F_try, Fp_inv, optimize=True)
            else:
                Fe_try = F_try
            Je_try = (
                Fe_try[0, 0] * (Fe_try[1, 1] * Fe_try[2, 2] - Fe_try[1, 2] * Fe_try[2, 1]) -
                Fe_try[0, 1] * (Fe_try[1, 0] * Fe_try[2, 2] - Fe_try[1, 2] * Fe_try[2, 0]) +
                Fe_try[0, 2] * (Fe_try[1, 0] * Fe_try[2, 1] - Fe_try[1, 1] * Fe_try[2, 0])
            )
            
            try:
                P_new, K4_new = constitutive_hyperelastic_3d(
                    F_try, C4, I2, I4, I4rt, Fp=Fp,
                    model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
                    v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
                if not np.any(np.isnan(P_new)) and not np.any(np.isnan(K4_new)) and np.all(Je_try > 1e-4):
                    F = F_try
                    P, K4 = P_new, K4_new
                    success_step = True
                    break
            except Exception:
                pass
                
        if not success_step:
            break

        lam = P.copy()

        GP = G_op(P)
        res = np.linalg.norm(GP) / (np.sqrt(GP.size) * (np.mean(np.abs(P)) + 1e-20))
        if res < tol and i > 0:
            return F, P, K4, i + 1

    # If the loop finishes without returning, it means it didn't converge or broke
    GP = G_op(P)
    res = np.linalg.norm(GP) / (np.sqrt(GP.size) * (np.mean(np.abs(P)) + 1e-20))
    raise ValueError(f"AL solver did not converge or exploded. Final relative residual: {res:.2e}")


# ---------------------------------------------------------------------------
# Core Newton-CG finite-strain solver (single load step)
# ---------------------------------------------------------------------------

def _newton_cg_step(F, F_bar, Ghat4, C4, I2, I4, I4rt,
                    tol_NW=1e-5, tol_CG=1e-6, max_NW=20,
                    model_type="svk", plane_mode="plane_strain",
                    A_m=0.0, B_m=0.0, C_m=0.0,
                    v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
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
    P, K4, _ = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, model_type=model_type, plane_mode=plane_mode,
        A_m=A_m, B_m=B_m, C_m=C_m,
        v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
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
            dFm, _ = sp.bicgstab(A_op, rhs, rtol=tol_CG, maxiter=150)
        except TypeError:
            dFm, _ = sp.bicgstab(A_op, rhs, tol=tol_CG, maxiter=150)
        dF     = dFm.reshape(ndim, ndim, nx, ny)

        # Update F
        if i_NW == 0:
            F = F + DbarF_grid + dF    # apply macro jump + micro fluctuation
        else:
            F = F + dF                 # pure Newton update

        # Recompute constitutive response at updated F
        P, K4, _ = constitutive_hyperelastic_2d(
            F, C4, I2, I4, I4rt, model_type=model_type, plane_mode=plane_mode,
            A_m=A_m, B_m=B_m, C_m=C_m,
            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)

        # Convergence check (skip iteration 0 as in de Geus code)
        res_norm = np.linalg.norm(dFm) / (np.linalg.norm(F) + 1e-20)
        if res_norm < tol_NW and i_NW > 0:
            return F, P, K4, i_NW + 1

    return F, P, K4, max_NW


def finite_strain_solver_step_2d(
    F, F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
    driving_component=(0, 0), P_target=None, P_mask=None,
    E_avg=100e9, nu_avg=0.3,
    tol_NW=1e-5, tol_CG=1e-6, max_NW=50,
    tol_macro=1e6, max_iter_macro=20,
    enable_console=True, model_type="svk", plane_mode="plane_strain",
    A_m=0.0, B_m=0.0, C_m=0.0,
    solver="al", pixel=1.0,
    v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0
):
    """
    Solves a single step of the finite strain problem in 2D, enforcing mixed BCs on F_bar.
    Supports an optional plastic deformation gradient field Fp (eigenstrain).

    Parameters
    ----------
    solver : 'al' (default), 'newton_cg', or 'dbfft'
        'al'        — Augmented Lagrangian (robust, fixed reference stiffness).
        'newton_cg' — Original Newton-CG (faster per iteration, less robust).
        'dbfft'     — Displacement-Based FFT solver (extremely robust).
    """
    ndim = 2
    nx, ny = F.shape[2], F.shape[3]
    if P_target is None:
        P_target = np.zeros((ndim, ndim))
    if P_mask is None:
        P_mask = np.zeros((ndim, ndim), dtype=bool)

    F_start = F.copy()
    F_bar_current = F_bar.copy()

    F_final = F_start.copy()
    P_final = None
    Sig_final = None
    K4_final = None
    max_err = 0.0

    for it_mac in range(max_iter_macro):

        # ---- Inner solve: AL, DBFFT, or Newton-CG ----
        if solver == "al":
            F_curr, P, K4, F33, _nitr = _al_step_2d(
                F_start.copy(), F_bar_current, Ghat4, C4, I2, I4, I4rt, Fp=Fp,
                tol=tol_NW, max_iter=max_NW,
                model_type=model_type, plane_mode=plane_mode,
                A_m=A_m, B_m=B_m, C_m=C_m,
                v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
        elif solver == "dbfft":
            Lx, Ly = nx * pixel, ny * pixel
            Xi = _get_frequencies_2d(nx, ny, Lx, Ly)
            F_curr, P, K4, F33, _nitr = _dbfft_step_2d(
                F_start.copy(), F_bar_current, Xi, C4, I2, I4, I4rt, Fp=Fp,
                tol=tol_NW, tol_CG=tol_CG, max_iter=max_NW,
                model_type=model_type, plane_mode=plane_mode,
                A_m=A_m, B_m=B_m, C_m=C_m,
                v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
        else:
            # --- Original Newton-CG path ---
            DbarF = F_bar_current - F_start.mean(axis=(2, 3))
            DbarF_grid = np.einsum('ij,xy->ijxy', DbarF, np.ones((nx, ny)))

            P, K4, F33 = constitutive_hyperelastic_2d(
                F_start, C4, I2, I4, I4rt, Fp=Fp,
                model_type=model_type, plane_mode=plane_mode,
                A_m=A_m, B_m=B_m, C_m=C_m,
                v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)

            def G_op(A2):
                return _project(A2, Ghat4)

            def K_dF_op(dFm_flat):
                dF = dFm_flat.reshape(ndim, ndim, nx, ny)
                return _trans2(_ddot42(K4, _trans2(dF)))

            def G_K_dF(dFm_flat):
                return G_op(K_dF_op(dFm_flat)).reshape(-1)

            A_op = sp.LinearOperator(
                shape=(F_start.size, F_start.size),
                matvec=G_K_dF, dtype='float64')

            F_curr = F_start.copy()
            if Fp is not None:
                det = Fp[0,0]*Fp[1,1] - Fp[0,1]*Fp[1,0]
                det_safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
                Fp_inv = np.zeros_like(Fp)
                Fp_inv[0,0] =  Fp[1,1] / det_safe
                Fp_inv[1,1] =  Fp[0,0] / det_safe
                Fp_inv[0,1] = -Fp[0,1] / det_safe
                Fp_inv[1,0] = -Fp[1,0] / det_safe
            else:
                Fp_inv = None

            for i_NW in range(max_NW):
                rhs = (-G_op(K_dF_op(DbarF_grid.reshape(-1))).reshape(-1)
                       if i_NW == 0 else -G_op(P).reshape(-1))
                try:
                    dFm, _ = sp.bicgstab(A_op, rhs, rtol=tol_CG, maxiter=150)
                except TypeError:
                    dFm, _ = sp.bicgstab(A_op, rhs, tol=tol_CG, maxiter=150)
                dF  = dFm.reshape(ndim, ndim, nx, ny)
                dX  = (DbarF_grid + dF) if i_NW == 0 else dF

                alpha = 1.0
                for _ in range(16):
                    F_trial = F_curr + alpha * dX
                    Fe_t = (np.einsum('ijxy,jkxy->ikxy', F_trial, Fp_inv, optimize=True)
                            if Fp_inv is not None else F_trial)
                    Je = Fe_t[0,0]*Fe_t[1,1] - Fe_t[0,1]*Fe_t[1,0]
                    if np.any(Je <= 1e-4) or np.any(np.isnan(Je)):
                        alpha *= 0.5; continue
                    try:
                        P_t, K4_t, F33_t = constitutive_hyperelastic_2d(
                            F_trial, C4, I2, I4, I4rt, Fp=Fp,
                            model_type=model_type, plane_mode=plane_mode,
                            A_m=A_m, B_m=B_m, C_m=C_m,
                            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
                        if np.any(np.isnan(P_t)) or np.any(np.isnan(K4_t)):
                            alpha *= 0.5; continue
                        F_curr, P, K4, F33 = F_trial, P_t, K4_t, F33_t; break
                    except Exception:
                        alpha *= 0.5
                else:
                    raise ValueError("Newton-CG line search failed to find a valid step.")

                if np.linalg.norm(dFm) / (np.linalg.norm(F_curr) + 1e-20) < tol_NW and i_NW > 0:
                    break
            else:
                raise ValueError(f"Newton-CG inner solver did not converge within {max_NW} iterations.")

        # --- Shared: compute Cauchy stress, check mixed BCs ---
        Sig_field = cauchy_from_P(P, F_curr, F33=F33)
        Sig_mac   = Sig_field.mean(axis=(2, 3))

        F_final   = F_curr
        P_final   = P
        Sig_final = Sig_field
        K4_final  = K4

        if not np.any(P_mask):
            break

        stress_err = np.zeros((ndim, ndim))
        stress_err[P_mask] = P_target[P_mask] - Sig_mac[P_mask]
        max_err = np.max(np.abs(stress_err[P_mask]))

        if max_err < tol_macro:
            break

        i_drv, j_drv = driving_component
        d_F_mat = (stress_err - nu_avg * np.trace(stress_err) * np.eye(ndim)) / E_avg
        d_F_mat = np.clip(d_F_mat, -0.01, 0.01)

        for ii in range(ndim):
            for jj in range(ndim):
                if not (P_mask[ii, jj] and ii == jj):
                    continue
                if (ii, jj) == (i_drv, j_drv):
                    continue
                F_bar_current[ii, jj] += d_F_mat[ii, jj]

        # Next outer iteration starts from the latest converged F
        F_start = F_curr
    else:
        if np.any(P_mask):
            raise ValueError(f"Outer BC loop did not converge after {max_iter_macro} iterations. Final max stress error: {max_err:.2e} Pa (threshold: {tol_macro:.2e} Pa).")

    return F_final, P_final, Sig_final, K4_final, F_bar_current



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
    tol_CG=1e-6,
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
    model_type="svk",
    A_m=0.0, B_m=0.0, C_m=0.0,
    solver="al",
    v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0
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
    F_bar_init = None

    for s in range(n_steps + 1):
        eps_s = eps_steps[s]

        # Build target F̄ and stress masks for this step
        F_bar, F_mask, P_tgt, P_mask = build_finite_strain_bc(
            driving_component, eps_s, mixed_targets, plane_mode,
            F_bar_initial=F_bar_init
        )

        F, P_field, Sig_field, K4, F_bar = finite_strain_solver_step_2d(
            F, F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
            driving_component=driving_component, P_target=P_tgt, P_mask=P_mask,
            E_avg=E.mean(), nu_avg=nu.mean(),
            tol_NW=tol_NW, tol_CG=tol_CG, max_NW=max_NW,
            tol_macro=tol_macro, max_iter_macro=max_iter_macro,
            enable_console=enable_console,
            model_type=model_type,
            plane_mode=plane_mode,
            A_m=A_m, B_m=B_m, C_m=C_m,
            solver=solver, pixel=pixel,
            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4
        )
        F_bar_init = F_bar

        F_mac = F.mean(axis=(2, 3))
        P_mac = P_field.mean(axis=(2, 3))
        Sig_mac = Sig_field.mean(axis=(2, 3))

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

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()

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
                              mixed_targets, ndim=3, F_bar_initial=None):
    """
    Translate the standard MGKMC config keys into 3D finite-strain BC masks.
    """
    if F_bar_initial is None:
        F_bar = np.eye(ndim)          # start from identity
    else:
        F_bar = F_bar_initial.copy()
        
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
                       tol_NW=1e-5, tol_CG=1e-6, max_NW=20,
                       model_type="svk", A_m=0.0, B_m=0.0, C_m=0.0,
                       v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0):
    """
    Run 3D Newton-CG iterations to enforce G : P(F) = 0.
    """
    ndim = 3
    nx, ny, nz = F.shape[2], F.shape[3], F.shape[4]

    DbarF      = F_bar - F.mean(axis=(2, 3, 4))
    DbarF_grid = np.einsum('ij,xyz->ijxyz', DbarF, np.ones((nx, ny, nz)))

    P, K4 = constitutive_hyperelastic_3d(
        F, C4, I2, I4, I4rt, model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
        v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)

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
            dFm, _ = sp.bicgstab(A_op, rhs, rtol=tol_CG, maxiter=150)
        except TypeError:
            dFm, _ = sp.bicgstab(A_op, rhs, tol=tol_CG, maxiter=150)
        dF     = dFm.reshape(ndim, ndim, nx, ny, nz)

        if i_NW == 0:
            F = F + DbarF_grid + dF
        else:
            F = F + dF

        P, K4 = constitutive_hyperelastic_3d(
            F, C4, I2, I4, I4rt, model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)

        res_norm = np.linalg.norm(dFm) / (np.linalg.norm(F) + 1e-20)
        if res_norm < tol_NW and i_NW > 0:
            return F, P, K4, i_NW + 1

    return F, P, K4, max_NW


def finite_strain_solver_step_3d(
    F, F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
    driving_component=(0, 0), P_target=None, P_mask=None,
    E_avg=100e9, nu_avg=0.3,
    tol_NW=1e-5, tol_CG=1e-6, max_NW=50,
    tol_macro=1e6, max_iter_macro=20,
    enable_console=True, model_type="svk",
    A_m=0.0, B_m=0.0, C_m=0.0,
    solver="al", pixel=1.0,
    v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0
):
    """
    Solves a single step of the finite strain problem in 3D, enforcing mixed BCs on F_bar.
    Supports an optional plastic deformation gradient field Fp (eigenstrain).

    Parameters
    ----------
    solver : 'al' (default), 'newton_cg', or 'dbfft'
    """
    ndim = 3
    nx, ny, nz = F.shape[2], F.shape[3], F.shape[4]
    if P_target is None:
        P_target = np.zeros((ndim, ndim))
    if P_mask is None:
        P_mask = np.zeros((ndim, ndim), dtype=bool)

    F_start = F.copy()
    F_bar_current = F_bar.copy()

    F_final = F_start.copy()
    P_final = None
    Sig_final = None
    K4_final = None
    max_err = 0.0

    for it_mac in range(max_iter_macro):

        # ---- Inner solve: AL, DBFFT, or Newton-CG ----
        if solver == "al":
            F_curr, P, K4, _nitr = _al_step_3d(
                F_start.copy(), F_bar_current, Ghat4, C4, I2, I4, I4rt, Fp=Fp,
                tol=tol_NW, max_iter=max_NW,
                model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
                v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
        elif solver == "dbfft":
            Lx, Ly, Lz = nx * pixel, ny * pixel, nz * pixel
            Xi = _get_frequencies_3d(nx, ny, nz, Lx, Ly, Lz)
            F_curr, P, K4, _nitr = _dbfft_step_3d(
                F_start.copy(), F_bar_current, Xi, C4, I2, I4, I4rt, Fp=Fp,
                tol=tol_NW, tol_CG=tol_CG, max_iter=max_NW,
                model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
                v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
        else:
            # --- Original Newton-CG path ---
            DbarF = F_bar_current - F_start.mean(axis=(2, 3, 4))
            DbarF_grid = np.einsum('ij,xyz->ijxyz', DbarF, np.ones((nx, ny, nz)))

            P, K4 = constitutive_hyperelastic_3d(
                F_start, C4, I2, I4, I4rt, Fp=Fp,
                model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
                v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)

            def G_op(A2):
                return _project_3d(A2, Ghat4)

            def K_dF_op(dFm_flat):
                dF = dFm_flat.reshape(ndim, ndim, nx, ny, nz)
                return _trans2_3d(_ddot42_3d(K4, _trans2_3d(dF)))

            def G_K_dF(dFm_flat):
                return G_op(K_dF_op(dFm_flat)).reshape(-1)

            A_op = sp.LinearOperator(
                shape=(F_start.size, F_start.size),
                matvec=G_K_dF, dtype='float64')

            F_curr = F_start.copy()
            Fp_inv = _invert_Fp_3d(Fp) if Fp is not None else None

            for i_NW in range(max_NW):
                rhs = (-G_op(K_dF_op(DbarF_grid.reshape(-1))).reshape(-1)
                       if i_NW == 0 else -G_op(P).reshape(-1))
                try:
                    dFm, _ = sp.bicgstab(A_op, rhs, rtol=tol_CG, maxiter=150)
                except TypeError:
                    dFm, _ = sp.bicgstab(A_op, rhs, tol=tol_CG, maxiter=150)
                dF  = dFm.reshape(ndim, ndim, nx, ny, nz)
                dX  = (DbarF_grid + dF) if i_NW == 0 else dF

                alpha = 1.0
                for _ in range(16):
                    F_trial = F_curr + alpha * dX
                    Fe_t = (np.einsum('ijxyz,jkxyz->ikxyz', F_trial, Fp_inv, optimize=True)
                            if Fp_inv is not None else F_trial)
                    Je = (Fe_t[0,0]*(Fe_t[1,1]*Fe_t[2,2]-Fe_t[1,2]*Fe_t[2,1])
                         -Fe_t[0,1]*(Fe_t[1,0]*Fe_t[2,2]-Fe_t[1,2]*Fe_t[2,0])
                         +Fe_t[0,2]*(Fe_t[1,0]*Fe_t[2,1]-Fe_t[1,1]*Fe_t[2,0]))
                    if np.any(Je <= 1e-4) or np.any(np.isnan(Je)):
                        alpha *= 0.5; continue
                    try:
                        P_t, K4_t = constitutive_hyperelastic_3d(
                            F_trial, C4, I2, I4, I4rt, Fp=Fp,
                            model_type=model_type, A_m=A_m, B_m=B_m, C_m=C_m,
                            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4)
                        if np.any(np.isnan(P_t)) or np.any(np.isnan(K4_t)):
                            alpha *= 0.5; continue
                        F_curr, P, K4 = F_trial, P_t, K4_t; break
                    except Exception:
                        alpha *= 0.5
                else:
                    raise ValueError("Newton-CG line search failed to find a valid step.")

                if np.linalg.norm(dFm) / (np.linalg.norm(F_curr) + 1e-20) < tol_NW and i_NW > 0:
                    break
            else:
                raise ValueError(f"Newton-CG inner solver did not converge within {max_NW} iterations.")

        # --- Shared: Cauchy stress + mixed BC check ---
        Sig_field = cauchy_from_P_3d(P, F_curr)
        Sig_mac   = Sig_field.mean(axis=(2, 3, 4))

        F_final   = F_curr
        P_final   = P
        Sig_final = Sig_field
        K4_final  = K4

        if not np.any(P_mask):
            break

        stress_err = np.zeros((ndim, ndim))
        stress_err[P_mask] = P_target[P_mask] - Sig_mac[P_mask]
        max_err = np.max(np.abs(stress_err[P_mask]))

        if max_err < tol_macro:
            break

        i_drv, j_drv = driving_component
        d_F_mat = (stress_err - nu_avg * np.trace(stress_err) * np.eye(ndim)) / E_avg
        d_F_mat = np.clip(d_F_mat, -0.01, 0.01)

        for ii in range(ndim):
            for jj in range(ndim):
                if not (P_mask[ii, jj] and ii == jj):
                    continue
                if (ii, jj) == (i_drv, j_drv):
                    continue
                F_bar_current[ii, jj] += d_F_mat[ii, jj]

        F_start = F_curr
    else:
        if np.any(P_mask):
            raise ValueError(f"Outer BC loop did not converge after {max_iter_macro} iterations. Final max stress error: {max_err:.2e} Pa (threshold: {tol_macro:.2e} Pa).")

    return F_final, P_final, Sig_final, K4_final, F_bar_current


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
    tol_CG=1e-6,
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
    model_type="svk",
    A_m=0.0, B_m=0.0, C_m=0.0,
    solver="al",
    v1=0.0, v2=0.0, v3=0.0, g1=0.0, g2=0.0, g3=0.0, g4=0.0
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
    F_bar_init = None

    for s in range(n_steps + 1):
        eps_s = eps_steps[s]

        F_bar, F_mask, P_tgt, P_mask = build_finite_strain_bc_3d(
            driving_component, eps_s, mixed_targets, ndim=ndim,
            F_bar_initial=F_bar_init
        )

        F, P_field, Sig_field, K4, F_bar = finite_strain_solver_step_3d(
            F, F_bar, Ghat4, C4, I2, I4, I4rt, Fp=None,
            driving_component=driving_component, P_target=P_tgt, P_mask=P_mask,
            E_avg=E.mean(), nu_avg=nu.mean(),
            tol_NW=tol_NW, tol_CG=tol_CG, max_NW=max_NW,
            tol_macro=tol_macro, max_iter_macro=max_iter_macro,
            enable_console=enable_console,
            model_type=model_type,
            A_m=A_m, B_m=B_m, C_m=C_m,
            solver=solver, pixel=pixel,
            v1=v1, v2=v2, v3=v3, g1=g1, g2=g2, g3=g3, g4=g4
        )
        F_bar_init = F_bar

        F_mac = F.mean(axis=(2, 3, 4))
        P_mac = P_field.mean(axis=(2, 3, 4))
        Sig_mac = Sig_field.mean(axis=(2, 3, 4))

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

    _total_time = _time.time() - _t0
    _m, _s = divmod(_total_time, 60)
    _h, _m = divmod(_m, 60)
    _duration_str = f"\nSimulation Finish Time: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {_total_time:.2f} seconds ({int(_h):d}h {int(_m):02d}m {int(_s):02d}s)\n"
    if _log_f:
        _log_f.write(_duration_str)
        _log_f.close()
    if _glog_f:
        _glog_f.close()

    return (
        np.array(F_macro_list),
        np.array(Sig_macro_list),
        np.array(P_macro_list),
        F_list,
        Sig_list,
    )

