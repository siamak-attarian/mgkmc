# mgkmc/elasticity_helpers.py
"""
Elasticity helpers for defining macroscopic strain tensors for specific loading cases.
These functions use average material properties to estimate the required strain tensor
for stress-controlled directions.

Supported cases:
1. All strains given.
2. Plane stress in Z, Y fixed, X strain input.
3. Uniaxial stress (Y and Z stress free), X strain input.
4. Pure shear strain in XY.
"""

import numpy as np

__all__ = [
    "get_strain_tensor",
    "get_plane_stress_z_fixed_y",
    "get_uniaxial_stress_x",
    "get_pure_shear_xy",
]


def get_strain_tensor(eps_tensor: np.ndarray, **kwargs) -> np.ndarray:
    """
    Case 1: All strains given directly.
    Returns the input tensor as is.
    """
    return np.array(eps_tensor)


def get_plane_stress_z_fixed_y(eps_xx: float, E: float, nu: float) -> np.ndarray:
    """
    Case 2: Plane stress in Z (sigma_zz=0), Y fixed (eps_yy=0), X strain input.
    
    Uses average Poisson's ratio to estimate eps_zz:
    eps_zz = -nu / (1 - nu) * eps_xx
    """
    nu_avg = np.mean(nu)
    eps_zz = -nu_avg / (1 - nu_avg) * eps_xx
    
    eps = np.zeros((3, 3))
    eps[0, 0] = eps_xx
    eps[1, 1] = 0.0
    eps[2, 2] = eps_zz
    return eps


def get_uniaxial_stress_x(eps_xx: float, E: float, nu: float) -> np.ndarray:
    """
    Case 3: Uniaxial stress in X (sigma_yy=sigma_zz=0), X strain input.
    
    Uses average Poisson's ratio to estimate transverse strains:
    eps_yy = eps_zz = -nu * eps_xx
    """
    nu_avg = np.mean(nu)
    eps_trans = -nu_avg * eps_xx
    
    eps = np.zeros((3, 3))
    eps[0, 0] = eps_xx
    eps[1, 1] = eps_trans
    eps[2, 2] = eps_trans
    return eps


def get_pure_shear_xy(eps_xy: float, **kwargs) -> np.ndarray:
    """
    Case 4: Pure shear strain in XY.
    Note: eps_xy is the tensorial shear strain (gamma_xy / 2).
    All other components are zero.
    """
    eps = np.zeros((3, 3))
    eps[0, 1] = eps[1, 0] = eps_xy
    return eps
