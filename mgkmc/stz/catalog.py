import numpy as np

# ============================================================
#  EXACT GLASS STZ MODE GENERATOR (MATCHES C CODE)
# ============================================================

def stz_mode_glass(gamma0):
    """
    Generate one GLASS STZ mode matching the C code elastic.c logic:

        gxx = 0.5 * gamma0 * gaussian(0,1)
        gyy = -gxx
        gxy = 0.5 * gamma0 * gaussian(0,1)

    Plane strain:
        [ gxx   gxy   0 ]
        [ gxy  -gxx   0 ]
        [  0     0    0 ]
    """
    gxx = 0.5 * gamma0 * np.random.normal()
    gxy = 0.5 * gamma0 * np.random.normal()
    gyy = -gxx

    G = np.zeros((3,3))
    G[0,0] = gxx
    G[1,1] = gyy
    G[0,1] = G[1,0] = gxy
    return G


def stz_catalog_glass(M, gamma0):
    """
    Generate M independent STZ modes for a voxel,
    exactly matching the C-code random glass initialization.
    Returns numpy array of shape (M, 3, 3).
    """
    catalog = np.zeros((M, 3, 3))
    for i in range(M):
        catalog[i] = stz_mode_glass(gamma0)
    return catalog


# ============================================================
#  FULLY 3D ISOTROPIC STZ MODE GENERATOR
# ============================================================

def stz_mode_glass_3d(gamma0):
    """
    Generate one fully 3D STZ mode:
    A symmetric deviatoric tensor with random orientation in 3D space,
    scaled such that its equivalent strain magnitude has the same distribution
    (Rayleigh with scale gamma0) as the 2D glass mode.
    """
    # Generate random 3x3 matrix from Gaussian
    M = np.random.normal(size=(3, 3))
    # Make symmetric
    M_sym = 0.5 * (M + M.T)
    # Make deviatoric (subtract trace to conserve volume)
    G = M_sym - (1.0 / 3.0) * np.trace(M_sym) * np.eye(3)
    # Normalize G so that equivalent strain is exactly 1.0
    # gamma_eq = sqrt(2 * G : G)
    norm = np.sqrt(np.sum(G**2))
    if norm < 1e-12:
        return np.zeros((3, 3))
    G = G * (1.0 / (np.sqrt(2.0) * norm))
    
    # Scale by Rayleigh-distributed magnitude with scale gamma0
    # Sample Rayleigh: scale * sqrt(-2 * ln(U))
    mag = gamma0 * np.sqrt(-2.0 * np.log(np.random.uniform(1e-15, 1.0)))
    return G * mag


def stz_catalog_glass_3d(M, gamma0):
    """
    Generate M independent 3D STZ modes for a voxel.
    Returns numpy array of shape (M, 3, 3).
    """
    catalog = np.zeros((M, 3, 3))
    for i in range(M):
        catalog[i] = stz_mode_glass_3d(gamma0)
    return catalog

