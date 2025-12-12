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
