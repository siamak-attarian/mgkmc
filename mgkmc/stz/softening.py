import numpy as np

def von_mises_strain(e):
    """Return equivalent Mises strain for a 3×3 deformation tensor."""
    s = e - np.trace(e)/3 * np.eye(3)
    return np.sqrt(3/2 * np.sum(s*s))

def update_softening(voxel, gamma, jp=100, jt=300, g_max=None):
    vm = von_mises_strain(gamma)
    voxel.g_p += jp * vm**2
    
    # Enforce limit on permanent softening if specified
    if g_max is not None and voxel.g_p > g_max:
         voxel.g_p = g_max
         
    voxel.g_t  = jt * vm**2
