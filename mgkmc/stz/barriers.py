import numpy as np

def compute_barrier(voxel, volume, softening_scheme="isotropic", debug=False):
    """
    Compute Q[m] = voxel.Q0[m] * exp(-g_eff) - work
    softening_scheme: "isotropic" or "directional"
    """
    # Base softening parameter
    g_base = voxel.g_p + voxel.g_t
    
    Q = np.zeros(voxel.M)
    
    # Pre-calculate directional modifiers if needed
    if softening_scheme == "directional" and voxel.prev_gamma is not None:
         # Norm of previous gamma
         norm_prev = np.sqrt(np.sum(voxel.prev_gamma**2))
         if norm_prev < 1e-12:
              modifiers = np.ones(voxel.M) # Fallback if prev is zero
         else:
              modifiers = np.zeros(voxel.M)
              for m, gamma in enumerate(voxel.catalog):
                   # Dot product
                   dot = np.sum(voxel.prev_gamma * gamma)
                   norm_curr = np.sqrt(np.sum(gamma**2))
                   
                   cosine = dot / (norm_prev * norm_curr + 1e-12)
                   
                   # Square Forward Modifier: (1 + cos)^2 / 4
                   # cos=1 -> mod=1 (Full softening)
                   # cos=-1 -> mod=0 (No softening)
                   # cos=0 -> mod=0.25 (Weak softening)
                   modifiers[m] = (1.0 + cosine)**2 / 4.0
    else:
         # Isotropic: Modifier is 1.0 for all directions
         modifiers = np.ones(voxel.M)

    for m, gamma in enumerate(voxel.catalog):
        GPa_nm3_to_eV = 6.241509
        # voxel.sigma is in Pa, convert to GPa
        sigma_GPa = voxel.sigma / 1e9 
        work = 0.5 * volume * np.sum(sigma_GPa * gamma) * GPa_nm3_to_eV
        
        # Apply modified softening
        # Effective g = modifier * g_base
        g_eff = modifiers[m] * g_base
        
        Q[m] = voxel.Q0[m] * np.exp(-g_eff) - work
        
        if debug and m == 0:
            print(f"  [DEBUG] Mode {m}: Q0={voxel.Q0[m]:.4f}, g_base={g_base:.4f}, Mod={modifiers[m]:.4f}, exp(-g_eff)={np.exp(-g_eff):.4f}")

    voxel.Q = Q
    return Q
