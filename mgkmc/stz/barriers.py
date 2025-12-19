import numpy as np

def compute_barrier(voxel, volume, softening_scheme="isotropic", debug=False, current_time=0.0, tau=np.inf):
    """
    Compute Q[m] = voxel.Q0[m] * exp(-g_eff) - work
    softening_scheme: "isotropic" or "directional"
    current_time: Current simulation time (for decay)
    tau: Decay time constant (t_Temp)
    """
    # Transient Softening Decay
    # g_t(t) = g_t(0) * exp(-(t - t_last)/tau)
    g_t_curr = 0.0
    if voxel.g_t > 0:
        if tau == np.inf:
            g_t_curr = voxel.g_t # No decay (infinite tau or T=0)
        elif voxel.last_event_time == -np.inf:
            g_t_curr = 0.0 # Never flipped, no transient heat
        else:
            dt = current_time - voxel.last_event_time
            if dt < 0: dt = 0 # Safety for numerical jitter
            g_t_curr = voxel.g_t * np.exp(-dt / tau)

    # Base softening parameter
    g_base = voxel.g_p + g_t_curr
    
    Q = np.zeros(voxel.M)
    
    # Pre-calculate directional modifiers if needed
    if softening_scheme == "directional" and voxel.prev_gamma is not None:
         # Norm of previous gamma
         norm_prev = np.sqrt(np.sum(voxel.prev_gamma**2))
         if norm_prev < 1e-12:
              modifiers = np.ones(voxel.M) # Fallback if prev is zero
         else:
              # VECTORIZED: Process all modes at once
              # catalog is already numpy array of shape (M, 3, 3)
              catalog_array = voxel.catalog  # Shape: (M, 3, 3)
              
              # Dot products: sum over last two dimensions
              # prev_gamma is (3, 3), broadcast to (1, 3, 3) for element-wise multiply
              dots = np.sum(catalog_array * voxel.prev_gamma[np.newaxis, :, :], axis=(1, 2))  # Shape: (M,)
              
              # Norms of current gammas
              norms_curr = np.sqrt(np.sum(catalog_array**2, axis=(1, 2)))  # Shape: (M,)
              
              # Cosine similarities
              cosines = dots / (norm_prev * norms_curr + 1e-12)  # Shape: (M,)
              
              # Square Forward Modifier: (1 + cos)^2 / 4
              # cos=1 -> mod=1 (Full softening)
              # cos=-1 -> mod=0 (No softening)
              # cos=0 -> mod=0.25 (Weak softening)
              modifiers = (1.0 + cosines)**2 / 4.0  # Shape: (M,)
    else:
         # Isotropic: Modifier is 1.0 for all directions
         modifiers = np.ones(voxel.M)

    # VECTORIZED: Calculate work and Q for all modes at once
    GPa_nm3_to_eV = 6.241509
    sigma_GPa = voxel.sigma / 1e9  # Convert Pa to GPa, shape (3, 3)
    
    # Work calculation: 0.5 * volume * sum(sigma_GPa * gamma) for each mode
    # catalog is (M, 3, 3), sigma_GPa is (3, 3)
    # Broadcast and sum over last two dimensions
    work = 0.5 * volume * np.sum(voxel.catalog * sigma_GPa[np.newaxis, :, :], axis=(1, 2)) * GPa_nm3_to_eV  # Shape: (M,)
    
    # Effective softening: g_eff = modifier * g_base for each mode
    g_eff = modifiers * g_base  # Shape: (M,)
    
    # Q calculation: Q[m] = Q0[m] * exp(-g_eff[m]) - work[m]
    Q = voxel.Q0 * np.exp(-g_eff) - work  # Shape: (M,)
    
    # Debug output for first mode if requested
    if debug:
        print(f"  [DEBUG] Mode 0: Q0={voxel.Q0[0]:.4f}, g_base={g_base:.4f} (gp={voxel.g_p:.4f}, gt={g_t_curr:.4f}), Mod={modifiers[0]:.4f}")
    
    voxel.Q = Q
    return Q
