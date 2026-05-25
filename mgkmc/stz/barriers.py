import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def compute_barrier(Q_field, Q0_field, sigma_field, catalog, volume, 
                    soft_prop, last_event_time, current_time, 
                    prev_strain_dir, softening_cap,
                    softening_scheme_idx=0, tau=np.inf):
    """
    Compute barriers for ALL voxels in place.
    
    Parameters
    ----------
    Q_field : (nx, ny, nz, M) - Updated in place
    Q0_field : (nx, ny, nz, M)
    sigma_field : (nx, ny, nz, 3, 3)
    catalog : (nx, ny, nz, M, 3, 3)
    volume : float
    soft_prop : (nx, ny, nz, 4) - [g_p, g_t, unused, unused]
    last_event_time : (nx, ny, nz)
    current_time : float
    prev_strain_dir : (nx, ny, nz, 3, 3) - Normalized strain direction of last event
    softening_cap : float - e.g. 0.78
    softening_scheme_idx : int 0=isotropic, 1=directional
    tau : float
    """
    nx, ny, nz, M = Q_field.shape
    GPa_nm3_to_eV = 6.241509
    
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # 1. Transient Softening
                g_t = soft_prop[x,y,z,1]
                t_last = last_event_time[x,y,z]
                
                g_t_curr = 0.0
                if g_t > 0:
                    if tau == np.inf:
                        g_t_curr = g_t
                    elif t_last == -np.inf:
                        g_t_curr = 0.0
                    else:
                        dt = current_time - t_last
                        if dt < 0: dt = 0
                        g_t_curr = g_t * np.exp(-dt / tau)
                
                g_p = soft_prop[x,y,z,0]
                g_base = g_p + g_t_curr
                
                # Apply the maximum softening boundary (eta_max) to the total softening
                if softening_cap > 0 and g_base > softening_cap:
                    g_base = softening_cap
                    
                # 2. Iterate Modes
                # Using explicit loops for Numba speed on small arrays
                for m in range(M):
                    if softening_scheme_idx == 1: # Directional
                         # Legacy Logic: Modifier = (1 + cos_theta)^2 / 4
                         # We need dot product and norms of (catalog mode) vs (prev_strain_dir)
                         
                         dot_prod = 0.0
                         norm_mode_sq = 0.0
                         norm_prev_sq = 0.0
                         
                         for i in range(3):
                             for j in range(3):
                                 val_m = catalog[x,y,z,m,i,j]
                                 val_p = prev_strain_dir[x,y,z,i,j]
                                 
                                 dot_prod += val_m * val_p
                                 norm_mode_sq += val_m**2
                                 norm_prev_sq += val_p**2
                         
                         norm_prev = np.sqrt(norm_prev_sq)
                         norm_mode = np.sqrt(norm_mode_sq)
                         
                         if norm_prev < 1e-12 or norm_mode < 1e-12:
                             modifier = 1.0
                         else:
                             cos_theta = dot_prod / (norm_mode * norm_prev)
                             # Restored Formula
                             modifier = (1.0 + cos_theta)**2 / 4.0
                    else:
                        modifier = 1.0
                    
                    g_eff = modifier * g_base
                    
                    # Work: 0.5 * V * sum(sig * gamma)
                    # Contract sigma (3,3) with catalog[x,y,z,m] (3,3)
                    w_sum = 0.0
                    for i in range(3):
                        for j in range(3):
                             w_sum += sigma_field[x,y,z,i,j] * catalog[x,y,z,m,i,j]
                             
                    # Sigma is in Pa, need GPa
                    w_val = 0.5 * volume * (w_sum / 1e9) * GPa_nm3_to_eV
                    
                    # Q = Q0 * exp(-g) - W
                    Q_field[x,y,z,m] = Q0_field[x,y,z,m] * np.exp(-g_eff) - w_val

