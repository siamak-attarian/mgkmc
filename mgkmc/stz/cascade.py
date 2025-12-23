import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def find_unstable(Q_field, threshold=0.0):
    """
    Find all unstable modes (Q < threshold).
    Returns list of (x, y, z, m) tuples? 
    Numba doesn't like lists of tuples. 
    Return (N, 4) array.
    """
    nx, ny, nz, M = Q_field.shape
    
    # Pass 1: Count
    count = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Only check minimum Q? Or all unstable modes?
                # AQS usually flips the MOST unstable or just the first one encountered?
                # Usually we check min(Q) per voxel? Or just any Q < 0?
                # Let's check all modes.
                for m in range(M):
                    if Q_field[x,y,z,m] < threshold:
                        count += 1
                        
    # Pass 2: Fill
    # Flattened array [x, y, z, m]
    out = np.empty((count, 4), dtype=np.int64)
    k = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for m in range(M):
                    if Q_field[x,y,z,m] < threshold:
                        out[k, 0] = x
                        out[k, 1] = y
                        out[k, 2] = z
                        out[k, 3] = m
                        k += 1
    return out

@jit(nopython=True, cache=True)
def apply_flip_soa(eps_plastic_field, key_field, soft_prop_field, last_event_time_field, 
                   catalog, x, y, z, m, 
                   current_time, jp, jt, g_max):
    """
    Apply flip to SoA arrays.
    """
    # 1. Update Plastic Strain
    # eps += gamma
    for i in range(3):
        for j in range(3):
            eps_plastic_field[x,y,z,i,j] += catalog[x,y,z,m,i,j]
            
    # 2. Update Softening (Legacy Logic)
    # Calculate Von Mises of the specific mode m
    # vm = sqrt(1.5 * sum(gamma_ij^2))
    sum_sq = 0.0
    for i in range(3):
        for j in range(3):
            sum_sq += catalog[x,y,z,m,i,j]**2
    vm = np.sqrt(1.5 * sum_sq)
    
    gp = soft_prop_field[x,y,z,0]
    
    # g_p += jp * vm^2
    gp_new = gp + jp * vm**2
    
    # Clamp g_p if cap is set
    if g_max > 0 and gp_new > g_max:
        gp_new = g_max
        
    soft_prop_field[x,y,z,0] = gp_new
    
    # g_t = jt * vm^2 (Set to peak value, decays later)
    # Note: legacy code set g_t = jt * vm**2 (not additive?)
    # "voxel.g_t = jt * vm**2" -> It seems it resets or sets the magnitude?
    # Usually transient heating is additive? 
    # Checking legacy again: "voxel.g_t = jt * vm**2" -> It is assignment (=), not (+=).
    # But usually heat accumulates? 
    # Wait, if I flip again immediately, do I lose previous heat?
    # The legacy code clearly wrote: voxel.g_t = jt * vm**2
    # So we will follow that strictly.
    gt_new = jt * vm**2
    soft_prop_field[x,y,z,1] = gt_new
    
    # 3. Update Timestamp
    last_event_time_field[x,y,z] = current_time


