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
                   current_time, jp, jt, g_max, jn_frac):
    """
    Apply flip to SoA arrays.
    """
    nx = eps_plastic_field.shape[0]
    ny = eps_plastic_field.shape[1]
    nz = eps_plastic_field.shape[2]

    # 1. Update Plastic Strain
    # eps += gamma
    for i in range(3):
        for j in range(3):
            eps_plastic_field[x,y,z,i,j] += catalog[x,y,z,m,i,j]
            
     # 2. Update Softening (Von Mises equivalent strain squared)
    e11 = catalog[x,y,z,m,0,0]
    e22 = catalog[x,y,z,m,1,1]
    e33 = catalog[x,y,z,m,2,2]
    e12 = catalog[x,y,z,m,0,1]
    e13 = catalog[x,y,z,m,0,2]
    e23 = catalog[x,y,z,m,1,2]

    # Eq (18) from the 2013 paper
    sum_sq = (e12**2 + e23**2 + e13**2) + \
             ((e22 - e33)**2 + (e33 - e11)**2 + (e11 - e22)**2) / 6.0
    
    gp = soft_prop_field[x,y,z,0]
    
    # g_p += jp * sum_sq (Consistent with C code: PermSoftening += DeltaSoftening * PermSoft)
    gp_new = gp + jp * sum_sq
    
    # Clamp g_p if cap is set
    if g_max > 0 and gp_new > g_max:
        gp_new = g_max
        
    soft_prop_field[x,y,z,0] = gp_new
    
    # g_t = jt * sum_sq (Consistent with C code: TempSoftening = DeltaSoftening * TempSoft)
    gt_new = jt * sum_sq
    soft_prop_field[x,y,z,1] = gt_new
    
    # 3. Update Timestamp
    last_event_time_field[x,y,z] = current_time

    # 4. Neighbor Softening
    if jn_frac > 0.0:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx_n = (x + dx + nx) % nx
                    ny_n = (y + dy + ny) % ny
                    nz_n = (z + dz + nz) % nz
                    
                    # Permanent Softening
                    gp_n = soft_prop_field[nx_n, ny_n, nz_n, 0]
                    gp_n_new = gp_n + jn_frac * jp * sum_sq
                    if g_max > 0 and gp_n_new > g_max:
                        gp_n_new = g_max
                    soft_prop_field[nx_n, ny_n, nz_n, 0] = gp_n_new
                    
                    # Transient Softening (Additive for neighbors)
                    gt_n = soft_prop_field[nx_n, ny_n, nz_n, 1]
                    soft_prop_field[nx_n, ny_n, nz_n, 1] = gt_n + jn_frac * jt * sum_sq


