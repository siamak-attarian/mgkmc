import numpy as np
from numba import jit
from scipy.stats import rayleigh

def compute_rates_2d(Q_field, volume, temperature, nu0=1e13, instability_mode="cascade"):
    """
    Compute KMC rates for all modes (2D version).
    """
    nx, ny, M = Q_field.shape
    if isinstance(temperature, (int, float, np.number)):
        temperature_arr = np.full((nx, ny), float(temperature))
    else:
        temperature_arr = np.asarray(temperature, dtype=np.float64)
    return _compute_rates_2d_jit(Q_field, volume, temperature_arr, nu0, instability_mode)

@jit(nopython=True, cache=True)
def _compute_rates_2d_jit(Q_field, volume, temperature_arr, nu0=1e13, instability_mode="cascade"):
    """
    JIT core for computing KMC rates with local temperature array.
    """
    nx, ny, M = Q_field.shape
    kB = 8.617e-5
    
    count = 0
    for x in range(nx):
        for y in range(ny):
            for m in range(M):
                q = Q_field[x,y,m]
                if instability_mode == "kmc" or q > 0:
                    count += 1
    
    rates = np.empty(count, dtype=np.float64)
    indices = np.empty(count, dtype=np.int64)
    
    k = 0
    total_rate = 0.0
    for x in range(nx):
        for y in range(ny):
            T = temperature_arr[x, y]
            beta = 1.0 / (kB * T) if T > 0.0 else np.inf
            for m in range(M):
                q = Q_field[x,y,m]
                include = (instability_mode == "kmc" or q > 0)
                
                if include:
                    if q <= 0:
                        r = volume * nu0
                    else:
                        r = volume * nu0 * np.exp(-q * beta)
                    
                    rates[k] = r
                    total_rate += r
                    indices[k] = (x * ny + y) * M + m
                    k += 1
                           
    return rates, indices, total_rate

@jit(nopython=True, cache=True)
def select_event_2d(rates, total_rate):
    """Select event index from rates array."""
    if total_rate <= 0: return -1
    r = np.random.uniform(0, total_rate)
    current_sum = 0.0
    for i in range(len(rates)):
        current_sum += rates[i]
        if current_sum >= r: return i
    return len(rates)-1

@jit(nopython=True, cache=True)
def decode_index_2d(flat_idx, ny, M):
    """Decode flat index back to x, y, m for 2D."""
    m = flat_idx % M
    temp = flat_idx // M
    y = temp % ny
    x = temp // ny
    return x, y, m

def stz_catalog_glass_2d(M, gamma0, stz_mode="simple_shear"):
    """Generate M independent 2x2 STZ modes for 2D.
    If stz_mode is 'pure_shear', generates traditional pure shear (causes exponential scaling in finite strain).
    If stz_mode is 'simple_shear', generates simple shear mapping with the same symmetric part.
    """
    catalog = np.zeros((M, 2, 2))
    for i in range(M):
        gxx = 0.5 * gamma0 * np.random.normal()
        gxy = 0.5 * gamma0 * np.random.normal()
        if stz_mode == "pure_shear":
            catalog[i, 0, 0] = gxx
            catalog[i, 1, 1] = -gxx
            catalog[i, 0, 1] = catalog[i, 1, 0] = gxy
        else:
            R = np.sqrt(gxx**2 + gxy**2)
            sign = 1.0 if np.random.rand() > 0.5 else -1.0
            catalog[i, 0, 0] = gxx
            catalog[i, 1, 1] = -gxx
            catalog[i, 0, 1] = gxy + sign * R
            catalog[i, 1, 0] = gxy - sign * R
    return catalog

@jit(nopython=True, cache=True)
def compute_barrier_2d(Q_field, Q0_field, sigma_field, catalog, volume, 
                       soft_prop, last_event_time, current_time, 
                       prev_strain_dir, softening_cap,
                       softening_scheme_idx=0, tau=np.inf):
    """
    Compute barriers for all 2D modes in place.
    """
    nx, ny, M = Q_field.shape
    GPa_nm3_to_eV = 6.241509
    
    for x in range(nx):
        for y in range(ny):
            g_t = soft_prop[x,y,1]
            t_last = last_event_time[x,y]
            
            g_t_curr = 0.0
            if g_t > 0:
                if tau == np.inf: g_t_curr = g_t
                elif t_last == -np.inf: g_t_curr = 0.0
                else:
                    dt = current_time - t_last
                    if dt < 0: dt = 0
                    g_t_curr = g_t * np.exp(-dt / tau)
            
            g_p = soft_prop[x,y,0]
            g_base = g_p + g_t_curr
            if softening_cap > 0 and g_base > softening_cap:
                g_base = softening_cap
            
            for m in range(M):
                modifier = 1.0
                if softening_scheme_idx == 1: # Directional
                    dot_prod = 0.0
                    norm_mode_sq = 0.0
                    norm_prev_sq = 0.0
                    for i in range(2):
                        for j in range(2):
                            val_m = catalog[x,y,m,i,j]
                            val_p = prev_strain_dir[x,y,i,j]
                            dot_prod += val_m * val_p
                            norm_mode_sq += val_m**2
                            norm_prev_sq += val_p**2
                    
                    norm_prev = np.sqrt(norm_prev_sq)
                    norm_mode = np.sqrt(norm_mode_sq)
                    if norm_prev < 1e-12 or norm_mode < 1e-12:
                        modifier = 1.0
                    else:
                        cos_theta = dot_prod / (norm_mode * norm_prev)
                        modifier = (1.0 + cos_theta)**2 / 4.0
                
                g_eff = modifier * g_base
                
                w_sum = 0.0
                for i in range(2):
                    for j in range(2):
                        w_sum += sigma_field[x,y,i,j] * catalog[x,y,m,i,j]
                
                w_val = 0.5 * volume * (w_sum / 1e9) * GPa_nm3_to_eV
                Q_field[x,y,m] = Q0_field[x,y,m] * np.exp(-g_eff) - w_val

@jit(nopython=True, cache=True)
def find_unstable_2d(Q_field, threshold=0.0):
    """Find all unstable modes in 2D grid."""
    nx, ny, M = Q_field.shape
    count = 0
    for x in range(nx):
        for y in range(ny):
            for m in range(M):
                if Q_field[x,y,m] < threshold:
                    count += 1
    
    out = np.empty((count, 3), dtype=np.int64)
    k = 0
    for x in range(nx):
        for y in range(ny):
            for m in range(M):
                if Q_field[x,y,m] < threshold:
                    out[k, 0], out[k, 1], out[k, 2] = x, y, m
                    k += 1
    return out

@jit(nopython=True, cache=True)
def apply_flip_soa_2d(eps_plastic_field, soft_prop_field, last_event_time_field, 
                      catalog, x, y, m, 
                      current_time, jp, jt, g_max, jn_frac):
    """Apply flip logic to 2D state arrays."""
    nx, ny = eps_plastic_field.shape[:2]

    # 1. Update Plastic Strain
    for i in range(2):
        for j in range(2):
            eps_plastic_field[x,y,i,j] += catalog[x,y,m,i,j]
            
    # 2. Update Softening (2D VM equivalent strain)
    e11, e22, e12 = catalog[x,y,m,0,0], catalog[x,y,m,1,1], catalog[x,y,m,0,1]
    # sum_sq = (1/6)*((e22-0)^2 + (0-e11)^2 + (e11-e22)^2) + e12^2
    sum_sq = (e12**2) + (e22**2 + e11**2 + (e11 - e22)**2) / 6.0
    
    gp_new = soft_prop_field[x,y,0] + jp * sum_sq
    if g_max > 0 and gp_new > g_max: gp_new = g_max
    soft_prop_field[x,y,0] = gp_new
    soft_prop_field[x,y,1] = jt * sum_sq 
    last_event_time_field[x,y] = current_time

    # 3. Neighbor Softening (2D neighbors: 8-connectivity)
    if jn_frac > 0.0:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                nx_n, ny_n = (x + dx + nx) % nx, (y + dy + ny) % ny
                
                gp_n = soft_prop_field[nx_n, ny_n, 0] + jn_frac * jp * sum_sq
                if g_max > 0 and gp_n > g_max: gp_n = g_max
                soft_prop_field[nx_n, ny_n, 0] = gp_n
                
                # Transient softening for neighbors
                soft_prop_field[nx_n, ny_n, 1] += jn_frac * jt * sum_sq

# --- 2D Barrier Generators ---

def get_gaussian_generator_2d(mean=2.0, std=0.6, min_cutoff=0.1, max_cutoff=None):
    """Gaussian barrier generator for 2D grids."""
    def generator(shape):
        barriers = np.random.normal(loc=mean, scale=std, size=shape)
        if min_cutoff is not None or max_cutoff is not None:
            barriers = np.clip(barriers, a_min=min_cutoff, a_max=max_cutoff)
        return barriers
    return generator

def get_rayleigh_generator_2d(loc=1.0, scale=0.5, min_cutoff=0.1, max_cutoff=None):
    """Rayleigh barrier generator for 2D grids."""
    def generator(shape):
        barriers = rayleigh(loc=loc, scale=scale).rvs(size=shape)
        if min_cutoff is not None or max_cutoff is not None:
            barriers = np.clip(barriers, a_min=min_cutoff, a_max=max_cutoff)
        return barriers
    return generator

def get_modified_rayleigh_generator_2d(mean=2.0, std=0.6, min_cutoff=0.1, max_cutoff=None):
    """Modified Rayleigh barrier generator for 2D grids: f(x) ~ x * exp(-(x-m)^2/(2s^2))"""
    def generator(shape):
        if isinstance(shape, int):
            N = shape
        else:
            N = np.prod(np.array(shape))
            
        samples = np.empty(N)
        n_accepted = 0
        a = max(0.0, mean - 6*std)
        b = mean + 6*std
        x_max = (mean + np.sqrt(mean**2 + 4*std**2)) / 2.0
        c_max = x_max * np.exp(-(x_max - mean)**2 / (2 * std**2))
        
        while n_accepted < N:
            n_needed = N - n_accepted
            n_propose = max(int(n_needed * 3), 100)
            x_prop = np.random.uniform(a, b, n_propose)
            u_prop = np.random.uniform(0, c_max, n_propose)
            pdf_prop = x_prop * np.exp(-(x_prop - mean)**2 / (2 * std**2))
            accepted = x_prop[u_prop < pdf_prop]
            num_acc = len(accepted)
            if num_acc > 0:
                take = min(num_acc, n_needed)
                samples[n_accepted:n_accepted+take] = accepted[:take]
                n_accepted += take
                
        barriers = samples.reshape(shape)
        if min_cutoff is not None or max_cutoff is not None:
            barriers = np.clip(barriers, a_min=min_cutoff, a_max=max_cutoff)
        return barriers
    return generator

def get_modified_rayleigh_with_exponential_generator_2d(mean=2.0, std=0.6, epsilon=0.1, ratio=0.8, min_cutoff=0.1, max_cutoff=None):
    """Mixture of Modified Rayleigh and Exponential barrier generator for 2D grids."""
    def generator(shape):
        if isinstance(shape, int):
            N = shape
        else:
            N = np.prod(np.array(shape))
            
        barriers = np.empty(N)
        u_mix = np.random.uniform(0.0, 1.0, N)
        mask_exp = u_mix < ratio
        n_ray = np.sum(~mask_exp)
        
        if n_ray > 0:
            m_ray_gen = get_modified_rayleigh_generator_2d(mean, std, min_cutoff, max_cutoff)
            barriers[~mask_exp] = m_ray_gen(n_ray).flatten()
        
        n_exp = N - n_ray
        if n_exp > 0:
            barriers[mask_exp] = np.random.exponential(scale=epsilon, size=n_exp)
            
        barriers = barriers.reshape(shape)
        if min_cutoff is not None or max_cutoff is not None:
            barriers = np.clip(barriers, a_min=min_cutoff, a_max=max_cutoff)
        return barriers
    return generator

def get_barrier_generator_2d(generator_type, **kwargs):
    """Factory function to get a 2D barrier generator by type."""
    gtype = generator_type.lower()
    if gtype == "gaussian":
        return get_gaussian_generator_2d(**kwargs)
    elif gtype == "rayleigh":
        return get_rayleigh_generator_2d(**kwargs)
    elif gtype == "modified_rayleigh":
        return get_modified_rayleigh_generator_2d(**kwargs)
    elif gtype == "modified_rayleigh_with_exponential":
        return get_modified_rayleigh_with_exponential_generator_2d(**kwargs)
    else:
        raise ValueError(f"Unknown 2D barrier generator type: {generator_type}")
