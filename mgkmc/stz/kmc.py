import numpy as np
from numba import jit

def compute_rates(Q_field, volume, temperature, nu0=1e13, instability_mode="cascade"):
    """
    Compute KMC rates for all modes (3D version).
    """
    nx, ny, nz, M = Q_field.shape
    if isinstance(temperature, (int, float, np.number)):
        temperature_arr = np.full((nx, ny, nz), float(temperature))
    else:
        temperature_arr = np.asarray(temperature, dtype=np.float64)
    return _compute_rates_jit(Q_field, volume, temperature_arr, nu0, instability_mode)

@jit(nopython=True, cache=True)
def _compute_rates_jit(Q_field, volume, temperature_arr, nu0=1e13, instability_mode="cascade"):
    """
    JIT core for computing KMC rates with local temperature array (3D).
    """
    nx, ny, nz, M = Q_field.shape
    kB = 8.617e-5
    
    # Pass 1: Count valid events
    count = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for m in range(M):
                    q = Q_field[x,y,z,m]
                    if instability_mode == "kmc":
                        # In KMC mode, we pick EVERYTHING (thermal and unstable)
                        count += 1
                    else:
                        # Classic mode: only thermal events (Q > 0)
                        if q > 0:
                            count += 1
    
    # Pass 2: Fill
    rates = np.empty(count, dtype=np.float64)
    indices = np.empty(count, dtype=np.int64)
    
    k = 0
    total_rate = 0.0
    
    for x in range(nx):
        for y in range(ny):
             for z in range(nz):
                 T = temperature_arr[x, y, z]
                 beta = 1.0 / (kB * T) if T > 0.0 else np.inf
                 for m in range(M):
                     q = Q_field[x,y,z,m]
                     
                     include = False
                     if instability_mode == "kmc":
                         include = True
                     elif q > 0:
                         include = True
                     
                     if include:
                         if q <= 0:
                             # Unstable or marginally stable: Rate = nu0
                             # This caps the rate and avoids overflow
                             r = volume * nu0
                         else:
                             # Thermal event: Standard Arrhenius
                             r = volume * nu0 * np.exp(-q * beta)
                         
                         rates[k] = r
                         total_rate += r
                         # Encode index
                         indices[k] = ((x * ny + y) * nz + z) * M + m
                         k += 1
                          
    return rates, indices, total_rate

@jit(nopython=True, cache=True)
def select_event(rates, total_rate):
    """
    Select event index from rates array.
    """
    if total_rate <= 0:
        return -1
        
    r = np.random.uniform(0, total_rate)
    
    current_sum = 0.0
    for i in range(len(rates)):
        current_sum += rates[i]
        if current_sum >= r:
            return i
            
    return len(rates)-1

@jit(nopython=True, cache=True)
def decode_index(flat_idx, ny, nz, M):
    """Decode flat index back to x, y, z, m"""
    m = flat_idx % M
    temp = flat_idx // M
    z = temp % nz
    temp = temp // nz
    y = temp % ny
    x = temp // ny
    return x, y, z, m
