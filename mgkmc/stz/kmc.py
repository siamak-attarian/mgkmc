import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def compute_rates(Q_field, volume, temperature, nu0=1e13, instability_mode="cascade"):
    """
    Compute KMC rates for all modes using Numba.
    Returns:
       rates_flat: 1D array of all valid rates
       indices_flat: 1D array of encoded indices
       total_rate: sum of all rates
    """
    nx, ny, nz, M = Q_field.shape
    kB = 8.617e-5
    beta = 1.0 / (kB * temperature) if temperature > 0 else 0.0
    
    # We'll return everything and let selection handle zeros?
    # No, selection needs compact array.
    
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
    # We need to map back to (x,y,z,m). 
    # Storing 4 ints per event is expensive? 
    # Use encoded index: idx = ((x*ny + y)*nz + z)*M + m
    indices = np.empty(count, dtype=np.int64)
    
    k = 0
    total_rate = 0.0
    
    for x in range(nx):
        for y in range(ny):
             for z in range(nz):
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
