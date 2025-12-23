import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def compute_rates(Q_field, volume, temperature, nu0=1e13):
    """
    Compute KMC rates for all modes using Numba.
    Returns:
       rates_flat: 1D array of all valid rates (Q>0)
       indices_flat: 1D array of encoded indices (x*Ny*Nz*M + ...) or mapped?
       
    For rejection-free KMC, we need a cumulative sum.
    
    Strategy:
    1. Calculate rate for every mode.
    2. Store non-zero rates in a pre-allocated buffer?
    
    Or return dense arrays and let Python filter? 
    Python filtering is slow.
    
    Better Strategy:
    Return `rates` of shape (Nx, Ny, Nz, M).
    Let caller flatten? Or flatten inside?
    
    Flattening inside Numba is fast.
    """
    nx, ny, nz, M = Q_field.shape
    kB = 8.617e-5
    beta = 1.0 / (kB * temperature) if temperature > 0 else 0.0
    
    # We'll return everything and let selection handle zeros?
    # No, selection needs compact array.
    
    # Pass 1: Count valid events (Q > 0)
    count = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for m in range(M):
                    if Q_field[x,y,z,m] > 0:
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
                     if q > 0:
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
        return -1, np.inf
        
    r = np.random.uniform(0, total_rate)
    
    current_sum = 0.0
    for i in range(len(rates)):
        current_sum += rates[i]
        if current_sum >= r:
            dt = 1.0 / total_rate
            return i, dt
            
    return len(rates)-1, 1.0/total_rate

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
