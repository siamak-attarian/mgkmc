import numpy as np

def compute_rates(grid, volume, temperature, nu0=1e13, strain_rate_sensitivity=0.0, 
                  applied_strain_rate=1.0, current_time=0.0):
    """
    Compute KMC rates for all STZ modes in the grid.
    
    Formula: rate = volume * nu0 * exp( -Q / (kB * T) )
    
    Parameters
    ----------
    grid : np.ndarray (nx, ny, nz) of Voxel objects
    volume : float
        Voxel volume (nm^3)
    temperature : float
        Temperature (Kelvin)
    nu0 : float
        Attempt frequency (Hz)
    strain_rate_sensitivity : float
        Sensitivity exponent 's'.
    applied_strain_rate : float
        Macroscopic strain rate (1/s).
    current_time : float
        Current simulation time (s).
        
    Returns
    -------
    rates : np.ndarray
        Flat array of rates for all modes in all voxels.
    map_indices : list
        List of (x, y, z, m) corresponding to the flat rates array.
    total_rate : float
        Sum of all rates.
    """
    kB = 8.617e-5 # eV/K
    
    rates = []
    map_indices = []
    
    # We iterate systematically
    nx, ny, nz = grid.shape
    
    # Pre-calculate thermal factor
    beta = 1.0 / (kB * temperature) if temperature > 0 else np.inf
    
    # Optimization: using list append is safe. 
    # For very large grids, we might want to vectorize, but prompt says "Use simple Python loops first".
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                voxel = grid[x,y,z]
                # Q should already be computed and stored in voxel.Q
                # However, if using strain_rate_sensitivity, Q needs modification.
                # But Q depends on 'rate_factor' which depends on local strain rate.
                # Local strain rate depends on (time - last_event_time).
                
                # Retrieve base Q (computed by barriers.py)
                # We assume voxel.Q is up to date relative to the current stress state.
                # Note: voxel.Q comes from compute_barrier() which sets Q = Q0*exp(-g)-W.
                
                # Filter modes with Q <= 0 (Athermal)
                # These should be handled by cascade, but if they exist here, we ignore them according to instructions.
                # "Ignore modes where Q <= 0 (those are athermal)."
                
                for m in range(voxel.M):
                    Q_val = voxel.Q[m]
                    
                    if Q_val <= 0:
                        continue
                        
                    # 5. STRAIN RATE SENSITIVITY
                    if strain_rate_sensitivity > 0:
                         # rate_factor = 1 + (local_strain / (delta_t * applied_strain_rate))
                         # local_eps = sqrt(eps_xx^2 + eps_yy^2 + 2*eps_xy^2)
                         # We need local epsilon. 'voxel' object needs to store it?
                         # Voxel object tracks plastic strain 'eps_plastic'. 
                         # But 'local_eps' usually refers to total local strain or plastic strain increment?
                         # Prompt: "Compute local equivalent strain: local_eps = ..."
                         # ... "Compute local strain rate estimate: local_rate = local_eps / (current_time - last_event_time + eps)"
                         # This implies tracking 'last_event_time' per voxel.
                         
                         local_eps_tensor = voxel.eps_plastic # This is cumulative plastic strain?
                         # Or is it local TOTAL strain?
                         # Usually in STZ: 'local_strain' driving the rate might be the accumulated strain since last reset?
                         # "local_eps = sqrt(eps_xx^2 + ...)" looks like Invariant.
                         # Assuming 'voxel.eps_plastic' is what is meant, or the elastic+plastic?
                         # Given "rate_factor = 1 + local_rate / applied_strain_rate", it acts as a viscosity.
                         # If the voxel hasn't flipped in a long time, 'local_rate' is low.
                         
                         # Let's verify what 'local_eps' implies.
                         # "local_eps = sqrt(eps_xx^2 + eps_yy^2 + 2*eps_xy^2)"
                         # If this is computed on the plastic strain tensor.
                         e = voxel.eps_plastic
                         local_dstrain = np.sqrt(e[0,0]**2 + e[1,1]**2 + 2*e[0,1]**2) # Simplified 2D-like or full 3D trace?
                         # The formula given is 2D-ish (xx, yy, xy).
                         # Let's assume full tensor norm: sqrt(0.5 * sum(eps_ij * eps_ij)) or similar?
                         # Prompt formula: "sqrt(eps_xx^2 + eps_yy^2 + 2*eps_xy^2)". 
                         # This looks like the effective shear strain in 2D.
                         # I will implement it as given.
                         
                         delta_t = current_time - getattr(voxel, 'last_event_time', 0.0) + 1e-12
                         local_rate = local_dstrain / delta_t
                         
                         rate_factor = 1.0 + (local_rate / applied_strain_rate)
                         
                         # Modify barrier
                         # Q = Q * (rate_factor ** s)
                         Q_val = Q_val * (rate_factor ** strain_rate_sensitivity)

                    # Compute Rate
                    r = volume * nu0 * np.exp( -Q_val * beta )
                    rates.append(r)
                    map_indices.append((x, y, z, m))

    return np.array(rates), map_indices, np.sum(rates)

def select_event(rates, total_rate):
    """
    Select an event using roulette wheel selection.
    
    Parameters
    ----------
    rates : np.ndarray
        Array of rates.
    total_rate : float
        Sum of rates.
        
    Returns
    -------
    index : int
        Index into the rates array (and map_indices).
    dt_kmc : float
        Mean residence time (1/total_rate).
    """
    if total_rate <= 0:
        return None, float('inf')
        
    r = np.random.uniform(0, total_rate)
    
    # Find first event where cumulative >= r
    cumulative = np.cumsum(rates)
    idx = np.searchsorted(cumulative, r)
    
    # Safety check
    if idx >= len(rates):
        idx = len(rates) - 1
        
    # Deterministic Mean Residence Time (1/Rate)
    dt_kmc = 1.0 / total_rate
    print('Total rate:', total_rate)
    return idx, dt_kmc
