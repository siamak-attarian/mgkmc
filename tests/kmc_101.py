import numpy as np

def analyze_kmc_sweeps():
    # Physics Constants
    kB = 8.617e-5  # eV/K
    nu0 = 1e13     # Hz
    volume = 1.0   # nm^3
    
    # Simulation Parameters
    N_voxels = 128*128*1
    M_modes = 20
    Total_Modes = N_voxels * M_modes
    
    T = 300.0
    beta = 1.0 / (kB * T)
    dt_elastic = 1.0 # arbitrary reference time step (seconds)
    
    QH_mean = 2.0 # eV
    
    print(f"============================================================")
    print(f"               KMC SWEEP ANALYSIS (T={T} K)                 ")
    print(f"============================================================")
    print(f"Total Modes: {Total_Modes:,}")
    print(f"Mean Barrier: {QH_mean} eV")
    print(f"Elastic dt Reference: {dt_elastic} s")
    print("-" * 60)
    
    # ---------------------------------------------------------
    # Sweep 1: Cutoff
    # ---------------------------------------------------------
    cutoffs = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9]
    std_fixed = 0.3
    
    print(f"\n--- SWEEP 1: BARRIER CUTOFF (Fixed Std = {std_fixed} eV) ---")
    print(f"Note: Barriers < Cutoff are set TO Cutoff (Clipping)")
    print(f"{'Cutoff (eV)':<12} | {'Total Rate (Hz)':<15} | {'Avg dt_kmc (s)':<15} | {'P(No Event)':<12}")
    print("-" * 65)
    
    np.random.seed(42)
    # Generate base distribution
    # Using float64 for precision with exponentials
    Q_base = np.random.normal(QH_mean, std_fixed, Total_Modes)
    
    for c in cutoffs:
        # Clip barriers: if Q < c, set Q = c
        Q_clipped = np.maximum(Q_base, c)
        
        rates = volume * nu0 * np.exp(-Q_clipped * beta)
        total_rate = np.sum(rates)
        
        if total_rate <= 0:
            dt_kmc = np.inf
            prob = 1.0
        else:
            dt_kmc = 1.0 / total_rate
            # P(No Event) = exp(-dt_elastic / dt_kmc) = exp(-R_tot * dt_elastic)
            prob = np.exp(-dt_elastic / dt_kmc)
            
        print(f"{c:<12.1f} | {total_rate:<15.2e} | {dt_kmc:<15.2e} | {prob:<12.2e}")

    # ---------------------------------------------------------
    # Sweep 2: Standard Deviation
    # ---------------------------------------------------------
    stds = [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]
    cutoff_fixed = 0.5
    
    print(f"\n--- SWEEP 2: DISORDER/STD (Fixed Cutoff = {cutoff_fixed} eV) ---")
    print(f"{'Std (eV)':<12} | {'Total Rate (Hz)':<15} | {'Avg dt_kmc (s)':<15} | {'P(No Event)':<12}")
    print("-" * 65)

    for s in stds:
        np.random.seed(42) # Reset seed for fair comparison across stds
        Q_base = np.random.normal(QH_mean, s, Total_Modes)
        
        # Clip
        Q_clipped = np.maximum(Q_base, cutoff_fixed)
        
        rates = volume * nu0 * np.exp(-Q_clipped * beta)
        total_rate = np.sum(rates)
        
        if total_rate <= 0:
            dt_kmc = np.inf
            prob = 1.0
        else:
            dt_kmc = 1.0 / total_rate
            prob = np.exp(-dt_elastic / dt_kmc)
            
        print(f"{s:<12.2f} | {total_rate:<15.2e} | {dt_kmc:<15.2e} | {prob:<12.2e}")

if __name__ == "__main__":
    analyze_kmc_sweeps()
