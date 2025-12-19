import numpy as np
import os
from mgkmc import AthermalSimulation
from mgkmc.stz.barriers import compute_barrier
from mgkmc.stz.kmc import compute_rates

def main():
    print("=" * 60)
    print("AQS INITIAL STATE DIAGNOSTIC")
    print("=" * 60)

    # 1. Setup Parameters (Matching Demo 6)
    SEED = 42
    np.random.seed(SEED)
    
    nx, ny, nz = 128, 128, 1
    # Material properties
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)
    
    pixel = 0.7
    M = 20
    gamma0 = 0.14
    
    TEMPERATURE = 250.0 # K
    STRAIN_RATE = 1e-4

    # Barrier Stats
    MEAN_Q = 2.0
    STD_Q = 0.6
    MIN_Q = 0.5

    print(f"Stats: Mean Q={MEAN_Q}, Std Q={STD_Q}, Min Q={MIN_Q}")
    print(f"Temp: {TEMPERATURE} K, Rate: {STRAIN_RATE} 1/s")

    def my_barrier_generator(n_modes):
        random_barriers = np.random.normal(loc=MEAN_Q, scale=STD_Q, size=n_modes)
        clipped_barriers = np.clip(random_barriers, a_min=MIN_Q, a_max=None)
        return clipped_barriers
    
    # 2. Initialize Simulation
    sim = AthermalSimulation(
        nx, ny, nz, M, gamma0, E, nu, pixel=pixel,
        barrier_generator=my_barrier_generator,
        temperature=TEMPERATURE,
        strain_rate=STRAIN_RATE,
        softening_enabled=True,
        softening_scheme="directional",
        softening_params={"jp": 11, "jt": 33},
        softening_cap=-np.log(0.4),
        solver_args={"tol": 1e-4, "max_iter": 50},
        output_dir="diagnostic_output"
    )

    print("\n[Diagnostic] Simulation Initialized.")
    print("[Diagnostic] Calculating Initial Barriers and Rates...")
    
    # Need to run one elastic solve to get stress?
    # Actually, initially stress is zero if we don't load.
    # But let's check what sim.sig_field is (should be zeros)
    sigma_max = np.abs(sim.sig_field).max()
    print(f"[Diagnostic] Initial Max Stress: {sigma_max:.4e} Pa")

    # 3. Compute Barriers for ALL voxels
    # We iterate and collect stats
    all_Q = []
    all_rates = []
    
    # Also collect detailed info for sorting
    # (x, y, z, m, Q, Rate)
    detailed_data = []

    # Constants
    k_B = 8.617e-5 # eV/K
    beta = 1.0 / (k_B * TEMPERATURE) if TEMPERATURE > 0 else np.inf
    nu0 = 1e13 # Attempt frequency

    print("[Diagnostic] Scanning grid...")
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                voxel = sim.grid[x,y,z] # z=0
            # Ensure barriers are set
            if not hasattr(voxel, 'Q') or voxel.Q is None:
                 compute_barrier(voxel, sim.volume, current_time=0.0, tau=np.inf)
            
            # Record Q values
            all_Q.extend(voxel.Q)

            # Compute rates directly to be sure
            # R = nu0 * exp(-beta * Q)
            qs = np.array(voxel.Q)
            rates = nu0 * np.exp(-beta * qs)
            all_rates.extend(rates)

            for m in range(M):
                detailed_data.append((x, y, 0, m, qs[m], rates[m]))

    all_Q = np.array(all_Q)
    all_rates = np.array(all_rates)
    
    # 4. Statistics
    print("\n" + "-"*40)
    print("INITIAL BARRIER STATISTICS")
    print("-"*40)
    print(f"Total Modes: {len(all_Q)}")
    print(f"Min Q:  {all_Q.min():.4f} eV")
    print(f"Max Q:  {all_Q.max():.4f} eV")
    print(f"Mean Q: {all_Q.mean():.4f} eV")
    print(f"Std Q:  {all_Q.std():.4f} eV")
    
    print("\n" + "-"*40)
    print("INITIAL RATE STATISTICS (T=250K)")
    print("-"*40)
    total_rate = np.sum(all_rates)
    print(f"Total System Rate (sum r_i): {total_rate:.4e} Hz")
    if total_rate > 0:
        dt_expected = 1.0 / total_rate
        print(f"Expected Time to First Event (1/R_tot): {dt_expected:.4e} s")
    else:
        print("Expected Time: Infinite (Rate=0)")
    
    # Elastic Step Comparison
    # Elastic step for d_strain ~ 1e-4 or so?
    # In Demo 6, we apply rate 1e-4. If target strain increment is say 1e-5 (for a step)
    # dt_elastic = d_eps / rate = 1e-5 / 1e-4 = 0.1 s.
    # Actually, in run_mixed, loop calculates dt_elastic based on strain_increment or similar.
    # Here just showing values.

    # 5. Top 10 Lowest Barriers / Highest Rates
    print("\n" + "-"*40)
    print("TOP 20 MOST ACTIVE MODES")
    print("-"*40)
    print(f"{'Rank':<5} {'Loc(x,y,z,m)':<15} {'Q (eV)':<10} {'Rate (Hz)':<12} {'Prob (%)':<10}")
    
    # Sort by Rate (Descending)
    dtype = [('x', int), ('y', int), ('z', int), ('m', int), ('Q', float), ('Rate', float)]
    structured_data = np.array(detailed_data, dtype=dtype)
    sorted_data = np.sort(structured_data, order='Rate')[::-1] # Descending

    for i in range(20):
        d = sorted_data[i]
        prob = (d['Rate'] / total_rate) * 100 if total_rate > 0 else 0
        loc_str = f"({d['x']},{d['y']},{d['z']},{d['m']})"
        print(f"{i+1:<5} {loc_str:<15} {d['Q']:.4f}     {d['Rate']:.4e}     {prob:.5f}")

if __name__ == "__main__":
    main()
