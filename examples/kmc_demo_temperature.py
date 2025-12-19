import numpy as np
import matplotlib.pyplot as plt
import os
from mgkmc.aqs import AthermalSimulation
from mgkmc.stz.catalog import stz_catalog_glass

def run_simulation(T, output_dir):
    print(f"--- Running Simulation at T={T}K ---")
    nx, ny, nz = 16, 16, 16  # Slightly larger for better averaging
    M = 20
    gamma0 = 0.1
    
    # Material properties
    E = np.ones((nx, ny, nz)) * 70.0 # GPa
    nu = np.ones((nx, ny, nz)) * 0.3
    
    # Setup Simulation
    # Use slower strain rate for Thermal case to allow KMC to happen
    # T=300K, Barrier ~ 0.5-1.0eV.
    # nu0 = 1e13.
    # If Q=0.5eV, rate ~ 1e13 * exp(-20) ~ 2e4 Hz. 
    # If strain_rate = 1e4, dt_elastic = 1e-4. KMC happens frequently.
    # If strain_rate = 1.0, dt_elastic = 1.0. KMC happens VERY frequently.
    rate = 1.0 
    
    sim = AthermalSimulation(nx, ny, nz, M, gamma0, E, nu, 
                             output_dir=output_dir,
                             temperature=T,
                             strain_rate=rate,
                             softening_enabled=True, # Enable softening for realistic curves
                             softening_params={"jp": 50, "jt": 100})
    
    # Run slightly past yield
    strain_inc = np.zeros((3,3))
    dt_strain = 0.001 # 0.1% per step
    strain_inc[0,1] = dt_strain
    strain_inc[1,0] = dt_strain # Simple Shear
    
    n_steps = 40 # Total 4% strain (Enough to see yield)
    
    # We want to collect stress-strain data
    # The simulation logs to global_log.txt, we can read that later or instrument custom logging here
    # Accessing history directly from sim object is easier if we added it (we did!)
    
    sim.run(n_global_steps=n_steps, strain_increment_tensor=strain_inc)
    
    return sim.history_global # List of (eps_xx, sig_xx) - wait, we applied xy shear.
    # The history_global in aqs.py logs (eps[0,0], sig[0,0]). 
    # We need Shear stress!
    # Let's fix log reading or just extract from history if it stored what we need.
    # aqs.py: self.history_global.append((eps_macro_curr[0,0], sig_macro_curr[0,0]/1e9))
    # It hardcoded [0,0] (xx component).
    # We need to parse the log file for full tensor or modify AQS to store what we want.
    # Easier to parse log for this demo.

def parse_log(log_path):
    # Log format: Step Eps_xx ... Eps_yz Sig_xx ... Sig_yz ...
    #  0     1      2      3      4      5      6      7       8     9      10     11     12
    # Step, Exx, Eyy, Ezz, Exy, Exz, Eyz, Sxx, Syy, Szz, Sxy, Sxz, Syz
    
    data = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines[1:]: # Skip header
        parts = line.split()
        if not parts: continue
        # Exy is index 4 (0-based from conversion) -> actually column 4 in values
        # Step is col 0.
        # Exx(1), Eyy(2), Ezz(3), Exy(4)
        # Sxy is index 10.
        
        exy = float(parts[4])
        sxy = float(parts[10]) / 1e9 # GPa
        data.append((exy, sxy))
        
    return np.array(data)

# Run T=0
out_t0 = "demo_output_t0"
run_simulation(0.0, out_t0)
data_t0 = parse_log(os.path.join(out_t0, "global_log.txt"))

# Run T=300
out_t300 = "demo_output_t300"
run_simulation(300.0, out_t300)
data_t300 = parse_log(os.path.join(out_t300, "global_log.txt"))

# Run T=600 (Enhanced softening)
out_t600 = "demo_output_t600"
run_simulation(600.0, out_t600)
data_t600 = parse_log(os.path.join(out_t600, "global_log.txt"))

# Plot
plt.figure(figsize=(8, 6))
plt.plot(data_t0[:,0], data_t0[:,1], label="T=0K (Athermal)", linewidth=2)
plt.plot(data_t300[:,0], data_t300[:,1], label="T=300K", linewidth=2)
plt.plot(data_t600[:,0], data_t600[:,1], label="T=600K", linewidth=2)

plt.xlabel("Shear Strain")
plt.ylabel("Shear Stress (GPa)")
plt.title("Temperature-Dependent Yielding (KMC)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("kmc_temperature_comparison.png")
print("Plot saved to kmc_temperature_comparison.png")
