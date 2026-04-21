"""
Example 03: Temperature and Rate Effects
========================================
Demonstrates running a simulation at a finite temperature using the KMC mode,
which activates the true Kinetic Monte Carlo random timing progression.
"""
import os
import yaml
import numpy as np
from mgkmc import ThermalSimulation

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    nx, ny, nz = 32, 32, 1
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)

    print("Initializing Finite-Temperature KMC Simulation...")
    sim = ThermalSimulation(
        nx, ny, nz,
        M=config['system']['M'],
        gamma0=config['system']['gamma0'],
        E_field=E,
        nu_field=nu,
        pixel=config['system']['pixel'],
        output_dir="output_temp",
        temperature=300.0,                   # Set to 300 K
        strain_rate=1e6,                     # Slower loading rate explicitly specified
        jp=config['physics']['jp'],
        jt=config['physics']['jt'],
        instability_mode="kmc",              # Finite temp explicitly operates via sequential KMC
        nu0=config['dynamics']['nu0']
    )

    print("Running...")
    sim.run_mixed(
        n_global_steps=100,
        strain_rate=1e6,
        component=(0, 1), # Pure shear
        stress_targets={(0,0): 0.0, (1,1): 0.0, (2,2): 0.0},
        vtk_interval=50,
        vtk_mode="kmc", # Export VTK specifically during any KMC event for fine temporal scale
        enable_kmc_log=True,
        enable_console_log=True,
        summary_filename="summary_temperature.txt"
    )
    
    print("Simulation Complete!")

if __name__ == "__main__":
    main()
