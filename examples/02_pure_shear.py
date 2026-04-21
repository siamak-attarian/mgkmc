"""
Example 02: Pure Shear
======================
Simulates a metallic glass sample under pure shear (driving eps_xy)
with other macroscopic stress components fully relaxed to 0.
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

    print("Initializing Pure Shear Simulation...")
    sim = ThermalSimulation(
        nx, ny, nz,
        M=config['system']['M'],
        gamma0=config['system']['gamma0'],
        E_field=E,
        nu_field=nu,
        pixel=config['system']['pixel'],
        output_dir="output_shear",
        temperature=0.0,
        strain_rate=1e7,
        jp=config['physics']['jp'],
        jt=config['physics']['jt']
    )

    print("Running...")
    # Pure shear: drive the xy component (0,1)
    # The solver will iteratively relax normal stress components to exactly zero (or custom targets)
    sim.run_mixed(
        n_global_steps=200,
        strain_rate=1e7,
        component=(0, 1),
        stress_targets={(0,0): 0.0, (1,1): 0.0, (2,2): 0.0}, 
        vtk_interval=50,
        vtk_mode="elastic",
        enable_console_log=True,
        summary_filename="summary_shear.txt"
    )
    
    print("Simulation Complete!")

if __name__ == "__main__":
    main()
