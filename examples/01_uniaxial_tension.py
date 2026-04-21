"""
Example 01: Uniaxial Tension
============================
Simulates a metallic glass sample under uniaxial tension along the x-axis.
"""
import os
import yaml
import numpy as np
from mgkmc import ThermalSimulation

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Base parameters
    nx = 32
    ny = 32
    nz = 1
    
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)

    print("Initializing Uniaxial Tension Simulation...")
    sim = ThermalSimulation(
        nx, ny, nz,
        M=config['system']['M'],
        gamma0=config['system']['gamma0'],
        E_field=E,
        nu_field=nu,
        pixel=config['system']['pixel'],
        output_dir="output_uniaxial",
        temperature=0.0,                     # Athermal quasi-static
        strain_rate=1e7,
        jp=config['physics']['jp'],
        jt=config['physics']['jt'],
        softening_scheme="directional"
    )

    print("Running...")
    # Uniaxial Tension: we drive strain in the xx component (0,0)
    # and require yy (1,1) and zz (2,2) stresses to rigidly be 0.0 GPa
    sim.run_mixed(
        n_global_steps=200,
        strain_rate=1e7,
        component=(0, 0),         
        stress_targets={(1, 1): 0.0, (2, 2): 0.0},
        vtk_interval=50,
        vtk_mode="elastic",
        enable_console_log=True,
        summary_filename="summary_uniaxial.txt"
    )
    
    print("Simulation Complete!")

if __name__ == "__main__":
    main()
