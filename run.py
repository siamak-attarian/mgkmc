"""
MGKMC Main Execution Script
===========================

This script acts as the primary entry point for launching simulations.
It parses `config.yaml` for system, material, physics, boundary conditions,
and output settings, and automatically instantiates and executes the 
`ThermalSimulation` environment.

Usage:
------
    python run.py
"""

import os
import sys
import yaml
import numpy as np

# Add local path to ensure mgkmc is importable if run from source dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mgkmc import ThermalSimulation
from mgkmc.microstructure import generate_field

def main():
    # ---------------------------------------------------------
    # 1. Load Configuration
    # ---------------------------------------------------------
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find {config_path} in the current directory.")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Apply Seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    
    simulation_type = config.get('simulation_type', 'kmc').lower()
    print(f"Loaded configuration from {config_path}")
    print(f"Simulation Type selected: '{simulation_type}'")
    
    # ---------------------------------------------------------
    # 2. Extract System & Material Properties
    # ---------------------------------------------------------
    sys_conf = config['system']
    nx, ny, nz = sys_conf['nx'], sys_conf['ny'], sys_conf['nz']
    shape = (nx, ny, nz)
    
    # Material Elastic Modulus (E) and Poisson's ratio (nu)
    mat_conf = config['material']
    E_field = generate_field(
        mat_conf['E']['mode'], 
        shape, 
        constant_val=mat_conf['E'].get('value', 70.0),
        params=mat_conf['E'].get('parameters', {})
    )
    nu_field = generate_field(
        mat_conf['nu']['mode'], 
        shape, 
        constant_val=mat_conf['nu'].get('value', 0.3),
        params=mat_conf['nu'].get('parameters', {})
    )

    # ---------------------------------------------------------
    # 3. Setup Physics & Dynamics
    # ---------------------------------------------------------
    phys_conf = config['physics']
    dyn_conf = config['dynamics']
    out_conf = config['output']
    
    print("Initializing ThermalSimulation environment...")
    sim = ThermalSimulation(
        nx, ny, nz,
        M=sys_conf['M'],
        gamma0=sys_conf['gamma0'],
        E_field=E_field,
        nu_field=nu_field,
        pixel=sys_conf['pixel'],
        
        # Physics Parameters
        jp=phys_conf.get('jp', 20),
        jt=phys_conf.get('jt', 20),
        neighbor_softening_fraction=phys_conf.get('neighbor_softening_fraction', 0.0),
        softening_scheme=phys_conf.get('softening_scheme', 'isotropic'),
        softening_cap=phys_conf.get('softening_cap', 2.0),
        tau=phys_conf.get('tau', np.inf),
        q_act_temp=phys_conf.get('q_act_temp', 0.37),
        stability_threshold=phys_conf.get('stability_threshold', 0.0),
        redraw_directions=phys_conf.get('redraw_directions', True),
        redraw_barriers=phys_conf.get('redraw_barriers', True),
        
        # Outputs
        output_dir=out_conf.get('directory', 'output'),
        
        # Dynamics Parameters
        temperature=float(dyn_conf.get('temperature', 0.0)),
        strain_rate=float(dyn_conf.get('physical_strain_rate', 1.0e7)),
        strain_rate_sensitivity=float(dyn_conf.get('strain_rate_sensitivity', 0.0)),
        nu0=float(dyn_conf.get('nu0', 1.0e13)),
        instability_mode=dyn_conf.get('instability_mode', 'cascade'),
        cascade_timing=dyn_conf.get('cascade_timing', 'none'),
        scale_rate_by_volume=dyn_conf.get('scale_rate_by_volume', True),
        fast_patching=dyn_conf.get('fast_patching', None)
    )

    # ---------------------------------------------------------
    # 4. Prepare Boundary Conditions & Run
    # ---------------------------------------------------------
    bc_conf = config['boundary_conditions']
    component = tuple(bc_conf['driving_component'])
    
    # Parse string keys "[1, 1]" into integer tuples (1, 1)
    stress_targets = {}
    for k_str, val in bc_conf.get('mixed_targets', {}).items():
        # Clean "[1, 1]" -> "1, 1" -> (1, 1)
        k_clean = k_str.strip('[]()').split(',')
        if len(k_clean) == 2:
            stress_targets[(int(k_clean[0]), int(k_clean[1]))] = float(val)
            
    print(f"\n--- Starting execution ---")
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Temperature: {sim.temperature} K, Strain Rate: {sim.strain_rate:.2e} 1/s")
    print(f"Driving Strain Component: {component}")
    print(f"Target Stresses (Relaxed Components): {dict(stress_targets)}")
    
    if simulation_type == "linear_elastic":
        n_steps = 0
    else:
        n_steps = int(dyn_conf.get('n_steps', 100))
        
    # Execute mixed loading
    sim.run_mixed(
        n_global_steps=n_steps,
        strain_rate=float(dyn_conf.get('physical_strain_rate', 1e7)),
        component=component,
        stress_targets=stress_targets,
        mixed_tol=float(bc_conf.get('mixed_tol', 1e-4)),
        
        # Output interval setups
        vtk_interval=out_conf.get('vtk_interval', 10),
        vtk_mode=out_conf.get('vtk_mode', 'none'),
        checkpoint_interval=out_conf.get('checkpoint_interval', None),
        checkpoint_mode=out_conf.get('checkpoint_mode', 'none'),
        checkpoint_elastic_only=out_conf.get('checkpoint_elastic_only', True),
        enable_save_q=out_conf.get('enable_save_q', False),
        save_q_interval=out_conf.get('save_q_interval', None),
        
        # Logging flags
        enable_console_log=out_conf.get('enable_console', True),
        summary_filename=out_conf.get('summary_filename', 'summary_log.txt'),
        enable_summary_log=out_conf.get('enable_summary_log', True),
        enable_global_log=out_conf.get('enable_global_log', True),
        enable_cascade_log=out_conf.get('enable_cascade_log', True),
        enable_kmc_log=out_conf.get('enable_kmc_log', True),
        track_cascades=out_conf.get('track_cascades', False)
    )
    
    # If the user requested pure linear_elastic analysis, we halt here.
    if simulation_type == "linear_elastic":
        print("\n'linear_elastic' mode selected. The simulation will not proceed into plastic yielding.")
        print("Initial elastic equilibrium has been established and exported. Exiting.")
        return
    
    print("\nSimulation successfully completed via run.py.")

if __name__ == "__main__":
    main()
