import os
import sys
import yaml
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgkmc.kmc_simulator import KmcSimulation2D
from mgkmc.finite_strain_simulator import finite_strain_solver_step_2d

def debug_run():
    config_path = r"D:\GoogleDrive\2-MGKMC\mgkmc\examples\7-KMC\config_neo_hookean.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config.get('seed', 1))
    
    sys_conf = config['system']
    nx, ny = sys_conf['nx'], sys_conf['ny']
    
    mat_conf = config['material']
    mu_val = float(mat_conf['mu'])
    lambda_val = float(mat_conf['lambda'])
    
    # Calculate E and nu
    denom = lambda_val + mu_val
    E_val = mu_val * (3.0 * lambda_val + 2.0 * mu_val) / denom
    nu_val = lambda_val / (2.0 * denom)
    
    E_field = np.full((nx, ny), E_val * 1e9)
    nu_field = np.full((nx, ny), nu_val)
    
    sim = KmcSimulation2D(
        nx, ny,
        M=sys_conf['M'],
        gamma0=sys_conf['gamma0'],
        E_field=E_field,
        nu_field=nu_field,
        pixel=sys_conf.get('pixel', 1.0),
        plane_mode=sys_conf.get('plane_mode', 'plane_stress'),
        barrier_generator=config['barriers']['type'],
        barrier_kwargs=config['barriers']['kwargs'],
        
        jp=config['physics'].get('jp', 20),
        jt=config['physics'].get('jt', 0),
        neighbor_softening_fraction=config['physics'].get('neighbor_softening_fraction', 0.0),
        softening_scheme=config['physics'].get('softening_scheme', 'isotropic'),
        softening_cap=config['physics'].get('softening_cap', 0.9),
        q_act_temp=config['physics'].get('q_act_temp', 0.37),
        redraw_directions=config['physics'].get('redraw_directions', True),
        redraw_barriers=config['physics'].get('redraw_barriers', True),
        
        output_dir="debug_output",
        temperature=float(config['dynamics'].get('temperature', 300.0)),
        strain_rate=float(config['dynamics'].get('physical_strain_rate', 1.0e9)),
        nu0=float(config['dynamics'].get('nu0', 1.0e12)),
        stability_threshold=config['physics'].get('stability_threshold', 0.0),
        fast_patching=config['dynamics'].get('fast_patching', None),
        instability_mode=config['dynamics'].get('instability_mode', 'kmc'),
        cascade_timing=config['dynamics'].get('cascade_timing', 'none'),
        scale_rate_by_volume=config['dynamics'].get('scale_rate_by_volume', False),
        
        strain_assumption=sys_conf.get('strain_assumption', 'finite_strain'),
        hyperelastic_model=sys_conf.get('hyperelastic_model', 'neo_hookean'),
        A_m=float(mat_conf.get('A', 0.0)),
        B_m=float(mat_conf.get('B', 0.0)),
        C_m=float(mat_conf.get('C', 0.0)),
        solver=sys_conf.get('solver', 'al')
    )
    
    # We want to run it step-by-step
    sim.run_simulation(
        n_global_steps=390,
        step_size=0.0001,
        component=(0, 0),
        stress_targets={(1, 1): 0.0},
        mixed_tol=1e4,
        mixed_max_iter=10,
        enable_console_log=True,
        enable_summary_log=True,
        enable_global_log=True
    )

if __name__ == "__main__":
    debug_run()
