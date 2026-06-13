import os
import sys
import yaml
import numpy as np

# Ensure mgkmc is importable
sys.path.append(r"D:\GoogleDrive\2-MGKMC\mgkmc")
from mgkmc import KmcSimulation2D

def run_sim(strain_assumption):
    config_path = r"D:\GoogleDrive\2-MGKMC\mgkmc\examples\7-KMC\config_neo_hookean.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    np.random.seed(config.get('seed', 1))
    
    sys_conf = config['system']
    nx, ny = sys_conf['nx'], sys_conf['ny']
    
    mat_conf = config['material']
    mu_val = float(mat_conf.get('mu', 65.0)) * 1e9
    lambda_val = float(mat_conf.get('lambda', 25.0)) * 1e9
    
    # Calculate E and nu
    denom = lambda_val + mu_val
    E_val = mu_val * (3.0 * lambda_val + 2.0 * mu_val) / denom
    nu_val = lambda_val / (2.0 * denom)
    
    E_field = np.full((nx, ny), E_val)
    nu_field = np.full((nx, ny), nu_val)
    
    phys_conf = config.get('physics', {})
    dyn_conf = config.get('dynamics', {})
    bar_conf = config.get('barriers', {})
    bar_type = bar_conf.get('type', 'gaussian')
    bar_kwargs = bar_conf.get('kwargs', {})
    
    sim = KmcSimulation2D(
        nx, ny,
        M=sys_conf['M'],
        gamma0=sys_conf['gamma0'],
        E_field=E_field,
        nu_field=nu_field,
        pixel=sys_conf.get('pixel', 1.0),
        plane_mode=sys_conf.get('plane_mode', 'plane_stress').lower(),
        barrier_generator=bar_type,
        barrier_kwargs=bar_kwargs,
        jp=phys_conf.get('jp', 20),
        jt=phys_conf.get('jt', 0),
        neighbor_softening_fraction=phys_conf.get('neighbor_softening_fraction', 0.0),
        softening_scheme=phys_conf.get('softening_scheme', 'isotropic'),
        softening_cap=phys_conf.get('softening_cap', 0.9),
        q_act_temp=phys_conf.get('q_act_temp', 0.37),
        redraw_directions=phys_conf.get('redraw_directions', True),
        redraw_barriers=phys_conf.get('redraw_barriers', True),
        output_dir="temp_out",
        temperature=float(dyn_conf.get('temperature', 300.0)),
        strain_rate=float(dyn_conf.get('physical_strain_rate', 1e9)),
        nu0=float(dyn_conf.get('nu0', 1e12)),
        stability_threshold=phys_conf.get('stability_threshold', 0.0),
        strain_assumption=strain_assumption,
        hyperelastic_model="neo_hookean"
    )
    
    # Run the simulation for 100 loading steps (1% strain)
    sim.driving_component = (0, 0)
    sim.stress_targets = {(1, 1): 0.0}
    sim.elastic_run(sim.eps_macro)
    
    strain_unit = np.zeros((2,2))
    strain_unit[sim.driving_component] = 1.0
    
    step_size = float(config['loading']['step_size'])
    n_steps = 600 # Run 6.0% strain
    dt_step = abs(step_size) / sim.strain_rate
    remaining_time = 0.0
    
    elastic_steps_done = 0
    total_kmc_steps = 0
    step = 1
    
    results = []
    
    while elastic_steps_done < n_steps:
        remaining_time += dt_step
        while remaining_time > 0:
            sim.update_barriers()
            
            eff_volume = sim.volume if sim.scale_rate_by_volume else 1.0
            from mgkmc.kmc_simulator_functions import compute_rates_2d, select_event_2d, decode_index_2d
            rates, indices, total_rate = compute_rates_2d(
                sim.Q, eff_volume, sim.temperature, sim.nu0, instability_mode="kmc"
            )
            
            if total_rate > 0:
                t_wait = -np.log(np.random.rand()) / total_rate
                if t_wait < remaining_time:
                    sim.time += t_wait
                    sim.eps_macro += strain_unit * (sim.strain_rate * t_wait)
                    remaining_time -= t_wait
                    idx_flat = indices[select_event_2d(rates, total_rate)]
                    x, y, m = decode_index_2d(idx_flat, sim.ny, sim.M)
                    
                    C = sim.catalog[x,y,m].copy()
                    if sim.strain_assumption == "finite_strain":
                        I_plus_C = np.eye(2) + C
                        det_I_plus_C = I_plus_C[0,0] * I_plus_C[1,1] - I_plus_C[0,1] * I_plus_C[1,0]
                        I_plus_C = I_plus_C / np.sqrt(max(1e-12, det_I_plus_C))
                        sim.F_plastic[x, y] = np.dot(I_plus_C, sim.F_plastic[x, y])
                    else:
                        sim.eps_plastic[x, y] += C
                        
                    # Softening
                    e11, e22, e12 = C[0,0], C[1,1], C[0,1]
                    sum_sq = (e12**2) + (e22**2 + e11**2 + (e11 - e22)**2) / 6.0
                    gp_new = sim.soft_prop[x,y,0] + sim.jp * sum_sq
                    if sim.softening_cap > 0 and gp_new > sim.softening_cap: gp_new = sim.softening_cap
                    sim.soft_prop[x,y,0] = gp_new
                    sim.last_event_time[x,y] = sim.time
                    
                    sim.prev_strain_dir[x,y] = C
                    sim.catalog[x,y,m] = stz_catalog_glass_2d(1, sim.gamma0)[0]
                    sim.Q0[x,y,m] = sim.barrier_generator((1,))[0]
                    
                    sig_mean = sim.elastic_run(sim.eps_macro)
                    total_kmc_steps += 1
                    
                    results.append((sim.eps_macro[0,0], sig_mean[0,0] / 1e9, total_kmc_steps))
                    continue
            
            sim.eps_macro += strain_unit * (sim.strain_rate * remaining_time)
            sim.time += remaining_time
            remaining_time = 0
            
        sig_mean = sim.elastic_run(sim.eps_macro)
        elastic_steps_done += 1
        results.append((sim.eps_macro[0,0], sig_mean[0,0] / 1e9, total_kmc_steps))
        
    return results

if __name__ == "__main__":
    from mgkmc.kmc_simulator_functions import stz_catalog_glass_2d
    print("Running Small Strain Simulation...")
    ss_results = run_sim("small_strain")
    print("Running Finite Strain Simulation...")
    fs_results = run_sim("finite_strain")
    
    print("\nComparison at selected steps:")
    print(f"{'Strain':<10} | {'Small Strain Stress (GPa)':<30} | {'Finite Strain Stress (GPa)':<30} | {'SS Flips':<10} | {'FS Flips':<10}")
    print("-" * 95)
    for i in range(0, min(len(ss_results), len(fs_results)), 50):
        ss_eps, ss_sig, ss_flips = ss_results[i]
        fs_eps, fs_sig, fs_flips = fs_results[i]
        print(f"{ss_eps:<10.5f} | {ss_sig:<30.6f} | {fs_sig:<30.6f} | {ss_flips:<10d} | {fs_flips:<10d}")
