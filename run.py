import numpy as np
import os
import shutil
import yaml
import argparse
import pyfftw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mgkmc import AthermalSimulation, generate_correlated_field

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_material_field(field_config, shape, seed=None):
    mode = field_config.get('mode', 'constant')
    
    if mode == "constant":
        return np.full(shape, field_config['value'])
    
    elif mode == "file":
        path = field_config['path']
        if path.endswith('.npy'):
            return np.load(path)
        else:
            return np.loadtxt(path).reshape(shape)
            
    elif mode == "generated":
        params = field_config['parameters']
        return generate_correlated_field(
            shape=shape,
            mean=params.get('mean', 1.0),
            std=params.get('std', 0.1),
            corr=params.get('corr', 10),
            clip_min=params.get('clip_min'),
            clip_max=params.get('clip_max'),
            seed=seed
        )
    else:
        raise ValueError(f"Unknown material mode: {mode}")

def main():
    parser = argparse.ArgumentParser(description="Run MGKMC Simulation from YAML config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", type=str, help="Path to HDF5 checkpoint to resume from (errors if not found)")
    parser.add_argument("--resume_or_initiate", type=str, help="Path to checkpoint to resume from (initiates fresh if not found)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return

    print("=" * 60)
    print(f"Loading Configuration: {args.config}")
    cfg = load_config(args.config)
    print("=" * 60)

    # 1. RNG Setup
    seed = cfg.get('seed', 42)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")

    # ========================================
    # Resume Logic Detection
    # ========================================
    resume_path = None
    should_resume = False
    
    # 1. Check CLI
    if args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            print(f"Error: Checkpoint file '{resume_path}' not found (requested via --resume).")
            return
        should_resume = True
    elif args.resume_or_initiate:
        resume_path = args.resume_or_initiate
        if os.path.exists(resume_path):
            should_resume = True
            print(f"Checkpoint found at '{resume_path}'. Resuming...")
        else:
            print(f"Checkpoint not found at '{resume_path}'. Starting fresh...")
            should_resume = False
    
    # 2. Check Config if not set via CLI
    if not args.resume and not args.resume_or_initiate:
        res_cfg = cfg.get('resume', {})
        res_mode = res_cfg.get('mode', 'none')
        res_path = res_cfg.get('checkpoint_path')
        
        if res_mode == "resume":
            if res_path and os.path.exists(res_path):
                resume_path = res_path
                should_resume = True
                print(f"Resuming from config: {resume_path}")
            else:
                print(f"Error: Checkpoint '{res_path}' not found (requested in config).")
                return
        elif res_mode == "resume_or_initiate":
            if res_path and os.path.exists(res_path):
                resume_path = res_path
                should_resume = True
                print(f"Resuming from config: {resume_path}")
            else:
                print(f"Checkpoint not found in config. Starting fresh...")
                should_resume = False

    # 2. Geometry
    sys_cfg = cfg['system']
    num_threads = sys_cfg.get('num_threads', 1)
    pyfftw.config.NUM_THREADS = num_threads
    print(f"FFT Threads set to {num_threads}")
    
    nx, ny, nz = int(sys_cfg['nx']), int(sys_cfg['ny']), int(sys_cfg['nz'])
    shape = (nx, ny, nz)
    
    # 3. Material Fields
    mat_cfg = cfg['material']
    E = get_material_field(mat_cfg['E'], shape, seed=seed)
    nu = get_material_field(mat_cfg['nu'], shape, seed=seed+1 if seed else None)
    
    # 4. Initialize Simulation
    out_cfg = cfg['output']
    output_dir = out_cfg['directory']
    
    if os.path.exists(output_dir) and not should_resume:
        print(f"Cleaning output directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except OSError:
            pass
    elif should_resume:
        print(f"Preserving existing output directory for resume: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the config file for records
    if out_cfg.get('enable_config_backup', True):
        shutil.copy(args.config, os.path.join(output_dir, "parameters.yaml"))
        print(f"Config backed up to {os.path.join(output_dir, 'parameters.yaml')}")
            
    phys_cfg = cfg['physics']
    bar_cfg = cfg['barriers']
    dyn_cfg = cfg['dynamics']
    det_cfg = cfg['detection']
    
    # Handle softening JP/JT override if disabled
    jp = float(phys_cfg['jp'])
    jt = float(phys_cfg['jt'])
    if not phys_cfg.get('enable_softening', True):
        jp = 0.0
        jt = 0.0
        print("Softening DISABLED manually.")

    if should_resume:
        print(f"Loading state from {resume_path}...")
        sim = AthermalSimulation.load_checkpoint(resume_path)
        # Re-apply some output settings from current config
        sim.output_dir = output_dir
    else:
        sim = AthermalSimulation(
            nx, ny, nz,
            M=int(sys_cfg['M']),
            gamma0=float(sys_cfg['gamma0']),
            E_field=E,
            nu_field=nu,
            pixel=float(sys_cfg['pixel']),
            barrier_generator=bar_cfg['type'],
            barrier_kwargs=bar_cfg['kwargs'],
            output_dir=output_dir,
            softening_scheme=phys_cfg['softening_scheme'],
            softening_cap=float(phys_cfg['softening_cap']),
            jp=jp,
            jt=jt,
            temperature=float(dyn_cfg['temperature']),
            strain_rate=float(dyn_cfg['physical_strain_rate']),
            stability_threshold=float(phys_cfg['stability_threshold']),
            redraw_directions=phys_cfg.get('redraw_directions', True),
            redraw_barriers=phys_cfg.get('redraw_barriers', True),
            max_cascade_steps_pct=float(det_cfg.get('max_cascade_steps_pct', 0.3)),
            nu0=float(dyn_cfg.get('nu0', 1e13)),
            q_act_temp=float(phys_cfg.get('q_act_temp', 0.37))
        )

    # 5. Run Mixed BC Simulation
    bc_cfg = cfg['boundary_conditions']
    
    # Convert string keys like "[1, 1]" back to tuples (1, 1)
    mixed_targets = {}
    for k, v in bc_cfg.get('mixed_targets', {}).items():
        # Evaluate string representation of tuple/list
        key = tuple(eval(k)) if isinstance(k, str) else tuple(k)
        mixed_targets[key] = float(v)

    driving_comp = tuple(bc_cfg['driving_component'])
    
    # Calculate strain rate per step
    eps_target = float(dyn_cfg['eps_target'])
    n_steps = int(dyn_cfg['n_steps'])
    strain_rate_per_step = eps_target / n_steps
    
    sim.run_mixed(
        n_global_steps=n_steps,
        strain_rate=strain_rate_per_step,
        component=driving_comp,
        stress_targets=mixed_targets,
        mixed_tol=float(bc_cfg['mixed_tol']) * 1e6, # Convert MPa to Pa
        
        # Checkpoint & Detection
        checkpoint_interval=out_cfg['checkpoint_interval'],
        checkpoint_path=os.path.join(output_dir, "checkpoint"),
        checkpoint_mode=out_cfg['checkpoint_mode'],
        
        # Logging
        enable_console_log=out_cfg.get('enable_console', True),
        summary_filename=out_cfg.get('summary_filename', "summary_log.txt"),
        
        stop_on_stress_drop=det_cfg['stop_on_stress_drop'],
        stress_drop_lookback=det_cfg.get('stress_drop_lookback', 1),
        stress_drop_component=driving_comp,
        stop_post_drop_steps=det_cfg['stop_post_drop_steps'],
        ignore_drop_steps=det_cfg['ignore_drop_steps'],
        checkpoint_elastic_only=out_cfg.get('checkpoint_elastic_only', False),
        enable_save_q=out_cfg.get('enable_save_q', False),
        save_q_interval=out_cfg.get('save_q_interval'),
        save_q_elastic_only=out_cfg.get('save_q_elastic_only', False),
        max_kmc_steps_pct=float(det_cfg.get('max_kmc_steps_pct', 0.3)),
        enable_global_log=out_cfg.get('enable_global_log', True),
        enable_cascade_log=out_cfg.get('enable_cascade_log', True),
        enable_kmc_log=out_cfg.get('enable_kmc_log', True),
        vtk_interval=out_cfg.get('vtk_interval'),
        vtk_mode=out_cfg.get('vtk_mode', 'none'),
        track_cascades=out_cfg.get('track_cascades', False),
        append_logs=should_resume
    )

    print(f"\nSimulation complete. Results in '{output_dir}'")

    # 6. Optional Plotting
    if out_cfg.get('enable_plotting', False):
        try:
            hist_global = np.array(sim.history_global)
            plt.figure(figsize=(10, 6))
            if len(hist_global) > 0:
                plt.plot(hist_global[:,0]*100, hist_global[:,1], 'b-o', markersize=2)
            plt.xlabel('Strain (%)')
            plt.ylabel('Stress (GPa)')
            plt.title(f"MGKMC Simulation: {args.config}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "stress_strain.png"))
            plt.close() # Release memory/resources
            print("Plot generated.")
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
