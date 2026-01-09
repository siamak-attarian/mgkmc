import numpy as np
import os
import time
from mgkmc.stz.cascade import find_unstable, apply_flip_soa
from mgkmc.stz.update_fft import update_stress_fft_full
from mgkmc.stz.catalog import stz_catalog_glass
from mgkmc.analysis import export_to_vtk
from mgkmc.stz.kmc import compute_rates, select_event, decode_index
from mgkmc.stz.barriers import compute_barrier
from mgkmc.checkpoint import save_checkpoint, load_checkpoint
from mgkmc.stz.barrier_generators import get_barrier_generator

class AthermalSimulation:
    def __init__(self, 
                 nx, ny, nz, 
                 M, gamma0, 
                 E_field, nu_field,
                 pixel=1.0,  
                 barrier_generator=None,
                 barrier_kwargs={},
                 softening_scheme="isotropic", # "isotropic" or "directional"
                 softening_cap=2.0,
                 jp=10.0, jt=30.0,
                 tau=np.inf, # Transient decay time (Set to inf for no decay)
                 output_dir="output",
                 temperature=0.0, # Kelvin
                 strain_rate=1.0, # 1/s, used for KMC decision
                 strain_rate_sensitivity=0.0, # 's' exponent
                 stability_threshold=0.0 # eV, threshold for athermal instability
                 ):
        """
        Initialize Athermal Quasi-Static Simulation (with Thermal extensions) using Numba/SoA.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Simulation Parameters
        self.M = M
        self.gamma0 = gamma0
        self.volume = pixel**3
        self.pixel = pixel
        self.softening_scheme = softening_scheme
        self.softening_cap = softening_cap
        self.jp = jp
        self.jt = jt
        self.tau = tau
        self.temperature = temperature
        self.strain_rate = strain_rate
        self.strain_rate_sensitivity = strain_rate_sensitivity
        self.stability_threshold = stability_threshold
        
        # Grid Setup (Arrays)
        self.grid_shape = (nx, ny, nz)
        # self.E = E_field # Move assignments below to use corrected values
        # self.nu = nu_field
        
        # ================= SoA Arrays =================
        # 1. State Fields
        # Handle Unit Convention: If user passes GPa (e.g. 70.0), convert to Pa (70e9)
        # Heuristic: If mean(E) < 1e6, assume GPa.
        if E_field.mean() < 1e6:
            print(" [AthermalSimulation] Detected E in GPa (mean < 1 MPa). Converting to Pa (*1e9).")
            self.E_field = E_field * 1e9
        else:
            self.E_field = E_field
            
        self.nu_field = nu_field
        self.eps_field = np.zeros((nx, ny, nz, 3, 3)) # Elastic + Plastic? No, usually total?
        # AQS update_stress_fft returns: eps_total(x) and sigma(x).
        
        self.sig_field = np.zeros((nx, ny, nz, 3, 3))
        
        # Legacy aliases (used in solver calls)
        self.E = self.E_field
        self.nu = self.nu_field
        
        # 2. Plasticity Fields
        self.eps_plastic = np.zeros((nx, ny, nz, 3, 3))
        
        # 3. Softening [g_p, g_t, unused, unused]
        self.soft_prop = np.zeros((nx, ny, nz, 4)) 

        # 4. Barriers
        self.Q = np.zeros((nx, ny, nz, M))
        self.Q0 = np.zeros((nx, ny, nz, M))
        
        # 5. Catalog (Nx, Ny, Nz, M, 3, 3)
        # This consumes memory but allows fast indexing
        self.catalog = np.zeros((nx, ny, nz, M, 3, 3))
        
        # 6. Timing and Direction
        self.last_event_time = np.full((nx, ny, nz), -np.inf)
        self.prev_strain_dir = np.zeros((nx, ny, nz, 3, 3))
        
        # ================= Initialization =================
        # Q0
        if barrier_generator is None:
             # Default legacy behavior
             self.barrier_generator = get_barrier_generator("gaussian", mean=2.0, std=0.6)
        elif isinstance(barrier_generator, str):
             # Built-in generator
             self.barrier_generator = get_barrier_generator(barrier_generator, **barrier_kwargs)
        else:
             # Custom function
             self.barrier_generator = barrier_generator
             
        self.Q0 = self.barrier_generator((nx, ny, nz, M))

        # Catalog
        # Fill catalog with random modes
        # We need to call stz_catalog_glass M times per voxel? 
        # Or just vectorized fill?
        # stz_catalog_glass(M, gamma0) returns (M, 3, 3)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                     self.catalog[x,y,z] = stz_catalog_glass(M, gamma0)

        # Analysis State
        self.eps_macro = np.zeros((3,3)) 
        self.time = 0.0
        
        self.history_global = [] 
        self.history_detailed = [] 
        
        self.solver_args = {}

    def _init_logs(self, summary_filename="summary_log.txt"):
        # Create log files with headers
        self.log_global_path = os.path.join(self.output_dir, "global_log.txt")
        self.cascade_log_path = os.path.join(self.output_dir, "cascade_log.txt")
        self.summary_log_path = os.path.join(self.output_dir, summary_filename)
        
        # Clear existing
        open(self.log_global_path, 'w').close()
        open(self.cascade_log_path, 'w').close()
        
        # Summary Log Header
        # [Timestamp] [Elapsed] Step Type Eps_xx Sig_xx KMC Cascade Flips
        summary_header = f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} {'Type':<10} {'Eps_xx':<12} {'Sig_xx(GPa)':<12} {'KMC':<8} {'Cascade':<8} {'Flips':<8}\n"
        with open(self.summary_log_path, "w") as f:
            f.write(summary_header)
            f.write("-" * len(summary_header) + "\n")

        # Global Log Header (Legacy Format)
        header_fmt = "{:<10} {:<12} {:<10} " + " ".join(["{:<15}"]*12) + " {:<14} {:<17}"
        headers = [
            "GlobalStep", "ElasticStep", "KMCStep",
            "Eps_xx", "Eps_yy", "Eps_zz", "Eps_xy", "Eps_xz", "Eps_yz",
            "Sig_xx(GPa)", "Sig_yy(GPa)", "Sig_zz(GPa)", "Sig_xy(GPa)", "Sig_xz(GPa)", "Sig_yz(GPa)",
            "CascadeSteps", "TotalCascadeFlips"
        ]
        
        with open(self.log_global_path, "w") as f:
            f.write(header_fmt.format(*headers) + "\n")
            
        # Detailed Cascade Log Header (Aligned)
        # GlobalStep(12) LocalStep(12) NumUnstable(15) FlippedVoxels(30)
        cascade_header_fmt = "{:<12} {:<12} {:<15} {:<30}"
        with open(self.cascade_log_path, "w") as f:
            f.write(cascade_header_fmt.format(
                "GlobalStep", "LocalStep", "NumUnstable", 
                "FlippedVoxels(x,y,z,mode)"
            ) + "\n")

    def log_global(self, global_step, elastic_step, kmc_step, eps, sig, cascade_steps, total_flips):
        # eps, sig here are MACROSCOPIC 3x3
        # Flatten tensors for logging
        # We want: xx, yy, zz, xy, xz, yz
        indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
        
        # Consistent formatting
        line_fmt = "{:<10d} {:<12d} {:<10d} " + " ".join(["{:<15.6e}"]*12) + " {:<14d} {:<17d}"
        
        values = [global_step, elastic_step, kmc_step]
        values.extend([eps[i,j] for i,j in indices])
        values.extend([sig[i,j]/1e9 for i,j in indices])
        values.append(cascade_steps)
        values.append(total_flips)
        
        with open(self.log_global_path, "a") as f:
            f.write(line_fmt.format(*values) + "\n")

    def log_kmc(self, global_step, kmc_step, dt_kmc, dt_elastic, event_idx, barrier_ev):
        path = os.path.join(self.output_dir, "kmc_log.txt")
        fmt_header = "{:<10} {:<10} {:<15} {:<15} {:<15} {:<20} {:<15}\n"
        fmt_data   = "{:<10d} {:<10d} {:<15.6e} {:<15.6e} {:<15.6e} {:<20} {:<15.6f}\n"
        
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(fmt_header.format("GlobalStep", "KMCStep", "DtElastic", "DtKMC", "e^(-DtE/DtK)", "Event(x,y,z,m)", "Barrier(eV)"))
        
        if dt_kmc > 0:
             ratio = np.exp(-dt_elastic / dt_kmc)
        else:
             ratio = 0.0

        x, y, z, m = event_idx
        event_str = f"({x},{y},{z},{m})"
        with open(path, "a") as f:
            f.write(fmt_data.format(global_step, kmc_step, dt_elastic, dt_kmc, ratio, event_str, barrier_ev))

    def log_cascade(self, global_step, local_step, event_idx, barrier, n_unstable=1):
        # Aligned format to match header
        # GlobalStep(12) LocalStep(12) NumUnstable(15) FlippedVoxels(30)
        
        path = self.cascade_log_path
        
        # Handle batch or single
        # event_idx can be array of shape (N, 4) or single tuple/list (4,)
        
        try:
            # Check if it's a batch (array or list of lists)
            if hasattr(event_idx, 'ndim') and event_idx.ndim == 2:
                # It is a 2D array [[x,y,z,m], ...]
                indices = event_idx
            elif isinstance(event_idx, (list, tuple)) and len(event_idx) == 4 and isinstance(event_idx[0], (int, np.integer)):
                 # Single tuple (x,y,z,m)
                 indices = [event_idx]
            else:
                 indices = event_idx # Hope it's iterable of 4-tuples
        except:
             indices = []

        if len(indices) > 50:
            flip_str = f"{len(indices)} voxels flipped (truncated)"
        else:
            # Format: (x,y,z,m);(x,y,z,m)
            parts = []
            for row in indices:
                x, y, z, m = row
                parts.append(f"({x},{y},{z},{m})")
            flip_str = ";".join(parts)
        
        line_fmt = "{:<12d} {:<12d} {:<15d} {:<30}\n"
        
        with open(path, "a") as f:
            f.write(line_fmt.format(global_step, local_step, n_unstable, flip_str))

    def update_barriers(self):
        # Call Numba Kernel
        # self.softening_scheme -> int for Numba
        # 0=isotropic, 1=directional
        scheme = 0 if self.softening_scheme == "isotropic" else 1
        
        compute_barrier(self.Q, self.Q0, self.sig_field, self.catalog, self.volume,
                        self.soft_prop, self.last_event_time, self.time, 
                        self.prev_strain_dir, self.softening_cap,
                        scheme, self.tau)

    def _run_cascade(self, global_step):
        local_step = 0
        total_flips = 0

        while True:
            # 1. Update Barriers (Fast Numba)
            self.update_barriers()
            
            # 2. Find Unstable (Fast Numba)
            # Returns arrays of indices [[x,y,z,m], ...]
            unstable_indices = find_unstable(self.Q, threshold=self.stability_threshold)
            n_unstable = len(unstable_indices)

            if n_unstable == 0:
                break
            
            idx_pick = 0 # Simple first pick
            ux, uy, uz, um = unstable_indices[idx_pick]

                
            # Legacy Logic: Flip ALL unstable sites at once
            self.log_cascade(global_step, local_step, unstable_indices, 0.0, n_unstable=n_unstable)
            
            # Flip Loop
            for k in range(n_unstable):
                 ux, uy, uz, um = unstable_indices[k]
                 
                 # Apply Flip (Update Plastic Strain & Softening)
                 apply_flip_soa(self.eps_plastic, None, self.soft_prop, self.last_event_time,
                                self.catalog, ux, uy, uz, um, self.time, 
                                self.jp, self.jt, self.softening_cap)
                 
                 # Update direction
                 self.prev_strain_dir[ux,uy,uz] = self.catalog[ux,uy,uz,um]
                 
                 # Renew Catalog (Assumption: simple renewal)
                 self.catalog[ux,uy,uz,um] = stz_catalog_glass(1, self.gamma0)[0]
                 
                 # Reset Barrier (New Q0) - Consistent with the configured generator
                 # We generate one new value using the generator (shape=(1,))
                 self.Q0[ux,uy,uz,um] = self.barrier_generator((1,))[0]
            
            total_flips += n_unstable
            
            # 3. Global Elastic Relax (FFT) - REQUIRED after batch flip
            self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                   self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
            )
            
            local_step += 1
            if local_step > 20000: # Safety
                 print("Cascade limit reached (20000 steps)")
                 break
                 
        # Final Mean Values
        eps_curr = self.eps_field.mean(axis=(0,1,2))
        sig_curr = self.sig_field.mean(axis=(0,1,2))
        
        return local_step, total_flips, eps_curr, sig_curr

    def run_mixed(self, n_global_steps, strain_rate, component=(0,1), 
                  stress_targets={}, mixed_tol=1e-4, mixed_max_iter=50,
                  kmc_mode="accumulate", # "accumulate" or "on_demand"
                  checkpoint_interval=None, checkpoint_path="checkpoint", 
                  checkpoint_mode="periodic", # "periodic", "current", "last", or "none"
                  stop_on_stress_drop=None, stress_drop_component=(0,1), stop_post_drop_steps=0,
                  vtk_mode=None, ignore_drop_steps=0,
                  checkpoint_elastic_only=False,
                  enable_console_log=True,
                  summary_filename="summary_log.txt",
                  stress_drop_lookback=1):
        
        # Setup defaults
        if not stress_targets: # Check if empty dict
             if component == (0,0):
                 stress_targets[(1,1)] = 0.0
                 stress_targets[(2,2)] = 0.0
             elif component == (0,1): # Shear
                 stress_targets[(0,0)] = 0.0
                 stress_targets[(1,1)] = 0.0
                 stress_targets[(2,2)] = 0.0

        self._init_logs(summary_filename=summary_filename)
        
        import time
        from datetime import datetime
        start_time_total = time.time()
        
        # Elastic Time Step
        if self.strain_rate > 0:
             dt_elastic = abs(strain_rate) / self.strain_rate
        else:
             dt_elastic = 1.0
             
        # Initial Relax
        _, _, eps_curr, sig_curr = self._run_cascade(global_step=0)
        
        # Checkpoint Counters
        elastic_chk_id = 0
        
        # Save Initial State (Step 0) - Always treated as Elastic
        # Only save if mode is periodic or current
        if checkpoint_interval is not None and checkpoint_mode in ["periodic", "current"]:
             if checkpoint_mode == "current":
                 cp_name = f"{checkpoint_path}.h5"
             elif checkpoint_elastic_only:
                 cp_name = f"{checkpoint_path}_elastic_{elastic_chk_id:06d}.h5"
                 elastic_chk_id += 1
             else:
                 cp_name = f"{checkpoint_path}_000000.h5"
             
             self.save_checkpoint(cp_name, step=0)
        
        # Detection state
        stress_history = [sig_curr[stress_drop_component]]
        stop_drop_triggered = False
        stop_countdown = stop_post_drop_steps
        
        step = 1
        elastic_steps_done = 0
        total_kmc_steps = 0
        
        # Helper for Legacy Compliance
        E_avg = np.mean(self.E)
        nu_avg = np.mean(self.nu)
        
        def get_correction_legacy(sigma_err):
             # Legacy formula (Simplified/Incorrect but consistent)
             tr_sig = np.trace(sigma_err)
             return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg

        last_step_type = "elastic" # Initialization

        while elastic_steps_done < n_global_steps:
             iteration_steps = 0
             iteration_flips = 0
             
             while True:
                 # 1. Stability Check
                 # Fast Numba check
                 self.update_barriers()
                 unstable_indices = find_unstable(self.Q, self.stability_threshold)
                 if len(unstable_indices) > 0:
                     l, f, _, _ = self._run_cascade(step)
                     iteration_steps += l
                     iteration_flips += f
                     continue
                 
                 # 2. Rates
                 rates_flat, indices_flat, total_rate = compute_rates(self.Q, self.volume, self.temperature)
                 
                 # 3. Decision
                 idx_flat, dt_kmc = select_event(rates_flat, total_rate)
                 trigger = np.exp(-dt_elastic / dt_kmc) if dt_kmc > 0 else 0.0
                 
                 if self.temperature > 0 and np.random.uniform() > trigger:
                      # KMC Event
                      if idx_flat == -1: continue 
                      
                      # Decode
                      x, y, z, m = decode_index(indices_flat[idx_flat], self.grid_shape[1], self.grid_shape[2], self.M)
                      barrier_val = self.Q[x,y,z,m]
                      
                      apply_flip_soa(self.eps_plastic, None, self.soft_prop, self.last_event_time,
                                     self.catalog, x, y, z, m, self.time, 
                                     self.jp, self.jt, self.softening_cap)
                      
                      self.prev_strain_dir[x,y,z] = self.catalog[x,y,z,m]
                      self.catalog[x,y,z,m] = stz_catalog_glass(1, self.gamma0)[0]
                      self.log_kmc(step, total_kmc_steps, dt_kmc, dt_elastic, (x,y,z,m), barrier_val)
                      
                      self.time += dt_kmc
                      total_kmc_steps += 1
                      last_step_type = "kmc"
                      
                      # Update State
                      self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                           self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
                      )
                      
                      if kmc_mode == "on_demand":
                           break
                           
                 else:
                      # Elastic Event
                      self.time += dt_elastic
                      
                      # Apply Strain + Relax
                      eps_inc = np.zeros((3,3))
                      eps_inc[component] = strain_rate # Target increment
                      self.eps_macro += eps_inc
                      
                      # Update State after Increment (Required so cascade/check sees change)
                      self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                           self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
                      )
                      
                      # Mixed Relax Loop (Legacy Consistent)
                      converged = False
                      for it in range(mixed_max_iter):
                          # A. Cascade (Relax Plasticity)
                          l, f, _, sig_M = self._run_cascade(step)
                          iteration_steps += l
                          iteration_flips += f
                          
                          # B. Check Stress Convergence
                          stress_err_tensor = np.zeros((3,3))
                          max_err = 0.0
                          for idx_t, target_val in stress_targets.items():
                              err = target_val - sig_M[idx_t] # Target - Current
                              stress_err_tensor[idx_t] = err
                              max_err = max(max_err, abs(err))
                              
                          if max_err < mixed_tol:
                              converged = True
                              break
                              
                          # C. Apply Correction (Legacy Formula)
                          eps_corr = get_correction_legacy(stress_err_tensor)
                          eps_corr[component] = 0.0 # Don't touch driven component
                          
                          self.eps_macro += eps_corr
                          
                          # D. Update Elastic State (Required for next cascade check)
                          self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                               self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
                          )
                          
                      if not converged:
                          print(f"Warning: Mixed loop did not converge at step {step} (Err={max_err:.2e})")
                          
                      elastic_steps_done += 1
                      last_step_type = "elastic"
                      break
             
             # Log Global
             eps_curr = self.eps_field.mean(axis=(0,1,2))
             sig_curr = self.sig_field.mean(axis=(0,1,2))
             self.log_global(step, elastic_steps_done, total_kmc_steps, eps_curr, sig_curr, iteration_steps, iteration_flips)
             self.history_global.append((eps_curr[0,0], sig_curr[0,0]/1e9))
             
             curr_stress_val = sig_curr[stress_drop_component]
             curr_strain_val = eps_curr[stress_drop_component]
             
             # Timing
             now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
             elapsed = time.time() - start_time_total
             
             status_msg = f"[{now}] [{elapsed:8.2f}s] Step {step:4d}: Type={last_step_type.upper():<8}, KMC={total_kmc_steps:4d}, Cascade={iteration_steps:4d}, Flips={iteration_flips:4d}, Eps_xx={curr_strain_val:8.6f}, Sig_xx={curr_stress_val/1e9:6.3f} GPa"
             
             # Write to Summary Log
             # Formatted for the header defined in _init_logs
             summary_line = f"{now:<20} {elapsed:<12.2f} {step:<8d} {last_step_type.upper():<10} {curr_strain_val:<12.6f} {curr_stress_val/1e9:<12.3f} {total_kmc_steps:<8d} {iteration_steps:<8d} {iteration_flips:<8d}\n"
             with open(self.summary_log_path, "a") as f:
                 f.write(summary_line)
            
             # Checkpoint Logic
             should_save = False
             if checkpoint_interval and step % checkpoint_interval == 0:
                 if checkpoint_mode in ["periodic", "current"]:
                     should_save = True
                     if checkpoint_elastic_only and last_step_type != "elastic":
                         should_save = False
             
             if should_save:
                 if checkpoint_mode == "periodic":
                     if checkpoint_elastic_only:
                         # Sequential Elastic Checkpoint
                         cp_name = f"{checkpoint_path}_elastic_{elastic_chk_id:06d}.h5"
                         elastic_chk_id += 1
                     else:
                         # Standard Global Step Checkpoint
                         cp_name = f"{checkpoint_path}_{step:06d}.h5"
                 else: # checkpoint_mode == "current"
                     cp_name = f"{checkpoint_path}.h5"
                 self.save_checkpoint(cp_name, step=step)

             # Stress Drop Detection Logic
             if stop_on_stress_drop is not None and not stop_drop_triggered:
                 if step > ignore_drop_steps:
                     # Look back logic
                     lookback = max(1, stress_drop_lookback)
                     if len(stress_history) >= lookback:
                         ref_stress = stress_history[-lookback]
                         if abs(ref_stress) > 1e-6:
                             drop_frac = (ref_stress - curr_stress_val) / ref_stress
                         else:
                             drop_frac = 0.0
                         
                         if drop_frac > stop_on_stress_drop:
                             print(f"\n[ALERT] Shear Band Detected! Stress drop {drop_frac*100:.1f}% > {stop_on_stress_drop*100:.1f}% (Trend over {lookback} steps) at step {step}")
                             stop_drop_triggered = True
                             status_msg += " [SB DETECTED]"
             
             stress_history.append(curr_stress_val)
             if enable_console_log:
                 print(status_msg)
             
             if stop_drop_triggered:
                 if stop_countdown > 0:
                     stop_countdown -= 1
                 else:
                     print(f"Stopping criteria: {stop_post_drop_steps} steps after detection.")
                     break
             
             step += 1

        # Final Save
        if checkpoint_mode == "last":
             self.save_checkpoint(f"{checkpoint_path}_final.h5", step=step-1)
        elif checkpoint_mode in ["periodic", "current"]:
             # Optional: Ensure the very last state is saved if it wasn't just saved
             if checkpoint_interval is not None and (step-1) % checkpoint_interval != 0:
                  self.save_checkpoint(f"{checkpoint_path}_final.h5", step=step-1)


    def run(self, *args, **kwargs):
        print("Use run_mixed instead.")
        pass

    def save_checkpoint(self, path, step=None):
         if not path.endswith('.h5'):
             path += '.h5'
             
         # Disable HDF5 locking to avoid conflicts with cloud sync (Google Drive)
         os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
             
         import h5py
         import time
         from datetime import datetime
         
         max_retries = 3
         for attempt in range(max_retries):
             try:
                 with h5py.File(path, "w") as f:
                     # Metadata
                     meta = f.create_group('metadata')
                     meta.attrs['nx'] = self.grid_shape[0]
                     meta.attrs['ny'] = self.grid_shape[1]
                     meta.attrs['nz'] = self.grid_shape[2]
                     meta.attrs['M'] = self.M
                     meta.attrs['gamma0'] = self.gamma0
                     meta.attrs['pixel'] = self.pixel
                     meta.attrs['timestamp'] = datetime.now().isoformat()
                     
                     if step is not None:
                         meta.attrs['step'] = step
                     else:
                         meta.attrs['step'] = 0
                     
                     # Fields (SoA Direct Dump)
                     fields = f.create_group('fields')
                     fields.create_dataset('eps_field', data=self.eps_field, compression='gzip')
                     fields.create_dataset('sig_field', data=self.sig_field, compression='gzip')
                     fields.create_dataset('E_field', data=self.E_field, compression='gzip')
                     fields.create_dataset('nu_field', data=self.nu_field, compression='gzip')
                     
                     # SoA State
                     grid = f.create_group('grid')
                     grid.create_dataset('eps_plastic', data=self.eps_plastic, compression='gzip')
                     grid.create_dataset('soft_prop', data=self.soft_prop, compression='gzip')
                     grid.create_dataset('Q', data=self.Q, compression='gzip')
                     grid.create_dataset('Q0', data=self.Q0, compression='gzip')
                     grid.create_dataset('catalog', data=self.catalog, compression='gzip')
                     grid.create_dataset('last_event_time', data=self.last_event_time, compression='gzip')
                 
                 # Success
                 return
                 
             except (ImportError, Exception) as e:
                 if attempt < max_retries - 1:
                     # print(f"Warning: Checkpoint attempt {attempt+1} failed: {e}. Retrying...")
                     time.sleep(0.5)
                 else:
                     print(f"Error saving checkpoint after {max_retries} attempts: {e}")



