import numpy as np
import os
import time
from datetime import datetime
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
                 stability_threshold=0.0, # eV, threshold for athermal instability
                 redraw_directions=True, # Redraw all modes in voxel after flip
                 redraw_barriers=True    # Redraw all Q0 in voxel after flip
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
        self.redraw_directions = redraw_directions
        self.redraw_barriers = redraw_barriers
        
        # Grid Setup (Arrays)
        self.grid_shape = (nx, ny, nz)
        
        # ================= SoA Arrays =================
        # 1. State Fields
        if E_field.mean() < 1e6:
            print(" [AthermalSimulation] Detected E in GPa (mean < 1 MPa). Converting to Pa (*1e9).")
            self.E_field = E_field * 1e9
        else:
            self.E_field = E_field
            
        self.nu_field = nu_field
        self.eps_field = np.zeros((nx, ny, nz, 3, 3))
        self.sig_field = np.zeros((nx, ny, nz, 3, 3))
        
        # Legacy aliases
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
        self.catalog = np.zeros((nx, ny, nz, M, 3, 3))
        
        # 6. Timing and Direction
        self.last_event_time = np.full((nx, ny, nz), -np.inf)
        self.prev_strain_dir = np.zeros((nx, ny, nz, 3, 3))
        
        # ================= Initialization =================
        # Q0
        if barrier_generator is None:
             self.barrier_generator = get_barrier_generator("gaussian", mean=2.0, std=0.6)
        elif isinstance(barrier_generator, str):
             self.barrier_generator = get_barrier_generator(barrier_generator, **barrier_kwargs)
        else:
             self.barrier_generator = barrier_generator
             
        self.Q0 = self.barrier_generator((nx, ny, nz, M))

        # Catalog
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                     self.catalog[x,y,z] = stz_catalog_glass(M, self.gamma0)

        # Analysis State
        self.eps_macro = np.zeros((3,3)) 
        self.time = 0.0
        self.history_global = [] 
        self.solver_args = {}

    def _init_logs(self, summary_filename="summary_log.txt"):
        self.log_global_path = os.path.join(self.output_dir, "global_log.txt")
        self.cascade_log_path = os.path.join(self.output_dir, "cascade_log.txt")
        self.summary_log_path = os.path.join(self.output_dir, summary_filename)
        
        open(self.log_global_path, 'w').close()
        open(self.cascade_log_path, 'w').close()
        
        summary_header = f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} {'Type':<10} {'Eps_xx':<12} {'Sig_xx(GPa)':<12} {'KMC':<8} {'Cascade':<8} {'Flips':<8}\n"
        with open(self.summary_log_path, "w") as f:
            f.write(summary_header)
            f.write("-" * len(summary_header) + "\n")

        header_fmt = "{:<10} {:<12} {:<10} " + " ".join(["{:<15}"]*12) + " {:<14} {:<17}"
        headers = [
            "GlobalStep", "ElasticStep", "KMCStep",
            "Eps_xx", "Eps_yy", "Eps_zz", "Eps_xy", "Eps_xz", "Eps_yz",
            "Sig_xx(GPa)", "Sig_yy(GPa)", "Sig_zz(GPa)", "Sig_xy(GPa)", "Sig_xz(GPa)", "Sig_yz(GPa)",
            "CascadeSteps", "TotalCascadeFlips"
        ]
        with open(self.log_global_path, "w") as f:
            f.write(header_fmt.format(*headers) + "\n")
            
        cascade_header_fmt = "{:<12} {:<12} {:<15} {:<30}"
        with open(self.cascade_log_path, "w") as f:
            f.write(cascade_header_fmt.format("GlobalStep", "LocalStep", "NumUnstable", "FlippedVoxels(x,y,z,mode)") + "\n")

    def log_global(self, global_step, elastic_step, kmc_step, eps, sig, cascade_steps, total_flips):
        indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
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
        ratio = np.exp(-dt_elastic / dt_kmc) if dt_kmc > 0 else 0.0
        x, y, z, m = event_idx
        event_str = f"({x},{y},{z},{m})"
        with open(path, "a") as f:
            f.write(fmt_data.format(global_step, kmc_step, dt_elastic, dt_kmc, ratio, event_str, barrier_ev))

    def log_cascade(self, global_step, local_step, event_idx, barrier, n_unstable=1):
        path = self.cascade_log_path
        try:
            if hasattr(event_idx, 'ndim') and event_idx.ndim == 2:
                indices = event_idx
            elif isinstance(event_idx, (list, tuple)) and len(event_idx) == 4 and isinstance(event_idx[0], (int, np.integer)):
                indices = [event_idx]
            else:
                indices = event_idx
        except:
            indices = []

        if len(indices) > 50:
            flip_str = f"{len(indices)} voxels flipped (truncated)"
        else:
            parts = [f"({x},{y},{z},{m})" for x, y, z, m in indices]
            flip_str = ";".join(parts)
        
        line_fmt = "{:<12d} {:<12d} {:<15d} {:<30}\n"
        with open(path, "a") as f:
            f.write(line_fmt.format(global_step, local_step, n_unstable, flip_str))

    def update_barriers(self):
        scheme = 0 if self.softening_scheme == "isotropic" else 1
        compute_barrier(self.Q, self.Q0, self.sig_field, self.catalog, self.volume,
                        self.soft_prop, self.last_event_time, self.time, 
                        self.prev_strain_dir, self.softening_cap,
                        scheme, self.tau)

    def _run_cascade(self, global_step):
        local_step = 0
        total_flips = 0

        while True:
            self.update_barriers()
            unstable_indices = find_unstable(self.Q, self.stability_threshold)
            n_unstable = len(unstable_indices)
            if n_unstable == 0:
                break
            
            # Sort by barrier value (lowest first) to prioritize the most unstable modes
            q_values = self.Q[unstable_indices[:,0], unstable_indices[:,1], unstable_indices[:,2], unstable_indices[:,3]]
            sort_idx = np.argsort(q_values)
            unstable_indices = unstable_indices[sort_idx]

            flipped_indices = []
            flipped_voxels_in_batch = set()
            for k in range(n_unstable):
                 ux, uy, uz, um = unstable_indices[k]
                 voxel_id = (ux, uy, uz)
                 if voxel_id in flipped_voxels_in_batch:
                      continue
                 
                 flipped_voxels_in_batch.add(voxel_id)
                 flipped_indices.append((ux, uy, uz, um))
                 
                 apply_flip_soa(self.eps_plastic, None, self.soft_prop, self.last_event_time,
                                self.catalog, ux, uy, uz, um, self.time, 
                                self.jp, self.jt, self.softening_cap)
                 
                 self.prev_strain_dir[ux,uy,uz] = self.catalog[ux,uy,uz,um]
                 
                 if self.redraw_directions or self.redraw_barriers:
                      if self.redraw_directions:
                           self.catalog[ux,uy,uz] = stz_catalog_glass(self.M, self.gamma0)
                      if self.redraw_barriers:
                           self.Q0[ux,uy,uz] = self.barrier_generator((self.M,))
                 else:
                      self.catalog[ux,uy,uz,um] = stz_catalog_glass(1, self.gamma0)[0]
                      self.Q0[ux,uy,uz,um] = self.barrier_generator((1,))[0]
            
            n_flipped = len(flipped_indices)
            self.log_cascade(global_step, local_step, flipped_indices, 0.0, n_unstable=n_flipped)
            total_flips += n_flipped
            
            self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                   self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
            )
            
            local_step += 1
            if local_step > 20000:
                 print("Cascade limit reached (20000 steps)")
                 break
                 
        eps_curr = self.eps_field.mean(axis=(0,1,2))
        sig_curr = self.sig_field.mean(axis=(0,1,2))
        return local_step, total_flips, eps_curr, sig_curr

    def run_mixed(self, n_global_steps, strain_rate, component=(0,1), 
                  stress_targets={}, mixed_tol=1e-4, mixed_max_iter=50,
                  kmc_mode="accumulate", 
                  checkpoint_interval=None, checkpoint_path="checkpoint", 
                  checkpoint_mode="periodic", 
                  stop_on_stress_drop=None, stress_drop_component=(0,1), stop_post_drop_steps=20,
                  vtk_mode=None, ignore_drop_steps=0,
                  checkpoint_elastic_only=False,
                  enable_console_log=True,
                  summary_filename="summary_log.txt",
                  stress_drop_lookback=1,
                  enable_save_q=False,
                  save_q_interval=None,
                  save_q_elastic_only=False):
        
        if not stress_targets:
             if component == (0,0):
                 stress_targets[(1,1)] = 0.0
                 stress_targets[(2,2)] = 0.0
             elif component == (0,1):
                 stress_targets[(0,0)] = 0.0
                 stress_targets[(1,1)] = 0.0
                 stress_targets[(2,2)] = 0.0

        self._init_logs(summary_filename=summary_filename)
        start_time_total = time.time()
        
        dt_elastic = abs(strain_rate) / self.strain_rate if self.strain_rate > 0 else 1.0
        _, _, eps_curr, sig_curr = self._run_cascade(global_step=0)
        
        if enable_save_q and save_q_interval is not None:
             np.save(os.path.join(self.output_dir, "Q_step_000000.npy"), self.Q)

        elastic_chk_id = 0
        if checkpoint_interval is not None and checkpoint_mode in ["periodic", "current"]:
             cp_name = f"{checkpoint_path}.h5" if checkpoint_mode == "current" else \
                       (f"{checkpoint_path}_elastic_{elastic_chk_id:06d}.h5" if checkpoint_elastic_only else f"{checkpoint_path}_000000.h5")
             if checkpoint_elastic_only: elastic_chk_id += 1
             self.save_checkpoint(cp_name, step=0)
        
        stress_history = [sig_curr[stress_drop_component]]
        stop_drop_triggered = False
        stop_countdown = stop_post_drop_steps
        step = 1
        elastic_steps_done = 0
        total_kmc_steps = 0
        E_avg, nu_avg = np.mean(self.E), np.mean(self.nu)
        
        def get_correction_legacy(sigma_err):
             tr_sig = np.trace(sigma_err)
             return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg
        
        last_step_type = "elastic"
        kmc_baseline_stress = None

        while elastic_steps_done < n_global_steps:
             iteration_steps, iteration_flips = 0, 0
             while True:
                 self.update_barriers()
                 unstable_indices = find_unstable(self.Q, self.stability_threshold)
                 if len(unstable_indices) > 0:
                     l, f, _, _ = self._run_cascade(step)
                     iteration_steps += l
                     iteration_flips += f
                     continue
                 
                 rates_flat, indices_flat, total_rate = compute_rates(self.Q, self.volume, self.temperature)
                 idx_flat, dt_kmc = select_event(rates_flat, total_rate)
                 trigger = np.exp(-dt_elastic / dt_kmc) if dt_kmc > 0 else 0.0
                 
                 if self.temperature > 0 and np.random.uniform() > trigger:
                      if idx_flat == -1: continue 
                      x, y, z, m = decode_index(indices_flat[idx_flat], self.grid_shape[1], self.grid_shape[2], self.M)
                      barrier_val = self.Q[x,y,z,m]
                      apply_flip_soa(self.eps_plastic, None, self.soft_prop, self.last_event_time,
                                     self.catalog, x, y, z, m, self.time, self.jp, self.jt, self.softening_cap)
                      self.prev_strain_dir[x,y,z] = self.catalog[x,y,z,m]
                      
                      if self.redraw_directions or self.redraw_barriers:
                           if self.redraw_directions: self.catalog[x,y,z] = stz_catalog_glass(self.M, self.gamma0)
                           if self.redraw_barriers: self.Q0[x,y,z] = self.barrier_generator((self.M,))
                      else:
                           self.catalog[x,y,z,m] = stz_catalog_glass(1, self.gamma0)[0]
                           self.Q0[x,y,z,m] = self.barrier_generator((1,))[0]
                      
                      self.log_kmc(step, total_kmc_steps, dt_kmc, dt_elastic, (x,y,z,m), barrier_val)
                      self.time += dt_kmc
                      total_kmc_steps += 1
                      last_step_type = "kmc"
                      if kmc_baseline_stress is None:
                           kmc_baseline_stress = sig_curr[stress_drop_component]
                      
                      self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                           self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args)
                      
                      # Stress drop detection for KMC
                      sig_curr = self.sig_field.mean(axis=(0,1,2))
                      curr_stress_val = sig_curr[stress_drop_component]
                      if stop_on_stress_drop is not None and not stop_drop_triggered and step > ignore_drop_steps:
                           if kmc_baseline_stress is not None:
                                drop_frac = (kmc_baseline_stress - curr_stress_val) / kmc_baseline_stress if abs(kmc_baseline_stress) > 1e-6 else 0.0
                                if drop_frac > stop_on_stress_drop:
                                     print(f"\n[ALERT] KMC Stress Drop Detected! {drop_frac*100:.1f}% > {stop_on_stress_drop*100:.1f}% (Cumulative since start of KMC sequence) at step {step}")
                                     stop_drop_triggered = True
                      stress_history.append(curr_stress_val)
                      if stop_drop_triggered and kmc_mode == "accumulate": break
                      if kmc_mode == "on_demand": break
                 else:
                      self.time += dt_elastic
                      eps_inc = np.zeros((3,3))
                      eps_inc[component] = strain_rate
                      self.eps_macro += eps_inc
                      self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                           self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args)
                      
                      converged = False
                      for it in range(mixed_max_iter):
                          l, f, _, sig_M = self._run_cascade(step)
                          iteration_steps += l
                          iteration_flips += f
                          stress_err_tensor = np.zeros((3,3))
                          max_err = 0.0
                          for idx_t, target_val in stress_targets.items():
                              err = target_val - sig_M[idx_t]
                              stress_err_tensor[idx_t] = err
                              max_err = max(max_err, abs(err))
                          if max_err < mixed_tol:
                              converged = True
                              break
                          eps_corr = get_correction_legacy(stress_err_tensor)
                          eps_corr[component] = 0.0
                          self.eps_macro += eps_corr
                          self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                               self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args)
                      if not converged: print(f"Warning: Mixed loop did not converge at step {step} (Err={max_err:.2e})")
                      elastic_steps_done += 1
                      last_step_type = "elastic"
                      kmc_baseline_stress = None
                      
                      # Stress drop detection for Elastic
                      sig_curr = self.sig_field.mean(axis=(0,1,2))
                      curr_stress_val = sig_curr[stress_drop_component]
                      if stop_on_stress_drop is not None and not stop_drop_triggered and step > ignore_drop_steps:
                           lookback = max(1, stress_drop_lookback)
                           if len(stress_history) >= lookback:
                                ref_stress = stress_history[-lookback]
                                drop_frac = (ref_stress - curr_stress_val) / ref_stress if abs(ref_stress) > 1e-6 else 0.0
                                if drop_frac > stop_on_stress_drop:
                                     print(f"\n[ALERT] Elastic Stress Drop Detected! {drop_frac*100:.1f}% > {stop_on_stress_drop*100:.1f}% at step {step}")
                                     stop_drop_triggered = True
                      stress_history.append(curr_stress_val)
                      break
             
             eps_curr, sig_curr = self.eps_field.mean(axis=(0,1,2)), self.sig_field.mean(axis=(0,1,2))
             self.log_global(step, elastic_steps_done, total_kmc_steps, eps_curr, sig_curr, iteration_steps, iteration_flips)
             self.history_global.append((eps_curr[0,0], sig_curr[0,0]/1e9))
             curr_stress_val, curr_strain_val = sig_curr[stress_drop_component], eps_curr[stress_drop_component]
             now, elapsed = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time_total
             status_msg = f"[{now}] [{elapsed:8.2f}s] Step {step:4d}: Type={last_step_type.upper():<8}, KMC={total_kmc_steps:4d}, Cascade={iteration_steps:4d}, Flips={iteration_flips:4d}, Eps_xx={curr_strain_val:8.6f}, Sig_xx={curr_stress_val/1e9:6.3f} GPa"
             if stop_drop_triggered:
                  status_msg += " [SB DETECTED]"
             summary_line = f"{now:<20} {elapsed:<12.2f} {step:<8d} {last_step_type.upper():<10} {curr_strain_val:<12.6f} {curr_stress_val/1e9:<12.3f} {total_kmc_steps:<8d} {iteration_steps:<8d} {iteration_flips:<8d}\n"
             with open(self.summary_log_path, "a") as f: f.write(summary_line)
             
             should_save = checkpoint_interval and step % checkpoint_interval == 0 and checkpoint_mode in ["periodic", "current"]
             if should_save and checkpoint_elastic_only and last_step_type != "elastic": should_save = False
             if should_save:
                  cp_name = f"{checkpoint_path}.h5" if checkpoint_mode == "current" else \
                            (f"{checkpoint_path}_elastic_{elastic_chk_id:06d}.h5" if checkpoint_elastic_only else f"{checkpoint_path}_{step:06d}.h5")
                  if checkpoint_elastic_only: elastic_chk_id += 1
                  self.save_checkpoint(cp_name, step=step)

             if enable_save_q and save_q_interval and step % save_q_interval == 0:
                  should_save_q = True
                  if save_q_elastic_only and last_step_type != "elastic":
                       should_save_q = False
                  if should_save_q:
                       np.save(os.path.join(self.output_dir, f"Q_step_{step:06d}.npy"), self.Q)
             if enable_console_log: print(status_msg)
             if stop_drop_triggered:
                  if stop_countdown > 0: stop_countdown -= 1
                  else: print(f"Stopping criteria: {stop_post_drop_steps} steps after detection."); break
             step += 1

        if checkpoint_mode == "last": self.save_checkpoint(f"{checkpoint_path}_final.h5", step=step-1)
        elif checkpoint_mode in ["periodic", "current"] and checkpoint_interval is not None and (step-1) % checkpoint_interval != 0:
             self.save_checkpoint(f"{checkpoint_path}_final.h5", step=step-1)

        total_time = time.time() - start_time_total
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        duration_str = f"\nSimulation Finish Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {total_time:.2f} seconds ({int(h):d}h {int(m):02d}m {int(s):02d}s)\n"
        with open(self.summary_log_path, "a") as f: f.write(duration_str)
        if enable_console_log: print(duration_str)

    def run(self, *args, **kwargs):
        print("Use run_mixed instead.")
        pass

    def save_checkpoint(self, path, step=None):
         if not path.endswith('.h5'): path += '.h5'
         os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
         import h5py
         max_retries = 3
         for attempt in range(max_retries):
             try:
                 with h5py.File(path, "w") as f:
                     meta = f.create_group('metadata')
                     meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz'] = self.grid_shape
                     meta.attrs['M'], meta.attrs['gamma0'], meta.attrs['pixel'] = self.M, self.gamma0, self.pixel
                     meta.attrs['timestamp'], meta.attrs['step'] = datetime.now().isoformat(), step if step is not None else 0
                     fields = f.create_group('fields')
                     fields.create_dataset('eps_field', data=self.eps_field, compression='gzip')
                     fields.create_dataset('sig_field', data=self.sig_field, compression='gzip')
                     fields.create_dataset('E_field', data=self.E_field, compression='gzip')
                     fields.create_dataset('nu_field', data=self.nu_field, compression='gzip')
                     grid = f.create_group('grid')
                     grid.create_dataset('eps_plastic', data=self.eps_plastic, compression='gzip')
                     grid.create_dataset('soft_prop', data=self.soft_prop, compression='gzip')
                     grid.create_dataset('Q', data=self.Q, compression='gzip')
                     grid.create_dataset('Q0', data=self.Q0, compression='gzip')
                     grid.create_dataset('catalog', data=self.catalog, compression='gzip')
                     grid.create_dataset('last_event_time', data=self.last_event_time, compression='gzip')
                 return
             except (ImportError, Exception) as e:
                 if attempt < max_retries - 1: time.sleep(0.5)
                 else: print(f"Error saving checkpoint after {max_retries} attempts: {e}")
