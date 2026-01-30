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
                 redraw_barriers=True,    # Redraw all Q0 in voxel after flip
                 max_cascade_steps_pct=0.3, # Stop cascade if steps > this pct of voxels
                 nu0=1e13,                 # Attempt frequency (Hz)
                 q_act_temp=0.37           # Activation barrier for JT recovery (eV)
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
        self.max_cascade_steps_pct = max_cascade_steps_pct
        self.nu0 = nu0
        self.q_act_temp = q_act_temp

        # Dynamic calculation of tau if not provided
        if self.temperature > 0 and (self.tau is None or self.tau == np.inf):
            kB = 8.617e-5 # eV/K
            # tau = 1 / (nu0 * exp(-q_act_temp / (kB * T)))
            self.tau = 1.0 / (self.nu0 * np.exp(-self.q_act_temp / (kB * self.temperature)))
            print(f"Calculated Dynamic Softening Decay Time Constant (tau): {self.tau:.4e} s")
        elif self.tau == np.inf:
            print("Softening Decay Time Constant (tau): Infinite (No decay)")
        else:
            print(f"Using Provided Softening Decay Time Constant (tau): {self.tau:.4e} s")
        
        # Grid Setup (Arrays)
        self.grid_shape = (nx, ny, nz)
        self.total_voxels = nx * ny * nz
        
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

    def _init_logs(self, summary_filename="summary_log.txt", 
                   enable_summary_log=True, enable_global_log=True, 
                   enable_cascade_log=True, enable_kmc_log=True):
        self.log_global_path = os.path.join(self.output_dir, "global_log.txt")
        self.cascade_log_path = os.path.join(self.output_dir, "cascade_log.txt")
        self.summary_log_path = os.path.join(self.output_dir, summary_filename)
        self.kmc_log_path = os.path.join(self.output_dir, "kmc_log.txt")
        
        self.enable_summary_log = enable_summary_log
        self.enable_global_log = enable_global_log
        self.enable_cascade_log = enable_cascade_log
        self.enable_kmc_log = enable_kmc_log

        # Open handles with line buffering to reduce disk hits on WSL/NTFS
        self._f_global = open(self.log_global_path, "w", buffering=1) if enable_global_log else None
        self._f_cascade = open(self.cascade_log_path, "w", buffering=1) if enable_cascade_log else None
        self._f_summary = open(self.summary_log_path, "w", buffering=1) if enable_summary_log else None
        self._f_kmc = None # Lazy open
        
        if self._f_summary:
            summary_header = f"{'Timestamp':<20} {'Elapsed(s)':<12} {'Step':<8} {'Type':<10} {'Eps_xx':<12} {'Sig_xx(GPa)':<12} {'KMC':<8} {'Cascade':<8} {'Flips':<8} {'SimTime(s)':<15}\n"
            self._f_summary.write(summary_header)
            self._f_summary.write("-" * len(summary_header) + "\n")

        if self._f_global:
            header_fmt = "{:<10} {:<12} {:<10} " + " ".join(["{:<15}"]*12) + " {:<14} {:<17} {:<15}"
            headers = [
                "GlobalStep", "ElasticStep", "KMCStep",
                "Eps_xx", "Eps_yy", "Eps_zz", "Eps_xy", "Eps_xz", "Eps_yz",
                "Sig_xx(GPa)", "Sig_yy(GPa)", "Sig_zz(GPa)", "Sig_xy(GPa)", "Sig_xz(GPa)", "Sig_yz(GPa)",
                "CascadeSteps", "TotalCascadeFlips", "SimTime(s)"
            ]
            self._f_global.write(header_fmt.format(*headers) + "\n")
            
        if self._f_cascade:
            cascade_header_fmt = "{:<12} {:<12} {:<15} {:<30}"
            self._f_cascade.write(cascade_header_fmt.format("GlobalStep", "LocalStep", "NumUnstable", "FlippedVoxels(x,y,z,mode)") + "\n")

    def _close_logs(self):
        for attr in ['_f_global', '_f_cascade', '_f_summary', '_f_kmc']:
            f = getattr(self, attr, None)
            if f:
                f.close()
                setattr(self, attr, None)

    def log_global(self, global_step, elastic_step, kmc_step, time_sim, eps, sig, cascade_steps, total_flips):
        if self._f_global:
            fmt = "{:<10d} {:<12d} {:<10d} " + " ".join(["{:<15.6f}"]*6) + " " + " ".join(["{:<15.3f}"]*6) + " {:<14d} {:<17d} {:<15.6e}\n"
            data = [
                global_step, elastic_step, kmc_step,
                eps[0,0], eps[1,1], eps[2,2], eps[0,1], eps[0,2], eps[1,2],
                sig[0,0]/1e9, sig[1,1]/1e9, sig[2,2]/1e9, sig[0,1]/1e9, sig[0,2]/1e9, sig[1,2]/1e9,
                cascade_steps, total_flips, time_sim
            ]
            self._f_global.write(fmt.format(*data))

    def log_kmc(self, global_step, kmc_step, dt_kmc, dt_elastic, event_idx, barrier_ev):
        fmt_header = "{:<10} {:<10} {:<15} {:<15} {:<15} {:<20} {:<15}\n"
        fmt_data   = "{:<10d} {:<10d} {:<15.6e} {:<15.6e} {:<15.6e} {:<20} {:<15.6f}\n"
        
        if self._f_kmc is None:
            if not self.enable_kmc_log:
                return
            if not os.path.exists(self.kmc_log_path):
                 with open(self.kmc_log_path, "w") as f:
                      f.write(fmt_header.format("GlobalStep", "KMCStep", "DtElastic", "DtKMC", "e^(-DtE/DtK)", "Event(x,y,z,m)", "Barrier(eV)"))
            self._f_kmc = open(self.kmc_log_path, "a", buffering=1)

        # dt_elastic here refers to the time duration of the elastic increment that was being processed
        # dt_kmc here refers to the actual waiting time for the KMC event
        ratio = np.exp(-dt_elastic / dt_kmc) if dt_kmc > 0 else 0.0 # This ratio is not directly used in the new scheme, but kept for log consistency
        x, y, z, m = event_idx
        event_str = f"({x},{y},{z},{m})"
        self._f_kmc.write(fmt_data.format(global_step, kmc_step, dt_elastic, dt_kmc, ratio, event_str, barrier_ev))

    def log_cascade(self, global_step, local_step, event_idx, barrier, n_unstable=1):
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
        if self._f_cascade:
            self._f_cascade.write(line_fmt.format(global_step, local_step, n_unstable, flip_str))

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
            max_cascade_steps = int(self.max_cascade_steps_pct * self.total_voxels)
            if local_step > max_cascade_steps:
                 print(f"Cascade limit reached ({local_step} steps > {self.max_cascade_steps_pct*100:.1f}% of {self.total_voxels} voxels)")
                 return local_step, total_flips, self.eps_field.mean(axis=(0,1,2)), self.sig_field.mean(axis=(0,1,2)), True
                 
        eps_curr = self.eps_field.mean(axis=(0,1,2))
        sig_curr = self.sig_field.mean(axis=(0,1,2))
        return local_step, total_flips, eps_curr, sig_curr, False

    def run_mixed(self, n_global_steps, strain_rate, component=(0,1), 
                  stress_targets={}, mixed_tol=1e-4, mixed_max_iter=50,
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
                  save_q_elastic_only=False,
                  max_kmc_steps_pct=0.3,
                  enable_summary_log=True,
                  enable_global_log=True,
                  enable_cascade_log=True,
                  enable_kmc_log=True):
        
        if not stress_targets:
             if component == (0,0):
                 stress_targets[(1,1)] = 0.0
                 stress_targets[(2,2)] = 0.0
             elif component == (0,1):
                 stress_targets[(0,0)] = 0.0
                 stress_targets[(1,1)] = 0.0
                 stress_targets[(2,2)] = 0.0

        self._init_logs(summary_filename=summary_filename,
                        enable_summary_log=enable_summary_log,
                        enable_global_log=enable_global_log,
                        enable_cascade_log=enable_cascade_log,
                        enable_kmc_log=enable_kmc_log)
        start_time_total = time.time()
        
        # dt_elastic_increment is the time duration for one macroscopic elastic strain increment
        dt_elastic_increment = abs(strain_rate) / self.strain_rate if self.strain_rate > 0 else 1.0
        _, _, eps_curr, sig_curr, truncated = self._run_cascade(global_step=0)
        if truncated: return
        
        if enable_save_q and save_q_interval is not None:
             np.save(os.path.join(self.output_dir, "Q_step_000000.npy"), self.Q)

        # Pre-calculate unit strain tensor for loading
        strain_unit_tensor = np.zeros((3,3))
        strain_unit_tensor[component] = 1.0

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
        sequential_kmc_steps = 0
        max_sequential_kmc = int(max_kmc_steps_pct * self.total_voxels)

        def _do_logging(current_step, step_type, cascade_steps, cascade_flips):
            eps_curr, sig_curr = self.eps_field.mean(axis=(0,1,2)), self.sig_field.mean(axis=(0,1,2))
            self.log_global(current_step, elastic_steps_done, total_kmc_steps, self.time, eps_curr, sig_curr, cascade_steps, cascade_flips)
            self.history_global.append((eps_curr[0,0], sig_curr[0,0]/1e9))
            
            curr_stress_val, curr_strain_val = sig_curr[stress_drop_component], eps_curr[stress_drop_component]
            now, elapsed = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time_total
            
            status_msg = f"[{now}] [{elapsed:7.1f}s | {self.time:8.2e}s] Step {current_step:4d}: Type={step_type.upper():<8}, KMC={total_kmc_steps:4d}, Cascade={cascade_steps:d}, Eps_xx={curr_strain_val:8.6f}, Sig={curr_stress_val/1e9:6.3f} GPa"
            if stop_drop_triggered:
                status_msg += " [SB DETECTED]"
            
            summary_line = f"{now:<20} {elapsed:<12.2f} {current_step:<8d} {step_type.upper():<10} {curr_strain_val:<12.6f} {curr_stress_val/1e9:<12.3f} {total_kmc_steps:<8d} {cascade_steps:<8d} {cascade_flips:<8d} {self.time:<15.6e}\n"
            
            if self._f_summary:
                self._f_summary.write(summary_line)
            if enable_console_log:
                print(status_msg)
            
            return curr_stress_val

        while elastic_steps_done < n_global_steps:
             iteration_steps, iteration_flips = 0, 0
             remaining_time = dt_elastic_increment # Time left in the current macroscopic elastic increment
             
             while remaining_time > 0:
                 self.update_barriers()
                 unstable_indices = find_unstable(self.Q, self.stability_threshold)
                 if len(unstable_indices) > 0:
                    l, f, _, _, truncated = self._run_cascade(step)
                    iteration_steps += l
                    iteration_flips += f
                    if truncated: return
                    # After cascade, re-evaluate KMC rates for the remaining time
                    continue
                 
                 rates_flat, indices_flat, total_rate = compute_rates(self.Q, self.volume, self.temperature, self.nu0)
                 
                 if self.temperature > 0 and total_rate > 0:
                      # Use standard KMC waiting time (Poisson process)
                      u = np.random.uniform()
                      t_wait = -np.log(u) / total_rate
                      
                      # Competition: does event happen before the rest of this elastic increment?
                      if t_wait < remaining_time:
                           # Thermal event wins
                           # Apply partial strain for the time that passed
                           d_eps = strain_unit_tensor * (self.strain_rate * t_wait)
                           self.eps_macro += d_eps

                           idx_flat = select_event(rates_flat, total_rate) # Selects one event based on rates
                           if idx_flat == -1: # Should not happen if total_rate > 0, but as a safeguard
                                # We already advanced eps_macro, but no event. 
                                # Time will be advanced below.
                                self.time += remaining_time
                                remaining_time = 0
                                continue
                           
                           x, y, z, m = decode_index(indices_flat[idx_flat], self.grid_shape[1], self.grid_shape[2], self.M)
                           barrier_val = self.Q[x,y,z,m]
                           apply_flip_soa(self.eps_plastic, None, self.soft_prop, self.last_event_time,
                                          self.catalog, x, y, z, m, self.time + t_wait, self.jp, self.jt, self.softening_cap)
                           self.prev_strain_dir[x,y,z] = self.catalog[x,y,z,m]
                           
                           if self.redraw_directions or self.redraw_barriers:
                                if self.redraw_directions: self.catalog[x,y,z] = stz_catalog_glass(self.M, self.gamma0)
                                if self.redraw_barriers: self.Q0[x,y,z] = self.barrier_generator((self.M,))
                           else:
                                self.catalog[x,y,z,m] = stz_catalog_glass(1, self.gamma0)[0]
                                self.Q0[x,y,z,m] = self.barrier_generator((1,))[0]
                           
                           self.log_kmc(step, total_kmc_steps, remaining_time, t_wait, (x,y,z,m), barrier_val)
                           self.time += t_wait
                           remaining_time -= t_wait
                           total_kmc_steps += 1
                           last_step_type = "kmc"
                           
                           # Check for immediate cascades triggered by this KMC event
                           l_kmc, f_kmc, _, _, truncated = self._run_cascade(step)
                           if truncated: return
                           
                           # If no cascade happened, we still need to recompute stress for the lone KMC flip
                           if l_kmc == 0:
                               self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                                    self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args)
                           
                           # Log this KMC event specifically, including any cascades that happened just before or as a result of it
                           curr_stress_val = _do_logging(step, "kmc", iteration_steps + l_kmc, iteration_flips + f_kmc)
                           
                           # Reset accumulators for the rest of the increment
                           iteration_steps, iteration_flips = 0, 0
                           
                           # Stress drop detection for KMC
                           if stop_on_stress_drop is not None and not stop_drop_triggered and step > ignore_drop_steps:
                                if kmc_baseline_stress is None:
                                     kmc_baseline_stress = curr_stress_val 
                                drop_frac = (abs(kmc_baseline_stress) - abs(curr_stress_val)) / abs(kmc_baseline_stress) if abs(kmc_baseline_stress) > 1e-6 else 0.0
                                if drop_frac > stop_on_stress_drop:
                                     print(f"\n[ALERT] KMC Stress Drop Detected! {drop_frac*100:.1f}% > {stop_on_stress_drop*100:.1f}% (Cumulative since start of KMC sequence) at step {step}")
                                     stop_drop_triggered = True
                           
                           stress_history.append(curr_stress_val)
                           sequential_kmc_steps += 1
                           step += 1 # KMC counts as a step
                           
                           if sequential_kmc_steps > max_sequential_kmc:
                                print(f"\n[TERMINATE] KMC sequence limit reached! {sequential_kmc_steps} consecutive steps.")
                                return
                           continue
                      else:
                           # Elastic loading wins (uses up remaining time)
                           d_eps = strain_unit_tensor * (self.strain_rate * remaining_time)
                           self.eps_macro += d_eps
                           self.time += remaining_time
                           remaining_time = 0
                 else:
                      # Zero temperature or zero rate -> strictly elastic
                      d_eps = strain_unit_tensor * (self.strain_rate * remaining_time)
                      self.eps_macro += d_eps
                      self.time += remaining_time
                      remaining_time = 0
                      
             # Step-end stress update (incorporates all strain added in the KMC loop)
             self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                  self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args)
             
             converged = False
             for it in range(mixed_max_iter):
                 l, f, _, sig_M, truncated = self._run_cascade(step)
                 iteration_steps += l
                 iteration_flips += f
                 if truncated: return
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
             sequential_kmc_steps = 0
             
             # Final logging for this elastic step
             curr_stress_val = _do_logging(step, "elastic", iteration_steps, iteration_flips)
             
             # Stress drop detection for Elastic
             if stop_on_stress_drop is not None and not stop_drop_triggered and step > ignore_drop_steps:
                  lookback = max(1, stress_drop_lookback)
                  if len(stress_history) >= lookback:
                       ref_stress = stress_history[-lookback]
                       drop_frac = (abs(ref_stress) - abs(curr_stress_val)) / abs(ref_stress) if abs(ref_stress) > 1e-6 else 0.0
                       if drop_frac > stop_on_stress_drop:
                            print(f"\n[ALERT] Elastic Stress Drop Detected! {drop_frac*100:.1f}% > {stop_on_stress_drop*100:.1f}% at step {step}")
                            stop_drop_triggered = True
             stress_history.append(curr_stress_val)
             
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
        if self._f_summary:
             self._f_summary.write(duration_str)
        if enable_console_log: print(duration_str)
        self._close_logs()

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
                     meta.attrs['sim_time'] = self.time
                     meta.attrs['nu0'] = self.nu0
                     meta.attrs['q_act_temp'] = self.q_act_temp
                     meta.attrs['temperature'] = self.temperature
                     meta.attrs['strain_rate'] = self.strain_rate
                     meta.attrs['stability_threshold'] = self.stability_threshold
                     meta.attrs['jp'] = self.jp
                     meta.attrs['jt'] = self.jt
                     meta.attrs['softening_cap'] = self.softening_cap
                     meta.attrs['softening_scheme'] = self.softening_scheme
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
                     
                     state = f.create_group('state')
                     state.create_dataset('eps_macro', data=self.eps_macro)
                 return
             except (ImportError, Exception) as e:
                 if attempt < max_retries - 1: time.sleep(0.5)
                 else: print(f"Error saving checkpoint after {max_retries} attempts: {e}")

    @classmethod
    def load_checkpoint(cls, path):
         """
         Load AthermalSimulation from an HDF5 checkpoint.
         """
         import h5py
         with h5py.File(path, "r") as f:
             meta = f['metadata']
             nx, ny, nz = meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz']
             M, gamma0, pixel = meta.attrs['M'], meta.attrs['gamma0'], meta.attrs['pixel']
             
             # Fields
             fields = f['fields']
             E_field = fields['E_field'][:]
             nu_field = fields['nu_field'][:]
             
             # Create instance
             sim = cls(nx, ny, nz, M=M, gamma0=gamma0, E_field=E_field, nu_field=nu_field, 
                       pixel=pixel, temperature=meta.attrs['temperature'], 
                       strain_rate=meta.attrs['strain_rate'], jp=meta.attrs['jp'], jt=meta.attrs['jt'],
                       softening_cap=meta.attrs['softening_cap'], softening_scheme=meta.attrs['softening_scheme'],
                       nu0=meta.attrs.get('nu0', 1e13), q_act_temp=meta.attrs.get('q_act_temp', 0.37))
             
             # Restore dynamic state
             sim.time = meta.attrs.get('sim_time', 0.0)
             sim.eps_field = fields['eps_field'][:]
             sim.sig_field = fields['sig_field'][:]
             
             grid = f['grid']
             sim.eps_plastic = grid['eps_plastic'][:]
             sim.soft_prop = grid['soft_prop'][:]
             sim.Q = grid['Q'][:]
             sim.Q0 = grid['Q0'][:]
             sim.catalog = grid['catalog'][:]
             sim.last_event_time = grid['last_event_time'][:]
             
             if 'state' in f:
                  sim.eps_macro = f['state']['eps_macro'][:]
                  
             print(f"Loaded checkpoint from {path} (SimTime={sim.time:.4e}s, Step={meta.attrs['step']})")
             return sim
