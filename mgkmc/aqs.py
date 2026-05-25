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

class ThermalSimulation:
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
                 neighbor_softening_fraction=0.0,
                 output_dir="output",
                 temperature=0.0, # Kelvin
                 strain_rate=1.0, # 1/s, used for KMC decision
                 stability_threshold=0.0, # eV, threshold for athermal instability
                 redraw_directions=True, # Redraw all modes in voxel after flip
                 redraw_barriers=True,    # Redraw all Q0 in voxel after flip
                 max_cascade_steps_pct=0.3, # Stop cascade if steps > this pct of voxels
                 nu0=1e13,                 # Attempt frequency (Hz)
                 q_act_temp=0.37,          # Activation barrier for JT recovery (eV)
                 instability_mode="cascade", # "cascade" or "kmc"
                 cascade_timing="none",      # "none", "single", or "per_flip"
                 scale_rate_by_volume=True,
                 fast_patching=None
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
        self.neighbor_softening_fraction = neighbor_softening_fraction
        self.temperature = temperature
        self.strain_rate = strain_rate
        self.stability_threshold = stability_threshold
        self.redraw_directions = redraw_directions
        self.redraw_barriers = redraw_barriers
        self.max_cascade_steps_pct = max_cascade_steps_pct
        self.nu0 = nu0
        self.q_act_temp = q_act_temp
        self.instability_mode = instability_mode
        self.cascade_timing = cascade_timing
        self.scale_rate_by_volume = scale_rate_by_volume

        # Internal Physics Calculation: tau (Relaxation Time)
        if self.temperature > 0:
            kB = 8.617e-5 # eV/K
            # tau = 1 / (nu0 * exp(-q_act_temp / (kB * T)))
            self.tau = 1.0 / (self.nu0 * np.exp(-self.q_act_temp / (kB * self.temperature)))
            print(f" [ThermalSimulation] T={self.temperature}K > 0: Calculated Dynamic Softening Decay (tau): {self.tau:.4e} s")
        else:
            self.tau = np.inf
            print(" [ThermalSimulation] T=0K: Softening Decay (tau): Infinite (No decay)")
        
        # Grid Setup (Arrays)
        self.grid_shape = (nx, ny, nz)
        self.total_voxels = nx * ny * nz
        
        # ================= SoA Arrays =================
        # 1. State Fields
        if E_field.mean() < 1e6:
            print(" [ThermalSimulation] Detected E in GPa (mean < 1 MPa). Converting to Pa (*1e9).")
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
        self.flip_event_history = []  # List of (global_step, local_step, x, y, z, mode)
        self.solver_args = {}
        
        # Fast Patching (Predictor-Corrector) Setup
        self.fast_patching_enabled = fast_patching.get('enabled', False) if fast_patching else False
        self.patch_radius = fast_patching.get('patch_radius', 3) if fast_patching else 3
        self.sync_interval = fast_patching.get('sync_interval', 100) if fast_patching else 100
        self.flips_since_sync = 0
        self.sigma_macro_unit = None
        if self.fast_patching_enabled:
            self._precompute_patch_kernels()

    def _precompute_patch_kernels(self):
        print("\n [ThermalSimulation] Pre-computing Truncated Green's Function Stress Patches...")
        R = self.patch_radius
        nx, ny, nz = self.grid_shape
        # 5 orthogonal basis trace-free pure shear tensors
        bases = [
            np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]),
            np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1.0], [0, 0, 0], [1.0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
        ]
        self.patch_kernels = []
        self.patch_missing_mean = []
        for P in bases:
            eps_plas = np.zeros((nx, ny, nz, 3, 3))
            eps_plas[nx//2, ny//2, nz//2] = P
            eps_mac = np.zeros((3,3))
            _, sig_field, _, _ = update_stress_fft_full(
                eps_plas, eps_mac, self.E, self.nu, pixel=self.pixel, **self.solver_args
            )
            
            sig_rolled = np.roll(sig_field, shift=(-(nx//2), -(ny//2), -(nz//2)), axis=(0,1,2))
            crop = np.zeros((2*R+1, 2*R+1, 1 if nz==1 else 2*R+1, 3, 3))
            for dx in range(-R, R+1):
                for dy in range(-R, R+1):
                    for dz in range(-R if nz>1 else 0, R+1 if nz>1 else 1):
                        crop[dx+R, dy+R, dz+R if nz>1 else 0] = sig_rolled[dx % nx, dy % ny, dz % nz]
            self.patch_kernels.append(crop)
            
            full_mean = np.mean(sig_field, axis=(0,1,2))
            crop_sum = np.sum(crop, axis=(0,1,2))
            # The missing uniform background shift to conserve volume expansion relaxation
            missing_mean = full_mean - (crop_sum / (nx * ny * np.maximum(1, nz)))
            if nz == 1:
                missing_mean = full_mean - (crop_sum / (nx * ny))
            self.patch_missing_mean.append(missing_mean)
            
        print(f" [ThermalSimulation] 5 Patch Kernels of radius {R} computed and cached.\n")

    def export_vtk(self, filename):
        """Export current state to VTK."""
        from mgkmc.analysis import export_to_vtk
        export_to_vtk(filename, self.eps_field, self.sig_field, self.E, self.nu, 
                      pixel=self.pixel, eps_plastic_field=self.eps_plastic, 
                      soft_prop_field=self.soft_prop, match_matplotlib_orientation=True)

    def _init_logs(self, summary_filename="summary_log.txt", 
                   enable_summary_log=True, enable_global_log=True, 
                   enable_cascade_log=True, enable_kmc_log=True,
                   append=False):
        self.log_global_path = os.path.join(self.output_dir, "global_log.txt")
        self.cascade_log_path = os.path.join(self.output_dir, "cascade_log.txt")
        self.summary_log_path = os.path.join(self.output_dir, summary_filename)
        self.kmc_log_path = os.path.join(self.output_dir, "kmc_log.txt")
        
        self.enable_summary_log = enable_summary_log
        self.enable_global_log = enable_global_log
        self.enable_cascade_log = enable_cascade_log
        self.enable_kmc_log = enable_kmc_log

        mode = "a" if append else "w"
        
        self._f_global = open(self.log_global_path, mode, buffering=1) if enable_global_log else None
        self._f_cascade = open(self.cascade_log_path, mode, buffering=1) if enable_cascade_log else None
        self._f_summary = open(self.summary_log_path, mode, buffering=1) if enable_summary_log else None
        self._f_kmc = None # Lazy open

        def is_empty(path):
            return not os.path.exists(path) or os.path.getsize(path) == 0

        if self._f_summary and (not append or is_empty(self.summary_log_path)):
            summary_header = f"{'Timestamp':<22} {'Elapsed(s)':<12} {'Step':<8} {'Type':<15} {'Eps_xx':<12} {'Sig_xx(GPa)':<15} {'KMC':<8} {'Cascade':<8} {'Flips':<8} {'SimTime(s)':<15}\n"
            self._f_summary.write(summary_header)
            self._f_summary.write("-" * len(summary_header) + "\n")

        if self._f_global and (not append or is_empty(self.log_global_path)):
            header_fmt = "{:<10} {:<12} {:<10} " + " ".join(["{:<15}"]*12) + " {:<14} {:<17} {:<15}"
            headers = [
                "GlobalStep", "ElasticStep", "KMCStep",
                "Eps_xx", "Eps_yy", "Eps_zz", "Eps_xy", "Eps_xz", "Eps_yz",
                "Sig_xx(GPa)", "Sig_yy(GPa)", "Sig_zz(GPa)", "Sig_xy(GPa)", "Sig_xz(GPa)", "Sig_yz(GPa)",
                "CascadeSteps", "TotalCascadeFlips", "SimTime(s)"
            ]
            self._f_global.write(header_fmt.format(*headers) + "\n")
            
        if self._f_cascade and (not append or is_empty(self.cascade_log_path)):
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

        ratio = np.exp(-dt_elastic / dt_kmc) if dt_kmc > 0 else 0.0
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

    def _run_cascade(self, global_step, vtk_prefix=None, track_cascades=False, strain_unit_tensor=None, eps_target=None, component=(0,0), log_callback=None):
        local_step = 0
        total_flips = 0

        if vtk_prefix:
            fname = f"{vtk_prefix}_step{global_step:06d}_cascade0000.vtu"
            self.export_vtk(fname)

        while True:
            self.update_barriers()
            unstable_indices = find_unstable(self.Q, self.stability_threshold)
            n_unstable = len(unstable_indices)
            if n_unstable == 0:
                break
            
            q_values = self.Q[unstable_indices[:,0], unstable_indices[:,1], unstable_indices[:,2], unstable_indices[:,3]]
            sort_idx = np.argsort(q_values)
            unstable_indices = unstable_indices[sort_idx]
            n_unstable = len(unstable_indices)

            flipped_indices = []
            flipped_voxels_in_batch = set()
            for k in range(n_unstable):
                ux, uy, uz, um = unstable_indices[k]
                voxel_id = (ux, uy, uz)
                if voxel_id in flipped_voxels_in_batch:
                    continue
                
                flipped_voxels_in_batch.add(voxel_id)
                flipped_indices.append((ux, uy, uz, um))
                
                C = self.catalog[ux,uy,uz,um].copy()
                apply_flip_soa(self.eps_plastic, None, self.soft_prop, self.last_event_time,
                               self.catalog, ux, uy, uz, um, self.time, 
                               self.jp, self.jt, self.softening_cap, self.neighbor_softening_fraction)
                
                self.prev_strain_dir[ux,uy,uz] = C
                
                if self.redraw_directions or self.redraw_barriers:
                    if self.redraw_directions:
                        self.catalog[ux,uy,uz] = stz_catalog_glass(self.M, self.gamma0)
                    if self.redraw_barriers:
                        self.Q0[ux,uy,uz] = self.barrier_generator((self.M,))
                else:
                    self.catalog[ux,uy,uz,um] = stz_catalog_glass(1, self.gamma0)[0]
                    self.Q0[ux,uy,uz,um] = self.barrier_generator((1,))[0]
                
                if track_cascades:
                    self.flip_event_history.append((int(global_step), int(local_step), int(ux), int(uy), int(uz), int(um)))
            
            n_flipped = len(flipped_indices)
            self.log_cascade(global_step, local_step, flipped_indices, 0.0, n_unstable=n_flipped)
            total_flips += n_flipped
            
            # Cascade Timing: Advance time and strain PER ITERATION
            dt_cascade = 0.0
            if self.cascade_timing == "single": dt_cascade = 1.0 / self.nu0
            elif self.cascade_timing == "per_flip": dt_cascade = n_flipped / self.nu0
            
            if dt_cascade > 0:
                self.time += dt_cascade
                if strain_unit_tensor is not None:
                    self.eps_macro += strain_unit_tensor * (self.strain_rate * dt_cascade)

            self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
            )
            
            if log_callback:
                log_callback(local_step, n_flipped)
            
            # Check for eps_target after logging
            if eps_target is not None and self.eps_macro[component] >= eps_target:
                eps_curr = self.eps_field.mean(axis=(0,1,2))
                sig_curr = self.sig_field.mean(axis=(0,1,2))
                return local_step + 1, total_flips, eps_curr, sig_curr, False, True
            
            if vtk_prefix:
                fname = f"{vtk_prefix}_step{global_step:06d}_cascade{local_step+1:04d}.vtu"
                self.export_vtk(fname)
            
            local_step += 1
            max_cascade_steps = int(self.max_cascade_steps_pct * self.total_voxels)
            if local_step > max_cascade_steps:
                print(f"Cascade limit reached ({local_step} steps > {self.max_cascade_steps_pct*100:.1f}% of {self.total_voxels} voxels)")
                return local_step, total_flips, self.eps_macro.copy(), self.sig_field.mean(axis=(0,1,2)), True, False
                 
        eps_curr = self.eps_macro.copy()
        sig_curr = self.sig_field.mean(axis=(0,1,2))
        return local_step, total_flips, eps_curr, sig_curr, False, False

    def run_simulation(self, n_global_steps, step_size, component=(0,1), 
                  stress_targets={}, mixed_tol=1e-4, mixed_max_iter=50,
                  checkpoint_interval=None, checkpoint_path="checkpoint",  
                  stop_on_stress_drop=None, stress_drop_component=None, stop_post_drop_steps=20,
                  vtk_interval="none", vtk_elastic_only=True, 
                  track_cascades=False,
                  enable_console_log=True,
                  summary_filename="summary_log.txt",
                  enable_summary_log=True,
                  enable_global_log=True,
                  enable_cascade_log=True,
                  enable_kmc_log=True,
                  checkpoint_elastic_only=False,
                  enable_save_q=False,
                  save_q_interval=None,
                  save_q_elastic_only=False,
                  max_kmc_steps_pct=0.3,
                  ignore_drop_steps=0,
                  stress_drop_lookback=1,
                  append_logs=False,
                  eps_target=None,
                  instability_mode="cascade", # "cascade" or "kmc"
                  cascade_timing="single"): # "single" or "per_flip"
        
        if stress_drop_component is None:
            stress_drop_component = component

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
                        enable_kmc_log=enable_kmc_log,
                        append=append_logs)
        
        if enable_console_log:
            header = f"{'Timestamp':<22} {'Elapsed(s)':<12} {'Step':<8} {'Type':<15} {'Eps_xx':<12} {'Sig_xx(GPa)':<15} {'KMC':<8} {'Cascade':<8} {'Flips':<8} {'SimTime(s)':<15}"
            print(header)
            print("-" * len(header))

        start_time_total = time.time()
        
        elastic_steps_done = 0
        total_kmc_steps = 0
        cascade_event_count = 0  # Cumulative cascade events
        step = 1
        stop_drop_triggered = False
        stop_countdown = stop_post_drop_steps
        elastic_chk_id = 0
        E_avg, nu_avg = np.mean(self.E), np.mean(self.nu)
        
        def _do_logging(current_step, step_type, cascade_id, cascade_flips):
            """
            Unified logging.
            cascade_id: In 'cascade' mode, this is the cumulative event counter.
                       In 'kmc' mode, this is 0.
            cascade_flips: Number of flips in this specific entry.
            """
            eps_curr, sig_curr = self.eps_macro.copy(), self.sig_field.mean(axis=(0,1,2))
            # Global log uses original total_flips logic (sum of flips)
            self.log_global(current_step, elastic_steps_done, total_kmc_steps, self.time, eps_curr, sig_curr, cascade_id, cascade_flips)
            self.history_global.append((eps_curr[0,0], sig_curr[0,0]/1e9))
            
            curr_stress_val, curr_strain_val = sig_curr[stress_drop_component], eps_curr[stress_drop_component]
            now, elapsed = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time_total
            
            summary_line = f"{now:<22} {elapsed:<12.2f} {current_step:<8d} {step_type.upper():<15} {curr_strain_val:<12.6f} {curr_stress_val/1e9:<15.3f} {total_kmc_steps:<8d} {cascade_id:<8d} {cascade_flips:<8d} {self.time:<15.6e}\n"
            
            if self._f_summary:
                self._f_summary.write(summary_line)
            if enable_console_log:
                print(summary_line.strip())
            
            if vtk_interval is not None and vtk_interval not in ["none", "last"]:
                save_vtk = False
                if vtk_interval == "current":
                    save_vtk = True
                elif isinstance(vtk_interval, int):
                    if vtk_elastic_only:
                        # Count only elastic increments
                        if step_type.lower() in ["elastic", "init"] and elastic_steps_done % vtk_interval == 0:
                            save_vtk = True
                    else:
                        # Count global steps (includes KMC/Cascade events)
                        if current_step % vtk_interval == 0:
                            save_vtk = True
                
                if save_vtk:
                    if not vtk_elastic_only or step_type.lower() in ["elastic", "init"]:
                        vtk_fname = os.path.join(self.output_dir, f"vtk_step_{current_step:06d}.vtu")
                        self.export_vtk(vtk_fname)

            return curr_stress_val

        def get_correction_legacy(sigma_err):
            tr_sig = np.trace(sigma_err)
            return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg

        if vtk_interval != "none" and vtk_interval != "last":
            # Step 0 is always considered an elastic/initial state
            self.export_vtk(os.path.join(self.output_dir, "vtk_step_000000.vtu"))

        strain_unit_tensor = np.zeros((3,3))
        strain_unit_tensor[component] = 1.0

        def _cascade_log_callback(local_step, cascade_step_flips):
            # This is for internal cascade steps inside _run_cascade (if track_cascades is True)
            # Log each internal iteration to summary_log for better transparency
            _do_logging(step, "cascade", cascade_event_count, cascade_step_flips)

        # Initial relaxation (step 0)
        if self.instability_mode == "cascade":
            l, f, eps_curr, sig_curr, truncated, stopped_by_eps = self._run_cascade(
                global_step=0, track_cascades=track_cascades, strain_unit_tensor=strain_unit_tensor,
                eps_target=eps_target, component=component, log_callback=_cascade_log_callback
            )
            if f > 0:
                cascade_event_count += 1
                # Timing handled internally in _run_cascade loop
                pass
                # _do_logging(0, "cascade", cascade_event_count, f)  # Now handled iteration-by-iteration in _cascade_log_callback
            
            if truncated or stopped_by_eps: 
                self._close_logs()
                return
        else:
            # KMC mode relaxation happens inside the main loop loop
            # BUT we should check if there's an instability at step 0
            self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
            )
            sig_curr = self.sig_field.mean(axis=(0,1,2))
        
        stress_history = [sig_curr[stress_drop_component]]
        
        if checkpoint_interval is not None and checkpoint_interval not in ["none", "last"]:
            cp_name = f"{checkpoint_path}.h5" if checkpoint_interval == "current" else \
                      (f"{checkpoint_path}_elastic_{elastic_chk_id:06d}.h5" if checkpoint_elastic_only else f"{checkpoint_path}_000000.h5")
            if checkpoint_elastic_only: elastic_chk_id += 1
            self.save_checkpoint(cp_name, step=0)

        dt_step = abs(step_size) / self.strain_rate if self.strain_rate > 0 else 1.0
        max_sequential_kmc = int(max_kmc_steps_pct * self.total_voxels)
        kmc_baseline_stress = None
        sequential_kmc_steps = 0
        remaining_time = 0.0

        while elastic_steps_done < n_global_steps:
            remaining_time += dt_step 
            
            while remaining_time > 0:
                self.update_barriers()
                
                # BRANCH: CASCADE vs KMC unstable handling
                if self.instability_mode == "cascade":
                    unstable_indices = find_unstable(self.Q, self.stability_threshold)
                    if len(unstable_indices) > 0:
                        v_pfx = os.path.join(self.output_dir, "vtk_cascade") if (not vtk_elastic_only and track_cascades) else None
                        l, f, _, _, truncated, stopped_by_eps = self._run_cascade(
                            step, vtk_prefix=v_pfx, track_cascades=track_cascades, strain_unit_tensor=strain_unit_tensor,
                            eps_target=eps_target, component=component, log_callback=_cascade_log_callback
                        )
                        if f > 0:
                            cascade_event_count += 1
                            # Timing handled internally in _run_cascade loop
                            pass
                            # _do_logging(step, "cascade", cascade_event_count, f)  # Now handled iteration-by-iteration in _cascade_log_callback

                        if truncated or stopped_by_eps: 
                            self._close_logs()
                            return
                        continue

                # SHARED: Rate calculation
                # In 'kmc' mode, compute_rates will include Q <= 0
                eff_volume = self.volume if self.scale_rate_by_volume else 1.0
                rates_flat, indices_flat, total_rate = compute_rates(self.Q, eff_volume, self.temperature, self.nu0, instability_mode=self.instability_mode)
                
                if total_rate > 0:
                    u = np.random.uniform()
                    t_wait = -np.log(u) / total_rate
                    
                    if t_wait < remaining_time:
                        self.time += t_wait
                        self.eps_macro += strain_unit_tensor * (self.strain_rate * t_wait)
                        remaining_time -= t_wait

                        idx_flat = select_event(rates_flat, total_rate) 
                        if idx_flat == -1: continue
                        
                        x, y, z, m = decode_index(indices_flat[idx_flat], self.grid_shape[1], self.grid_shape[2], self.M)
                        is_instab = self.Q[x,y,z,m] <= self.stability_threshold
                        
                        C = self.catalog[x,y,z,m].copy()
                        
                        apply_flip_soa(self.eps_plastic, None, self.soft_prop, self.last_event_time,
                                       self.catalog, x, y, z, m, self.time, self.jp, self.jt, self.softening_cap, self.neighbor_softening_fraction)
                        self.prev_strain_dir[x,y,z] = C
                        
                        if self.redraw_directions or self.redraw_barriers:
                            if self.redraw_directions: self.catalog[x,y,z] = stz_catalog_glass(self.M, self.gamma0)
                            if self.redraw_barriers: self.Q0[x,y,z] = self.barrier_generator((self.M,))
                        else:
                            self.catalog[x,y,z,m] = stz_catalog_glass(1, self.gamma0)[0]
                            self.Q0[x,y,z,m] = self.barrier_generator((1,))[0]
                        
                        if getattr(self, 'fast_patching_enabled', False):
                            if self.sigma_macro_unit is None:
                                eps_plas = np.zeros_like(self.eps_plastic)
                                _, self.sigma_macro_unit, _, _ = update_stress_fft_full(
                                    eps_plas, strain_unit_tensor, self.E, self.nu, pixel=self.pixel, **self.solver_args
                                )
                            if t_wait > 0:
                                self.sig_field += self.sigma_macro_unit * (self.strain_rate * t_wait)
                            
                            c0, c1, c2, c3, c4 = C[0,0], -C[2,2], C[0,1], C[0,2], C[1,2]
                            patch = (c0 * self.patch_kernels[0] + c1 * self.patch_kernels[1] + 
                                     c2 * self.patch_kernels[2] + c3 * self.patch_kernels[3] + 
                                     c4 * self.patch_kernels[4])
                            
                            mean_shift = (c0 * self.patch_missing_mean[0] + c1 * self.patch_missing_mean[1] + 
                                          c2 * self.patch_missing_mean[2] + c3 * self.patch_missing_mean[3] + 
                                          c4 * self.patch_missing_mean[4])
                            
                            nx, ny, nz = self.grid_shape
                            R = self.patch_radius
                            for dx in range(-R, R+1):
                                for dy in range(-R, R+1):
                                    for dz in range(-R if nz>1 else 0, R+1 if nz>1 else 1):
                                        px, py, pz = (x+dx)%nx, (y+dy)%ny, (z+dz)%nz
                                        self.sig_field[px, py, pz] += patch[dx+R, dy+R, dz+R if nz>1 else 0]
                            
                            self.sig_field += mean_shift
                            
                            self.flips_since_sync += 1
                            if self.flips_since_sync >= self.sync_interval:
                                self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                                    self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
                                )
                                self.flips_since_sync = 0
                        else:
                            self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                                self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
                            )
                        
                        log_type = "kmc_instab" if is_instab else "kmc"
                        # In KMC mode, cascade counter is 0 as per request
                        curr_stress_val = _do_logging(step, log_type, 0 if self.instability_mode == "kmc" else cascade_event_count, 1)
                        total_kmc_steps += 1
                        
                        if is_instab: sequential_kmc_steps += 1
                        else: sequential_kmc_steps = 0
                        
                        if sequential_kmc_steps > max_sequential_kmc:
                            print(f"\n[TERMINATE] KMC instability sequence limit reached! {sequential_kmc_steps} steps.")
                            return
                        
                        stress_history.append(curr_stress_val)
                        step += 1
                        continue
                    else:
                        d_eps = strain_unit_tensor * (self.strain_rate * remaining_time)
                        self.eps_macro += d_eps
                        self.time += remaining_time
                        remaining_time = 0
                else:
                    d_eps = strain_unit_tensor * (self.strain_rate * remaining_time)
                    self.eps_macro += d_eps
                    self.time += remaining_time
                    remaining_time = 0
                    
            # End of remaining_time inner loop (Elastic increment update)
            self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
            )
            
            # Elastic relaxation loop (Mixed BCs)
            converged = False
            for it in range(mixed_max_iter):
                if self.instability_mode == "cascade":
                    v_pfx = os.path.join(self.output_dir, "vtk_cascade") if (not vtk_elastic_only and track_cascades) else None
                    l, f, _, sig_M, truncated, stopped_by_eps = self._run_cascade(
                        step, vtk_prefix=v_pfx, track_cascades=track_cascades, strain_unit_tensor=strain_unit_tensor,
                        eps_target=eps_target, component=component, log_callback=_cascade_log_callback
                    )
                    if f > 0:
                        cascade_event_count += 1
                        # Timing handled internally in _run_cascade loop
                        pass
                        # _do_logging(step, "cascade", cascade_event_count, f)  # Now handled iteration-by-iteration in _cascade_log_callback
                    if truncated or stopped_by_eps: 
                        self._close_logs()
                        return
                    sig_curr = sig_M
                else:
                    sig_curr = self.sig_field.mean(axis=(0,1,2))
                
                stress_err_tensor = np.zeros((3,3))
                max_err = 0.0
                for idx_t, target_val in stress_targets.items():
                    err = target_val - sig_curr[idx_t]
                    stress_err_tensor[idx_t] = err
                    max_err = max(max_err, abs(err))
                
                if max_err < mixed_tol:
                    converged = True
                    break
                
                eps_corr = get_correction_legacy(stress_err_tensor)
                eps_corr[component] = 0.0
                self.eps_macro += eps_corr
                self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
                    self.eps_plastic, self.eps_macro, self.E, self.nu, pixel=self.pixel, **self.solver_args
                )
            
            if not converged:
                print(f"Warning: Mixed loop did not converge at step {step} (Err={max_err:.2e})")

            elastic_steps_done += 1
            curr_stress_val = _do_logging(step, "elastic", 0 if self.instability_mode == "kmc" else cascade_event_count, 0)
            stress_history.append(curr_stress_val)
             
            if stop_on_stress_drop is not None and not stop_drop_triggered and step > ignore_drop_steps:
                lookback = max(1, stress_drop_lookback)
                if len(stress_history) >= lookback:
                    ref_stress = stress_history[-lookback]
                    drop_frac = (abs(ref_stress) - abs(curr_stress_val)) / abs(ref_stress) if abs(ref_stress) > 1e-6 else 0.0
                    if drop_frac > stop_on_stress_drop:
                        print(f"\n[ALERT] Stress Drop Detected! {drop_frac*100:.1f}% at step {step}")
                        stop_drop_triggered = True
             
            if checkpoint_interval is not None and checkpoint_interval not in ["none", "last"]:
                save_chk = False
                cp_name = None
                if checkpoint_interval == "current":
                    save_chk = True
                    cp_name = f"{checkpoint_path}.h5"
                elif isinstance(checkpoint_interval, int) and step % checkpoint_interval == 0:
                    save_chk = True
                    cp_name = f"{checkpoint_path}_elastic_{elastic_chk_id:06d}.h5" if checkpoint_elastic_only else f"{checkpoint_path}_{step:06d}.h5"
                    if checkpoint_elastic_only: elastic_chk_id += 1
                
                if save_chk and cp_name:
                    self.save_checkpoint(cp_name, step=step)
    
            if enable_save_q and save_q_interval and step % save_q_interval == 0:
                should_save_q = True
                if save_q_elastic_only and self.instability_mode != "elastic": # Changed from last_step_type to self.instability_mode
                    should_save_q = False
                if should_save_q:
                    np.save(os.path.join(self.output_dir, f"Q_step_{step:06d}.npy"), self.Q)

            if stop_drop_triggered:
                if stop_countdown > 0: stop_countdown -= 1
                else: break
            step += 1

        if vtk_interval == "last":
            self.export_vtk(os.path.join(self.output_dir, f"vtk_step_{step-1:06d}_final.vtu"))
        elif vtk_interval not in [None, "none", "last"] and isinstance(vtk_interval, int) and (step-1) % vtk_interval != 0:
            self.export_vtk(os.path.join(self.output_dir, f"vtk_step_{step-1:06d}_final.vtu"))

        if checkpoint_interval == "last": 
            self.save_checkpoint(f"{checkpoint_path}_final.h5", step=step-1)
        elif checkpoint_interval not in [None, "none", "last"] and isinstance(checkpoint_interval, int) and (step-1) % checkpoint_interval != 0:
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
                    meta.attrs['neighbor_softening_fraction'] = self.neighbor_softening_fraction
                    meta.attrs['softening_cap'] = self.softening_cap
                    meta.attrs['softening_scheme'] = self.softening_scheme
                    meta.attrs['scale_rate_by_volume'] = self.scale_rate_by_volume
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
                    
                    if hasattr(self, 'flip_event_history') and len(self.flip_event_history) > 0:
                        flips_arr = np.array(self.flip_event_history, dtype=np.int32)
                        f.create_dataset('history/flips', data=flips_arr, compression='gzip')
                return
            except (ImportError, Exception) as e:
                if attempt < max_retries - 1: time.sleep(0.5)
                else: print(f"Error saving checkpoint after {max_retries} attempts: {e}")

    @classmethod
    def load_checkpoint(cls, path):
        """
        Load ThermalSimulation from an HDF5 checkpoint.
        """
        import h5py
        with h5py.File(path, "r") as f:
            meta = f['metadata']
            nx, ny, nz = meta.attrs['nx'], meta.attrs['ny'], meta.attrs['nz']
            M, gamma0, pixel = meta.attrs['M'], meta.attrs['gamma0'], meta.attrs['pixel']
            
            fields = f['fields']
            E_field = fields['E_field'][:]
            nu_field = fields['nu_field'][:]
            
            sim = cls(nx, ny, nz, M=M, gamma0=gamma0, E_field=E_field, nu_field=nu_field, 
                      pixel=pixel, temperature=meta.attrs['temperature'], 
                      strain_rate=meta.attrs['strain_rate'], jp=meta.attrs['jp'], jt=meta.attrs['jt'],
                      neighbor_softening_fraction=meta.attrs.get('neighbor_softening_fraction', 0.0),
                      softening_cap=meta.attrs['softening_cap'], softening_scheme=meta.attrs['softening_scheme'],
                      nu0=meta.attrs.get('nu0', 1e13), q_act_temp=meta.attrs.get('q_act_temp', 0.37),
                      scale_rate_by_volume=meta.attrs.get('scale_rate_by_volume', True))
            
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
                
            if 'history/flips' in f:
                sim.flip_event_history = f['history/flips'][:].tolist()
               
            print(f"Loaded checkpoint from {path} (SimTime={sim.time:.4e}s, Step={meta.attrs['step']})")
            return sim
