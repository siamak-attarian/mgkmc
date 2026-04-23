import numpy as np
import os
import time
from datetime import datetime
from .linear_elastic_simulator import spectral_solver_2d
from .kmc_simulator_functions import (
    compute_rates_2d, select_event_2d, decode_index_2d, stz_catalog_glass_2d,
    compute_barrier_2d, find_unstable_2d, apply_flip_soa_2d, get_barrier_generator_2d
)
from .analysis.vtk import export_to_vtk

class KmcSimulation2D:
    def __init__(self, nx, ny, M, gamma0, E_field, nu_field, pixel=1.0,
                 barrier_generator=None, barrier_kwargs={}, softening_scheme="isotropic",
                 softening_cap=2.0, jp=10.0, jt=30.0, neighbor_softening_fraction=0.0,
                 q_act_temp=0.37, output_dir="output", temperature=0.0,
                 strain_rate=1.0, stability_threshold=0.0, nu0=1e13,
                 plane_mode="plane_strain", fast_patching=None,
                 instability_mode="cascade", cascade_timing="none", 
                 scale_rate_by_volume=True,
                 redraw_directions=True, redraw_barriers=True):
        
        self.nx, self.ny = nx, ny
        self.M, self.gamma0 = M, gamma0
        self.pixel, self.volume = pixel, pixel**3 # 3D STZ volume per paper Eq.12
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        # Physics params
        self.softening_scheme = softening_scheme
        self.softening_cap = softening_cap
        self.jp, self.jt = jp, jt
        self.neighbor_softening_fraction = neighbor_softening_fraction
        self.temperature = temperature
        self.strain_rate, self.stability_threshold = strain_rate, stability_threshold
        self.nu0 = nu0
        self.q_act_temp = q_act_temp
        self.plane_mode = plane_mode
        
        # Fast Patching Flags
        self.fast_patching_enabled = fast_patching.get('enabled', False) if fast_patching else False
        self.patch_radius = fast_patching.get('patch_radius', 5) if fast_patching else 5
        self.sync_interval = fast_patching.get('sync_interval', 100) if fast_patching else 100
        self.flips_since_sync = 0
        self.sigma_macro_unit = None
        self.instability_mode = instability_mode
        self.cascade_timing = cascade_timing
        self.scale_rate_by_volume = scale_rate_by_volume
        self.redraw_directions = redraw_directions
        self.redraw_barriers = redraw_barriers

        # Internal Physics Calculation: tau (Relaxation Time)
        if self.temperature > 0:
            kB = 8.617e-5 # eV/K
            # tau = 1 / (nu0 * exp(-q_act_temp / (kB * T)))
            self.tau = 1.0 / (self.nu0 * np.exp(-self.q_act_temp / (kB * self.temperature)))
            print(f" [KmcSimulation2D] T={self.temperature}K > 0: Calculated Dynamic Softening Decay (tau): {self.tau:.4e} s")
        else:
            self.tau = np.inf
            print(" [KmcSimulation2D] T=0K: Softening Decay (tau): Infinite (No decay)")

        # Fields (nx, ny)
        if isinstance(E_field, (int, float, np.number)):
            self.E_field = np.full((nx, ny), float(E_field))
        else:
            self.E_field = np.array(E_field)

        if self.E_field.mean() < 1e6: # Convert GPa to Pa if needed
            self.E_field *= 1e9
            
        if isinstance(nu_field, (int, float, np.number)):
            self.nu_field = np.full((nx, ny), float(nu_field))
        else:
            self.nu_field = np.array(nu_field)
        self.eps_field = np.zeros((nx, ny, 2, 2))
        self.sig_field = np.zeros((nx, ny, 2, 2))
        self.eps_plastic = np.zeros((nx, ny, 2, 2))
        self.soft_prop = np.zeros((nx, ny, 2)) # [g_p, g_t]
        self.last_event_time = np.full((nx, ny), -np.inf)
        self.eps_macro = np.zeros((2, 2))
        self.time = 0.0

        # Barriers & Orientations
        self.Q = np.zeros((nx, ny, M))
        if barrier_generator is None:
            self.barrier_generator = get_barrier_generator_2d("gaussian", mean=2.0, std=0.6)
        elif isinstance(barrier_generator, str):
            self.barrier_generator = get_barrier_generator_2d(barrier_generator, **barrier_kwargs)
        else:
            self.barrier_generator = barrier_generator
            
        self.Q0 = self.barrier_generator((nx, ny, M))
        self.catalog = np.zeros((nx, ny, M, 2, 2))
        for x in range(nx):
            for y in range(ny):
                self.catalog[x,y] = stz_catalog_glass_2d(M, gamma0)
        
        self.prev_strain_dir = np.zeros((nx, ny, 2, 2))
        self.solver_args = {}

        if self.fast_patching_enabled:
            self._precompute_patch_kernels(self.E_field)
        
        # Logs
        self._f_summary = None
        self._f_global = None

    def _init_logs(self, summary_filename="summary_log.txt", enable_summary_log=True, enable_global_log=True, append=False):
        self.summary_log_path = os.path.join(self.output_dir, summary_filename)
        self.global_log_path = os.path.join(self.output_dir, "global_log.txt")
        mode = "a" if append else "w"

        def is_empty(path):
            return not os.path.exists(path) or os.path.getsize(path) == 0

        if enable_summary_log:
            self._f_summary = open(self.summary_log_path, mode, buffering=1)
            if not append or is_empty(self.summary_log_path):
                header = f"{'Timestamp':<22} {'Elapsed(s)':<12} {'Step':<8} {'Type':<15} {'Eps_xx':<12} {'Sig_xx(GPa)':<15} {'KMC':<8} {'Cascade':<8} {'Flips':<8} {'SimTime(s)':<15}\n"
                self._f_summary.write(header)
                self._f_summary.write("-" * len(header) + "\n")

        if enable_global_log:
            self._f_global = open(self.global_log_path, mode, buffering=1)
            if not append or is_empty(self.global_log_path):
                header_fmt = "{:<10} {:<12} {:<10} " + " ".join(["{:<15}"]*12) + " {:<14} {:<17} {:<15}\n"
                headers = [
                    "GlobalStep", "ElasticStep", "KMCStep",
                    "Eps_xx", "Eps_yy", "Eps_zz", "Eps_xy", "Eps_xz", "Eps_yz",
                    "Sig_xx(GPa)", "Sig_yy(GPa)", "Sig_zz(GPa)", "Sig_xy(GPa)", "Sig_xz(GPa)", "Sig_yz(GPa)",
                    "CascadeSteps", "TotalCascadeFlips", "SimTime(s)"
                ]
                self._f_global.write(header_fmt.format(*headers))

    def _close_logs(self):
        for f in [self._f_summary, self._f_global]:
            if f: f.close()

    def log_global(self, global_step, elastic_step, kmc_step, time_sim, eps, sig, cascade_steps, total_flips):
        if self._f_global:
            fmt = "{:<10d} {:<12d} {:<10d} " + " ".join(["{:<15.6f}"]*6) + " " + " ".join(["{:<15.3f}"]*6) + " {:<14d} {:<17d} {:<15.6e}\n"
            
            # Plane Strain correction for Sig_zz
            sig_zz = 0.0
            if self.plane_mode == "plane_strain":
                nu_avg = self.nu_field.mean()
                sig_zz = nu_avg * (sig[0,0] + sig[1,1])

            data = [
                global_step, elastic_step, kmc_step,
                eps[0,0], eps[1,1], 0.0, eps[0,1], 0.0, 0.0,
                sig[0,0]/1e9, sig[1,1]/1e9, sig_zz/1e9, sig[0,1]/1e9, 0.0, 0.0,
                cascade_steps, total_flips, time_sim
            ]
            self._f_global.write(fmt.format(*data))

    def _precompute_patch_kernels(self, E_field):
        print(f"\n [KmcSimulation2D] Pre-computing Stress Patches (Radius={self.patch_radius})...")
        R = self.patch_radius
        nx, ny = self.nx, self.ny
        
        # 2 Deviatoric Bases for 2D: (xx=-yy) and (xy)
        bases = [
            np.array([[1.0, 0], [0, -1.0]]), # Basis 0: Pure Shear
            np.array([[0, 1.0], [1.0, 0]])   # Basis 1: Simple Shear
        ]
        
        self.patch_kernels = []
        self.patch_missing_mean = []
        
        for P in bases:
            eps_plas = np.zeros((nx, ny, 2, 2))
            eps_plas[nx//2, ny//2] = P
            eps_mac = np.zeros((2,2))
            _, sig_field, _, _ = spectral_solver_2d(
                E_field, self.nu_field, eps_mac, eps_plastic=eps_plas,
                pixel=self.pixel, plane_mode=self.plane_mode, **self.solver_args
            )
            
            sig_rolled = np.roll(sig_field, shift=(-(nx//2), -(ny//2)), axis=(0,1))
            crop = np.zeros((2*R+1, 2*R+1, 2, 2))
            for dx in range(-R, R+1):
                for dy in range(-R, R+1):
                    crop[dx+R, dy+R] = sig_rolled[dx % nx, dy % ny]
            
            self.patch_kernels.append(crop)
            self.patch_missing_mean.append(np.mean(sig_field, axis=(0,1)) - np.sum(crop, axis=(0,1))/(nx*ny))

        print(" [KmcSimulation2D] Fast Patching Kernels Ready.")

    def elastic_run(self, eps_macro):
        """Standard 2D elastic equilibrium step."""
        self.eps_field, self.sig_field, _, _ = spectral_solver_2d(
            self.E_field, self.nu_field, eps_macro, eps_plastic=self.eps_plastic,
            pixel=self.pixel, plane_mode=self.plane_mode, **self.solver_args
        )
        return self.sig_field.mean(axis=(0,1))

    def update_barriers(self):
        scheme_idx = 1 if self.softening_scheme == "directional" else 0
        compute_barrier_2d(
            self.Q, self.Q0, self.sig_field, self.catalog, self.volume,
            self.soft_prop, self.last_event_time, self.time, self.prev_strain_dir,
            self.softening_cap, scheme_idx, self.tau
        )

    def _run_cascade(self, step, component=(0,0), log_callback=None):
        total_flips = 0
        local_step = 0
        while True:
            self.update_barriers()
            unstable_indices = find_unstable_2d(self.Q, self.stability_threshold)
            n_unstable = len(unstable_indices)
            if n_unstable == 0: break
            
            # Sort by Q value (most negative first)
            q_values = self.Q[unstable_indices[:,0], unstable_indices[:,1], unstable_indices[:,2]]
            sort_idx = np.argsort(q_values)
            unstable_indices = unstable_indices[sort_idx]
            
            flipped_voxels_in_batch = set()
            flips_in_this_batch = 0
            
            for k in range(n_unstable):
                x, y, m = unstable_indices[k]
                voxel_id = (x, y)
                if voxel_id in flipped_voxels_in_batch:
                    continue
                
                flipped_voxels_in_batch.add(voxel_id)
                flips_in_this_batch += 1
                
                C = self.catalog[x,y,m].copy()
                apply_flip_soa_2d(self.eps_plastic, self.soft_prop, self.last_event_time, self.catalog, x, y, m, self.time, self.jp, self.jt, self.softening_cap, self.neighbor_softening_fraction)
                self.prev_strain_dir[x,y] = C
                
                # Redraw catalog/barriers after flip
                if self.redraw_directions or self.redraw_barriers:
                    if self.redraw_directions:
                        self.catalog[x,y] = stz_catalog_glass_2d(self.M, self.gamma0)
                    if self.redraw_barriers:
                        self.Q0[x,y] = self.barrier_generator((self.M,))
                else:
                    self.catalog[x,y,m] = stz_catalog_glass_2d(1, self.gamma0)[0]
                    self.Q0[x,y,m] = self.barrier_generator((1,))[0]
                
                if self.fast_patching_enabled:
                    # Predictor: Update Stress locally using kernels
                    gxx, gxy = C[0,0], C[0,1]
                    patch = gxx * self.patch_kernels[0] + gxy * self.patch_kernels[1]
                    mean_shift = gxx * self.patch_missing_mean[0] + gxy * self.patch_missing_mean[1]
                    
                    R = self.patch_radius
                    for dx in range(-R, R+1):
                        for dy in range(-R, R+1):
                            px, py = (x+dx)%self.nx, (y+dy)%self.ny
                            self.sig_field[px, py] += patch[dx+R, dy+R]
                    self.sig_field += mean_shift

            if not self.fast_patching_enabled:
                self.elastic_run(self.eps_macro)
            
            total_flips += flips_in_this_batch
            local_step += 1
            if log_callback: log_callback(local_step, flips_in_this_batch)
            if local_step > self.nx * self.ny: break
            
        if self.fast_patching_enabled and total_flips > 0:
            # Corrector: Final Sync
            self.elastic_run(self.eps_macro)
            
        return local_step, total_flips, self.eps_macro.copy(), self.sig_field.mean(axis=(0,1)), False, False

    def run_simulation(self, n_global_steps, step_size, component=(0,0), 
                  stress_targets={}, mixed_tol=1e-4, mixed_max_iter=50,
                  checkpoint_interval="none", checkpoint_path="checkpoint",  
                  vtk_interval="none", vtk_elastic_only=True, 
                  track_cascades=False, enable_console_log=True,
                  summary_filename="summary_log.txt", enable_summary_log=True,
                  enable_global_log=True, max_kmc_steps_pct=0.3, **kwargs):
        
        self._init_logs(summary_filename, enable_summary_log, enable_global_log)
        start_time_total = time.time()
        elastic_steps_done, total_kmc_steps, cascade_event_count, step = 0, 0, 0, 1
        sequential_kmc_steps = 0
        max_sequential_kmc = int(max_kmc_steps_pct * self.nx * self.ny)
        
        def _do_logging(s, s_type, c_id, c_flips):
            epsM, sigM = self.eps_macro.copy(), self.sig_field.mean(axis=(0,1))
            self.log_global(s, elastic_steps_done, total_kmc_steps, self.time, epsM, sigM, c_id, c_flips)

            now, elapsed = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time_total
            curr_strain_val = epsM[component]
            curr_stress_val = sigM[component]
            
            summary_line = f"{now:<22} {elapsed:<12.2f} {s:<8d} {s_type.upper():<15} {curr_strain_val:<12.6f} {curr_stress_val/1e9:<15.3f} {total_kmc_steps:<8d} {c_id:<8d} {c_flips:<8d} {self.time:<15.6e}\n"
            
            if self._f_summary:
                self._f_summary.write(summary_line)
            if enable_console_log:
                if s == 0: # Print header on first call if console enabled
                    header = f"{'Timestamp':<22} {'Elapsed(s)':<12} {'Step':<8} {'Type':<15} {'Eps_xx':<12} {'Sig_xx(GPa)':<15} {'KMC':<8} {'Cascade':<8} {'Flips':<8} {'SimTime(s)':<15}"
                    print(header)
                    print("-" * len(header))
                print(summary_line.strip())

            if vtk_interval != "none" and s % (vtk_interval if isinstance(vtk_interval, int) else 1) == 0:
                export_to_vtk(os.path.join(self.output_dir, f"step_{s:05d}.vtu"), self.eps_field, self.sig_field, self.E_field, self.nu_field, pixel=self.pixel)

        self.elastic_run(self.eps_macro)
        _do_logging(0, "INIT", 0, 0)
        strain_unit = np.zeros((2,2))
        strain_unit[component] = 1.0
        dt_step = abs(step_size) / self.strain_rate if self.strain_rate > 0 else 1.0
        remaining_time = 0.0

        while elastic_steps_done < n_global_steps:
            remaining_time += dt_step
            while remaining_time > 0:
                self.update_barriers()
                
                # Instability Handling
                if self.instability_mode == "cascade":
                    unstable = find_unstable_2d(self.Q, self.stability_threshold)
                    if len(unstable) > 0:
                        _, f, _, _, _, _ = self._run_cascade(step, component=component)
                        if f > 0: cascade_event_count += 1
                        continue

                eff_volume = self.volume if self.scale_rate_by_volume else 1.0
                rates, indices, total_rate = compute_rates_2d(self.Q, eff_volume, self.temperature, self.nu0, instability_mode=self.instability_mode)
                if total_rate > 0:
                    t_wait = -np.log(np.random.rand()) / total_rate
                    if t_wait < remaining_time:
                        self.time += t_wait
                        self.eps_macro += strain_unit * (self.strain_rate * t_wait)
                        remaining_time -= t_wait
                        idx_flat = indices[select_event_2d(rates, total_rate)]
                        x, y, m = decode_index_2d(idx_flat, self.ny, self.M)
                        
                        is_instab = self.Q[x,y,m] <= self.stability_threshold
                        C = self.catalog[x,y,m].copy()
                        
                        apply_flip_soa_2d(self.eps_plastic, self.soft_prop, self.last_event_time, self.catalog, x, y, m, self.time, self.jp, self.jt, self.softening_cap, self.neighbor_softening_fraction)
                        self.prev_strain_dir[x,y] = C
                        
                        # Redraw catalog/barriers after flip (per paper Section 2.1.2)
                        if self.redraw_directions or self.redraw_barriers:
                            if self.redraw_directions:
                                self.catalog[x,y] = stz_catalog_glass_2d(self.M, self.gamma0)
                            if self.redraw_barriers:
                                self.Q0[x,y] = self.barrier_generator((self.M,))
                        else:
                            self.catalog[x,y,m] = stz_catalog_glass_2d(1, self.gamma0)[0]
                            self.Q0[x,y,m] = self.barrier_generator((1,))[0]
                        
                        if self.fast_patching_enabled:
                            if self.sigma_macro_unit is None:
                                _, self.sigma_macro_unit, _, _ = spectral_solver_2d(
                                    self.E_field, self.nu_field, strain_unit,
                                    eps_plastic=np.zeros_like(self.eps_plastic), 
                                    pixel=self.pixel, plane_mode=self.plane_mode, **self.solver_args
                                )
                            
                            # Predictor: Update Stress locally
                            if t_wait > 0:
                                self.sig_field += self.sigma_macro_unit * (self.strain_rate * t_wait)
                            
                            # Kernel superposition
                            gxx, gxy = C[0,0], C[0,1]
                            patch = gxx * self.patch_kernels[0] + gxy * self.patch_kernels[1]
                            mean_shift = gxx * self.patch_missing_mean[0] + gxy * self.patch_missing_mean[1]
                            
                            R = self.patch_radius
                            for dx in range(-R, R+1):
                                for dy in range(-R, R+1):
                                    px, py = (x+dx)%self.nx, (y+dy)%self.ny
                                    self.sig_field[px, py] += patch[dx+R, dy+R]
                            self.sig_field += mean_shift
                            
                            self.flips_since_sync += 1
                            if self.flips_since_sync >= self.sync_interval:
                                self.elastic_run(self.eps_macro)
                                self.flips_since_sync = 0
                        else:
                            self.elastic_run(self.eps_macro)
                        
                        log_type = "KMC_INSTAB" if is_instab else "KMC"
                        if is_instab: sequential_kmc_steps += 1
                        else: sequential_kmc_steps = 0
                        
                        if sequential_kmc_steps > max_sequential_kmc:
                            print(f"\n[TERMINATE] KMC instability sequence limit reached! {sequential_kmc_steps} steps.")
                            return

                        total_kmc_steps += 1
                        _do_logging(step, log_type, 0 if self.instability_mode == "kmc" else cascade_event_count, 1)
                        step += 1
                        continue
                if self.fast_patching_enabled:
                    if self.sigma_macro_unit is not None:
                        self.sig_field += self.sigma_macro_unit * (self.strain_rate * remaining_time)
                
                self.eps_macro += strain_unit * (self.strain_rate * remaining_time)
                self.time += remaining_time
                remaining_time = 0
            
            for it in range(mixed_max_iter):
                sigM = self.elastic_run(self.eps_macro)
                err_max = 0.0
                for idx, target in stress_targets.items():
                    if idx[0] < 2 and idx[1] < 2:
                        err = target - sigM[idx]
                        err_max = max(err_max, abs(err))
                        self.eps_macro[idx] += err / self.E_field.mean()
                if err_max < mixed_tol: break
            
            elastic_steps_done += 1
            _do_logging(step, "ELAST", cascade_event_count, 0)
            step += 1
        total_time = time.time() - start_time_total
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        duration_str = f"\nSimulation Finish Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Duration: {total_time:.2f} seconds ({int(h):d}h {int(m):02d}m {int(s):02d}s)\n"
        
        if self._f_summary:
            self._f_summary.write(duration_str)
        if enable_console_log: 
            print(duration_str)

        self._close_logs()
