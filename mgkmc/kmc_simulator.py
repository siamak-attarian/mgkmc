import numpy as np
import os
import time
from scipy.linalg import expm
from datetime import datetime
from .linear_elastic_simulator import spectral_solver_2d, spectral_solver_secant_2d
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
                 redraw_directions=True, redraw_barriers=True,
                 enable_thermal=False, Cp=420.0, rho=6125.0,
                 thermal_diffusivity=3.0e-6, thermal_coords="pixel",
                 temperature_cap=1000.0, thermostat=False, tau_bath=0.0,
                 strain_assumption="small_strain", hyperelastic_model="svk",
                 A_m=0.0, B_m=0.0, C_m=0.0, solver="al", stz_mode="pure_shear",
                 d=0.0, k=0.0):
        
        self.nx, self.ny = nx, ny
        self.M, self.gamma0 = M, gamma0
        self.pixel, self.volume = pixel, pixel**3 # 3D STZ volume per paper Eq.12
        self.output_dir = output_dir
        self.solver = solver
        self.stz_mode = stz_mode
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
        
        # Thermal parameters
        self.enable_thermal = enable_thermal
        self.Cp = Cp
        self.rho = rho
        self.thermal_diffusivity = thermal_diffusivity
        self.thermal_coords = thermal_coords.lower()
        self.temperature_cap = temperature_cap
        self.thermostat = thermostat
        self.tau_bath = tau_bath
        self.Tlocal = np.full((nx, ny), float(self.temperature), dtype=np.float64)

        from .fft import compute_wave_vectors_2d
        Lx, Ly = (nx, ny) if self.thermal_coords == "pixel" else (nx * pixel, ny * pixel)
        kx, ky = compute_wave_vectors_2d(nx, ny, Lx, Ly)
        self.k2 = kx**2 + ky**2
        self.diffusivity_scaled = self.thermal_diffusivity
        if self.thermal_coords == "physical":
            self.diffusivity_scaled *= 1e18 # Convert m^2/s to nm^2/s
        
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
                self.catalog[x,y] = stz_catalog_glass_2d(M, gamma0, stz_mode=self.stz_mode)
        
        self.prev_strain_dir = np.zeros((nx, ny, 2, 2))
        self.solver_args = {}

        self.strain_assumption = strain_assumption
        self.hyperelastic_model = hyperelastic_model
        self.d = d
        self.k = k
        if self.strain_assumption == "finite_strain" or self.hyperelastic_model == "secant_degradation":
            self.fast_patching_enabled = False
        if self.strain_assumption == "finite_strain":
            from .finite_strain_simulator import _make_identity_tensors_2d, build_ghat4_2d, build_C4_2d
            self.I2_fs, self.I4_fs, self.I4rt_fs, self.I4s_fs, self.II_fs = _make_identity_tensors_2d(nx, ny)
            Lx, Ly = nx * pixel, ny * pixel
            self.Ghat4_fs = build_ghat4_2d(nx, ny, Lx, Ly, even_grid=(nx%2==0 or ny%2==0))
            self.C4_fs = build_C4_2d(self.E_field, self.nu_field, self.I4s_fs, self.II_fs, plane_mode=self.plane_mode)
            
            self.F_field = np.einsum('ij,xy->xyij', np.eye(2), np.ones((nx, ny)))
            self.F_plastic = np.einsum('ij,xy->xyij', np.eye(2), np.ones((nx, ny)))
            self.F_macro = np.eye(2)
            self.A_m = A_m
            self.B_m = B_m
            self.C_m = C_m

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
                header = f"{'Timestamp':<22} {'Elapsed(s)':<12} {'Step':<8} {'Type':<15} {'Eps_xx':<12} {'Sig_xx(GPa)':<15} {'KMC':<8} {'Cascade':<8} {'Flips':<8} {'SimTime(s)':<15}"
                if self.enable_thermal:
                    header += " T_avg(K)"
                header += "\n"
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

    def _min_substeps_for_flip(self, C, gamma_step=0.02):
        """Minimum N so that each sub-step has effective shear amplitude < gamma_step."""
        gamma_eff = np.sqrt(0.5 * np.sum(C**2))   # von-Mises equivalent shear
        return max(2, int(np.ceil(gamma_eff / gamma_step)))

    def elastic_run(self, eps_macro):
        """Standard 2D elastic equilibrium step, supporting both small and finite strain."""
        if getattr(self, "strain_assumption", "small_strain") == "finite_strain":
            from .finite_strain_simulator import build_finite_strain_bc, finite_strain_solver_step_2d
            drv_comp = getattr(self, "driving_component", (0, 0))
            stress_tgts = getattr(self, "stress_targets", {})
            eps_s = eps_macro[drv_comp]
            
            F_bar, F_mask, P_tgt, P_mask = build_finite_strain_bc(
                drv_comp, eps_s, stress_tgts, self.plane_mode,
                F_bar_initial=getattr(self, "F_macro", None)
            )
            
            F_in = np.einsum('xyij->ijxy', self.F_field)
            Fp_in = np.einsum('xyij->ijxy', self.F_plastic)
            
            F_out, P_out, Sig_out, K4_out, F_bar_updated = finite_strain_solver_step_2d(
                F_in, F_bar, self.Ghat4_fs, self.C4_fs, self.I2_fs, self.I4_fs, self.I4rt_fs, Fp=Fp_in,
                driving_component=drv_comp, P_target=P_tgt, P_mask=P_mask,
                E_avg=self.E_field.mean(), nu_avg=self.nu_field.mean(),
                enable_console=False,
                model_type=self.hyperelastic_model,
                plane_mode=self.plane_mode,
                A_m=self.A_m, B_m=self.B_m, C_m=self.C_m,
                solver=self.solver, pixel=self.pixel
            )
            
            self.F_field = np.einsum('ijxy->xyij', F_out)
            self.sig_field = np.einsum('ijxy->xyij', Sig_out)
            
            from .finite_strain_simulator import _dot22, _trans2
            E_GL = 0.5 * (_dot22(_trans2(F_out), F_out) - self.I2_fs)
            self.eps_field = np.einsum('ijxy->xyij', E_GL)
            
            self.F_macro = F_bar_updated
            for ii in range(2):
                for jj in range(2):
                    if ii == jj:
                        self.eps_macro[ii, jj] = self.F_macro[ii, jj] - 1.0
                    else:
                        self.eps_macro[ii, jj] = self.F_macro[ii, jj]
            
            sig_mean = self.sig_field.mean(axis=(0,1))
            if np.any(np.isnan(self.sig_field)) or np.any(np.isnan(self.F_field)) or np.max(np.abs(sig_mean)) / 1e9 > 20.0:
                print("\n" + "="*80)
                print("[ALERT] Weird/unstable stress detected in elastic_run!")
                print(f"Macro Strain (diagonal): {[self.eps_macro[i,i] for i in range(2)]}")
                print(f"Mean Stress (GPa):\n{sig_mean / 1e9}")
                print(f"Mean F_field:\n{self.F_field.mean(axis=(0,1))}")
                print(f"Max F_plastic:\n{np.max(np.abs(self.F_plastic), axis=(0,1))}")
                if getattr(self, "last_flip", None) is not None:
                    print(f"Last flipped voxel (x, y, mode): {self.last_flip}")
                print("="*80 + "\n")
                raise ValueError(f"Simulation stopped due to unstable stress: {sig_mean/1e9} GPa")
            
            return sig_mean
        else:
            if getattr(self, "hyperelastic_model", "linear") == "secant_degradation":
                if not hasattr(self, "lam_field"):
                    from .elasticity import compute_lame_2d
                    self.lam_field, self.mu_field = compute_lame_2d(self.E_field, self.nu_field, plane_mode=self.plane_mode)
                self.eps_field, self.sig_field, _, _ = spectral_solver_secant_2d(
                    self.lam_field, self.mu_field, self.d, self.k,
                    eps_macro, eps_plastic=self.eps_plastic,
                    pixel=self.pixel, plane_mode=self.plane_mode, **self.solver_args
                )
            else:
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

    def heat_conducting_2d(self, dt):
        """Solve 2D heat equation in Fourier space."""
        if not self.enable_thermal or dt <= 0:
            return
        from .fft import fft_field, ifft_field
        T_hat = fft_field(self.Tlocal)
        T_hat *= np.exp(-self.k2 * self.diffusivity_scaled * dt)
        self.Tlocal = ifft_field(T_hat)
        
        if self.thermostat:
            if self.tau_bath > 0:
                self.Tlocal = self.temperature + (self.Tlocal - self.temperature) * np.exp(-dt / self.tau_bath)
            else:
                self.Tlocal = self.Tlocal - (np.mean(self.Tlocal) - self.temperature)

        if self.temperature_cap > 0:
            self.Tlocal = np.clip(self.Tlocal, a_min=None, a_max=self.temperature_cap)

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
            flipped_events = []
            
            for k in range(n_unstable):
                x, y, m = unstable_indices[k]
                voxel_id = (x, y)
                if voxel_id in flipped_voxels_in_batch:
                    continue
                
                flipped_voxels_in_batch.add(voxel_id)
                flips_in_this_batch += 1
                flipped_events.append((x, y, m))
                
            if self.strain_assumption == "finite_strain":
                F_plastic_backup = self.F_plastic.copy()
                F_field_backup = self.F_field.copy()
                sig_field_backup = self.sig_field.copy()
                eps_field_backup = self.eps_field.copy()
                F_macro_backup = self.F_macro.copy()
                eps_macro_backup = self.eps_macro.copy()
                soft_prop_backup = self.soft_prop.copy()
                last_event_time_backup = self.last_event_time.copy()
                prev_strain_dir_backup = self.prev_strain_dir.copy()
                catalog_backup = self.catalog.copy()
                Q0_backup = self.Q0.copy()
                Tlocal_backup = self.Tlocal.copy()
                time_backup = self.time

                N_start = 2
                if len(flipped_events) > 0:
                    N_start = max(self._min_substeps_for_flip(self.catalog[x, y, m]) for x, y, m in flipped_events)
                N_start = min(N_start, 40)

                success = False
                for N in range(N_start, N_start + 20):
                    self.F_plastic = F_plastic_backup.copy()
                    self.F_field = F_field_backup.copy()
                    self.sig_field = sig_field_backup.copy()
                    self.eps_field = eps_field_backup.copy()
                    self.F_macro = F_macro_backup.copy()
                    self.eps_macro = eps_macro_backup.copy()
                    self.soft_prop = soft_prop_backup.copy()
                    self.last_event_time = last_event_time_backup.copy()
                    self.prev_strain_dir = prev_strain_dir_backup.copy()
                    self.catalog = catalog_backup.copy()
                    self.Q0 = Q0_backup.copy()
                    self.Tlocal = Tlocal_backup.copy()
                    self.time = time_backup

                    try:
                        sig_befores = {}
                        eps_befores = {}
                        C_saved = {}
                        for x, y, m in flipped_events:
                            sig_befores[(x,y)] = self.sig_field[x, y].copy()
                            eps_befores[(x,y)] = self.eps_field[x, y].copy()
                            C_saved[(x,y,m)] = self.catalog[x, y, m].copy()

                        for step_idx in range(1, N + 1):
                            for x, y, m in flipped_events:
                                C = self.catalog[x, y, m].copy()
                                C_sub = C / N
                                delta_Fp = expm(C_sub)
                                self.F_plastic[x, y] = np.dot(delta_Fp, self.F_plastic[x, y])

                                if step_idx == N:
                                    e11, e22, e12 = C[0,0], C[1,1], C[0,1]
                                    sum_sq = (e12**2) + (e22**2 + e11**2 + (e11 - e22)**2) / 6.0
                                    gp_new = self.soft_prop[x, y, 0] + self.jp * sum_sq
                                    if self.softening_cap > 0 and gp_new > self.softening_cap:
                                        gp_new = self.softening_cap
                                    self.soft_prop[x, y, 0] = gp_new
                                    self.soft_prop[x, y, 1] = self.jt * sum_sq
                                    self.last_event_time[x, y] = self.time

                                    if self.neighbor_softening_fraction > 0.0:
                                        nx, ny = self.nx, self.ny
                                        for dx in (-1, 0, 1):
                                            for dy in (-1, 0, 1):
                                                if dx == 0 and dy == 0: continue
                                                nx_n, ny_n = (x + dx + nx) % nx, (y + dy + ny) % ny
                                                gp_n = self.soft_prop[nx_n, ny_n, 0] + self.neighbor_softening_fraction * self.jp * sum_sq
                                                if self.softening_cap > 0 and gp_n > self.softening_cap:
                                                    gp_n = self.softening_cap
                                                self.soft_prop[nx_n, ny_n, 0] = gp_n
                                                self.soft_prop[nx_n, ny_n, 1] += self.neighbor_softening_fraction * self.jt * sum_sq
                                    
                                    self.prev_strain_dir[x, y] = C

                                    if self.enable_thermal:
                                        DeltaHeat = np.sum(self.sig_field[x, y] * C)
                                        delta_T = abs(DeltaHeat) / (self.rho * self.Cp)
                                        self.Tlocal[x, y] += delta_T
                                        if self.temperature_cap > 0:
                                            self.Tlocal[x, y] = min(self.Tlocal[x, y], self.temperature_cap)

                                    if self.redraw_directions or self.redraw_barriers:
                                        if self.redraw_directions:
                                            self.catalog[x, y] = stz_catalog_glass_2d(self.M, self.gamma0, stz_mode=self.stz_mode)
                                        if self.redraw_barriers:
                                            self.Q0[x, y] = self.barrier_generator((self.M,))
                                    else:
                                        self.catalog[x, y, m] = stz_catalog_glass_2d(1, self.gamma0, stz_mode=self.stz_mode)[0]
                                        self.Q0[x, y, m] = self.barrier_generator((1,))[0]

                            self.elastic_run(self.eps_macro)
                        
                        if hasattr(self, 'flip_callback'):
                            for x, y, m in flipped_events:
                                sig_after = self.sig_field[x, y].copy()
                                eps_after = self.eps_field[x, y].copy()
                                self.flip_callback(self, x, y, m, C_saved[(x,y,m)], sig_befores[(x,y)], sig_after, eps_befores[(x,y)], eps_after, 'cascade')
                        
                        success = True
                        break
                    except ValueError as e:
                        print(f"Warning: sub-stepping with N={N} failed: {e}. Trying N={N+1}...")
                
                if not success:
                    raise ValueError("Mechanical solver failed to converge under sub-stepping.")
            else:
                sig_befores = {}
                eps_befores = {}
                C_saved = {}
                for x, y, m in flipped_events:
                    sig_befores[(x,y)] = self.sig_field[x, y].copy()
                    eps_befores[(x,y)] = self.eps_field[x, y].copy()
                    C_saved[(x,y,m)] = self.catalog[x, y, m].copy()

                for x, y, m in flipped_events:
                    with open(os.path.join(self.output_dir, "flipped_voxels_log.txt"), 'a') as f_log:
                        f_log.write(f"{step},{x},{y},cascade\n")
                    apply_flip_soa_2d(self.eps_plastic, self.soft_prop, self.last_event_time, self.catalog, x, y, m, self.time, self.jp, self.jt, self.softening_cap, self.neighbor_softening_fraction)
                    self.prev_strain_dir[x, y] = self.catalog[x, y, m].copy()
                    
                    if self.enable_thermal:
                        DeltaHeat = np.sum(self.sig_field[x, y] * self.catalog[x, y, m])
                        delta_T = abs(DeltaHeat) / (self.rho * self.Cp)
                        self.Tlocal[x, y] += delta_T
                        if self.temperature_cap > 0:
                            self.Tlocal[x, y] = min(self.Tlocal[x, y], self.temperature_cap)
                    
                    if self.redraw_directions or self.redraw_barriers:
                        if self.redraw_directions:
                            self.catalog[x, y] = stz_catalog_glass_2d(self.M, self.gamma0, stz_mode=self.stz_mode)
                        if self.redraw_barriers:
                            self.Q0[x, y] = self.barrier_generator((self.M,))
                    else:
                        self.catalog[x, y, m] = stz_catalog_glass_2d(1, self.gamma0, stz_mode=self.stz_mode)[0]
                        self.Q0[x, y, m] = self.barrier_generator((1,))[0]
                    
                    if self.fast_patching_enabled:
                        gxx, gxy = self.catalog[x, y, m][0,0], self.catalog[x, y, m][0,1]
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

                if hasattr(self, 'flip_callback'):
                    for x, y, m in flipped_events:
                        sig_after = self.sig_field[x, y].copy()
                        eps_after = self.eps_field[x, y].copy()
                        self.flip_callback(self, x, y, m, C_saved[(x,y,m)], sig_befores[(x,y)], sig_after, eps_befores[(x,y)], eps_after, 'cascade')
            
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
                  enable_global_log=True, max_kmc_steps_pct=0.3, max_cascade_steps_pct=0.3, **kwargs):
        
        self.driving_component = component
        self.stress_targets = stress_targets
        self.last_flip = None
        
        self._init_logs(summary_filename, enable_summary_log, enable_global_log)
        start_time_total = time.time()
        elastic_steps_done, total_kmc_steps, cascade_event_count, step = 0, 0, 0, 1
        sequential_kmc_steps = 0
        max_sequential_kmc = int(max_kmc_steps_pct * self.nx * self.ny)
        max_cascade_limit = int(max_cascade_steps_pct * self.nx * self.ny)
        
        with open(os.path.join(self.output_dir, "flipped_voxels_log.txt"), 'w') as f_log:
            f_log.write("Step,X,Y,Type\n")
        
        def _do_logging(s, s_type, c_id, c_flips):
            epsM, sigM = self.eps_macro.copy(), self.sig_field.mean(axis=(0,1))
            self.log_global(s, elastic_steps_done, total_kmc_steps, self.time, epsM, sigM, c_id, c_flips)

            now, elapsed = datetime.now().strftime("%Y-%m-%d %H:%M:%S"), time.time() - start_time_total
            curr_strain_val = epsM[component]
            curr_stress_val = sigM[component]
            
            summary_line = f"{now:<22} {elapsed:<12.2f} {s:<8d} {s_type.upper():<15} {curr_strain_val:<12.6f} {curr_stress_val/1e9:<15.3f} {total_kmc_steps:<8d} {c_id:<8d} {c_flips:<8d} {self.time:<15.6e}"
            if self.enable_thermal:
                avg_T = np.mean(self.Tlocal)
                summary_line += f" {avg_T:<15.2f}"
            summary_line += "\n"
            
            if self._f_summary:
                self._f_summary.write(summary_line)
            if enable_console_log:
                if s == 0: # Print header on first call if console enabled
                    header = f"{'Timestamp':<22} {'Elapsed(s)':<12} {'Step':<8} {'Type':<15} {'Eps_xx':<12} {'Sig_xx(GPa)':<15} {'KMC':<8} {'Cascade':<8} {'Flips':<8} {'SimTime(s)':<15}"
                    if self.enable_thermal:
                        header += " T_avg(K)"
                    print(header)
                    print("-" * len(header))
                print(summary_line.strip())

            if vtk_interval is not None and vtk_interval not in ["none", "last"]:
                save_vtk = False
                if vtk_interval == "current":
                    save_vtk = True
                elif isinstance(vtk_interval, int):
                    if vtk_elastic_only:
                        # Count only elastic increments
                        if s_type.upper() in ["ELAST", "INIT"] and elastic_steps_done % vtk_interval == 0:
                            save_vtk = True
                    else:
                        # Count global steps (includes KMC/Cascade events)
                        if s % vtk_interval == 0:
                            save_vtk = True
                
                if save_vtk:
                    if not vtk_elastic_only or s_type.upper() in ["ELAST", "INIT"]:
                        export_to_vtk(os.path.join(self.output_dir, f"step_{s:05d}.vtu"), self.eps_field, self.sig_field, self.E_field, self.nu_field, pixel=self.pixel, Tlocal=self.Tlocal if self.enable_thermal else None)

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
                        
                        if cascade_event_count > max_cascade_limit:
                            print(f"\n[TERMINATE] Cascade instability sequence limit reached! {cascade_event_count} batches.")
                            return
                        continue
                    
                    cascade_event_count = 0

                eff_volume = self.volume if self.scale_rate_by_volume else 1.0
                rates, indices, total_rate = compute_rates_2d(self.Q, eff_volume, self.Tlocal if self.enable_thermal else self.temperature, self.nu0, instability_mode=self.instability_mode)
                if total_rate > 0:
                    t_wait = -np.log(np.random.rand()) / total_rate
                    if t_wait < remaining_time:
                        self.time += t_wait
                        self.heat_conducting_2d(t_wait)
                        self.eps_macro += strain_unit * (self.strain_rate * t_wait)
                        remaining_time -= t_wait
                        idx_flat = indices[select_event_2d(rates, total_rate)]
                        x, y, m = decode_index_2d(idx_flat, self.ny, self.M)
                        self.last_flip = (x, y, m)
                        with open(os.path.join(self.output_dir, "flipped_voxels_log.txt"), 'a') as f_log:
                            f_log.write(f"{step},{x},{y},kmc\n")
                        
                        is_instab = self.Q[x,y,m] <= self.stability_threshold
                        C = self.catalog[x,y,m].copy()
                        if self.strain_assumption == "finite_strain":
                            sig_before = self.sig_field[x, y].copy()
                            eps_before = self.eps_field[x, y].copy()
                            C_saved = C.copy()

                            F_plastic_backup = self.F_plastic.copy()
                            F_field_backup = self.F_field.copy()
                            sig_field_backup = self.sig_field.copy()
                            eps_field_backup = self.eps_field.copy()
                            F_macro_backup = self.F_macro.copy()
                            eps_macro_backup = self.eps_macro.copy()
                            soft_prop_backup = self.soft_prop.copy()
                            last_event_time_backup = self.last_event_time.copy()
                            prev_strain_dir_backup = self.prev_strain_dir.copy()
                            catalog_backup = self.catalog.copy()
                            Q0_backup = self.Q0.copy()
                            Tlocal_backup = self.Tlocal.copy()
                            time_backup = self.time

                            N_start = self._min_substeps_for_flip(C)
                            N_start = min(N_start, 40)

                            success = False
                            for N in range(N_start, N_start + 20):
                                self.F_plastic = F_plastic_backup.copy()
                                self.F_field = F_field_backup.copy()
                                self.sig_field = sig_field_backup.copy()
                                self.eps_field = eps_field_backup.copy()
                                self.F_macro = F_macro_backup.copy()
                                self.eps_macro = eps_macro_backup.copy()
                                self.soft_prop = soft_prop_backup.copy()
                                self.last_event_time = last_event_time_backup.copy()
                                self.prev_strain_dir = prev_strain_dir_backup.copy()
                                self.catalog = catalog_backup.copy()
                                self.Q0 = Q0_backup.copy()
                                self.Tlocal = Tlocal_backup.copy()
                                self.time = time_backup

                                try:
                                    for step_idx in range(1, N + 1):
                                        C_sub = C / N
                                        delta_Fp = expm(C_sub)
                                        self.F_plastic[x, y] = np.dot(delta_Fp, self.F_plastic[x, y])

                                        if step_idx == N:
                                            e11, e22, e12 = C[0,0], C[1,1], C[0,1]
                                            sum_sq = (e12**2) + (e22**2 + e11**2 + (e11 - e22)**2) / 6.0
                                            gp_new = self.soft_prop[x,y,0] + self.jp * sum_sq
                                            if self.softening_cap > 0 and gp_new > self.softening_cap: gp_new = self.softening_cap
                                            self.soft_prop[x,y,0] = gp_new
                                            self.soft_prop[x,y,1] = self.jt * sum_sq
                                            self.last_event_time[x,y] = self.time

                                            if self.neighbor_softening_fraction > 0.0:
                                                nx, ny = self.nx, self.ny
                                                for dx in (-1, 0, 1):
                                                    for dy in (-1, 0, 1):
                                                        if dx == 0 and dy == 0: continue
                                                        nx_n, ny_n = (x + dx + nx) % nx, (y + dy + ny) % ny
                                                        gp_n = self.soft_prop[nx_n, ny_n, 0] + self.neighbor_softening_fraction * self.jp * sum_sq
                                                        if self.softening_cap > 0 and gp_n > self.softening_cap: gp_n = self.softening_cap
                                                        self.soft_prop[nx_n, ny_n, 0] = gp_n
                                                        self.soft_prop[nx_n, ny_n, 1] += self.neighbor_softening_fraction * self.jt * sum_sq
                                            
                                            self.prev_strain_dir[x,y] = C

                                            if self.enable_thermal:
                                                DeltaHeat = np.sum(self.sig_field[x, y] * C)
                                                delta_T = abs(DeltaHeat) / (self.rho * self.Cp)
                                                self.Tlocal[x, y] += delta_T
                                                if self.temperature_cap > 0:
                                                    self.Tlocal[x, y] = min(self.Tlocal[x, y], self.temperature_cap)

                                            if self.redraw_directions or self.redraw_barriers:
                                                if self.redraw_directions:
                                                    self.catalog[x,y] = stz_catalog_glass_2d(self.M, self.gamma0, stz_mode=self.stz_mode)
                                                if self.redraw_barriers:
                                                    self.Q0[x,y] = self.barrier_generator((self.M,))
                                            else:
                                                self.catalog[x,y,m] = stz_catalog_glass_2d(1, self.gamma0, stz_mode=self.stz_mode)[0]
                                                self.Q0[x,y,m] = self.barrier_generator((1,))[0]

                                        self.elastic_run(self.eps_macro)
                                    
                                    success = True
                                    break
                                except ValueError as e:
                                    print(f"Warning: sub-stepping with N={N} failed: {e}. Trying N={N+1}...")
                            
                            if success and hasattr(self, 'flip_callback'):
                                sig_after = self.sig_field[x, y].copy()
                                eps_after = self.eps_field[x, y].copy()
                                self.flip_callback(self, x, y, m, C_saved, sig_before, sig_after, eps_before, eps_after, 'kmc')

                            if not success:
                                raise ValueError("Mechanical solver failed to converge under sub-stepping.")
                        else:
                            sig_before = None
                            eps_before = None
                            if hasattr(self, 'flip_callback'):
                                sig_before = self.sig_field[x, y].copy()
                                eps_before = self.eps_field[x, y].copy()
                            apply_flip_soa_2d(self.eps_plastic, self.soft_prop, self.last_event_time, self.catalog, x, y, m, self.time, self.jp, self.jt, self.softening_cap, self.neighbor_softening_fraction)
                            self.prev_strain_dir[x,y] = C
                            
                            if self.enable_thermal:
                                DeltaHeat = np.sum(self.sig_field[x, y] * C)
                                delta_T = abs(DeltaHeat) / (self.rho * self.Cp)
                                self.Tlocal[x, y] += delta_T
                                if self.temperature_cap > 0:
                                    self.Tlocal[x, y] = min(self.Tlocal[x, y], self.temperature_cap)
                            
                            if self.redraw_directions or self.redraw_barriers:
                                if self.redraw_directions:
                                    self.catalog[x,y] = stz_catalog_glass_2d(self.M, self.gamma0, stz_mode=self.stz_mode)
                                if self.redraw_barriers:
                                    self.Q0[x,y] = self.barrier_generator((self.M,))
                            else:
                                self.catalog[x,y,m] = stz_catalog_glass_2d(1, self.gamma0, stz_mode=self.stz_mode)[0]
                                self.Q0[x,y,m] = self.barrier_generator((1,))[0]
                            
                            if self.fast_patching_enabled:
                                if self.sigma_macro_unit is None:
                                    _, self.sigma_macro_unit, _, _ = spectral_solver_2d(
                                        self.E_field, self.nu_field, strain_unit,
                                        eps_plastic=np.zeros_like(self.eps_plastic), 
                                        pixel=self.pixel, plane_mode=self.plane_mode, **self.solver_args
                                    )
                                
                                if t_wait > 0:
                                    self.sig_field += self.sigma_macro_unit * (self.strain_rate * t_wait)
                                
                                gxx, gxy = C[0,0], C[0,1]
                                patch = gxx * self.patch_kernels[0] + gxy * self.patch_kernels[1]
                                mean_shift = gxx * self.patch_missing_mean[0] + gxy * self.patch_missing_mean[1]
                                
                                R = self.patch_radius
                                for dx in range(-R, R+1):
                                    for dy in range(-R, R+1):
                                        px, py = (x+dx)%self.nx, (y+dy)%self.ny
                                        self.sig_field[px, py] += patch[dx+R, dy+R]
                                self.sig_field += mean_shift
                                
                                if hasattr(self, 'flip_callback') and sig_before is not None:
                                    sig_after = self.sig_field[x, y].copy()
                                    eps_after = self.eps_field[x, y].copy()
                                    self.flip_callback(self, x, y, m, C, sig_before, sig_after, eps_before, eps_after, 'kmc')
                                
                                self.flips_since_sync += 1
                                if self.flips_since_sync >= self.sync_interval:
                                    self.elastic_run(self.eps_macro)
                                    self.flips_since_sync = 0
                            else:
                                self.elastic_run(self.eps_macro)
                                if hasattr(self, 'flip_callback') and sig_before is not None:
                                    sig_after = self.sig_field[x, y].copy()
                                    eps_after = self.eps_field[x, y].copy()
                                    self.flip_callback(self, x, y, m, C, sig_before, sig_after, eps_before, eps_after, 'kmc')
                        
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
                self.heat_conducting_2d(remaining_time)
                self.time += remaining_time
                remaining_time = 0
            
            if self.strain_assumption == "finite_strain":
                self.elastic_run(self.eps_macro)
            else:
                E_avg = self.E_field.mean()
                nu_avg = self.nu_field.mean()
                for it in range(mixed_max_iter):
                    sigM = self.elastic_run(self.eps_macro)
                    stress_err = np.zeros((2, 2))
                    err_max = 0.0
                    for idx, target in stress_targets.items():
                        if idx[0] < 2 and idx[1] < 2:
                            err = target - sigM[idx]
                            stress_err[idx] = err
                            err_max = max(err_max, abs(err))
                    if err_max < mixed_tol:
                        break
                    
                    # Coupled Poisson-corrected correction
                    tr_sig = np.trace(stress_err)
                    d_eps = (stress_err - nu_avg * tr_sig * np.eye(2)) / E_avg
                    for idx in stress_targets.keys():
                        if idx[0] < 2 and idx[1] < 2:
                            self.eps_macro[idx] += d_eps[idx]
            
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
