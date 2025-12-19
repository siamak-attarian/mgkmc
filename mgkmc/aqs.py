import numpy as np
import os
import time
from mgkmc.stz.grid import initialize_grid
from mgkmc.stz.cascade import find_unstable, apply_flip
from mgkmc.stz.update_fft import update_stress_fft, update_stress_fft_full
from mgkmc.elasticity import stress_from_strain
from mgkmc.stz.catalog import stz_catalog_glass
from mgkmc.postprocess import export_to_vtk
from mgkmc.stz.kmc import compute_rates, select_event
from mgkmc.stz.barriers import compute_barrier

class AthermalSimulation:
    def __init__(self, 
                 nx, ny, nz, 
                 M, gamma0, 
                 E_field, nu_field,
                 pixel=1.0,  
                 barrier_generator=None,
                 mode_generator=stz_catalog_glass,
                 output_dir="aqs_output",
                 softening_enabled=False,
                 softening_params=None, # e.g. {"jp": 10, "jt": 30}
                 softening_scheme="isotropic", # "isotropic" or "directional"
                 softening_cap=0.51, # Default from C code: -log(0.6)
                 solver_args=None,
                 debug_first_flip=False,
                 temperature=0.0, # Kelvin
                 strain_rate=1.0, # 1/s, used for KMC decision
                 strain_rate_sensitivity=0.0, # 's' exponent
                 stability_threshold=0.0 # eV, threshold for athermal instability
                 ):
        """
        Initialize Athermal Quasi-Static Simulation (with Thermal extensions).
        
        Parameters
        ----------
        nx, ny, nz : int
            Grid dimensions
        M : int
            Number of STZ modes per voxel
        gamma0 : float
            Characteristic shear strain (dimensionless)
        E_field : np.ndarray
            Young's modulus field in GPa (will be converted to Pa internally)
        nu_field : np.ndarray
            Poisson's ratio field (dimensionless)
        pixel : float
            Voxel size in nm (default: 1.0)
        barrier_generator : callable, optional
            Function to generate activation barriers (eV)
        mode_generator : callable, optional
            Function to generate STZ mode catalog
        output_dir : str
            Output directory for logs and VTK files
        softening_enabled : bool
            Enable barrier softening (default: False)
        solver_args : dict, optional
            Arguments for spectral solver
        debug_first_flip : bool
            Enable detailed diagnostics for first flip event
        temperature : float
            Temperature in Kelvin (default: 0.0)
        strain_rate : float
            Applied strain rate in 1/s (default: 1.0)
        strain_rate_sensitivity : float
            Sensitivity exponent for strain rate (default: 0.0)
        """
        
        self.nx, self.ny, self.nz = nx, ny, nz
        self.M = M
        self.gamma0 = gamma0 # Renamed from GAMMA0
        self.pixel = pixel
        self.volume = pixel**3
        self.output_dir = output_dir
        self.debug_first_flip = debug_first_flip
        self.first_flip_occurred = False
        self.softening_scheme = softening_scheme
        self.softening_cap = softening_cap

        # KMC / Thermal Parameters
        self.temperature = temperature
        self.strain_rate = strain_rate
        self.strain_rate_sensitivity = strain_rate_sensitivity
        self.stability_threshold = stability_threshold
        self.time = 0.0 # Simulation time in seconds
        
        # Softening Decay Time Constant (t_Temp)
        # Defaults to inf (no decay) until temperature is set/run
        self.tau = np.inf 
        
        # Softening Parameters (initialized to 0, will be set if softening_enabled)
        self.jp = 0.0
        self.jt = 0.0
        
        if self.temperature > 0:
            print(f"Thermal KMC ENABLED: T={self.temperature}K, Rate={self.strain_rate}/s")
        
        # Softening parameters
        if softening_enabled:
            params = softening_params if softening_params else {"jp": 100, "jt": 300}
            self.jp = params.get("jp", 100)
            self.jt = params.get("jt", 300)
            print(f"Softening ENABLED: jp={self.jp}, jt={self.jt}")
        else:
            self.jp, self.jt = 0, 0
            print("Softening DISABLED")

        # Solver config
        self.solver_args = solver_args if solver_args else {"max_iter": 200, "tol": 1e-6}
        
        # Store initial fields (in GPa for E)
        self.E_field_initial = E_field.copy()  # GPa
        self.nu_field_initial = nu_field.copy()
        
        # Convert E from GPa to Pa for internal use
        self.E = E_field * 1e9  # GPa → Pa
        self.nu = nu_field
        self.mode_generator = mode_generator
        self.barrier_generator = barrier_generator
        
        # Checkpoint tracking
        self.current_step = 0
        self.loading_func_name = None
        self.loading_params = None
        self.strain_increment_tensor = None
        self.vtk_mode = None

        # Initialize Grid
        print(f"Initializing {nx}x{ny}x{nz} grid with M={M}, gamma0={gamma0}...")
        self.grid = initialize_grid(
            nx, ny, nz, M, gamma0, 
            barrier_generator=barrier_generator, 
            mode_generator=mode_generator
        )

        # Macroscopic State
        self.eps_macro = np.zeros((3,3))

        # Initial fields (zero)
        self.eps_field = np.zeros((nx,ny,nz,3,3))
        self.sig_field = np.zeros((nx,ny,nz,3,3))

        # Setup Logging
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.global_log_path = os.path.join(output_dir, "global_log.txt")
        self.cascade_log_path = os.path.join(output_dir, "detailed_cascade.txt")
        
        # History for plotting
        self.history_global = [] # List of (eps_xx, sig_xx) tuples
        self.history_detailed = [] # List of (eps_xx, sig_xx) tuples including cascades
        self.flip_event_history = [] # List of tuples: (global_step, local_step, x, y, z, m)
        
        self._init_logs()
    
    def save_checkpoint(self, filename):
        """
        Save complete simulation state to checkpoint file.
        
        Parameters
        ----------
        filename : str
            Path to checkpoint file (HDF5 format)
        """
        from mgkmc.checkpoint import save_checkpoint
        save_checkpoint(self, filename)
    
    def export_vtk(self, filename):
        """
        Export current state to VTK.
        
        Parameters
        ----------
        filename : str
            Output path for VTK file.
        """
        from mgkmc.postprocess import export_to_vtk
        export_to_vtk(filename, self.eps_field, self.sig_field, self.E, self.nu, self.pixel,
                      grid=self.grid, include_plastic=True, match_matplotlib_orientation=True)
    
    @classmethod
    def load_checkpoint(cls, filename):
        """
        Load simulation from checkpoint file.
        
        Parameters
        ----------
        filename : str
            Path to checkpoint file
        
        Returns
        -------
        sim : AthermalSimulation
            Reconstructed simulation instance
        """
        from mgkmc.checkpoint import load_checkpoint
        return load_checkpoint(filename)
    
    def get_plastic_strain_field(self):
        """
        Extract plastic strain field from all voxels.
        
        Returns
        -------
        eps_plastic : np.ndarray (nx, ny, nz, 3, 3)
            Plastic strain tensor field
        """
        eps_plastic = np.zeros((self.nx, self.ny, self.nz, 3, 3))
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    eps_plastic[i, j, k] = self.grid[i, j, k].eps_plastic
        return eps_plastic
    
    def get_softening_fields(self):
        """
        Extract softening parameter fields from all voxels.
        
        Returns
        -------
        g_p : np.ndarray (nx, ny, nz)
            Plastic softening field
        g_t : np.ndarray (nx, ny, nz)
            Transient softening field
        """
        g_p = np.zeros((self.nx, self.ny, self.nz))
        g_t = np.zeros((self.nx, self.ny, self.nz))
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    g_p[i, j, k] = self.grid[i, j, k].g_p
                    g_t[i, j, k] = self.grid[i, j, k].g_t
        return g_p, g_t

    def _init_logs(self):
        # Global Log Header
        # Define widths for alignment
        # Step: 6, Floats: 15, Ints: 14
        header_fmt = "{:<6} " + " ".join(["{:>15}"]*12) + " {:>14} {:>14}"
        headers = [
            "Step", 
            "Eps_xx", "Eps_yy", "Eps_zz", "Eps_xy", "Eps_xz", "Eps_yz",
            "Sig_xx", "Sig_yy", "Sig_zz", "Sig_xy", "Sig_xz", "Sig_yz",
            "CascadeSteps", "TotalFlips"
        ]
        
        with open(self.global_log_path, "w") as f:
            f.write(header_fmt.format(*headers) + "\n")
            
        # Detailed Cascade Log Header
        with open(self.cascade_log_path, "w") as f:
            headers = [
                "GlobalStep", "LocalStep", "NumUnstable", 
                "FlippedVoxels(x,y,z,mode)" 
            ]
            f.write("\t".join(headers) + "\n")

    def log_global(self, step, eps, sig, cascade_steps, total_flips):
        # eps, sig here are MACROSCOPIC 3x3
        # Flatten tensors for logging
        # Tensor order: xx, xy, xz, yx, yy, yz, zx, zy, zz
        # We want: xx, yy, zz, xy, xz, yz
        indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
        
        # Consistent formatting
        line_fmt = "{:<6d} " + " ".join(["{:>15.6e}"]*12) + " {:>14d} {:>14d}"
        
        values = [step]
        values.extend([eps[i,j] for i,j in indices])
        values.extend([sig[i,j] for i,j in indices])
        values.append(cascade_steps)
        values.append(total_flips)
        
        with open(self.global_log_path, "a") as f:
            f.write(line_fmt.format(*values) + "\n")

    def log_cascade(self, global_step, local_step, unstable_list):
        n_unstable = len(unstable_list)
        # Limit string length if huge
        if n_unstable > 50:
             flip_str = f"{n_unstable} voxels flipped (truncated)"
        else:
             flip_str = ";".join([f"({x},{y},{z},{m})" for x,y,z,m in unstable_list])
        
        row = [
            f"{global_step}",
            f"{local_step}",
            f"{n_unstable}",
            flip_str
        ]
        
        with open(self.cascade_log_path, "a") as f:
            f.write("\t".join(row) + "\n")

    def log_kmc(self, global_step, kmc_step, dt, event_idx):
        path = os.path.join(self.output_dir, "kmc_log.txt")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("GlobalStep\tKMCStep\tDt\tEvent(x,y,z,m)\n")
        
        x, y, z, m = event_idx
        with open(path, "a") as f:
            f.write(f"{global_step}\t{kmc_step}\t{dt:.6e}\t({x},{y},{z},{m})\n")

    def update_barriers(self):
        """Re-compute barriers for all voxels given current time (updates decay)."""
        Nx, Ny, Nz = self.grid.shape
        for x in range(Nx):
            for y in range(Ny):
                 for z in range(Nz):
                      compute_barrier(self.grid[x,y,z], self.volume, 
                                      softening_scheme=self.softening_scheme,
                                      current_time=self.time, tau=self.tau)

    def _run_cascade(self, global_step):
        """
        Run the avalanche loop until stability.
        Uses self.eps_field and self.sig_field as current state.
        Updates self.eps_field, self.sig_field, and returns (local_step, total_flips, eps_macro, sig_macro)
        """
        local_step = 0
        total_flips = 0

        # We need an initial solve to get valid fields for checking barriers?
        # Actually barriers depend on sigma. If sigma is old, it might be wrong.
        # But we assume calling function did an update.
        
        # We need to return the LAST macro state
        # Initialize with current macro (if NO cascade happens)
        # We need to re-calculate macro from field or get it from update?
        # A bit messy to get macro from stored field without integration.
        # We'll rely on the update loop.
        
        # If we enter consistent, we have self.eps_field, self.sig_field correct.
        eps_macro_out = self.eps_macro.copy()
        sig_macro_out = np.mean(self.sig_field, axis=(0,1,2))
        
        enable_debug = self.debug_first_flip and not self.first_flip_occurred

        while True:
            # 1. Check stability (using decayed barriers)
            unstable = find_unstable(self.grid, self.volume, 
                                     softening_scheme=self.softening_scheme,
                                     threshold=self.stability_threshold,
                                     debug_first_flip=enable_debug,
                                     current_time=self.time, tau=self.tau)
            
            if not unstable:
                break
            
            # Mark that first flip has occurred
            if enable_debug and len(unstable) > 0:
                self.first_flip_occurred = True
                print(f"\n[DEBUG] ========== FIRST FLIP EVENT at Global Step {global_step} ==========")
                print(f"[DEBUG] Macroscopic stress (Pa): {sig_macro_out}")
                print(f"[DEBUG] Macroscopic strain: {eps_macro_out}")
            
            # 2. Log this cascade event
            self.log_cascade(global_step, local_step, unstable)
            
            # 3. Flip all unstable sites
            n_flips = len(unstable)
            total_flips += n_flips
            
            # Record detailed flip history
            for x, y, z, m in unstable:
                self.flip_event_history.append((global_step, local_step, x, y, z, m))
            
            # Stop criteria: >80% of elements flipped
            n_elements = self.nx * self.ny * self.nz
            if total_flips > 0.8 * n_elements:
                raise RuntimeError(f"Simulation stopped: More than 80% of elements flipped ({total_flips} > {0.8 * n_elements:.1f}) in a single cascade.")
            
            
            for x, y, z, m in unstable:
                voxel = self.grid[x,y,z]
                # Update plastic strain & softening (Propagate time)
                apply_flip(voxel, m, jp=self.jp, jt=self.jt, g_max=self.softening_cap, current_time=self.time)
                
                # Renew catalog
                if hasattr(self, 'mode_generator'):
                     voxel.set_catalog(self.mode_generator(self.M, self.gamma0))
                else:
                     voxel.set_catalog(stz_catalog_glass(self.M, self.gamma0))
                
                # Reform barriers
                voxel.reset_barriers(self.barrier_generator)
            
            # 4. Global Elastic Relax (FFT)
            self.eps_field, self.sig_field, eps_macro_out, sig_macro_out = update_stress_fft_full(
                self.grid, self.eps_macro, self.E, self.nu, 
                pixel=self.pixel, **self.solver_args
            )
            
            # Record detailed history
            self.history_detailed.append((eps_macro_out[0,0], sig_macro_out[0,0]/1e9))
            
            # Optional: Intermediate VTK
            if self.vtk_mode == "detailed":
                 fname = os.path.join(self.output_dir, f"step_{global_step:04d}_local_{local_step:04d}.vtu")
                 export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, 
                              grid=self.grid, include_plastic=True, match_matplotlib_orientation=True)

            local_step += 1
            
        return local_steps if 'local_steps' in locals() else local_step, total_flips, eps_macro_out, sig_macro_out

    def run(self, n_global_steps, strain_increment_tensor=None, vtk_mode="global", 
            loading_func=None, loading_params=None,
            checkpoint_interval=None, checkpoint_path=None, keep_checkpoints=False,
            stop_on_stress_drop=None, stress_drop_component=(0,0), stop_post_drop_steps=0,
            kmc_mode="accumulate"):
        """
        Run Simulation (KMC or AQS).
        
        Parameters
        ----------
        n_global_steps : int
            Number of elastic loading steps
        strain_increment_tensor : np.ndarray (3,3), optional
            Strain increment per ELastic step (for pure strain control)
        vtk_mode : str
            "global", "detailed", or None
        loading_func : callable, optional
            Function that returns strain tensor given parameters
        loading_params : dict, optional
            Parameters for loading_func 
        kmc_mode : str ("accumulate" or "on_demand")
            "accumulate" (Default): Nested loop. Multiple KMC events can occur before one elastic step.
                                    Time advances correctly. 'step' counts Elastic increments.
            "on_demand": C-style. Flattened loop. One iteration is either one KMC event OR one Elastic step.
                         'step' counts iterations (Events). Stops when target strain/steps reached.
        """
        print("Starting Simulation...")
        self.vtk_mode = vtk_mode
        self.mode_generator = getattr(self, 'mode_generator', stz_catalog_glass)
        
        # Checkpoint setup
        if checkpoint_interval is not None and checkpoint_path is None:
            checkpoint_path = os.path.join(self.output_dir, "checkpoint")
        
        # Store loading configuration
        if loading_func is not None:
            self.loading_func_name = loading_func.__name__ if hasattr(loading_func, '__name__') else str(loading_func)
            self.loading_params = loading_params
            self.strain_increment_tensor = None
        else:
            self.loading_func_name = None
            self.loading_params = None
            self.strain_increment_tensor = strain_increment_tensor
            
        # Determine dt_elastic (Time step for elastic loading)
        # Assuming strain_increment_tensor[0,0] is the main driver
        if self.strain_rate > 0 and strain_increment_tensor is not None:
            # Estimate dt from eps_xx increment
            d_eps = abs(strain_increment_tensor[0,0])
            if d_eps == 0:
                 d_eps = np.max(np.abs(strain_increment_tensor))
            dt_elastic = d_eps / self.strain_rate
        else:
            # Fallback or Loading Func
            # If loading func, we assume dt_elastic corresponds to (1 / n_steps * total_time)?
            # We'll default to 1.0/strain_rate if logical, or just 1.0
            dt_elastic = 1.0 / (self.strain_rate if self.strain_rate > 0 else 1.0)

        print(f"Elastic Time Step: {dt_elastic:.4e} s (based on rate {self.strain_rate})")

        # Stop logic state
        stop_drop_triggered = False
        stop_countdown = stop_post_drop_steps
        prev_stress_val = 0.0
        
        # Loading Setup (Same as before)
        if loading_func is not None:
            if loading_params is None:
                raise ValueError("loading_params required when loading_func is provided")
            use_loading_func = True
            eps_target = loading_params.get("eps_xx", 0.1)
        else:
            if strain_increment_tensor is None:
                raise ValueError("Either strain_increment_tensor or loading_func must be provided")
            use_loading_func = False

        # ==========================
        # Step 0: Initial Relaxation
        # ==========================
        self.current_step = 0
        print("Step 0 (Initial Relaxation)...")
        self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
            self.grid, self.eps_macro, self.E, self.nu, 
            pixel=self.pixel, **self.solver_args
        )
        
        local_steps, total_flips, eps_macro_curr, sig_macro_curr = self._run_cascade(global_step=0)
        self.log_global(0, eps_macro_curr, sig_macro_curr, local_steps, total_flips)
        
        prev_stress_val = sig_macro_curr[stress_drop_component]
        
        if vtk_mode == "global":
             fname = os.path.join(self.output_dir, "step_0000.vtu")
             export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, 
                          grid=self.grid, include_plastic=True, match_matplotlib_orientation=True)

        # ==========================
        # Main Loop (Hybrid KMC / Elastic)
        # ==========================
        step = 1
        kmc_substeps = 0
        elastic_steps_done = 0
        
        while elastic_steps_done < n_global_steps:

            self.current_step = step
            iteration_steps = 0
            iteration_flips = 0
            
            # KMC / Elastic Decision Loop
            while True:
                # 1. Check Athermal Stability first
                # (Pass current_time so stability check accounts for decay too!)
                unstable = find_unstable(self.grid, self.volume, 
                                         softening_scheme=self.softening_scheme,
                                         threshold=self.stability_threshold,
                                         current_time=self.time, tau=self.tau)
                if unstable:
                    # Trigger Avalanche
                    l_steps, t_flips, eps_macro_curr, sig_macro_curr = self._run_cascade(global_step=step)
                    iteration_steps += l_steps
                    iteration_flips += t_flips
                    # Time does NOT advance during avalanche (athermal)
                    continue
                
                # 2. Update Barriers (Decay) & Compute KMC Rates
                # Ensure Q is up to date with time decay
                self.update_barriers()
                
                if self.temperature > 0:
                    rates, indices, total_rate = compute_rates(
                        self.grid, self.volume, self.temperature, 
                        strain_rate_sensitivity=self.strain_rate_sensitivity,
                        applied_strain_rate=self.strain_rate,
                        current_time=self.time
                    )
                    idx, dt_kmc = select_event(rates, total_rate)
                else:
                    dt_kmc = float('inf')
                    idx = None
                
                # 3. Decision (Probabilistic Hybrid)
                # dt_kmc is now the Mean Residence Time (t_res = 1/R)
                # Probability of thermal event in dt_elastic: P = 1 - exp(-dt_elastic / t_res)
                # Equivalent check: eta > exp(-dt_elastic / t_res)
                
                trigger_threshold = np.exp(-dt_elastic / dt_kmc)
                if self.temperature > 0 and np.random.uniform() > trigger_threshold:
                    # THERMAL EVENT
                    x, y, z, m = indices[idx]
                    voxel = self.grid[x,y,z]
                    
                    # Apply Flip
                    apply_flip(voxel, m, jp=self.jp, jt=self.jt, g_max=self.softening_cap, current_time=self.time)
                    voxel.set_catalog(self.mode_generator(self.M, self.gamma0))
                    
                    # Log KMC
                    self.log_kmc(step, kmc_substeps, dt_kmc, (x,y,z,m))
                    self.time += dt_kmc
                    kmc_substeps += 1
                    
                    # Update Stress Logic
                    # We just flipped one voxel. This changes eps_plastic.
                    # We MUST update stress field to restore equilibrium and update Q for next step.
                    self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
                        self.grid, self.eps_macro, self.E, self.nu, 
                        pixel=self.pixel, **self.solver_args
                    )
                    
                    if kmc_mode == "on_demand":
                        # C-Style: Break loop. One event = One step.
                        # Do NOT increment elastic steps.
                        break
                    if kmc_mode == "on_demand":
                         # C-Style: We break the loop here. ONE event is ONE step.
                         # But we didn't do elastic loading, so elastic_steps_done does NOT increment.
                         break
                    
                else:
                    # ELASTIC EVENT
                    # Advance time
                    self.time += dt_elastic
                    
                    # Apply Loading
                    if use_loading_func:
                        current_params = loading_params.copy()
                        current_params["eps_xx"] = eps_target * (step / n_global_steps)
                        target_eps = loading_func(**current_params)
                        self.eps_macro = target_eps
                    else:
                        self.eps_macro += strain_increment_tensor
                    
                    # Update Stress
                    self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
                        self.grid, self.eps_macro, self.E, self.nu, 
                        pixel=self.pixel, **self.solver_args
                    )
                    
                    # Finalize Elastic Step
                    elastic_steps_done += 1
                    break
            
            # --- End of Decision Loop (Elastic Event Occurred) ---
            
            # Run Post-Elastic Cascade (Plastic Corrector)
            # This handles any instability caused by the elastic step
            local_steps, total_flips, eps_macro_curr, sig_macro_curr = self._run_cascade(global_step=step)
            
            # Log Global Step
            # For "on_demand", we log every step (even if just KMC), OR we log only on elastic?
            # C code logs every N events. Here we'll log every *elastic* step if accumulate,
            # but for on_demand we might get spam.
            # Let's keep existing behavior: Log only when this outer loop iterates.
            # In "accumulate": Outer loop = 1 Elastic Step (many KMC).
            # In "on_demand": Outer loop = 1 Event (KMC or Elastic).
            
            self.log_global(step, eps_macro_curr, sig_macro_curr, local_steps, total_flips)
            self.history_global.append((eps_macro_curr[0,0], sig_macro_curr[0,0]/1e9))
            
            curr_stress_val = sig_macro_curr[stress_drop_component]
            
            if kmc_mode == "on_demand" and dt_kmc < dt_elastic:
                event_type = "KMC"
            else:
                event_type = "ELASTIC"

            status_msg = f"Step {step} ({event_type}): KMC Events={kmc_substeps}, Cascade Steps={local_steps}, Flips={total_flips}, Sig_xx={curr_stress_val/1e9:.2f} GPa"
            kmc_substeps = 0 # Reset counter for next step
            
            # Update Outer Loop Counter
            step += 1
            
            # Checkpoint Logic
            if checkpoint_interval and step % checkpoint_interval == 0:
                if keep_checkpoints:
                    cp_name = f"{checkpoint_path}_{step:06d}.h5"
                else:
                    cp_name = f"{checkpoint_path}.h5"
                self.save_checkpoint(cp_name)

            # Stress Drop Detection Logic
            if stop_on_stress_drop is not None and not stop_drop_triggered:
                if abs(prev_stress_val) > 1e-6:
                    drop_frac = (prev_stress_val - curr_stress_val) / prev_stress_val
                else:
                    drop_frac = 0.0
                
                if drop_frac > stop_on_stress_drop:
                    print(f"\n[ALERT] Shear Band Detected! Stress drop {drop_frac*100:.1f}% > {stop_on_stress_drop*100:.1f}% at step {step}")
                    stop_drop_triggered = True
                    status_msg += " [SB DETECTED]"
            
            prev_stress_val = curr_stress_val
            
            print(status_msg)

            if vtk_mode == "global":
                 fname = os.path.join(self.output_dir, f"step_{step:04d}.vtu")
                 export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, 
                               grid=self.grid, include_plastic=True, match_matplotlib_orientation=True)

            # Stop Handling
            if stop_drop_triggered:
                if stop_countdown > 0:
                    stop_countdown -= 1
                else:
                    print(f"Stopping criteria: {stop_post_drop_steps} steps after detection.")
                    break
            
            step += 1

        print("Simulation Complete.")

    def _apply_strain_increment(self, eps_inc):
        """
        Apply homogeneous strain increment to all fields.
        
        Parameters
        ----------
        eps_inc : np.ndarray
            3x3 strain increment tensor
        """
        # Update macroscopic strain state
        self.eps_macro += eps_inc
        
        # Update fields and Sync Voxel objects (Crucial for find_unstable)
        self.eps_field, self.sig_field, _, _ = update_stress_fft_full(
            self.grid, self.eps_macro, self.E, self.nu, 
            pixel=self.pixel, **self.solver_args
        )

    def run_mixed(self, n_global_steps, strain_rate, component=(0,0), 
                  stress_targets=None,
                  mixed_tol=1e-4, mixed_max_iter=10,
                  vtk_mode="global",
                  checkpoint_interval=None, checkpoint_path=None, keep_checkpoints=False,
                  stop_on_stress_drop=None, stress_drop_component=(0,0), stop_post_drop_steps=0,
                  kmc_mode="accumulate"):
        """
        Run simulation with mixed boundary conditions (supports KMC).
        Drives strictly one strain component, while relaxing others to satisfy stress targets.
        
        Parameters
        ----------
        n_global_steps : int
            Number of steps
        strain_rate : float
            Strain increment per step for the driven component
        component : tuple
            Index of driven component, e.g. (0,0) for eps_xx
        stress_targets : dict, optional
            Target stress values for relaxed components.
            Default: All non-driven diagonal components target 0.0 (uniaxial tension).
            Format: {(1,1): 0.0, (2,2): 0.0}
        mixed_tol : float
            Tolerance for stress convergence (Pa)
        mixed_max_iter : int
            Maximum iterations for stress relaxation per step
        vtk_mode : str
            VTK output mode
        checkpoint_interval : int, optional
            Save checkpoint every N steps
        checkpoint_path : str, optional
            Base path for checkpoints
        keep_checkpoints : bool
            Save unique files if True
        stop_on_stress_drop : float, optional
            Stop if stress drops by this fraction
        stress_drop_component : tuple
            Which stress component to check
        stop_post_drop_steps : int
            Steps to continue after drop
        kmc_mode : str ("accumulate" or "on_demand")
            See run() docstring.
        """
        if stress_targets is None:
            # Default for uniaxial x-tension: relax yy and zz to zero
            stress_targets = {}
            if component == (0,0):
                stress_targets[(1,1)] = 0.0
                stress_targets[(2,2)] = 0.0
        
        self.vtk_mode = vtk_mode # Set for _run_cascade usage
        self.mode_generator = getattr(self, 'mode_generator', stz_catalog_glass)
        
        # Checkpoint setup
        if checkpoint_interval is not None and checkpoint_path is None:
            checkpoint_path = os.path.join(self.output_dir, "checkpoint")

        # Log setup
        self.loading_func_name = "mixed_control"
        self.loading_params = {
            "rate": strain_rate, 
            "comp": component,
            "targets": {str(k): v for k,v in stress_targets.items()}
        }
        
        print(f"Starting AQS Mixed Simulation: {n_global_steps} steps")
        print(f"Driving component {component} with rate {strain_rate}")
        print(f"Relaxing components: {list(stress_targets.keys())}")
        
        # Determine Elastic Time Step
        # strain_rate arg is "increment per step"
        # self.strain_rate is "1/s"
        if self.strain_rate > 0:
            dt_elastic = abs(strain_rate) / self.strain_rate
        else:
            dt_elastic = 1.0
        
        print(f"Elastic Time Step: {dt_elastic:.4e} s (based on rate {self.strain_rate})")
        
        # Compliance approximation for correction (isotropic)
        E_avg = self.E.mean()
        nu_avg = self.nu.mean()
        
        def get_correction(sigma_err):
            tr_sig = np.trace(sigma_err)
            return (sigma_err - nu_avg * tr_sig * np.eye(3)) / E_avg

        # Stop logic state
        stop_drop_triggered = False
        stop_countdown = stop_post_drop_steps
        prev_stress_val = 0.0

        # Ensure Step 0 is relaxed
        local_steps, total_flips, eps_macro_curr, sig_macro_curr = self._run_cascade(global_step=0)
        self.log_global(0, eps_macro_curr, sig_macro_curr, local_steps, total_flips)
        
        prev_stress_val = sig_macro_curr[stress_drop_component]
        
        if vtk_mode == "global":
             fname = os.path.join(self.output_dir, "step_0000.vtu")
             export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, 
                          grid=self.grid, include_plastic=True, match_matplotlib_orientation=True)

        step = 1
        kmc_substeps = 0
        elastic_steps_done = 0
        
        while elastic_steps_done < n_global_steps:
            self.current_step = step
            iteration_steps = 0
            iteration_flips = 0
            
            # KMC Decision Loop
            while True:
                # 1. Check Stability (should be stable from previous step, but check)
                unstable = find_unstable(self.grid, self.volume, 
                                         softening_scheme=self.softening_scheme,
                                         threshold=self.stability_threshold)
                if unstable:
                    l_steps, t_flips, _, _ = self._run_cascade(global_step=step)
                    continue
                
                # 2. Compute Rates
                if self.temperature > 0:
                    rates, indices, total_rate = compute_rates(
                        self.grid, self.volume, self.temperature,
                        strain_rate_sensitivity=self.strain_rate_sensitivity,
                        applied_strain_rate=self.strain_rate,
                        current_time=self.time
                    )
                    idx, dt_kmc = select_event(rates, total_rate)
                else:
                    dt_kmc = float('inf')
                    idx = None
                
                # 3. Decision (Probabilistic Hybrid)
                trigger_threshold = np.exp(-dt_elastic / dt_kmc)
                if self.temperature > 0 and np.random.uniform() > trigger_threshold:
                    # THERMAL EVENT
                    x, y, z, m = indices[idx]
                    voxel = self.grid[x,y,z]
                    
                    apply_flip(voxel, m, jp=self.jp, jt=self.jt, g_max=self.softening_cap)
                    voxel.set_catalog(self.mode_generator(self.M, self.gamma0))
                    
                    self.log_kmc(step, kmc_substeps, dt_kmc, (x,y,z,m))
                    self.time += dt_kmc
                    kmc_substeps += 1
                    
                    # Update Stress (without mixed relaxation)
                    self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
                        self.grid, self.eps_macro, self.E, self.nu, 
                        pixel=self.pixel, **self.solver_args
                    )
                    
                    if kmc_mode == "on_demand":
                         break
                    
                    if kmc_mode == "on_demand":
                        # C-Style: One event per step. Break inner loop.
                        # Do NOT increment elastic_steps_done
                        break
                else:
                    # ELASTIC EVENT
                    self.time += dt_elastic
                    
                    # 1. Apply Driving Strain
                    eps_inc = np.zeros((3,3))
                    eps_inc[component] = strain_rate
                    self._apply_strain_increment(eps_inc)
                    
                    # 2. Iterative Relaxation Loop (Mixed BC)
                    converged = False
                    
                    for it in range(mixed_max_iter):
                        l_steps, t_flips, eps_M, sig_M = self._run_cascade(global_step=step)
                        iteration_flips += t_flips
                        iteration_steps += l_steps
                        
                        stress_err_tensor = np.zeros((3,3))
                        max_err = 0.0
                        for idx_t, target in stress_targets.items():
                            err = target - sig_M[idx_t]
                            stress_err_tensor[idx_t] = err
                            max_err = max(max_err, abs(err))
                        
                        if max_err < mixed_tol:
                            converged = True
                            break
                        
                        eps_corr = get_correction(stress_err_tensor)
                        eps_corr[component] = 0.0
                        self._apply_strain_increment(eps_corr)
                    
                    if not converged:
                        print(f"Warning: Mixed control did not converge at step {step} (Max Err: {max_err:.2e})")
                    
                    # Break decision loop to finalize step
                    break
            
            # --- End of Decision Loop ---

            # Final values
            eps_macro_curr = self.eps_field.mean(axis=(0,1,2))
            sig_macro_curr = self.sig_field.mean(axis=(0,1,2))

            # Log
            iteration_steps = locals().get('iteration_steps', 0)
            iteration_flips = locals().get('iteration_flips', 0)
            self.log_global(step, eps_macro_curr, sig_macro_curr, iteration_steps, iteration_flips)
            self.history_global.append((eps_macro_curr[0,0], sig_macro_curr[0,0]/1e9))
            
            curr_stress_val = sig_macro_curr[stress_drop_component]
            status_msg = f"Step {step}: KMC={kmc_substeps}, Cascade={iteration_steps}, Flips={iteration_flips}, Sig_xx={curr_stress_val/1e9:.2f} GPa"
            kmc_substeps = 0

            # Checkpoint Logic
            if checkpoint_interval and step % checkpoint_interval == 0:
                if keep_checkpoints:
                    cp_name = f"{checkpoint_path}_{step:06d}.h5"
                else:
                    cp_name = f"{checkpoint_path}.h5"
                self.save_checkpoint(cp_name)

            # Stress Drop Detection Logic
            if stop_on_stress_drop is not None and not stop_drop_triggered:
                if abs(prev_stress_val) > 1e-6:
                    drop_frac = (prev_stress_val - curr_stress_val) / prev_stress_val
                else:
                    drop_frac = 0.0
                
                if drop_frac > stop_on_stress_drop:
                    print(f"\n[ALERT] Shear Band Detected! Stress drop {drop_frac*100:.1f}% > {stop_on_stress_drop*100:.1f}% at step {step}")
                    stop_drop_triggered = True
                    status_msg += " [SB DETECTED]"
            
            prev_stress_val = curr_stress_val
            print(status_msg)

            if vtk_mode == "global":
                 fname = os.path.join(self.output_dir, f"step_{step:04d}.vtu")
                 export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, 
                               grid=self.grid, include_plastic=True, match_matplotlib_orientation=True)

            if stop_drop_triggered:
                if stop_countdown > 0:
                    stop_countdown -= 1
                else:
                    print(f"Stopping criteria: {stop_post_drop_steps} steps after detection.")
                    break
            
            step += 1
            
        print("Mixed Simulation Complete.")
