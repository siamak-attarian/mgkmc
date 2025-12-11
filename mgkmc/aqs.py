import numpy as np
import os
import time
from mgkmc.stz.grid import initialize_grid
from mgkmc.stz.cascade import find_unstable, apply_flip
from mgkmc.stz.update_fft import update_stress_fft, update_stress_fft_full
from mgkmc.stz.catalog import stz_catalog_glass
from mgkmc.postprocess import export_to_vtk

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
                 debug_first_flip=False):
        """
        Initialize Athermal Quasi-Static Simulation.
        
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
        
        # Convert E from GPa to Pa for internal use
        self.E = E_field * 1e9  # GPa → Pa
        self.nu = nu_field
        self.mode_generator = mode_generator
        self.barrier_generator = barrier_generator

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
        
        self._init_logs()

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
        # But we need eps_macro_out to return.
        # Let's do a quick mean?
        eps_macro_out = self.eps_field.mean(axis=(0,1,2))
        sig_macro_out = self.sig_field.mean(axis=(0,1,2))

        while True:
            # 1. Check for stability
            # Enable debug for first flip event
            enable_debug = self.debug_first_flip and not self.first_flip_occurred
            unstable = find_unstable(self.grid, self.volume, 
                                     softening_scheme=self.softening_scheme, 
                                     debug_first_flip=enable_debug)
            
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
            
            # Stop criteria: >20% of elements flipped
            n_elements = self.nx * self.ny * self.nz
            if total_flips > 0.5 * n_elements:
                raise RuntimeError(f"Simulation stopped: More than 50% of elements flipped ({total_flips} > {0.2 * n_elements:.1f}) in a single cascade.")
            
            
            for x, y, z, m in unstable:
                voxel = self.grid[x,y,z]
                # Update plastic strain & softening
                apply_flip(voxel, m, jp=self.jp, jt=self.jt, g_max=self.softening_cap)
                # Regenerate catalog for this voxel
                # Note: Currently regenerating ALL modes for the voxel. 
                # Ideally might only regenerate the used one? 
                # But standard STZ usually renews the local config.
                # using the flexible mode generator from init
                # But we didn't save the generator function in self... 
                # Wait, "generation_function" in old code was passed in.
                # I should store it or just use the default logic?
                # User asked for flexibility. I should probably store `mode_generator` in `__init__`.
                # I'll fix this in the code below.
                if hasattr(self, 'mode_generator'):
                     voxel.set_catalog(self.mode_generator(self.M, self.gamma0))
                else:
                     # Fallback
                     voxel.set_catalog(stz_catalog_glass(self.M, self.gamma0))
                
                # Reform barriers (crucial for AQS stability)
                voxel.reset_barriers(self.barrier_generator)
            
            # 4. Global Elastic Relax (FFT)
            # Updates voxel.sigma and returns macro state (which we might overwrite later but needed for equilibrium)
            self.eps_field, self.sig_field, eps_macro_out, sig_macro_out = update_stress_fft_full(
                self.grid, self.eps_macro, self.E, self.nu, 
                pixel=self.pixel, **self.solver_args
            )
            
            # Record detailed history
            self.history_detailed.append((eps_macro_out[0,0], sig_macro_out[0,0]/1e9))
            
            # Optional: Intermediate VTK for detailed local steps?
            if self.vtk_mode == "detailed":
                 fname = os.path.join(self.output_dir, f"step_{global_step:04d}_local_{local_step:04d}.vtu")
                 export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, match_matplotlib_orientation=True)

            local_step += 1
            
        return local_step, total_flips, eps_macro_out, sig_macro_out

    def run(self, n_global_steps, strain_increment_tensor=None, vtk_mode="global", 
            loading_func=None, loading_params=None):
        """
        Run AQS simulation.
        
        Parameters
        ----------
        n_global_steps : int
            Number of loading steps
        strain_increment_tensor : np.ndarray (3,3), optional
            Strain increment per step (for pure strain control)
        vtk_mode : str
            "global", "detailed", or None
        loading_func : callable, optional
            Function that returns strain tensor given parameters (e.g., get_uniaxial_stress_x)
            If provided, uses this instead of strain_increment_tensor
        loading_params : dict, optional
            Parameters for loading_func (e.g., {"eps_xx": 0.1, "E": E, "nu": nu})
        """
        print("Starting AQS Simulation...")
        self.vtk_mode = vtk_mode
        self.mode_generator = getattr(self, 'mode_generator', stz_catalog_glass)
        
        # Determine loading mode
        if loading_func is not None:
            # Mixed BC mode: use loading function
            if loading_params is None:
                raise ValueError("loading_params required when loading_func is provided")
            use_loading_func = True
            # Extract target from params (assume eps_xx is the control parameter)
            eps_target = loading_params.get("eps_xx", 0.1)
        else:
            # Pure strain control mode
            if strain_increment_tensor is None:
                raise ValueError("Either strain_increment_tensor or loading_func must be provided")
            use_loading_func = False

        # ==========================
        # Global Step 0: Initial Relaxation
        # ==========================
        print("Step 0 (Initial Relaxation)...")
        self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
            self.grid, self.eps_macro, self.E, self.nu, 
            pixel=self.pixel, **self.solver_args
        )
        
        local_steps, total_flips, eps_macro_curr, sig_macro_curr = self._run_cascade(global_step=0)
        self.log_global(0, eps_macro_curr, sig_macro_curr, local_steps, total_flips)
        
        if vtk_mode == "global":
             fname = os.path.join(self.output_dir, "step_0000.vtu")
             export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, match_matplotlib_orientation=True)

        # ==========================
        # Global Loading Loop
        # ==========================
        for step in range(1, n_global_steps + 1):
            # Determine target strain for this step
            if use_loading_func:
                # Update loading params for current step
                current_params = loading_params.copy()
                current_params["eps_xx"] = eps_target * (step / n_global_steps)
                target_eps = loading_func(**current_params)
                self.eps_macro = target_eps
            else:
                # Apply strain increment
                self.eps_macro += strain_increment_tensor
            
            # Update Stress Field (Elastic Predictor)
            self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
                self.grid, self.eps_macro, self.E, self.nu, 
                pixel=self.pixel, **self.solver_args
            )
            
            # Record Predictor for Detailed Path
            self.history_detailed.append((eps_macro_curr[0,0], sig_macro_curr[0,0]/1e9))
            
            # Run Cascade (Plastic Corrector)
            local_steps, total_flips, eps_macro_curr, sig_macro_curr = self._run_cascade(global_step=step)
            
            # Log
            self.log_global(step, eps_macro_curr, sig_macro_curr, local_steps, total_flips)
            
            # Record Global Envelope
            self.history_global.append((eps_macro_curr[0,0], sig_macro_curr[0,0]/1e9))
            
            print(f"Step {step}: Cascade Steps={local_steps}, Flips={total_flips}, Sig_xx={sig_macro_curr[0,0]:.2e}")

            if vtk_mode == "global":
                 fname = os.path.join(self.output_dir, f"step_{step:04d}.vtu")
                 export_to_vtk(fname, self.eps_field, self.sig_field, self.E, self.nu, self.pixel, match_matplotlib_orientation=True)

        print("Simulation Complete.")
