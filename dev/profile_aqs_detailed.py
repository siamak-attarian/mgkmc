"""
Detailed profiling of AQS with instrumented timing.
This version adds timing measurements to each major operation.
"""
import numpy as np
import time
from mgkmc import ThermalSimulation
from mgkmc.elasticity import get_uniaxial_stress_x

# Monkey-patch the AQS run method to add timing
import mgkmc.aqs
from mgkmc.stz.update_fft import update_stress_fft_full
from mgkmc.stz.cascade import find_unstable, apply_flip

original_run = mgkmc.aqs.ThermalSimulation.run

def instrumented_run(self, n_global_steps, strain_increment_tensor=None, vtk_mode="global", 
                     loading_func=None, loading_params=None):
    """Instrumented version of run() with detailed timing."""
    
    print("Starting INSTRUMENTED AQS Simulation...")
    self.vtk_mode = vtk_mode
    
    # Timing accumulators
    timings = {
        'loading_func': 0.0,
        'update_stress_fft': 0.0,
        'find_unstable': 0.0,
        'apply_flip': 0.0,
        'logging': 0.0,
        'other': 0.0
    }
    
    # Determine loading mode
    if loading_func is not None:
        if loading_params is None:
            raise ValueError("loading_params required when loading_func is provided")
        use_loading_func = True
        eps_target = loading_params.get("eps_xx", 0.1)
    else:
        if strain_increment_tensor is None:
            raise ValueError("Either strain_increment_tensor or loading_func must be provided")
        use_loading_func = False
    
    # Initial relaxation
    print("Step 0 (Initial Relaxation)...")
    self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
        self.grid, self.eps_macro, self.E, self.nu, 
        pixel=self.pixel, **self.solver_args
    )
    
    # Initial cascade (simplified, no timing)
    local_steps = 0
    total_flips = 0
    self.log_global(0, eps_macro_curr, sig_macro_curr, local_steps, total_flips)
    
    # Main loop
    for step in range(1, n_global_steps + 1):
        step_start = time.perf_counter()
        
        # 1. Loading function
        if use_loading_func:
            t0 = time.perf_counter()
            current_params = loading_params.copy()
            current_params["eps_xx"] = eps_target * (step / n_global_steps)
            target_eps = loading_func(**current_params)
            self.eps_macro = target_eps
            timings['loading_func'] += time.perf_counter() - t0
        else:
            self.eps_macro += strain_increment_tensor
        
        # 2. Update stress (spectral solver)
        t0 = time.perf_counter()
        self.eps_field, self.sig_field, eps_macro_curr, sig_macro_curr = update_stress_fft_full(
            self.grid, self.eps_macro, self.E, self.nu, 
            pixel=self.pixel, **self.solver_args
        )
        timings['update_stress_fft'] += time.perf_counter() - t0
        
        # 3. Find unstable
        t0 = time.perf_counter()
        unstable = find_unstable(self.grid, self.volume, 
                                 softening_scheme=self.softening_scheme, 
                                 debug_first_flip=False)
        timings['find_unstable'] += time.perf_counter() - t0
        
        # 4. Apply flips (if any)
        if unstable:
            t0 = time.perf_counter()
            for x, y, z, m in unstable:
                voxel = self.grid[x,y,z]
                apply_flip(voxel, m, jp=self.jp, jt=self.jt, g_max=self.softening_cap)
                voxel.set_catalog(self.mode_generator(self.M, self.gamma0))
                voxel.reset_barriers(self.barrier_generator)
            timings['apply_flip'] += time.perf_counter() - t0
        
        # 5. Logging
        t0 = time.perf_counter()
        self.log_global(step, eps_macro_curr, sig_macro_curr, 0, len(unstable) if unstable else 0)
        self.history_global.append((eps_macro_curr[0,0], sig_macro_curr[0,0]/1e9))
        timings['logging'] += time.perf_counter() - t0
        
        step_time = time.perf_counter() - step_start
        timings['other'] += step_time - sum([timings[k] for k in ['loading_func', 'update_stress_fft', 'find_unstable', 'apply_flip', 'logging']])
        
        print(f"Step {step}: Time={step_time:.4f}s, Sig_xx={sig_macro_curr[0,0]:.2e}")
    
    print("\nSimulation Complete.")
    
    # Print timing breakdown
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN")
    print("=" * 60)
    total = sum(timings.values())
    for key, val in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * val / total if total > 0 else 0
        print(f"  {key:20s}: {val:8.4f} s ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total:8.4f} s")
    print("=" * 60)

# Monkey-patch
mgkmc.aqs.ThermalSimulation.run = instrumented_run

def main():
    print("=" * 60)
    print("DETAILED AQS PROFILING")
    print("=" * 60)
    
    SEED = 42
    np.random.seed(SEED)
    
    nx, ny, nz = 128, 128, 1
    pixel = 0.7
    M = 20
    gamma0 = 0.14
    
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)
    
    ENABLE_SOFTENING = True
    SOFTENING_SCHEME = "directional"
    SOFTENING_PARAMS = {"jp": 11, "jt": 33}
    SOFTENING_CAP = 0.51
    OUTPUT_DIR = "aqs_profile_detailed"
    
    def my_barrier_generator(n_modes):
        random_barriers = np.random.normal(loc=2.0, scale=0.6, size=n_modes)
        return np.clip(random_barriers, a_min=0.5, a_max=None)
    
    sim = ThermalSimulation(
        nx, ny, nz, M=M, gamma0=gamma0,
        E_field=E, nu_field=nu, pixel=pixel,
        barrier_generator=my_barrier_generator,
        output_dir=OUTPUT_DIR,
        softening_enabled=ENABLE_SOFTENING,
        softening_params=SOFTENING_PARAMS,
        softening_scheme=SOFTENING_SCHEME,
        softening_cap=SOFTENING_CAP,
        solver_args=None,
        debug_first_flip=False
    )
    
    eps_target = 0.01
    n_steps = 10
    
    sim.run(
        n_steps, 
        vtk_mode=None,
        loading_func=get_uniaxial_stress_x,
        loading_params={
            "eps_xx": eps_target,
            "E": E.mean(),
            "nu": nu.mean(),
        }
    )

if __name__ == "__main__":
    main()
