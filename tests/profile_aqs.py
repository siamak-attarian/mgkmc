import numpy as np
import time
from mgkmc import AthermalSimulation
from mgkmc.elasticity import get_uniaxial_stress_x

def main():
    """
    Profile AQS simulation to identify bottlenecks.
    """
    print("=" * 60)
    print("AQS PROFILING - Finding Bottlenecks")
    print("=" * 60)
    
    # Same setup as aqs_demo_2.py but smaller for quick profiling
    SEED = 42
    np.random.seed(SEED)
    
    nx, ny, nz = 128, 128, 1
    pixel = 0.7
    M = 20
    gamma0 = 0.14
    
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)
    
    # Softening
    ENABLE_SOFTENING = True
    SOFTENING_SCHEME = "directional"
    SOFTENING_PARAMS = {"jp": 11, "jt": 33}
    SOFTENING_CAP = 0.51
    
    OUTPUT_DIR = "aqs_profile_temp"
    
    def my_barrier_generator(n_modes):
        random_barriers = np.random.normal(loc=2.0, scale=0.6, size=n_modes)
        min_barrier = 0.5
        clipped_barriers = np.clip(random_barriers, a_min=min_barrier, a_max=None)
        return clipped_barriers
    
    print("\nInitializing simulation...")
    init_start = time.perf_counter()
    
    sim = AthermalSimulation(
        nx, ny, nz,
        M=M, 
        gamma0=gamma0,
        E_field=E, 
        nu_field=nu,
        pixel=pixel,
        barrier_generator=my_barrier_generator,
        output_dir=OUTPUT_DIR,
        softening_enabled=ENABLE_SOFTENING,
        softening_params=SOFTENING_PARAMS,
        softening_scheme=SOFTENING_SCHEME,
        softening_cap=SOFTENING_CAP,
        solver_args=None,
        debug_first_flip=False
    )
    
    init_time = time.perf_counter() - init_start
    print(f"Initialization took: {init_time:.4f} s")
    
    # Run just a few steps for profiling
    eps_target = 0.01  # 1% strain
    n_steps = 10  # Just 10 steps for quick profiling
    
    print(f"\nRunning {n_steps} steps for profiling...")
    print("=" * 60)
    
    total_start = time.perf_counter()
    
    sim.run(
        n_steps, 
        vtk_mode=None,  # No VTK output for speed
        loading_func=get_uniaxial_stress_x,
        loading_params={
            "eps_xx": eps_target,
            "E": E.mean(),
            "nu": nu.mean(),
        }
    )
    
    total_time = time.perf_counter() - total_start
    
    print("=" * 60)
    print(f"\nTotal simulation time: {total_time:.4f} s")
    print(f"Time per step: {total_time / n_steps:.4f} s")
    print(f"Expected spectral solver time per step: ~0.1 s")
    print(f"Overhead factor: {(total_time / n_steps) / 0.1:.1f}x")
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    
    # Now let's add detailed timing to the AQS code
    print("\nTo identify the bottleneck, we need to add timing to:")
    print("  1. Spectral solver calls (update_stress_fft_full)")
    print("  2. Cascade loop (find_unstable, apply_flip)")
    print("  3. Grid operations (extract_eps_plastic, push_solver_results)")
    print("  4. Logging and I/O")

if __name__ == "__main__":
    main()
