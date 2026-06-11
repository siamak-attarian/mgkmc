import numpy as np
import time
from mgkmc import run_simulation, get_uniaxial_stress_x

def main():
    """
    Speed test for uniaxial stress using get_uniaxial_stress_x.
    Runs 10 iterations and reports average timing.
    """
    print("=" * 60)
    print("SPECTRAL SOLVER SPEED TEST - UNIAXIAL STRESS")
    print("=" * 60)
    
    # System size
    nx, ny, nz = 128, 128, 1
    pixel = 0.7  # nm
    
    # Material properties (homogeneous)
    E = np.full((nx, ny, nz), 70.0)  # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    # Uniaxial stress parameters
    eps_xx = 1e-4  # 0.01% strain in x-direction
    E_avg = E.mean()
    nu_avg = nu.mean()
    
    # Solver parameters
    max_iter = 200
    tol = 1e-6
    n_steps = 1  # Single load step for speed test
    
    print(f"\nSystem Configuration:")
    print(f"  Grid size: {nx} x {ny} x {nz}")
    print(f"  Total voxels: {nx * ny * nz:,}")
    print(f"  Pixel size: {pixel} nm")
    print(f"  E: {E[0,0,0]} GPa")
    print(f"  nu: {nu[0,0,0]}")
    print(f"\nBoundary Conditions (Uniaxial Stress):")
    print(f"  eps_xx: {eps_xx} (prescribed)")
    print(f"  eps_yy: {-nu_avg * eps_xx:.6e} (calculated from nu)")
    print(f"  eps_zz: {-nu_avg * eps_xx:.6e} (calculated from nu)")
    print(f"\nSolver Parameters:")
    print(f"  Max iterations: {max_iter}")
    print(f"  Tolerance: {tol}")
    print(f"  Number of test runs: 10")
    
    print("\n" + "-" * 60)
    print("Running speed test (10 iterations)...")
    print("-" * 60)
    
    # Run 10 times and collect timings
    timings = []
    
    for run_idx in range(10):
        print(f"\nRun {run_idx + 1}/10:")
        
        start_time = time.perf_counter()
        
        eps_macro_list, sig_macro_list, eps_list, sig_list = run_simulation(
            E, nu,
            loading_func=get_uniaxial_stress_x,
            loading_params={"eps_xx": eps_xx, "E": E_avg, "nu": nu_avg},
            n_steps=n_steps,
            pixel=pixel,
            store=False,  # Don't store fields to save memory
            max_iter=max_iter,
            tol=tol,
            verbose=False
        )
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        timings.append(elapsed_time)
        
        print(f"  Time: {elapsed_time:.4f} s ({elapsed_time * 1000:.2f} ms)")
    
    # Calculate statistics
    timings = np.array(timings)
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    
    print("\n" + "=" * 60)
    print("TIMING STATISTICS (10 runs)")
    print("=" * 60)
    print(f"  Mean:   {mean_time:.4f} s ({mean_time * 1000:.2f} ms)")
    print(f"  Std:    {std_time:.4f} s ({std_time * 1000:.2f} ms)")
    print(f"  Min:    {min_time:.4f} s ({min_time * 1000:.2f} ms)")
    print(f"  Max:    {max_time:.4f} s ({max_time * 1000:.2f} ms)")
    
    # Show final results from last run
    eps_macro = eps_macro_list[-1]
    sig_macro = sig_macro_list[-1]
    
    print(f"\nFinal Results (from last run):")
    print(f"  Macroscopic strain (eps_macro):")
    print(f"    eps_xx: {eps_macro[0,0]:.6e}")
    print(f"    eps_yy: {eps_macro[1,1]:.6e}")
    print(f"    eps_zz: {eps_macro[2,2]:.6e}")
    print(f"  Macroscopic stress (sig_macro):")
    print(f"    sig_xx: {sig_macro[0,0]:.6e} Pa ({sig_macro[0,0]/1e9:.4f} GPa)")
    print(f"    sig_yy: {sig_macro[1,1]:.6e} Pa ({sig_macro[1,1]/1e9:.4f} GPa)")
    print(f"    sig_zz: {sig_macro[2,2]:.6e} Pa ({sig_macro[2,2]/1e9:.4f} GPa)")
    
    print("\n" + "=" * 60)
    print("SPEED TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
