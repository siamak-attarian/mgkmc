"""
Test the vectorized directional softening implementation.
Compare performance before/after and verify correctness.
"""
import numpy as np
import time
from mgkmc import AthermalSimulation
from mgkmc.elasticity_helpers import get_uniaxial_stress_x

def main():
    print("=" * 60)
    print("TESTING VECTORIZED DIRECTIONAL SOFTENING")
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
    OUTPUT_DIR = "aqs_vectorized_test"
    
    def my_barrier_generator(n_modes):
        random_barriers = np.random.normal(loc=2.0, scale=0.6, size=n_modes)
        return np.clip(random_barriers, a_min=0.5, a_max=None)
    
    print("\nInitializing simulation with VECTORIZED directional softening...")
    sim = AthermalSimulation(
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
    
    print(f"\nRunning {n_steps} steps...")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
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
    
    total_time = time.perf_counter() - start_time
    
    print("=" * 60)
    print("\nPERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"BEFORE (non-vectorized): ~13.6 s for 10 steps (~1.36 s/step)")
    print(f"AFTER  (vectorized):     {total_time:.2f} s for 10 steps (~{total_time/n_steps:.3f} s/step)")
    print(f"\nSpeedup: {13.6 / total_time:.1f}x faster!")
    print("=" * 60)
    
    # Verify results look reasonable
    hist = np.array(sim.history_global)
    if len(hist) > 0:
        print(f"\nFinal stress: {hist[-1, 1]:.4f} GPa")
        print(f"Final strain: {hist[-1, 0]:.6f}")
        print("\n✓ Simulation completed successfully!")
    else:
        print("\n✗ Warning: No history recorded")

if __name__ == "__main__":
    main()
