"""
Micro-benchmark to test the barrier calculation performance directly.
"""
import numpy as np
import time
from mgkmc.stz.voxel import Voxel
from mgkmc.stz.catalog import stz_catalog_glass
from mgkmc.stz.barriers import compute_barrier

def main():
    print("=" * 60)
    print("MICRO-BENCHMARK: Barrier Calculation")
    print("=" * 60)
    
    # Create a single voxel
    M = 20
    gamma0 = 0.14
    volume = 0.7**3
    
    def my_barrier_generator(n_modes):
        return np.clip(np.random.normal(loc=2.0, scale=0.6, size=n_modes), a_min=0.5, a_max=None)
    
    voxel = Voxel(M, barrier_generator=my_barrier_generator)
    voxel.set_catalog(stz_catalog_glass(M, gamma0))
    voxel.sigma = np.random.randn(3, 3) * 1e8  # Random stress
    voxel.prev_gamma = np.random.randn(3, 3) * 0.1  # Random previous gamma
    voxel.g_p = 0.1
    voxel.g_t = 0.05
    
    # Warm-up
    for _ in range(10):
        compute_barrier(voxel, volume, softening_scheme="directional")
    
    # Benchmark
    n_iterations = 100000
    print(f"\nRunning {n_iterations:,} barrier calculations...")
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        Q = compute_barrier(voxel, volume, softening_scheme="directional")
    elapsed = time.perf_counter() - start
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.4f} s")
    print(f"  Time per call: {elapsed / n_iterations * 1e6:.2f} µs")
    print(f"  Calls per second: {n_iterations / elapsed:,.0f}")
    
    # For 16,384 voxels
    voxels = 128 * 128 * 1
    time_per_step = (elapsed / n_iterations) * voxels
    print(f"\nFor {voxels:,} voxels:")
    print(f"  Time per step: {time_per_step:.4f} s")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
