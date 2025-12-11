import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mgkmc import AthermalSimulation

def run_sim(label, softening_enabled, softening_params=None, color='b'):
    print(f"\n[Run] {label}...")
    
    nx, ny, nz = 16, 16, 1
    # Use consistent seed for barriers/modes so differences are purely due to softening
    np.random.seed(42)
    
    # Setup
    M = 20
    gamma0 = 0.05
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)
    
    # Fixed barriers (normal dist)
    def my_barrier_generator(n_modes):
        np.random.seed(None) # rely on global seed? No, allow randomness but same sequence if reset?
        # Actually AQS re-calls this.
        # To strictly compare, we want valid initial state similarity.
        # But divergence happens quickly.
        # Let's just use standard generation.
        return np.clip(np.random.normal(2.0, 0.6, size=n_modes), 0.5, None)

    output_dir = f"output_{label}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    sim = AthermalSimulation(
        nx, ny, nz, M, gamma0, E, nu,
        output_dir=output_dir,
        softening_enabled=softening_enabled,
        softening_params=softening_params,
        debug_first_flip=False
    )
    
    # Run small strain
    steps = 200
    strain_inc = np.zeros((3,3))
    strain_inc[0,0] = 2e-4 # Larger steps to trigger stuff fast
    
    sim.run(steps, strain_inc, vtk_mode=None)
    
    return np.array(sim.history_global), color

def main():
    results = []
    
    # 1. No Softening
    hist1, c1 = run_sim("No_Softening", False, color='k')
    results.append(("No Softening", hist1, c1))
    
    # 2. Standard Softening (jp=10, jt=30)
    hist2, c2 = run_sim("Standard_Softening", True, {"jp": 10, "jt": 30}, color='b')
    results.append(("Standard (jp=10)", hist2, c2))
    
    # 3. Strong Softening (jp=30, jt=50)
    hist3, c3 = run_sim("Strong_Softening", True, {"jp": 30, "jt": 50}, color='r')
    results.append(("Strong (jp=30)", hist3, c3))
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    for label, hist, color in results:
        strain_pct = hist[:,0] * 100
        stress = hist[:,1]
        plt.plot(strain_pct, stress, color=color, label=label, linewidth=1.5)
        
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (GPa)")
    plt.title("Effect of Softening Parameters on AQS Response")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("softening_comparison.png")
    print("\nComparison plot saved to softening_comparison.png")

if __name__ == "__main__":
    main()
