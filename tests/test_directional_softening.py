import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from mgkmc import ThermalSimulation

def run_sim(label, scheme, softening_params, color='b'):
    print(f"\n[Run] {label} (Scheme: {scheme})...")
    
    nx, ny, nz = 16, 16, 1
    # Use consistent seed for barriers/modes so differences are purely due to softening logic
    np.random.seed(42)
    
    # Setup
    M = 20
    gamma0 = 0.05
    E = np.full((nx, ny, nz), 70.0)
    nu = np.full((nx, ny, nz), 0.3)
    
    output_dir = f"output_{label}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    sim = ThermalSimulation(
        nx, ny, nz, M, gamma0, E, nu,
        output_dir=output_dir,
        softening_enabled=True,
        softening_params=softening_params,
        softening_scheme=scheme,
        debug_first_flip=False
    )
    
    # Run small strain
    steps = 200
    strain_inc = np.zeros((3,3))
    strain_inc[0,0] = 2e-4
    
    sim.run(steps, strain_inc, vtk_mode=None)
    
    return np.array(sim.history_global), color

def main():
    results = []
    
    # Base parameters: moderate softening
    params = {"jp": 30, "jt": 50} 
    
    # 1. Isotropic Softening
    hist1, c1 = run_sim("Isotropic", "isotropic", params, color='b')
    results.append(("Isotropic", hist1, c1))
    
    # 2. Directional Softening
    hist2, c2 = run_sim("Directional", "directional", params, color='r')
    results.append(("Directional", hist2, c2))
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    for label, hist, color in results:
        strain_pct = hist[:,0] * 100
        stress = hist[:,1]
        plt.plot(strain_pct, stress, color=color, label=label, linewidth=1.5)
        
    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (GPa)")
    plt.title(f"Softening Scheme Comparison (jp={params['jp']}, jt={params['jt']})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("directional_comparison.png")
    print("\nComparison plot saved to directional_comparison.png")

if __name__ == "__main__":
    main()
