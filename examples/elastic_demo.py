import numpy as np
import matplotlib.pyplot as plt
from mgkmc.solver import run_mixed_simulation

def main():
    # 1. Geometry and Material Properties
    nx, ny, nz = 32, 32, 32
    # Heterogeneous random elastic modulus (average E = 50 GPa, nu = 0.3)
    E = np.random.normal(loc=50e9, scale=5e9, size=(nx, ny, nz))
    nu = np.full((nx, ny, nz), 0.3)
    
    # 2. Define Mixed Boundary Conditions
    # We want to pull in XX direction to 5% strain (0.05)
    # We want YY and ZZ directions to be stress-free (0.0 Pa)
    
    target_strain_mask = np.zeros((3, 3), dtype=bool)
    # Prescribe Strain for XX component
    target_strain_mask[0, 0] = True 
    
    target_values = np.zeros((3, 3))
    # Target value for Strain XX is 5% (0.05)
    target_values[0, 0] = 0.05
    # Target values for stress components (YY, ZZ) are 0.0 (already zeros)
    
    # 3. Exeucte the Purely Elastic Simulation
    print("Starting purely elastic simulation (5% Tension)...")
    eps_macro, sig_macro, eps_fields, sig_fields = run_mixed_simulation(
        E=E, 
        nu=nu,
        target_strain_mask=target_strain_mask,
        target_values=target_values,
        n_steps=20,          # Reach 5% strain in 20 steps
        pixel=1.0,           # Voxel size
        tol_macro=1e-4 * 1e6,# Mixed BC stress tolerance (1e-4 MPa = 100 Pa)
        store=False          # Just store macro values, omit full fields to save memory
    )
    
    # 4. Plot Linear Elastic Response
    strain_xx = [eps[0,0] * 100 for eps in eps_macro] # %
    stress_xx = [sig[0,0] / 1e9 for sig in sig_macro] # GPa
    
    plt.figure(figsize=(8,5))
    plt.plot(strain_xx, stress_xx, 'b-o', label="Heterogeneous Material")
    
    # Plot expected homogeneous line: sigma = E * eps
    plt.plot(strain_xx, np.array(strain_xx)/100 * 50, 'r--', label="Ideal Homogeneous (E=50GPa)")
    
    plt.xlabel('Strain XX (%)')
    plt.ylabel('Stress XX (GPa)')
    plt.title('Purely Linear Elasticity (No KMC/STZ)')
    plt.legend()
    plt.grid(True)
    plt.savefig("elastic_tension.png")
    print("Saved plot to elastic_tension.png")

if __name__ == "__main__":
    main()
