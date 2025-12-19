import numpy as np
import sys
import os

# Ensure we can import mgkmc
sys.path.append(os.getcwd())

from mgkmc.solver import spectral_solver_3d

def test_plastic_relaxation():
    print("\n--- Testing Plastic Relaxation ---")
    
    nx, ny, nz = 16, 16, 1
    pixel = 1.0
    
    E = np.full((nx, ny, nz), 70e9)
    nu = np.full((nx, ny, nz), 0.3)
    
    # 1. Total Strain imposed (Constrained, effectively)
    eps_macro = np.zeros((3,3)) # Zero macro strain for simplicity?
    # No, let's allow it to evolve or fix it?
    # Solver enforces eps_mean = eps_bar. 
    # Let's fix eps_bar = 0.
    
    # 2. Add plastic strain to one voxel
    eps_p = np.zeros((nx, ny, nz, 3, 3))
    
    # Pure shear plastic strain
    gamma = np.zeros((3,3))
    gamma[0,1] = 0.05
    gamma[1,0] = 0.05
    
    eps_p[8,8,0] = gamma
    
    # 3. Solve
    eps, sig, epsM, sigM = spectral_solver_3d(
        E, nu, eps_macro, eps_plastic=eps_p,
        pixel=pixel, verbose=False
    )
    
    # 4. Check Stress at that voxel
    # sigma = C : (eps - eps_p)
    # eps is determined by compatibility.
    # For a single inclusion in infinite medium, eps ~ beta * eps_p (where beta ~ 0.5)
    # So eps - eps_p ~ (beta - 1) * eps_p ~ -0.5 * eps_p
    # Sigma should be OPPOSITE to eps_p
    
    sigma_vox = sig[8,8,0]
    eps_vox = eps[8,8,0]
    
    print(f"Applied Eps_Plastic (xy): {gamma[0,1]}")
    print(f"Result Eps (xy): {eps_vox[0,1]:.6f}")
    print(f"Result Sigma (xy): {sigma_vox[0,1]/1e9:.4f} GPa")
    
    # Check sign
    # We applied POSITIVE plastic shear.
    # We expect NEGATIVE stress (residual stress trying to push it back).
    
    if sigma_vox[0,1] < 0:
        print("PASS: Stress opposes plastic strain (Relaxation).")
    else:
        print("FAIL: Stress aligns with plastic strain (Hardening/Wrong Sign).")

if __name__ == "__main__":
    test_plastic_relaxation()
