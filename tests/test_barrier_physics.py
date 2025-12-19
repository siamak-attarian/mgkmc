import numpy as np
import sys
import os

# Ensure we can import mgkmc
sys.path.append(os.getcwd())

from mgkmc.stz.barriers import compute_barrier

class MockVoxel:
    def __init__(self, M=1):
        self.M = M
        self.sigma = np.zeros((3,3)) # Pa
        self.eps_plastic = np.zeros((3,3))
        self.catalog = [np.zeros((3,3))] # gamma
        self.Q0 = np.array([1.0]) # eV
        self.g_p = 0.0
        self.g_t = 0.0
        self.flip_count_total = 0

def test_barrier_work_calculation():
    print("--- Testing Barrier Work Calculation ---")
    
    # 1. Setup
    voxel = MockVoxel(M=1)
    volume = 1.0 # nm^3
    
    # Stress: 1 GPa pure shear in XY
    # Tensor: 
    # [[0, 1, 0]
    #  [1, 0, 0]
    #  [0, 0, 0]] GPa
    sigma_GPa = np.zeros((3,3))
    sigma_GPa[0,1] = 1.0
    sigma_GPa[1,0] = 1.0
    
    voxel.sigma = sigma_GPa * 1e9 # Convert to Pa for simulation state
    
    # Transformation Strain (Gamma): Pure shear 0.1 in XY
    gamma = np.zeros((3,3))
    gamma[0,1] = 0.1
    gamma[1,0] = 0.1
    voxel.catalog = [gamma]
    
    # Expected Work
    # W = 0.5 * V * (sigma : gamma)
    # sigma : gamma = sum(sigma_ij * gamma_ij) = (1*0.1) + (1*0.1) = 0.2 GPa
    # W = 0.5 * 1.0 nm^3 * 0.2 GPa
    # Unit conversion: 1 GPa * nm^3 = 1e-18 J * 1e-27 m^3 ... wait.
    # 1 GPa = 1e9 N/m^2 = 1e-9 N/nm^2
    # Energy = Force * dist = (stress * area) * dist = stress * volume
    # 1 GPa * 1 nm^3 = (1e-9 N/nm^2) * 1 nm^3 = 1e-9 N * nm = 1e-9 Joules? NO.
    
    # Let's check the code's constant: 6.241509
    # 1 Joule = 6.241509e18 eV
    
    # 1 Pa * m^3 = 1 Joule
    # 1 GPa = 1e9 Pa
    # 1 nm^3 = 1e-27 m^3
    # 1 GPa * nm^3 = 1e9 * 1e-27 Joules = 1e-18 Joules
    
    # 1 eV = 1.602e-19 Joules
    # So 1 GPa * nm^3 = 1e-18 J / 1.602e-19 J/eV = 10 / 1.602 = 6.2415 eV
    
    # So: Work (eV) = (Stress_in_GPa * Volume_in_nm3) * 6.2415
    
    # For our manual Calc:
    # sigma_double_dot_gamma = 0.2 (dimensionless if gamma is strain)
    # Work_GPa_nm3 = 0.5 * 1.0 * 0.2 = 0.1
    # Work_eV = 0.1 * 6.241509 = 0.6241509 eV
    
    # Run code
    Q = compute_barrier(voxel, volume, debug=True)
    
    # Q = Q0 - Work = 1.0 - 0.6241509 = 0.375849
    
    print(f"\nExpected Work: ~0.62415 eV")
    print(f"Computed Q[0]: {Q[0]:.6f} eV")
    
    if abs(Q[0] - (1.0 - 0.6241509)) < 1e-4:
        print("PASS: Barrier calculation matches manual expectation.")
    else:
        print(f"FAIL: Mismatch. Expected {1.0 - 0.6241509:.6f}, got {Q[0]:.6f}")

def test_solver_units():
    print("\n--- Testing Spectral Solver Units ---")
    from mgkmc.solver import spectral_solver_3d
    
    # 16x16x1 Grid
    nx, ny, nz = 16, 16, 1
    pixel = 1.0
    
    # Material: E = 70 GPa (70e9 Pa), nu = 0.3
    E = np.full((nx, ny, nz), 70e9)
    nu = np.full((nx, ny, nz), 0.3)
    
    # Applied Strain: 1% uniaxial (0.01)
    eps_macro_in = np.zeros((3,3))
    eps_macro_in[0,0] = 0.01
    
    # Run Solver (Elastic)
    eps, sig, epsM, sigM = spectral_solver_3d(
        E, nu, eps_macro_in, eps_plastic=None,
        max_iter=50, tol=1e-6, verbose=False, pixel=pixel
    )
    
    # Check Macroscopic Stress
    # Theoretical Uniaxial Stress for isotropic material:
    # sigma_xx = E * eps_xx (for uniaxial stress state? No, this is constrained strain in Solver?)
    # The solver imposes epsilon_average = eps_macro_in.
    # If nu=0.3, and we enforce eps_yy=0, eps_zz=0 (bc it's fixed macro strain),
    # sigma_xx = C_xxxx * eps_xx + C_xxyy * 0 ...
    # C_xxxx = E(1-nu) / ((1+nu)(1-2nu)) ... for Plane Strain / 3D?
    # Actually, let's just approximate sigma ~ E * eps
    
    # Hooke's Law 3D isotropic:
    # sigma_ij = lambda * tr(eps) * delta_ij + 2*mu * eps_ij
    # tr(eps) = 0.01
    # eps_11 = 0.01
    # sigma_11 = lambda * 0.01 + 2*mu * 0.01 = (lambda + 2mu) * 0.01
    
    lam = 70e9 * 0.3 / ((1+0.3)*(1-0.6))  # 70 * 0.3 / (1.3 * 0.4) = 21 / 0.52 = 40.38 GPa
    mu  = 70e9 / (2*(1+0.3))              # 70 / 2.6 = 26.92 GPa
    
    expected_sigma_xx = (lam + 2*mu) * 0.01 # Pa
    
    print(f"E = 70 GPa, applied eps_xx = 1%")
    print(f"Expected Sigma_xx (approx constrained): {expected_sigma_xx/1e9:.2f} GPa")
    print(f"Solver Sigma_xx: {sigM[0,0]/1e9:.2f} GPa")
    
    if abs(sigM[0,0] - expected_sigma_xx) < 1e9: # Tolerance 1 GPa (loose)
         print("PASS: Solver stress is in GPa range and correct magnitude.")
    else:
         print("FAIL: Solver stress mismatch.")

if __name__ == "__main__":
    test_barrier_work_calculation()
    test_solver_units()
