import numpy as np
import sys
import os

# Ensure we can import mgkmc
sys.path.append(os.getcwd())

from mgkmc.stz.barriers import compute_barrier

def test_barrier_work_calculation():
    print("--- Testing Barrier Work Calculation ---")
    
    # 1. Setup
    nx, ny, nz, M = 1, 1, 1, 1
    Q = np.zeros((nx, ny, nz, M))
    Q0 = np.full((nx, ny, nz, M), 1.0)
    sig_field = np.zeros((nx, ny, nz, 3, 3))
    sig_field[0,0,0,0,1] = 1e9 # 1 GPa
    sig_field[0,0,0,1,0] = 1e9
    
    catalog = np.zeros((nx, ny, nz, M, 3, 3))
    catalog[0,0,0,0,0,1] = 0.1
    catalog[0,0,0,0,1,0] = 0.1
    
    volume = 1.0 # nm^3
    soft_prop = np.zeros((nx, ny, nz, 4))
    last_event_time = np.full((nx, ny, nz), -np.inf)
    time = 0.0
    prev_strain_dir = np.zeros((nx, ny, nz, 3, 3))
    softening_cap = 2.0
    scheme = 0
    tau = np.inf
    
    # Run code
    compute_barrier(Q, Q0, sig_field, catalog, volume,
                    soft_prop, last_event_time, time, 
                    prev_strain_dir, softening_cap,
                    scheme, tau)
    
    print(f"\nExpected Work: ~0.62415 eV")
    print(f"Computed Q[0]: {Q[0,0,0,0]:.6f} eV")
    
    if abs(Q[0,0,0,0] - (1.0 - 0.6241509)) < 1e-4:
        print("PASS: Barrier calculation matches manual expectation.")
    else:
        print(f"FAIL: Mismatch. Expected {1.0 - 0.6241509:.6f}, got {Q[-1]:.6f}")
        assert False

def test_solver_units():
    print("\n--- Testing Spectral Solver Units ---")
    from mgkmc.linear_elastic_simulator import spectral_solver_3d
    
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
