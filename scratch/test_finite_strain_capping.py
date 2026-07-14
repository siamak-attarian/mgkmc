import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path so we can import mgkmc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgkmc.finite_strain_simulator import (
    _make_identity_tensors_2d, build_ghat4_2d, build_C4_2d,
    finite_strain_solver_step_2d
)

def run_test():
    print("--- Running Finite Strain Capping Incremental Solver Verification ---")
    nx, ny = 4, 4
    pixel = 1.0
    
    # Material constants
    lam = 80.91e9
    mu = 23.76e9
    E_val = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu_val = lam / (2 * (lam + mu))
    
    E = np.full((nx, ny), E_val)
    nu = np.full((nx, ny), nu_val)
    
    # Initialize finite strain solver inputs
    I2_fs, I4_fs, I4rt_fs, I4s_fs, II_fs = _make_identity_tensors_2d(nx, ny)
    Lx, Ly = nx * pixel, ny * pixel
    Ghat4_fs = build_ghat4_2d(nx, ny, Lx, Ly)
    C4_fs = build_C4_2d(E, nu, I4s_fs, II_fs, plane_mode="plane_stress")
    
    # Define a localized plastic shear strain concentration (mimicking an STZ flip)
    Fp = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))
    Fp[0, 1, 2, 2] = -0.15  # Apply negative plastic shear (giving positive local elastic shear concentration)
    
    # Define loading steps up to 40% shear strain
    n_steps = 80
    shear_targets = np.linspace(0.0, 0.40, n_steps + 1)
    
    # 1. Run uncapped incremental simulation
    print("\nRunning uncapped incremental simulation...")
    F_nocap = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))
    eps_nocap, sig_nocap = [], []
    uncapped_failed = False
    
    for s in range(n_steps + 1):
        gamma = shear_targets[s]
        F_bar = np.array([[1.0, gamma],
                          [0.0, 1.0]])
        try:
            F_nocap, P_out, Sig_out, K4_out, F_bar_updated = finite_strain_solver_step_2d(
                F_nocap, F_bar, Ghat4_fs, C4_fs, I2_fs, I4_fs, I4rt_fs, Fp=Fp,
                driving_component=(0, 1), P_target=None, P_mask=None,
                E_avg=E.mean(), nu_avg=nu.mean(),
                tol_NW=1e-5, tol_CG=1e-6, max_NW=100,
                enable_console=False, model_type="landau", plane_mode="plane_stress",
                solver="dbfft",
                v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
                g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9,
                strain_capping_enabled=False
            )
            eps_nocap.append(gamma)
            sig_nocap.append(Sig_out.mean(axis=(2, 3))[0, 1] / 1e9)
        except ValueError as e:
            print(f"Uncapped solver diverged at strain {gamma*100:.1f}%: {e}")
            uncapped_failed = True
            break

    # 2. Run capped incremental simulation
    print("\nRunning capped incremental simulation...")
    F_cap = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))
    eps_cap, sig_cap = [], []
    capped_failed = False
    
    for s in range(n_steps + 1):
        gamma = shear_targets[s]
        F_bar = np.array([[1.0, gamma],
                          [0.0, 1.0]])
        try:
            F_cap, P_out, Sig_out, K4_out, F_bar_updated = finite_strain_solver_step_2d(
                F_cap, F_bar, Ghat4_fs, C4_fs, I2_fs, I4_fs, I4rt_fs, Fp=Fp,
                driving_component=(0, 1), P_target=None, P_mask=None,
                E_avg=E.mean(), nu_avg=nu.mean(),
                tol_NW=1e-5, tol_CG=1e-6, max_NW=100,
                enable_console=False, model_type="landau", plane_mode="plane_stress",
                solver="dbfft",
                v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
                g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9,
                strain_capping_enabled=True,
                strain_capping_limit=0.06,
                strain_capping_tangent_ratio=0.05
            )
            eps_cap.append(gamma)
            sig_cap.append(Sig_out.mean(axis=(2, 3))[0, 1] / 1e9)
        except ValueError as e:
            print(f"Capped solver diverged at strain {gamma*100:.1f}%: {e}")
            capped_failed = True
            break

    # Plot results
    plt.figure(figsize=(7, 5))
    plt.plot(np.array(eps_nocap) * 100, sig_nocap, 'r--', label='Without Capping (Softening Collapse)')
    plt.plot(np.array(eps_cap) * 100, sig_cap, 'b-', label='With Capping (limit=0.06, tangent_ratio=0.05)')
    plt.axvline(x=6.0, color='gray', linestyle=':', label='Capping Limit (6% strain)')
    plt.xlabel('Shear Strain xy (%)')
    plt.ylabel('Shear Cauchy Stress xy (GPa)')
    plt.title('Finite Strain Landau: Regularization via Strain Capping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(__file__), 'finite_strain_capping_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nVerification plot saved to: {plot_path}")
    
    # Assertions
    print(f"Uncapped simulation converged up to: {eps_nocap[-1]*100:.1f}% strain")
    print(f"Capped simulation converged up to: {eps_cap[-1]*100:.1f}% strain")
    
    assert uncapped_failed, "Uncapped simulation should have failed/diverged due to negative tangent stiffness."
    assert eps_cap[-1] > eps_nocap[-1], f"Capped simulation should survive to a higher strain ({eps_cap[-1]*100:.1f}%) than the uncapped simulation ({eps_nocap[-1]*100:.1f}%)."
    assert sig_cap[-1] > 0.0, "Capped stress should remain positive."
    print("\nALL TESTS PASSED SUCCESSFULLY! Capping regularization makes finite strain Landau solver extremely robust under large plastic flips.")

if __name__ == "__main__":
    run_test()
