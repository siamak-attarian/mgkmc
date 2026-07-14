import numpy as np
import os
import sys

# Add parent directory to path so we can import mgkmc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mgkmc.finite_strain_simulator import finite_strain_simulation_2d

def run_debug():
    nx, ny = 4, 4
    
    # Material constants
    lam = 80.91e9
    mu = 23.76e9
    E_val = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu_val = lam / (2 * (lam + mu))
    
    E = np.full((nx, ny), E_val)
    nu = np.full((nx, ny), nu_val)
    
    print("\nRunning without capping (20% strain)...")
    F_mac_nocap, Sig_mac_nocap, P_mac_nocap, F_nocap, Sig_nocap = finite_strain_simulation_2d(
        E=E, nu=nu,
        driving_component=(0, 0),
        eps_target=0.20,
        n_steps=20,
        mixed_targets={(1, 1): 0.0},
        plane_mode="plane_stress",
        model_type="landau",
        solver="dbfft",
        v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
        g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9,
        strain_capping_enabled=False,
        max_iter_macro=100
    )
    
    print("\nRunning with capping (20% strain, cap=0.15)...")
    F_mac_cap, Sig_mac_cap, P_mac_cap, F_cap, Sig_cap = finite_strain_simulation_2d(
        E=E, nu=nu,
        driving_component=(0, 0),
        eps_target=0.20,
        n_steps=20,
        mixed_targets={(1, 1): 0.0},
        plane_mode="plane_stress",
        model_type="landau",
        solver="dbfft",
        v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
        g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9,
        strain_capping_enabled=True,
        strain_capping_limit=0.15,
        strain_capping_tangent_ratio=0.05,
        strain_capping_type="piecewise",
        max_iter_macro=100
    )

    print("\nRunning with AUTOMATIC PIECEWISE capping (20% strain)...")
    F_mac_auto_pw, Sig_mac_auto_pw, _, _, _ = finite_strain_simulation_2d(
        E=E, nu=nu,
        driving_component=(0, 0),
        eps_target=0.20,
        n_steps=20,
        mixed_targets={(1, 1): 0.0},
        plane_mode="plane_stress",
        model_type="landau",
        solver="dbfft",
        v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
        g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9,
        strain_capping_enabled=True,
        strain_capping_limit=None,
        strain_capping_type="piecewise",
        max_iter_macro=100
    )

    print("\nRunning with AUTOMATIC SMOOTH capping (20% strain)...")
    F_mac_auto_sm, Sig_mac_auto_sm, _, _, _ = finite_strain_simulation_2d(
        E=E, nu=nu,
        driving_component=(0, 0),
        eps_target=0.20,
        n_steps=20,
        mixed_targets={(1, 1): 0.0},
        plane_mode="plane_stress",
        model_type="landau",
        solver="dbfft",
        v1=-199.7e9, v2=-75.4e9, v3=-23.1e9,
        g1=1432.0e9, g2=311.0e9, g3=145.0e9, g4=-273.0e9,
        strain_capping_enabled=True,
        strain_capping_limit=None,
        strain_capping_type="smooth",
        max_iter_macro=100
    )
    
    print("\nComparing last step values:")
    print(f"F_xx (Uncapped):           {F_mac_nocap[-1][0, 0]:.6f}")
    print(f"F_xx (Hard Cap 0.15):      {F_mac_cap[-1][0, 0]:.6f}")
    print(f"F_xx (Auto Piecewise):     {F_mac_auto_pw[-1][0, 0]:.6f}")
    print(f"F_xx (Auto Smooth):        {F_mac_auto_sm[-1][0, 0]:.6f}")
    print(f"Sig_xx (Uncapped):         {Sig_mac_nocap[-1][0, 0]/1e9:.6f} GPa")
    print(f"Sig_xx (Hard Cap 0.15):    {Sig_mac_cap[-1][0, 0]/1e9:.6f} GPa")
    print(f"Sig_xx (Auto Piecewise):   {Sig_mac_auto_pw[-1][0, 0]/1e9:.6f} GPa")
    print(f"Sig_xx (Auto Smooth):      {Sig_mac_auto_sm[-1][0, 0]/1e9:.6f} GPa")
    
    # Let's inspect equivalent strain at the final state
    F_final = F_cap[-1].copy()
    Ce = np.einsum('jixy,jkxy->ikxy', F_final, F_final)
    I2 = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))
    E_GL_2d = 0.5 * (Ce - I2)
    
    # Local E33 (plane stress)
    trE_2d = E_GL_2d[0, 0] + E_GL_2d[1, 1]
    # We solve for E33 locally
    v1_arr = -199.7e9
    v2_arr = -75.4e9
    v3_arr = -23.1e9
    g1_arr = 1432.0e9
    g2_arr = 311.0e9
    g3_arr = 145.0e9
    g4_arr = -273.0e9
    
    E33 = - (lam / (lam + 2.0 * mu)) * trE_2d
    for iteration in range(20):
        I1 = trE_2d + E33
        I2_inv = E_GL_2d[0,0]**2 + E_GL_2d[1,1]**2 + 2.0*E_GL_2d[0,1]*E_GL_2d[1,0] + E33**2
        I3_inv = E_GL_2d[0,0]**3 + E_GL_2d[1,1]**3 + E33**3 # approximate for diagonal
        A_coef = lam * I1 + 0.5 * v1_arr * (I1**2) + v2_arr * I2_inv + (1.0/6.0) * g1_arr * (I1**3) + g2_arr * I1 * I2_inv + (4.0/3.0) * g3_arr * I3_inv
        B_coef = 2.0 * (mu + v2_arr * I1 + 0.5 * g2_arr * (I1**2) + g4_arr * I2_inv)
        C_coef = 4.0 * (v3_arr + g3_arr * I1)
        S33 = A_coef + B_coef * E33 + C_coef * (E33**2)
        dAdE33 = lam + v1_arr * I1 + 0.5 * g1_arr * (I1**2) + g2_arr * I2_inv + 2.0 * (v2_arr + g2_arr * I1) * E33 + 4.0 * g3_arr * (E33**2)
        dBdE33 = 2.0 * (v2_arr + g2_arr * I1) + 4.0 * g4_arr * E33
        dCdE33 = 4.0 * g3_arr
        C3333 = dAdE33 + dBdE33 * E33 + dCdE33 * (E33**2) + B_coef + 2.0 * C_coef * E33
        dE33 = - S33 / C3333
        E33 = E33 + dE33
        if np.max(np.abs(dE33)) < 1e-12:
            break
            
    E_GL = np.zeros((3, 3, nx, ny))
    E_GL[0:2, 0:2] = E_GL_2d
    E_GL[2, 2] = E33
    
    tr_orig = E_GL[0, 0] + E_GL[1, 1] + E_GL[2, 2]
    E_dev = E_GL.copy()
    tr_third = tr_orig / 3.0
    for i in range(3):
        E_dev[i, i] -= tr_third
    E_dev_double_dot = np.zeros_like(tr_orig)
    for i in range(3):
        for j in range(3):
            E_dev_double_dot += E_dev[i, j] * E_dev[j, i]
    E_eq = np.sqrt(np.maximum(0.0, (2.0 / 3.0) * E_dev_double_dot))
    print(f"\nFinal State equivalent strain E_eq: {E_eq.mean():.6f}")

if __name__ == "__main__":
    run_debug()
