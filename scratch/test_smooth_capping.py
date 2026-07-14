import numpy as np
import matplotlib.pyplot as plt

def landau_stress_1d(E_eq, lam, mu, v1, v2, v3, g1, g2, g3, g4):
    # Uniaxial loading along xx direction, let's assume a simple shear or uniaxial strain
    # For a simple shear strain E_xy = E_eq, all other components zero
    # S_xy = 2 * (mu + g4 * E_xy^2) * E_xy
    # Let's use this simple shear relation as a proxy:
    # S = 2 * (mu + 1.5 * g4 * E_eq^2) * E_eq
    # Let's use the actual parameters from the yaml file:
    # mu = 23.76e9, g4 = -273e9
    # S = 2 * mu * E_eq + 3 * g4 * E_eq^3
    return 2.0 * mu * E_eq + 3.0 * g4 * (E_eq**3)

def tangent_modulus_1d(E_eq, mu, g4):
    # G_tan = d(0.5 * S)/dE_eq = mu + 4.5 * g4 * E_eq^2
    return mu + 4.5 * g4 * (E_eq**2)

def main():
    mu = 23.76 # GPa
    g4 = -273.0 # GPa
    
    E_arr = np.linspace(0, 0.25, 500)
    
    # 1. Uncapped Landau
    S_uncapped = landau_stress_1d(E_arr, None, mu, None, None, None, None, None, None, g4)
    G_uncapped = tangent_modulus_1d(E_arr, mu, g4)
    
    # 2. Hard Capping (current implementation)
    E_cap = 0.12
    G_tangent_ratio = 0.05
    G_tangent = G_tangent_ratio * mu
    
    S_hard = []
    G_hard = []
    for E in E_arr:
        if E <= E_cap:
            S_val = landau_stress_1d(E, None, mu, None, None, None, None, None, None, g4)
            G_val = tangent_modulus_1d(E, mu, g4)
        else:
            S_cap = landau_stress_1d(E_cap, None, mu, None, None, None, None, None, None, g4)
            S_val = S_cap + 2.0 * G_tangent * (E - E_cap)
            G_val = G_tangent
        S_hard.append(S_val)
        G_hard.append(G_val)
    S_hard = np.array(S_hard)
    G_hard = np.array(G_hard)
    
    # 3. Smooth Capping using tanh
    # E_capped = E_cap * tanh(E / E_cap)
    # S = S_Landau(E_capped) + 2 * G_tangent * (E - E_capped)
    # If G_tangent = 0, S = S_Landau(E_capped)
    S_smooth = []
    G_smooth = []
    for E in E_arr:
        E_capped = E_cap * np.tanh(E / E_cap)
        S_val = landau_stress_1d(E_capped, None, mu, None, None, None, None, None, None, g4)
        # derivative: dS/dE = dS_Landau/dE_capped * dE_capped/dE
        # dS_Landau/dE_capped = 2 * G_tan(E_capped)
        # dE_capped/dE = sech^2(E/E_cap)
        G_val = tangent_modulus_1d(E_capped, mu, g4) * (1.0 / (np.cosh(E / E_cap)**2))
        S_smooth.append(S_val)
        G_smooth.append(G_val)
    S_smooth = np.array(S_smooth)
    G_smooth = np.array(G_smooth)
    
    # 4. Smooth Capping with residual tangent (G_tangent > 0)
    S_smooth_res = []
    G_smooth_res = []
    for E in E_arr:
        E_capped = E_cap * np.tanh(E / E_cap)
        S_val = landau_stress_1d(E_capped, None, mu, None, None, None, None, None, None, g4) + 2.0 * G_tangent * (E - E_capped)
        G_val = tangent_modulus_1d(E_capped, mu, g4) * (1.0 / (np.cosh(E / E_cap)**2)) + G_tangent * (1.0 - 1.0 / (np.cosh(E / E_cap)**2))
        S_smooth_res.append(S_val)
        G_smooth_res.append(G_val)
    S_smooth_res = np.array(S_smooth_res)
    G_smooth_res = np.array(G_smooth_res)
    
    # 5. C1 Continuous transition to Linear (at E_trans where G_tan = 0.1 * mu)
    eta = 0.1
    E_trans = np.sqrt((1.0 - eta) * mu / (4.5 * np.abs(g4)))
    S_trans = landau_stress_1d(E_trans, None, mu, None, None, None, None, None, None, g4)
    G_trans = tangent_modulus_1d(E_trans, mu, g4)
    
    S_c1_linear = []
    G_c1_linear = []
    for E in E_arr:
        if E <= E_trans:
            S_val = landau_stress_1d(E, None, mu, None, None, None, None, None, None, g4)
            G_val = tangent_modulus_1d(E, mu, g4)
        else:
            S_val = S_trans + 2.0 * G_trans * (E - E_trans)
            G_val = G_trans
        S_c1_linear.append(S_val)
        G_c1_linear.append(G_val)
    S_c1_linear = np.array(S_c1_linear)
    G_c1_linear = np.array(G_c1_linear)

    # 6. C1 Continuous transition to Hyperbolic Tangent (asymptotic)
    # For E > E_trans, S = S_trans + dS_max * tanh(G_trans * (E - E_trans) / dS_max)
    # where dS_max is the additional stress capacity.
    # To keep it simple, let's set dS_max = 0.5 GPa
    dS_max = 0.5 # GPa
    S_c1_tanh = []
    G_c1_tanh = []
    for E in E_arr:
        if E <= E_trans:
            S_val = landau_stress_1d(E, None, mu, None, None, None, None, None, None, g4)
            G_val = tangent_modulus_1d(E, mu, g4)
        else:
            S_val = S_trans + 2.0 * dS_max * np.tanh(G_trans * (E - E_trans) / dS_max)
            G_val = G_trans * (1.0 / (np.cosh(G_trans * (E - E_trans) / dS_max)**2))
        S_c1_tanh.append(S_val)
        G_c1_tanh.append(G_val)
    S_c1_tanh = np.array(S_c1_tanh)
    G_c1_tanh = np.array(G_c1_tanh)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stress-Strain plot
    ax1.plot(E_arr, S_uncapped, 'k--', label='Uncapped Landau (unstable)')
    ax1.plot(E_arr, S_hard, label=f'Hard Capping (cap={E_cap})')
    ax1.plot(E_arr, S_smooth, label=f'Smooth Capping (tanh, cap={E_cap})')
    ax1.plot(E_arr, S_smooth_res, label=f'Smooth Capping + Residual (cap={E_cap})')
    ax1.plot(E_arr, S_c1_linear, label=f'C1 Linear (trans at {E_trans:.3f})')
    ax1.plot(E_arr, S_c1_tanh, label=f'C1 Tanh (asymptotic, trans at {E_trans:.3f})')
    ax1.set_xlabel('Equivalent Strain E_eq')
    ax1.set_ylabel('Stress S_xy (GPa)')
    ax1.set_title('Stress-Strain Curve Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-2, 8)
    
    # Modulus-Strain plot
    ax2.plot(E_arr, G_uncapped, 'k--', label='Uncapped Landau')
    ax2.plot(E_arr, G_hard, label='Hard Capping')
    ax2.plot(E_arr, G_smooth, label='Smooth Capping')
    ax2.plot(E_arr, G_smooth_res, label='Smooth Capping + Residual')
    ax2.plot(E_arr, G_c1_linear, label='C1 Linear')
    ax2.plot(E_arr, G_c1_tanh, label='C1 Tanh')
    ax2.set_xlabel('Equivalent Strain E_eq')
    ax2.set_ylabel('Tangent Modulus G_tan (GPa)')
    ax2.set_title('Tangent Modulus vs Strain')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-10, 25)
    
    plt.tight_layout()
    plt.savefig('mgkmc/scratch/smooth_capping_comparison.png')
    print("Plot saved to mgkmc/scratch/smooth_capping_comparison.png")

if __name__ == "__main__":
    main()
