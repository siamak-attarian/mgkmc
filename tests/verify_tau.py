import numpy as np
from mgkmc import ThermalSimulation

def test_tau_calculation():
    # Setup dummy data
    nx, ny, nz = 2, 2, 1
    E = np.ones((nx, ny, nz))
    nu = np.zeros((nx, ny, nz))
    
    # CASE 1: T=0 (Should be inf)
    sim0 = ThermalSimulation(nx, ny, nz, M=2, gamma0=0.1, E_field=E, nu_field=nu, temperature=0.0)
    print(f"Testing T=0: tau = {sim0.tau}")
    assert sim0.tau == np.inf
    
    # CASE 2: T=300K, q_act_temp=0.37
    T = 300.0
    q = 0.37
    nu0 = 1e13
    kB = 8.617e-5
    expected_tau = 1.0 / (nu0 * np.exp(-q / (kB * T)))
    
    sim300 = ThermalSimulation(nx, ny, nz, M=2, gamma0=0.1, E_field=E, nu_field=nu, temperature=T, q_act_temp=q, nu0=nu0)
    print(f"Testing T=300: Calculated tau = {sim300.tau:.4e}, Expected = {expected_tau:.4e}")
    assert np.isclose(sim300.tau, expected_tau)
    
    print("\nSUCCESS: tau calculation is correct.")

if __name__ == "__main__":
    test_tau_calculation()
