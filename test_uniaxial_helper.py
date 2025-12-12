"""
Quick test to verify get_uniaxial_stress_x is being used correctly in AQS
"""
import numpy as np
from mgkmc.elasticity_helpers import get_uniaxial_stress_x

# Test parameters
eps_xx = 0.001
E_avg = 70.0  # GPa
nu_avg = 0.3

# Call the function
eps_tensor = get_uniaxial_stress_x(eps_xx, E_avg, nu_avg)

print("Testing get_uniaxial_stress_x:")
print(f"Input: eps_xx = {eps_xx}, E = {E_avg} GPa, nu = {nu_avg}")
print(f"\nOutput strain tensor:")
print(eps_tensor)
print(f"\nExpected:")
print(f"  eps_xx = {eps_xx}")
print(f"  eps_yy = {-nu_avg * eps_xx}")
print(f"  eps_zz = {-nu_avg * eps_xx}")
print(f"\nActual:")
print(f"  eps_xx = {eps_tensor[0,0]}")
print(f"  eps_yy = {eps_tensor[1,1]}")
print(f"  eps_zz = {eps_tensor[2,2]}")

# Verify it matches
assert np.isclose(eps_tensor[0,0], eps_xx), "eps_xx mismatch!"
assert np.isclose(eps_tensor[1,1], -nu_avg * eps_xx), "eps_yy mismatch!"
assert np.isclose(eps_tensor[2,2], -nu_avg * eps_xx), "eps_zz mismatch!"

print("\n✓ Test passed! get_uniaxial_stress_x is working correctly.")
