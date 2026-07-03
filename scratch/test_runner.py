import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_landau_small_strain import (
    test_landau_2d_plane_strain,
    test_landau_2d_plane_stress,
    test_landau_3d,
    test_landau_simulation_2d,
)

if __name__ == '__main__':
    print("Running test_landau_2d_plane_strain...")
    test_landau_2d_plane_strain()
    print("-> PASS")

    print("Running test_landau_2d_plane_stress...")
    test_landau_2d_plane_stress()
    print("-> PASS")

    print("Running test_landau_3d...")
    test_landau_3d()
    print("-> PASS")

    print("Running test_landau_simulation_2d...")
    test_landau_simulation_2d()
    print("-> PASS")

    print("All tests passed successfully!")
