from .microstructure import generate_correlated_field

from .elasticity import (
    get_strain_tensor,
    get_plane_stress_z_fixed_y,
    get_uniaxial_stress_x,
    get_pure_shear_xy,
)

from .analysis import plot_fields

from .solver import run_simulation, run_mixed_simulation, spectral_solver_3d
from .analysis import export_simulation_vtk
from .aqs import ThermalSimulation

# Checkpoint and post-processing tools
from . import checkpoint
from . import analysis

