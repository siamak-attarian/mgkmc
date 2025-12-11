import numpy as np
from .voxel import Voxel
from .catalog import stz_catalog_glass

def initialize_grid(Nx, Ny, Nz, M, gamma0, barrier_generator=None, mode_generator=stz_catalog_glass):
    """
    Allocate grid[x,y,z] and assign voxel objects with initial catalog.
    """
    grid = np.empty((Nx, Ny, Nz), dtype=object)

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                v = Voxel(M, barrier_generator=barrier_generator)
                v.set_catalog(mode_generator(M, gamma0))
                grid[x,y,z] = v

    return grid
