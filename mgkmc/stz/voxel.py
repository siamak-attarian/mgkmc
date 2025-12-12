import numpy as np

class Voxel:
    """
    Represents a single voxel in the STZ model.
    Stores plastic strain, total strain, stress, and catalog modes.
    """

    def __init__(self, M, barrier_generator=None):
        self.M = M
        
        # initial random activation energies
        if barrier_generator is None:
             # Default fallback if not specified: mean=1.0, std=0.25
             self.Q0 = np.random.normal(loc=1.0, scale=0.25, size=M)
        else:
             self.Q0 = barrier_generator(M)
        # strain tensors
        self.eps_plastic = np.zeros((3,3))
        self.eps_total   = np.zeros((3,3))
        self.sigma       = np.zeros((3,3))

        # softening
        self.g_p = 0.0
        self.g_t = 0.0
        self.prev_gamma = None # for directional softening

        # event catalog (stored as numpy array for efficient vectorization)
        # Shape: (M, 3, 3) - M modes, each is a 3x3 strain tensor
        self.catalog = np.zeros((M, 3, 3))

        # bookkeeping
        self.flip_count_total = 0
        self.last_flip_global = -1
        self.last_flip_local  = -1

    def set_catalog(self, catalog):
        """Replace catalog modes."""
        # Convert to numpy array if it's a list
        if isinstance(catalog, list):
            self.catalog = np.array(catalog)
        else:
            self.catalog = catalog

    def reset_barriers(self, barrier_generator=None):
        if barrier_generator is None:
             self.Q0 = np.random.normal(loc=1.0, scale=0.25, size=self.M)
        else:
             self.Q0 = barrier_generator(self.M)
