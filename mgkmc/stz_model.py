import numpy as np

# ------------------------------------------------------------
# Utility: Generate a random traceless SFTS tensor (3×3)
# Based on Appendix B generalization to 3D
# ------------------------------------------------------------

def random_sfts(cstar=0.1):
    """
    Generate a random 3x3 traceless symmetric tensor
    using Gaussian sampling for invariants.
    """
    # independent Gaussian components
    a = np.random.normal(scale=cstar/2)
    b = np.random.normal(scale=cstar/2)
    c = np.random.normal(scale=cstar/2)

    # symmetric traceless matrix in a random basis
    # Start with diagonal traceless form
    M = np.array([
        [ a,      0,      0 ],
        [ 0,      b,      0 ],
        [ 0,      0, -(a+b)]
    ])

    # random rotation in 3D
    R = random_rotation_matrix()
    return R @ M @ R.T

def random_sfts_xy(cstar=0.1):
    """
    Generate a random 3x3 traceless symmetric tensor that
    is restricted to the XY plane. No XZ or YZ shear components.
    """
    # Step 1: independent Gaussian deviatoric components in 2D
    a = np.random.normal(scale=cstar/2)
    b = -a  # enforce trace = 0 in the xy block (2D)
    shear = np.random.normal(scale=cstar/2)

    # Base 2D deviatoric tensor BEFORE rotation
    E2 = np.array([
        [a,        shear],
        [shear,    b]
    ])

    # Step 2: random rotation in the XY plane
    theta = np.random.uniform(0, 2*np.pi)
    R2 = np.array([
        [ np.cos(theta), -np.sin(theta) ],
        [ np.sin(theta),  np.cos(theta) ]
    ])

    E2_rot = R2 @ E2 @ R2.T

    # Step 3: embed in 3D
    G = np.zeros((3,3))
    G[0:2, 0:2] = E2_rot

    return G

def random_rotation_matrix():
    """
    Generate a random 3D rotation matrix uniformly on SO(3).
    """
    q = np.random.normal(size=4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1-2*(x*x+z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),   2*(y*z + w*x),   1-2*(x*x+y*y)]
    ])


# ------------------------------------------------------------
# Voxel class: store all state variables
# ------------------------------------------------------------

class Voxel:
    def __init__(self, M):
        # total strain tensor (elastic + plastic)
        self.eps_total = np.zeros((3,3))

        # accumulated plastic (STZ) strain
        self.eps_plastic = np.zeros((3,3))

        # stress tensor
        self.sigma = np.zeros((3,3))

        # softening parameters
        self.g_p = 0.0   # permanent
        self.g_t = 0.0   # temporary (not relaxed until thermal part)

        # time since last event (not used in athermal regime)
        self.t_last = 0.0  

        # event catalog (M SFTS tensors)
        # self.catalog = [random_sfts() for _ in range(M)]
        self.catalog = [random_sfts_xy() for _ in range(M)]

        # bookkeeping
        self.flip_count_total = 0
        self.flip_count_this_step = 0

        self.last_flip_global = -1
        self.last_flip_local = -1

    # ----------------------------------------
    # regenerate catalog after a generation change
    # ----------------------------------------
    def new_generation(self, M):
        # self.catalog = [random_sfts() for _ in range(M)]
        self.catalog = [random_sfts_xy() for _ in range(M)]


# ------------------------------------------------------------
# Grid Initialization (C-order indexing: z, y, x)
# grid[z][y][x]
# ------------------------------------------------------------

def initialize_grid(Nx, Ny, Nz, M):
    grid = np.empty((Nz, Ny, Nx), dtype=object)

    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                grid[z,y,x] = Voxel(M)

    return grid


# ------------------------------------------------------------
# Compute Q(m) activation barriers for one voxel
# ------------------------------------------------------------

def compute_barriers(voxel, deltaF, volume):
    """
    Return array of Q[m] for each mode.
    Athermal regime: Q < 0 means unstable.
    """
    g = voxel.g_p + voxel.g_t  # thermal relaxation not included yet
    prefactor = deltaF * np.exp(g)

    Q = []
    for gamma in voxel.catalog:
        work = 0.5 * volume * np.sum(voxel.sigma * gamma)
        Q.append(prefactor - work)

    return np.array(Q)


# ------------------------------------------------------------
# Athermal Update of a Single Voxel
# ------------------------------------------------------------

def apply_athermal_event(voxel, mode_idx, deltaF, jp=10, jt=30):
    """
    Apply the selected STZ transformation mode.
    Update:
      - plastic strain
      - softening g_p, g_t
      - generation (new catalog)
    """
    gamma = voxel.catalog[mode_idx]

    # update plastic strain
    voxel.eps_plastic += gamma

    # softening update (Section 2.1.3)
    mises = von_mises_strain(gamma)
    voxel.g_p += jp * mises**2
    voxel.g_t  = jt * mises**2  # temporary softening

    # generation change → new catalog
    voxel.new_generation(len(voxel.catalog))

    # bookkeeping
    voxel.flip_count_total += 1
    voxel.flip_count_this_step += 1


def von_mises_strain(e):
    """Compute 3D von Mises equivalent strain of a 3×3 tensor."""
    s = e - np.trace(e)/3 * np.eye(3)
    return np.sqrt(3/2 * np.sum(s*s))


# ------------------------------------------------------------
# Placeholder: Update stress using the FFT solver
# ------------------------------------------------------------

def update_stress_fft(grid):
    """
    Replace this with your real FFT-based elastic solver.
    Must update voxel.sigma for every voxel.
    """
    # Example placeholder:
    pass


# ------------------------------------------------------------
# Find all unstable voxels (Q < 0)
# ------------------------------------------------------------

def find_unstable_voxels(grid, deltaF, volume):
    """
    Return list of tuples: (z, y, x, mode_idx, Q_value)
    """
    unstable = []

    Nz, Ny, Nx = grid.shape
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                voxel = grid[z,y,x]
                Q = compute_barriers(voxel, deltaF, volume)
                m = np.argmin(Q)
                if Q[m] < 0:
                    unstable.append((z, y, x, m, Q[m]))

    return unstable


# ------------------------------------------------------------
# Run one athermal avalanche (cascade)
# Returns: sequence of flipped voxels: [(z,y,x), ...]
# ------------------------------------------------------------

def run_athermal_cascade(grid, deltaF, volume, global_step, jp=10, jt=30):
    """
    Perform athermal avalanche:
        - Detect all voxels with Q < 0
        - Flip them simultaneously
        - Update stress
        - Repeat until no unstable voxels remain

    Returns list of voxel flips in order:
        cascade_sequence = [(local_step, z, y, x, mode)]
    """
    cascade_sequence = []
    local_step = 0

    while True:
        unstable = find_unstable_voxels(grid, deltaF, volume)

        if not unstable:
            break

        # record event order
        for (z,y,x,m,Q_val) in unstable:
            cascade_sequence.append((local_step, z, y, x, m))

        # apply updates
        for (z,y,x,m,Q_val) in unstable:
            voxel = grid[z,y,x]
            apply_athermal_event(voxel, m, deltaF, jp, jt)
            voxel.last_flip_global = global_step
            voxel.last_flip_local  = local_step

        # recompute stress after simultaneous update
        update_stress_fft(grid)

        local_step += 1

    return cascade_sequence


# ------------------------------------------------------------
# Global simulation loop
# ------------------------------------------------------------

def run_simulation(grid, deltaF, volume, nsteps):
    """
    Full simulation loop (athermal only).
    """
    all_cascades = []

    for global_step in range(nsteps):

        # Apply macroscopic strain increment (your code needed)
        # update_macro_strain(...)

        # Update stress field
        update_stress_fft(grid)

        # Run cascade
        cascade = run_athermal_cascade(grid, deltaF, volume,
                                       global_step=global_step)
        all_cascades.append(cascade)

        # Reset per-step counters
        for voxel in grid.flat:
            voxel.flip_count_this_step = 0

    return all_cascades
