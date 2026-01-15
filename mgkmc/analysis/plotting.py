import numpy as np

__all__ = ["plot_fields"]



def plot_fields(
    E: np.ndarray,
    eps: np.ndarray,
    sig: np.ndarray,
    fields: list[str] | None = None,
    slice_z: int | None = None,
    title: str = "",
) -> None:
    """Flexible plotting of field components.

    Default layout (if fields is None):
      Row 1: [E, nu]
      Row 2: [eps_xx, sig_xx]
      Row 3: [eps_yy, sig_yy]
      Row 4: [eps_zz, sig_zz]

    Parameters
    ----------
    E : np.ndarray
        Elastic modulus field, shape (nx, ny, nz).
    eps : np.ndarray
        Strain tensor field, shape (nx, ny, nz, 3, 3).
    sig : np.ndarray
        Stress tensor field, shape (nx, ny, nz, 3, 3).
    fields : list[str], optional
        List of fields to plot.
        Supported keys: 'E', 'nu', 'eps_xx', 'eps_yy', 'eps_zz', 'eps_xy', 'eps_yz', 'eps_xz',
        'sig_xx', 'sig_yy', 'sig_zz', 'sig_xy', 'sig_yz', 'sig_xz'.
    slice_z : int, optional
        Z-slice index. Defaults to nz // 2.
    title : str, optional
        Title suffix.
    """
    nx, ny, nz = E.shape
    if slice_z is None:
        slice_z = nz // 2

    # Default fields as requested:
    # Top row: E, nu (calculated from E and assumed relation or just E?)
    # Wait, 'nu' is not passed in explicitly as an array to this function in the old signature.
    # The old signature was plot_fields(E, eps, sig, title).
    # The user request says "plot E and nu at the top row".
    # I don't have 'nu' in the signature. I should update the signature to accept 'nu' OR
    # calculate it?
    # Looking at the usages, 'nu' is available in the calling scope.
    # I should update the signature to accept 'nu'.
    # BUT, to maintain backward compatibility (if desired) or just follow the plan...
    # The plan said: "Update plot_fields signature: def plot_fields(E, eps, sig, fields=None, slice_z=None, title="")".
    # It seems I missed adding 'nu' to the signature in the plan description, but the goal implies plotting 'nu'.
    # I MUST add 'nu' to the signature to plot it.
    pass

def plot_fields(
    E: np.ndarray,
    nu: np.ndarray, 
    eps: np.ndarray,
    sig: np.ndarray,
    fields: list[str] | None = None,
    slice_z: int | None = None,
    title: str = "",
) -> None:
    """Flexible plotting of field components.

    Default layout (if fields is None):
      Row 1: [E, nu]
      Row 2: [eps_xx, sig_xx]
      Row 3: [eps_yy, sig_yy]
      Row 4: [eps_zz, sig_zz]

    Parameters
    ----------
    E : np.ndarray
        Elastic modulus field, shape (nx, ny, nz).
    nu : np.ndarray
        Poisson ratio field, shape (nx, ny, nz).
    eps : np.ndarray
        Strain tensor field, shape (nx, ny, nz, 3, 3).
    sig : np.ndarray
        Stress tensor field, shape (nx, ny, nz, 3, 3).
    fields : list[str], optional
        List of fields to plot.
        Supported keys: 'E', 'nu', 'eps_xx', 'eps_yy', 'eps_zz', 'eps_xy', 'eps_yz', 'eps_xz',
        'sig_xx', 'sig_yy', 'sig_zz', 'sig_xy', 'sig_yz', 'sig_xz'.
    slice_z : int, optional
        Z-slice index. Defaults to nz // 2.
    title : str, optional
        Title suffix.
    """
    nx, ny, nz = E.shape
    if slice_z is None:
        slice_z = nz // 2

    # Define available data map
    # Tensor indices map
    idx_map = {
        'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2),
        'xy': (0, 1), 'yz': (1, 2), 'xz': (0, 2)
    }

    def get_data(name):
        if name == 'E':
            return E[:, :, slice_z], 'viridis', 'Elastic Modulus E'
        if name == 'nu':
            return nu[:, :, slice_z], 'viridis', 'Poisson Ratio nu'
        
        prefix, comp = name.split('_')
        i, j = idx_map[comp]
        if prefix == 'eps':
            return eps[:, :, slice_z, i, j], 'plasma', f'Strain ε_{comp}'
        if prefix == 'sig':
            return sig[:, :, slice_z, i, j], 'plasma', f'Stress σ_{comp}'
        raise ValueError(f"Unknown field: {name}")

    if fields is None:
        # Default layout: 4 rows, 2 columns
        # Row 1: E, nu
        # Row 2: eps_xx, sig_xx
        # Row 3: eps_yy, sig_yy
        # Row 4: eps_zz, sig_zz
        layout = [
            ['E', 'nu'],
            ['eps_xx', 'sig_xx'],
            ['eps_yy', 'sig_yy'],
            ['eps_zz', 'sig_zz']
        ]
        nrows, ncols = 4, 2
        fnames_flat = [] # Not used for flat list iteration, using layout
    else:
        # If fields are provided, we just plot them in a grid.
        # Simple heuristic: square-ish grid
        n = len(fields)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        # Flatten layout for iteration
        layout = [fields[i:i+ncols] for i in range(0, n, ncols)]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    # Flatten axes for easier indexing if we were just iterating, but here we have structured layout
    
    for r, row_fields in enumerate(layout):
        for c, fname in enumerate(row_fields):
            ax = axes[r, c]
            data, cmap, label = get_data(fname)
            im = ax.imshow(data.T, origin="lower", cmap=cmap)
            ax.set_title(f"{label} {title}")
            plt.colorbar(im, ax=ax)

    # Turn off empty subplots
    total_slots = nrows * ncols
    used_slots = sum(len(row) for row in layout)
    if used_slots < total_slots:
         for i in range(used_slots, total_slots):
             r, c = divmod(i, ncols)
             axes[r, c].axis('off')

    plt.tight_layout()
    plt.show()
