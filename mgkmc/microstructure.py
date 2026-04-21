import numpy as np
from scipy.ndimage import gaussian_filter

def generate_correlated_field(
    shape, mean=1.0, std=0.1, corr=10, clip_min=None, clip_max=None,
    seed=None, visualize=False, title="Correlated field"
):
    """
    Generates a spatially correlated random Gaussian scalar field.

    This function utilizes a combination of random noise generation 
    and multi-dimensional Gaussian filtering to produce continuous, realistic 
    spatial fluctuations of properties such as the Elastic Modulus.
    
    Parameters
    ----------
    shape : tuple of int
        The (nx, ny, nz) grid dimensions of the required field.
    mean : float, optional
        The statistical mean of the normal distribution. Default is 1.0.
    std : float, optional
        The standard deviation of the fluctuations. Default is 0.1.
    corr : float, optional
        The Gaussian filter standard deviation (correlation length) applied
        across the grid in pixels. Higher values yield smoother fields. Default is 10.
    clip_min : float, optional
        The lower bound limit to truncate the field values. Extrema below this 
        will be mapped to ``clip_min``.
    clip_max : float, optional
        The upper bound limit to truncate the field values.
    seed : int, optional
        The random state seed for reproducibility.
    visualize : bool, optional
        If True, uses matplotlib to interactively plot a mid-plane 
        cross section of the field.
    title : str, optional
        The visual title passed into matplotlib if visualize is True.
        
    Returns
    -------
    numpy.ndarray
        A 3D array of shape ``shape`` containing the correlated values.
    """
    if seed is not None:
        np.random.seed(seed)

    field = np.random.normal(size=shape)
    field = gaussian_filter(field, sigma=corr, mode='wrap')

    field = (field - field.mean()) / field.std()
    field = mean + std*field

    if clip_min is not None or clip_max is not None:
        field = np.clip(field, clip_min, clip_max)

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(field[:,:,shape[2]//2].T, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()

    return field

def generate_field(config_mode, shape, constant_val=None, params=None):
    """
    Dynamically routes field initialization requests from YAML configurations.

    Serves as an operational wrapper directing execution logic depending on whether
    a material property acts as a universal numeric constant, derives from 
    statistically generated spatial correlations, or is imported explicitly 
    from an existing disk file array.

    Parameters
    ----------
    config_mode : str
        The initialization mode flag. Accepted strings are: "constant",
        "generated", or "file".
    shape : tuple of int
        The (nx, ny, nz) shape of the simulation domain.
    constant_val : float, optional
        The fallback property baseline used if ``config_mode`` evaluates as "constant".
    params : dict, optional
        Supplementary keyword specifications nested under the material property
        in ``config.yaml``. Needed for parameters dictating "generated" mode scaling settings or "file" pathing.
        
    Returns
    -------
    numpy.ndarray
        The initialized material parameter grid matching the required shape.
        
    Raises
    ------
    FileNotFoundError
        If "file" mode is requested but the supplied numpy file path is inaccessible.
    ValueError
        If an unrecognized ``config_mode`` string is provided.
    """
    import os
    if config_mode == "constant":
        return np.full(shape, constant_val)
    elif config_mode == "generated":
        mean = params.get('mean', constant_val)
        std = params.get('std', 0.0)
        corr = params.get('corr', 10)
        clip_min = params.get('clip_min', None)
        clip_max = params.get('clip_max', None)
        return generate_correlated_field(
            shape, mean=mean, std=std, corr=corr, 
            clip_min=clip_min, clip_max=clip_max
        )
    elif config_mode == "file":
        path = params.get('path')
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing or invalid path for field file: {path}")
        return np.load(path)
    else:
        raise ValueError(f"Unknown field mode: {config_mode}")
