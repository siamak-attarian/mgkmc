import numpy as np
from scipy.ndimage import gaussian_filter

def generate_correlated_field(
    shape, mean=1.0, std=0.1, corr=10, clip_min=None, clip_max=None,
    seed=None, visualize=False, title="Correlated field"
):
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
