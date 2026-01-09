import numpy as np

def get_gaussian_generator(mean=2.0, std=0.6, min_cutoff=0.1, max_cutoff=None):
    """
    Returns a function that generates Gaussian barriers with the specified parameters.
    
    Parameters
    ----------
    mean : float, default 2.0
        The mean of the normal distribution.
    std : float, default 0.6
        The standard deviation of the normal distribution.
    min_cutoff : float, default 0.1
        The minimum value for the generated barriers.
    max_cutoff : float, optional
        The maximum value for the generated barriers.
        
    Returns
    -------
    generator : function
        A function that takes a shape tuple and returns a numpy array of barriers.
    """
    def generator(shape):
        barriers = np.random.normal(loc=mean, scale=std, size=shape)
        if min_cutoff is not None or max_cutoff is not None:
            barriers = np.clip(barriers, a_min=min_cutoff, a_max=max_cutoff)
        return barriers
    
    return generator

def get_barrier_generator(generator_type, **kwargs):
    """
    Factory function to get a barrier generator by type.
    
    Parameters
    ----------
    generator_type : str
        The type of generator (e.g., "gaussian").
    **kwargs : dict
        Additional arguments for the generator.
        
    Returns
    -------
    generator : function
        A function that takes a shape tuple and returns a numpy array of barriers.
    """
    if generator_type.lower() == "gaussian":
        return get_gaussian_generator(**kwargs)
    else:
        raise ValueError(f"Unknown barrier generator type: {generator_type}")
