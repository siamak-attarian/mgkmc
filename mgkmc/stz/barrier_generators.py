import numpy as np
from scipy.stats import rayleigh

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

def get_rayleigh_generator(loc=1.0, scale=0.5, min_cutoff=0.1, max_cutoff=None):
    """
    Returns a function that generates Rayleigh barriers with the specified parameters.
    
    Parameters
    ----------
    loc : float, default 1.0
        The location parameter of the Rayleigh distribution.
    scale : float, default 0.5
        The scale parameter of the Rayleigh distribution.
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
        barriers = rayleigh(loc=loc, scale=scale).rvs(size=shape)
        if min_cutoff is not None or max_cutoff is not None:
            barriers = np.clip(barriers, a_min=min_cutoff, a_max=max_cutoff)
        return barriers
    
    return generator

def get_modified_rayleigh_generator(mean=2.0, std=0.6, min_cutoff=0.1, max_cutoff=None):
    """
    Returns a function that generates modified Rayleigh barriers with the specified parameters:
    f(x) ~ x * exp(-(x - mean)**2 / (2 * std**2))
    
    Parameters
    ----------
    mean : float, default 2.0
    std : float, default 0.6
    min_cutoff : float, default 0.1
    max_cutoff : float, optional
    """
    def generator(shape):
        if isinstance(shape, int):
            N = shape
        else:
            N = np.prod(shape)
            
        samples = np.empty(N)
        n_accepted = 0
        
        a = max(0.0, mean - 6*std)
        b = mean + 6*std
        
        x_max = (mean + np.sqrt(mean**2 + 4*std**2)) / 2.0
        c_max = x_max * np.exp(-(x_max - mean)**2 / (2 * std**2))
        
        while n_accepted < N:
            n_needed = N - n_accepted
            n_propose = max(int(n_needed * 3), 100)
            
            x_prop = np.random.uniform(a, b, n_propose)
            u_prop = np.random.uniform(0, c_max, n_propose)
            
            pdf_prop = x_prop * np.exp(-(x_prop - mean)**2 / (2 * std**2))
            
            accepted = x_prop[u_prop < pdf_prop]
            num_acc = len(accepted)
            
            if num_acc > 0:
                take = min(num_acc, n_needed)
                samples[n_accepted:n_accepted+take] = accepted[:take]
                n_accepted += take
                
        barriers = samples.reshape(shape)
        if min_cutoff is not None or max_cutoff is not None:
            barriers = np.clip(barriers, a_min=min_cutoff, a_max=max_cutoff)
        return barriers

    return generator

def get_modified_rayleigh_with_exponential_generator(mean=2.0, std=0.6, epsilon=0.1, ratio=0.8, min_cutoff=0.1, max_cutoff=None):
    """
    Returns a function that generates a mixture of Modified Rayleigh and Exponential barriers.
    ratio dictates the probability (0 to 1) of drawing from Exponential.
    """
    def generator(shape):
        if isinstance(shape, int):
            N = shape
        else:
            N = np.prod(shape)
            
        barriers = np.empty(N)
        
        # Determine which samples come from which distribution
        u_mix = np.random.uniform(0.0, 1.0, N)
        mask_exp = u_mix < ratio
        mask_ray = ~mask_exp
        n_ray = np.sum(mask_ray)
        n_exp = N - n_ray
        
        # 1. Generate Modified Rayleigh Samples
        if n_ray > 0:
            samples_ray = np.empty(n_ray)
            n_accepted = 0
            
            a = max(0.0, mean - 6*std)
            b = mean + 6*std
            
            x_max = (mean + np.sqrt(mean**2 + 4*std**2)) / 2.0
            c_max = x_max * np.exp(-(x_max - mean)**2 / (2 * std**2))
            
            while n_accepted < n_ray:
                n_needed = n_ray - n_accepted
                n_propose = max(int(n_needed * 3), 100)
                
                x_prop = np.random.uniform(a, b, n_propose)
                u_prop = np.random.uniform(0, c_max, n_propose)
                
                pdf_prop = x_prop * np.exp(-(x_prop - mean)**2 / (2 * std**2))
                accepted = x_prop[u_prop < pdf_prop]
                num_acc = len(accepted)
                
                if num_acc > 0:
                    take = min(num_acc, n_needed)
                    samples_ray[n_accepted:n_accepted+take] = accepted[:take]
                    n_accepted += take
            
            barriers[mask_ray] = samples_ray
            
        # 2. Generate Exponential Samples
        if n_exp > 0:
            # np.random.exponential takes 'scale' = epsilon
            barriers[~mask_ray] = np.random.exponential(scale=epsilon, size=n_exp)
            
        barriers = barriers.reshape(shape)
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
    elif generator_type.lower() == "rayleigh":
        return get_rayleigh_generator(**kwargs)
    elif generator_type.lower() == "modified_rayleigh":
        return get_modified_rayleigh_generator(**kwargs)
    elif generator_type.lower() == "modified_rayleigh_with_exponential":
        return get_modified_rayleigh_with_exponential_generator(**kwargs)
    else:
        raise ValueError(f"Unknown barrier generator type: {generator_type}")
