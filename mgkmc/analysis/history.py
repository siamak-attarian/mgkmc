"""
Post-processing tools for analyzing simulation history.

Provides functions to load checkpoints, extract history data from logs,
and analyze cascade statistics.
"""

import numpy as np
import os
import h5py
from pathlib import Path


def load_simulation(checkpoint_file):
    """
    Load simulation from checkpoint file.
    
    Parameters
    ----------
    checkpoint_file : str
        Path to checkpoint HDF5 file
    
    Returns
    -------
    sim : AthermalSimulation
        Loaded simulation instance
    
    Examples
    --------
    >>> sim = load_simulation("checkpoint_step_1000.h5")
    >>> # Continue simulation
    >>> sim.run(n_global_steps=100, ...)
    """
    from mgkmc.checkpoint import load_checkpoint
    return load_checkpoint(checkpoint_file)


def extract_history(output_dir):
    """
    Extract history data from simulation output directory.
    
    Parses global_log.txt and detailed_cascade.txt into structured arrays.
    
    Parameters
    ----------
    output_dir : str
        Path to simulation output directory
    
    Returns
    -------
    history : dict
        Dictionary containing:
        - 'global': np.ndarray (n_steps, 14)
            Columns: step, eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
                    sig_xx, sig_yy, sig_zz, sig_xy, sig_xz, sig_yz,
                    cascade_steps, total_flips
        - 'cascade': list of dict
            Each entry: {'global_step': int, 'local_step': int, 
                        'num_unstable': int, 'flipped_voxels': list}
    
    Examples
    --------
    >>> hist = extract_history("aqs_output")
    >>> global_data = hist['global']
    >>> strain = global_data[:, 1]  # eps_xx
    >>> stress = global_data[:, 7]  # sig_xx
    """
    global_log_path = os.path.join(output_dir, "global_log.txt")
    cascade_log_path = os.path.join(output_dir, "detailed_cascade.txt")
    
    history = {}
    
    # Parse global log
    if os.path.exists(global_log_path):
        try:
            # Skip header line
            global_data = np.loadtxt(global_log_path, skiprows=1)
            history['global'] = global_data
        except Exception as e:
            print(f"Warning: Could not parse global_log.txt: {e}")
            history['global'] = None
    else:
        print(f"Warning: {global_log_path} not found")
        history['global'] = None
    
    # Parse cascade log
    if os.path.exists(cascade_log_path):
        cascade_data = []
        try:
            with open(cascade_log_path, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        entry = {
                            'global_step': int(parts[0]),
                            'local_step': int(parts[1]),
                            'num_unstable': int(parts[2]),
                            'flipped_voxels_str': parts[3]
                        }
                        cascade_data.append(entry)
            history['cascade'] = cascade_data
        except Exception as e:
            print(f"Warning: Could not parse detailed_cascade.txt: {e}")
            history['cascade'] = []
    else:
        print(f"Warning: {cascade_log_path} not found")
        history['cascade'] = []
    
    return history


def analyze_cascades(output_dir):
    """
    Analyze cascade statistics from simulation output.
    
    Parameters
    ----------
    output_dir : str
        Path to simulation output directory
    
    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'total_cascades': int
            Total number of cascade events
        - 'cascade_sizes': np.ndarray
            Array of cascade sizes (number of flips per cascade)
        - 'avalanche_steps': np.ndarray
            Number of local steps per global step
        - 'total_flips_per_step': np.ndarray
            Total flips per global step
        - 'mean_cascade_size': float
        - 'max_cascade_size': int
        - 'cascade_size_distribution': tuple (bins, counts)
            Histogram of cascade sizes
    
    Examples
    --------
    >>> stats = analyze_cascades("aqs_output")
    >>> print(f"Mean cascade size: {stats['mean_cascade_size']:.2f}")
    >>> print(f"Max cascade size: {stats['max_cascade_size']}")
    """
    hist = extract_history(output_dir)
    
    stats = {}
    
    # Extract from global log
    if hist['global'] is not None:
        global_data = hist['global']
        stats['avalanche_steps'] = global_data[:, -2].astype(int)  # cascade_steps column
        stats['total_flips_per_step'] = global_data[:, -1].astype(int)  # total_flips column
    else:
        stats['avalanche_steps'] = np.array([])
        stats['total_flips_per_step'] = np.array([])
    
    # Extract from cascade log
    cascade_data = hist['cascade']
    stats['total_cascades'] = len(cascade_data)
    
    if len(cascade_data) > 0:
        cascade_sizes = np.array([c['num_unstable'] for c in cascade_data])
        stats['cascade_sizes'] = cascade_sizes
        stats['mean_cascade_size'] = cascade_sizes.mean()
        stats['max_cascade_size'] = cascade_sizes.max()
        
        # Histogram
        if len(cascade_sizes) > 0:
            bins = np.logspace(0, np.log10(cascade_sizes.max() + 1), 20)
            counts, bin_edges = np.histogram(cascade_sizes, bins=bins)
            stats['cascade_size_distribution'] = (bin_edges, counts)
        else:
            stats['cascade_size_distribution'] = (np.array([]), np.array([]))
    else:
        stats['cascade_sizes'] = np.array([])
        stats['mean_cascade_size'] = 0.0
        stats['max_cascade_size'] = 0
        stats['cascade_size_distribution'] = (np.array([]), np.array([]))
    
    return stats


def compute_plastic_strain_evolution(checkpoint_file):
    """
    Compute plastic strain evolution from checkpoint.
    
    Parameters
    ----------
    checkpoint_file : str
        Path to checkpoint HDF5 file
    
    Returns
    -------
    plastic_stats : dict
        Dictionary containing:
        - 'eps_plastic_mean': float
            Mean von Mises plastic strain
        - 'eps_plastic_max': float
            Maximum von Mises plastic strain
        - 'eps_plastic_field': np.ndarray (nx, ny, nz, 3, 3)
            Full plastic strain tensor field
        - 'eps_plastic_vm': np.ndarray (nx, ny, nz)
            Von Mises plastic strain field
        - 'active_voxels': int
            Number of voxels with plastic strain > 0
    """
    with h5py.File(checkpoint_file, 'r') as f:
        eps_plastic = f['grid/eps_plastic'][:]
    
    nx, ny, nz = eps_plastic.shape[:3]
    
    # Compute von Mises plastic strain
    tr_eps_p = np.trace(eps_plastic, axis1=3, axis2=4)[..., None, None]
    eps_p_dev = eps_plastic - np.eye(3)[None,None,None,:,:] * tr_eps_p/3
    eps_plastic_vm = np.sqrt((2/3) * np.sum(eps_p_dev**2, axis=(3,4)))
    
    stats = {
        'eps_plastic_mean': eps_plastic_vm.mean(),
        'eps_plastic_max': eps_plastic_vm.max(),
        'eps_plastic_field': eps_plastic,
        'eps_plastic_vm': eps_plastic_vm,
        'active_voxels': np.sum(eps_plastic_vm > 1e-10)
    }
    
    return stats


def extract_stress_strain_curves(output_dir, component='xx'):
    """
    Extract stress-strain curves from simulation output.
    
    Parameters
    ----------
    output_dir : str
        Path to simulation output directory
    component : str
        Tensor component to extract ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')
    
    Returns
    -------
    curves : dict
        Dictionary containing:
        - 'strain': np.ndarray
            Strain values
        - 'stress': np.ndarray
            Stress values (in GPa)
        - 'component': str
            Component name
    
    Examples
    --------
    >>> curves = extract_stress_strain_curves("aqs_output", component='xx')
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(curves['strain'], curves['stress'])
    >>> plt.xlabel('Strain')
    >>> plt.ylabel('Stress (GPa)')
    """
    hist = extract_history(output_dir)
    
    if hist['global'] is None:
        raise ValueError(f"Could not load global log from {output_dir}")
    
    global_data = hist['global']
    
    # Column mapping
    component_map = {
        'xx': (1, 7),   # (eps_xx, sig_xx)
        'yy': (2, 8),   # (eps_yy, sig_yy)
        'zz': (3, 9),   # (eps_zz, sig_zz)
        'xy': (4, 10),  # (eps_xy, sig_xy)
        'xz': (5, 11),  # (eps_xz, sig_xz)
        'yz': (6, 12),  # (eps_yz, sig_yz)
    }
    
    if component not in component_map:
        raise ValueError(f"Invalid component '{component}'. Must be one of {list(component_map.keys())}")
    
    eps_col, sig_col = component_map[component]
    
    curves = {
        'strain': global_data[:, eps_col],
        'stress': global_data[:, sig_col] / 1e9,  # Convert Pa to GPa
        'component': component
    }
    
    return curves


def generate_summary_report(output_dir, checkpoint_file=None):
    """
    Generate a summary report of simulation results.
    
    Parameters
    ----------
    output_dir : str
        Path to simulation output directory
    checkpoint_file : str, optional
        Path to checkpoint file for plastic strain analysis
    
    Returns
    -------
    report : str
        Formatted summary report
    
    Examples
    --------
    >>> report = generate_summary_report("aqs_output", "checkpoint_final.h5")
    >>> print(report)
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SIMULATION SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Cascade statistics
    try:
        stats = analyze_cascades(output_dir)
        lines.append("CASCADE STATISTICS:")
        lines.append(f"  Total cascade events: {stats['total_cascades']}")
        lines.append(f"  Mean cascade size: {stats['mean_cascade_size']:.2f}")
        lines.append(f"  Max cascade size: {stats['max_cascade_size']}")
        lines.append(f"  Total flips: {stats['total_flips_per_step'].sum()}")
        lines.append("")
    except Exception as e:
        lines.append(f"Could not analyze cascades: {e}")
        lines.append("")
    
    # Stress-strain curve
    try:
        curves = extract_stress_strain_curves(output_dir, 'xx')
        lines.append("STRESS-STRAIN (xx component):")
        lines.append(f"  Max strain: {curves['strain'].max():.4f}")
        lines.append(f"  Max stress: {curves['stress'].max():.2f} GPa")
        lines.append(f"  Final strain: {curves['strain'][-1]:.4f}")
        lines.append(f"  Final stress: {curves['stress'][-1]:.2f} GPa")
        lines.append("")
    except Exception as e:
        lines.append(f"Could not extract stress-strain curves: {e}")
        lines.append("")
    
    # Plastic strain (if checkpoint provided)
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            plastic_stats = compute_plastic_strain_evolution(checkpoint_file)
            lines.append("PLASTIC DEFORMATION:")
            lines.append(f"  Mean plastic strain (von Mises): {plastic_stats['eps_plastic_mean']:.6f}")
            lines.append(f"  Max plastic strain (von Mises): {plastic_stats['eps_plastic_max']:.6f}")
            lines.append(f"  Active voxels: {plastic_stats['active_voxels']}")
            lines.append("")
        except Exception as e:
            lines.append(f"Could not analyze plastic strain: {e}")
            lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)
