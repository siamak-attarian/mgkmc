# examples/plot_demo.py
"""
Example script demonstrating Matplotlib visualisation of the mgkmc simulation results,
using the same workflow as run_simulation.py.

Run with:
    python examples/plot_demo.py
"""

import numpy as np
from mgkmc import generate_correlated_field, run_simulation, get_uniaxial_stress_x
from mgkmc.analysis import plot_fields


def main():
    # 1. Create the same microstructure as in run_simulation.py
    seed = 1
    np.random.seed(seed)
    nx, ny, nz = 32, 32, 1
    pixel = 0.5

    E = generate_correlated_field(
        shape=(nx, ny, nz),
        mean=70e9,
        std=70e9 * 0.1,
        corr=2,
        seed=seed,
        visualize=False,
    )
    nu = 0.30 * np.ones((nx, ny, nz))

    # 2. Run the same loading path as in run_simulation.py
    # Target: 2% uniaxial strain in X (Case 3: Uniaxial stress).
    epsM, sigM, eps_list, sig_list = run_simulation(
        E, nu,
        loading_func=get_uniaxial_stress_x,
        loading_params={
            "eps_xx": 0.02,
            "E": 70e9,
            "nu": 0.30
        },
        n_steps=10,
        pixel=pixel,
        max_iter=200,
    )

    # Use final fields for plotting
    eps = eps_list[-1]
    sig = sig_list[-1]

    # 3. Plot using the generic helper
    plot_fields(E, nu, eps, sig, title="run_simulation demo")


if __name__ == "__main__":
    main()
