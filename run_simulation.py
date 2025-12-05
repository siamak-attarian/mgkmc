import numpy as np
from mgkmc import generate_correlated_field, run_simulation, export_simulation_vtk, get_uniaxial_stress_x

if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)

    nx, ny, nz = 32,32,1
    pixel = 0.5

    # --- Material fields ---
    E = generate_correlated_field(
        shape=(nx, ny, nz),
        mean=70e9,
        std=70e9*0.1,
        corr=2,
        seed=seed,
        visualize=False
    )
    nu = 0.30 * np.ones((nx, ny, nz))

    # --- Run simulation ---
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
        max_iter=200
    )

    # --- Export VTK (choose any option) ---
    export_simulation_vtk(eps_list, sig_list, E, nu, pixel, steps="last", prefix="mgkmc_test_")

    print("Simulation complete.")

    # Show the default plots
    from mgkmc.plotting import plot_fields
    # Use final fields
    eps = eps_list[-1]
    sig = sig_list[-1]
    plot_fields(E, nu, eps, sig, title="Default Layout")
