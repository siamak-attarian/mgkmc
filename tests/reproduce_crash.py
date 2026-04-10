import numpy as np
import os
import shutil
from mgkmc import ThermalSimulation

def main():
    # 16x16x1 for speed
    nx, ny, nz = 16, 16, 1
    pixel = 1.0
    M = 20
    gamma0 = 0.05
    
    E = np.full((nx, ny, nz), 70.0)  # 70 GPa
    nu = np.full((nx, ny, nz), 0.3)
    
    # Low barriers to trigger flip early
    def my_barrier_generator(n_modes):
        random_barriers = np.random.normal(loc=1.0, scale=0.1, size=n_modes)
        min_barrier = 0.1
        clipped_barriers = np.clip(random_barriers, a_min=min_barrier, a_max=None)
        return clipped_barriers
    
    output_dir = "reproduce_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    sim = ThermalSimulation(
        nx, ny, nz,
        M=M, 
        gamma0=gamma0,
        E_field=E, 
        nu_field=nu,
        pixel=pixel,
        barrier_generator=my_barrier_generator,
        output_dir=output_dir,
        softening_enabled=False,
        debug_first_flip=True
    )

    strain_inc = np.zeros((3,3))
    strain_inc[0,0] = 5e-4 # Larger steps to get there faster
    
    n_steps = 200
    
    sim.run(n_steps, strain_inc, vtk_mode=None)
    
if __name__ == "__main__":
    main()
