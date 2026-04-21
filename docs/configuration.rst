Configuration Guide
===================

The repository includes a centralized ``config.yaml`` file which stores hyperparameters, output settings, and initialization bounds for simulation.

Core Sections
-------------

**System Setup**

- ``nx``, ``ny``, ``nz``: Grid spatial dimensions.
- ``pixel``: The length scale of a voxel in nanometers (nm).
- ``M``: Number of shear orientations inside a voxel.
- ``gamma0``: The canonical transformation strain scalar describing the volume average strain caused by an STZ flip.

**Material** 

- ``E``: The Elastic Modulus in GPa. Can be a constant or procedurally generated field.
- ``nu``: Poisson's ratio field.

**Physics & Softening**

Configures how materials yield due to Shear Transformation Zone flips:
- ``softening_scheme``: Use ``directional`` (respects directionality of strain) or ``isotropic`` yielding.
- ``jp`` & ``jt``: Coupling constants controlling permanent (plastic) vs transient (thermal) softening amounts per flip.
- ``tau``: Time constant over which transient thermal softening decays over time natively under Kinetic Monte Carlo mechanics.
- ``q_act_temp``: Base activation energy describing how rapidly the transient softening decays.

**Simulation Dynamics**

- ``fast_patching``: Toggles the Fast predictor-corrector implementation which dramatically accelerates event resolution by bypassing the exact full-grid FFT algorithm until periodic intervals.
- ``temperature``: Imposed global Kelvin. Values > 0 activate explicit thermal jumps beneath the athermal yield surface in the solver loop.
- ``physical_strain_rate``: Defines the loading rate boundary condition (1/s).
- ``instability_mode``: Choose between ``cascade`` collective macroscopic solver and pure sequential ``kmc`` dynamics.
