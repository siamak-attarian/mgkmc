Examples Guide
==============

MGKMC is extremely versatile. The ``examples/`` directory provides several curated simulation recipes allowing you to understand and adapt physical scenarios directly.

Example 01: Uniaxial Tension
----------------------------
**File:** ``01_uniaxial_tension.py``
Models macroscopic deformation when the bulk material is continuously pulled along the positive X-axis while other stress vectors are iteratively relaxed to mimic boundaries moving freely without lateral restriction. It showcases using the ``run_mixed`` solver architecture effectively.

Example 02: Pure Shear
----------------------
**File:** ``02_pure_shear.py``
Evaluates the material under a monotonic shearing motion alongside the XY boundary, highlighting exactly how the solver responds by mapping diagonal components and generating strain invariants correctly.

Example 03: Temperature Effects
-------------------------------
**File:** ``03_temperature_effects.py``
The base code initializes purely under "Athermal Quasi-static" (AQS) mechanics—ignoring statistical time distributions. This example instead executes an explicit loop with a preset kelvin threshold enabling Kinetic Monte Carlo to sample statistically finite random activation jumps representing thermal noise yielding material. 

Example 04: Checkpoints and Restarts
------------------------------------
**File:** ``04_checkpoints_and_restarts.py``
Mechanisms for saving an ongoing computation, extracting its grid structure, voxel orientation mapping, and stress/strain state—and accurately loading the HDF5 dump inside a fresh python instance safely preserving exactly equal random number generator reproducibility.
