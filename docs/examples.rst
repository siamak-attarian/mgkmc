Examples
========

The ``examples/`` directory contains six worked scenarios, each a set of YAML
configurations with a ``README.md`` describing what it demonstrates. They build
up from pure elasticity to full KMC plasticity, so working through them in order
is the fastest way to understand what the solver does.

Every config is self-contained: heterogeneous fields are generated at run time
from the configured seed, so nothing is read from disk.

Running an example
------------------

Work from inside the example directory so that the output lands beside the
config rather than at the repository root:

.. code-block:: bash

   cd examples/1_uniaxial_tension_2d_homogeneous
   python ../../run.py plane_stress.yaml

Each run writes a summary log, a global log, and — where the config enables it —
ParaView-readable ``.vtu`` files. The stress-strain curve can be plotted with:

.. code-block:: bash

   python ../../tools/plot_summary_log.py output_plane_stress/summary_log.txt

1 — Uniaxial tension, 2D, homogeneous
-------------------------------------

Elastic-only baseline on a homogeneous 128x128 sheet, in three variants that
differ only in how the non-driven components are treated: ``plane_strain``,
``plane_strain_in_zz`` and ``plane_stress``.

Because the material is homogeneous and linear, the answers are known in closed
form, which makes this example a correctness check as well as an introduction:
the apparent stiffness should come out as :math:`\lambda + 2\mu`,
:math:`E/(1-\nu^2)` and :math:`E` respectively.

2 — Uniaxial tension, 3D solver on a single layer
--------------------------------------------------

The same problem driven through the 3D code path on a grid one voxel deep.
Running these alongside example 1 checks that the 2D and 3D solvers agree.

3 — Heterogeneous elasticity
----------------------------

Young's modulus becomes a spatially correlated random field, so a uniform
applied strain produces a non-uniform stress field — the precursor to
localisation. Sub-directories cover a linear versus secant-degradation
comparison, and a set in which the Lame constants are varied independently.

4 — KMC plasticity
------------------

The first example that runs the Shear Transformation Zone model: barriers
biased by local stress, kinetic Monte Carlo event selection, spectral
re-equilibration and softening. Two parameter sets are provided, each in three
variants — exact full-FFT stress updates, the fast-patching approximation, and
the 3D code path — so the cost and benefit of fast patching can be measured
directly.

5 — Small strain versus finite strain, 2D
------------------------------------------

The small-strain solver against the finite-strain solver under three
hyperelastic laws (Saint Venant-Kirchhoff, neo-Hookean and Murnaghan). A
compression case is included, which exposes the tension-compression asymmetry
that the third-order Murnaghan constants introduce and a linear model cannot
represent.

6 — Finite strain, fully 3D
---------------------------

The finite-strain solver on a genuine three-dimensional grid, with both lateral
directions relaxed to zero stress.

Driving the library directly
----------------------------

Simulations can also be constructed in Python rather than through
``run.py``. The entry point is
:meth:`~mgkmc.aqs.ThermalSimulation.run_simulation`:

.. code-block:: python

   import numpy as np
   from mgkmc import ThermalSimulation

   nx, ny, nz = 32, 32, 1
   sim = ThermalSimulation(
       nx, ny, nz, M=20, gamma0=0.14,
       E_field=np.full((nx, ny, nz), 70.0),
       nu_field=np.full((nx, ny, nz), 0.3),
       pixel=0.7,
       temperature=300.0,
       strain_rate=1e7,
       output_dir="output_quickstart",
   )
   sim.run_simulation(
       n_global_steps=200,
       step_size=1e-4,
       component=(0, 0),
       stress_targets={(1, 1): 0.0, (2, 2): 0.0},
   )

For two-dimensional KMC runs, :class:`~mgkmc.kmc_simulator.KmcSimulation2D`
exposes the same ``run_simulation`` entry point.
