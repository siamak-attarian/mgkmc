Configuration Guide
===================

Simulations are driven by a single YAML file. ``config.yaml`` in the repository
root is the reference copy: every key below appears there with an inline comment,
and the example directories contain configs for specific scenarios.

.. code-block:: bash

   python run.py config.yaml

If no path is given, ``run.py`` looks for ``config.yaml`` in the current
directory. Output is written to the directory named by ``output.directory``,
resolved relative to the current working directory.

Units
-----

=============================  =========================================
Quantity                       Unit
=============================  =========================================
Stresses, elastic moduli       GPa (converted to Pa internally)
Strains                        dimensionless
Temperature                    K
Activation energies, barriers  eV
Lengths, voxel size            nm
Time                           s
=============================  =========================================

Top level
---------

``seed``
   Integer seed for the global random number generator. Leave null for a
   non-reproducible run.

``simulation_type``
   ``kmc`` runs the full kinetic Monte Carlo plasticity simulation. ``elastic``
   establishes elastic equilibrium and exits. The older spelling
   ``linear_elastic`` is still accepted as a synonym for ``elastic``.

system
------

``dimensionality``
   ``2d`` or ``3d``.

``plane_mode``
   ``plane_strain`` or ``plane_stress``. Ignored when ``dimensionality`` is
   ``3d``.

``strain_assumption``
   ``small_strain`` for linear kinematics, ``finite_strain`` for large
   deformation.

``hyperelastic_model``
   Constitutive law. Finite-strain models: ``svk`` (Saint Venant-Kirchhoff),
   ``neo_hookean`` (2D only), ``murnaghan``. Small-strain models:
   ``secant_degradation``, ``rose``, ``landau``, and ``linear`` (default Hooke).

``solver``
   Non-linear FFT solver used on the finite-strain path: ``al`` (augmented
   Lagrangian, robust), ``newton_cg`` (faster, more sensitive to local failure),
   or ``dbfft`` (displacement-based, preconditioned).

``3d_barriers``
   When true, use the fully 3D isotropic STZ eigenstrain catalogue rather than
   the reduced in-plane set.

``nx``, ``ny``, ``nz``
   Grid dimensions in voxels. Set ``nz: 1`` for 2D.

``pixel``
   Voxel edge length in nm.

``M``
   Number of candidate STZ orientations generated per voxel.

``gamma0``
   Reference shear strain of a single transformation event.

``num_threads``
   Thread count passed to PyFFTW.

material
--------

``E`` and ``nu`` are field specifications, each taking a ``mode``:

- ``constant`` -- uses ``value``
- ``generated`` -- a spatially correlated random field; takes ``parameters``
  with ``mean``, ``std``, ``corr`` (correlation length in voxels) and optional
  ``clip_min`` / ``clip_max``
- ``file`` -- loads a ``.npy`` array from ``parameters.path``

``lambda`` and ``mu`` may be given instead of ``E``/``nu`` to specify Lame
constants directly, using the same field-specification form.

Model-specific parameters, each used only by the matching
``hyperelastic_model``:

``A``, ``B``, ``C``
   Murnaghan third-order elastic constants (GPa).

``d``, ``k``
   ``secant_degradation``: maximum degradation fraction and degradation rate.

``eta``, ``mu_floor_fraction``
   ``rose``: softening rate, and the floor on shear modulus as a fraction of its
   initial value.

Strain capping regularises the Landau model, preventing unbounded softening at
large strain:

``strain_capping_enabled``
   Toggle. The remaining capping keys apply only when this is true.

``strain_capping_limit``
   Green-Lagrange shear-equivalent strain at which capping begins. Null or zero
   selects an automatically computed physical limit.

``strain_capping_tangent_ratio``
   Residual tangent shear modulus as a fraction of the initial value.

``strain_capping_type``
   ``piecewise`` (match Landau exactly, then linear with a floor) or ``smooth``
   (tanh mapping).

``strain_capping_smooth_power``
   Exponent for ``smooth`` capping; larger values stay closer to exact Landau
   before the transition.

physics
-------

``enable_softening``
   Toggle barrier softening after a flip.

``softening_scheme``
   ``isotropic`` (uniform barrier reduction) or ``directional`` (reduction
   depends on the projection of the shear direction).

``jp``, ``jt``
   Coupling constants for permanent (plastic) and transient (thermal) softening
   per flip.

``neighbor_softening_fraction``
   Fraction of the plastic softening also applied to adjacent voxels.

``q_act_temp``
   Activation barrier for the beta-relaxation process that recovers transient
   softening (eV). Together with ``dynamics.nu0`` and ``dynamics.temperature``
   this sets the transient decay time; that decay time is computed internally
   and is not itself a configuration key.

``softening_cap``
   Maximum allowed barrier reduction (eV).

``stability_threshold``
   Energy below which a mode is treated as athermally unstable (eV).

``redraw_directions``, ``redraw_barriers``
   Whether to redraw the candidate orientations of a voxel, and its baseline
   barriers, after it flips.

``stz_mode``
   Eigenstrain tensor layout: ``pure_shear`` or ``simple_shear``.

barriers
--------

``type``
   Distribution of initial activation barriers: ``gaussian``, ``rayleigh``,
   ``modified_rayleigh``, or ``modified_rayleigh_with_exponential``.

``kwargs``
   Passed through to the selected generator. For ``gaussian``: ``mean``,
   ``std``, ``min_cutoff``.

dynamics
--------

``fast_patching``
   Mapping with ``enabled``, ``patch_radius`` and ``sync_interval``. When
   enabled, the stress update after each event uses a precomputed local kernel
   instead of a full FFT, resynchronising every ``sync_interval`` steps. Must be
   disabled on the finite-strain path.

``temperature``
   Simulation temperature in K.

``physical_strain_rate``
   Applied macroscopic strain rate (1/s).

``nu0``
   Attempt frequency (Hz).

``cascade_mode``
   ``false`` (default) resolves unstable flips sequentially through KMC rates;
   ``true`` lets athermal avalanches cascade immediately. This key replaced the
   earlier string-valued ``instability_mode``, which is still accepted for
   backward compatibility.

``scale_rate_by_volume``
   Scale KMC rates by voxel volume.

thermal
-------

``enable_thermal``
   Toggle the coupled heat-generation and diffusion solver.

``Cp``, ``rho``, ``thermal_diffusivity``
   Specific heat (J/kg K), density (kg/m^3) and thermal diffusivity (m^2/s).

``thermal_coords``
   ``pixel`` or ``physical``, selecting the frame for the diffusion calculation.

``temperature_cap``
   Ceiling on local temperature (K).

``thermostat``, ``tau_bath``
   Toggle bath coupling and its relaxation time (s). A ``tau_bath`` of zero
   fixes the mean temperature.

loading
-------

``eps_target``
   Target macroscopic strain. Negative values load in compression.

``step_size``
   Strain increment per loading step. The step count follows from
   ``eps_target / step_size``.

boundary_conditions
-------------------

``driving_component``
   Strain component driven by the loading: ``xx``, ``yy``, ``xy``, and
   additionally ``zz``, ``xz``, ``yz`` in 3D.

``mixed_targets``
   Mapping from component name to a target stress in GPa for the non-driven
   components. Omitting a component leaves its strain fixed at zero instead.

``mixed_tol``
   Convergence tolerance for the lateral relaxation iteration, in MPa.

output
------

``directory``
   Output directory, relative to the working directory.

``duplicate_directory_action``
   What to do when it already exists: ``delete``, ``rename`` or ``overwrite``.

``checkpoint_interval``
   Integer step interval, or ``none`` / ``last`` / ``current``.

``checkpoint_elastic_only``
   Restrict checkpoints to elastic loading steps.

``enable_save_q``, ``save_q_interval``
   Toggle and interval for dumping the raw barrier state array.

``enable_plotting``
   Generate summary stress-strain plots.

``enable_console``
   Per-step progress output to the terminal.

``enable_config_backup``
   Copy the config into the output directory.

``summary_filename``
   Name of the summary log.

``enable_summary_log``, ``enable_global_log``, ``enable_cascade_log``, ``enable_kmc_log``
   Individual log toggles.

``vtk_interval``
   Integer step interval for ``.vtu`` export, or ``none`` / ``last``.

``vtk_elastic_only``
   Restrict VTK export to elastic loading steps.

``track_cascades``
   Write a VTK file for every step within a cascade. Produces many files.

.. note::

   ``output.save_q_elastic_only`` appears in ``config.yaml`` but is not
   currently passed through by ``run.py``, and the solver does not act on it.
   Setting it has no effect.

detection
---------

``stop_on_stress_drop``
   Stop early once stress falls by this fraction from its peak (0.20 = 20%).

``stop_post_drop_steps``
   Extra steps to run and log after a drop is detected.

``ignore_drop_steps``
   Initial steps during which drops are ignored, avoiding premature exit on
   transients.

``stress_drop_lookback``
   Rolling window used to evaluate the drop.

``max_cascade_steps_pct``, ``max_kmc_steps_pct``
   Truncate a cascade, or a sequence of KMC steps, once the number of events
   exceeds this fraction of the total voxel count. Guards against runaway loops.
