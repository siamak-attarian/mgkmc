# 5 — Small strain vs finite strain, 2D

Elastic-only comparison of the small-strain solver against the finite-strain
solver under three hyperelastic constitutive laws. At 8% applied strain the two
kinematic assumptions have visibly diverged, and the size of that gap is what
this example is for.

All configs: 32×32 voxels, `pixel: 1.0` nm, 2D plane stress, `xx` driven to
`eps_target: 0.08` in steps of `1e-4`, `yy` relaxed to zero stress.

| Config | Kinematics | Constitutive model | Elastic constants |
|---|---|---|---|
| `small_strain.yaml` | small strain | linear | `E = 70` GPa, `nu = 0.3` |
| `finite_strain_svk.yaml` | finite strain | Saint Venant–Kirchhoff | `E = 70` GPa, `nu = 0.3` |
| `finite_strain_nh.yaml` | finite strain | neo-Hookean | `E = 70` GPa, `nu = 0.3` |
| `finite_strain_murnaghan.yaml` | finite strain | Murnaghan | `lambda = 25`, `mu = 65` GPa |
| `finite_strain_murnaghan_negative.yaml` | finite strain | Murnaghan | as above, `eps_target: -0.08` |

Note that the two Murnaghan configs are specified directly in Lamé constants and
use different values from the other three, so they are best compared against each
other rather than against the SVK and neo-Hookean curves.

`finite_strain_murnaghan_negative.yaml` applies the same magnitude of strain in
**compression**. Running it against `finite_strain_murnaghan.yaml` shows the
tension–compression asymmetry that the third-order Murnaghan constants introduce
and that a linear model cannot represent.

## Run

```bash
cd examples/5_uniaxial_tension_finite_strain_2d
python ../../run.py small_strain.yaml
python ../../run.py finite_strain_svk.yaml
```

Each writes to its own output directory, so the summary logs can be overlaid
afterwards.
