# 3 — Uniaxial tension with heterogeneous elasticity

Still elastic-only, but the material is no longer uniform. Elastic constants vary
from voxel to voxel as a spatially correlated random field, so the stress field
under a uniform applied strain becomes non-uniform — the precursor to strain
localisation once STZ events are switched on in example 4.

All configs drive `xx` to `eps_target: 0.08` in steps of `1e-4`, with `yy`
(and `zz` in 3D) relaxed to zero stress.

## Top level — heterogeneous Young's modulus

| Config | Grid | Field |
|---|---|---|
| `plane_stress_2d.yaml` | 128×128, 2D plane stress | `E` generated: mean 70 GPa, std 5, correlation length 10 voxels, clipped below at 50 |
| `plane_stress_3d.yaml` | 128×128×1, 3D | same field, 3D code path |

`nu` is held constant at 0.3. The field is generated at run time from `seed: 1`,
so no external data file is needed.

## `compare_linear_nonlinear/` — constitutive law, same microstructure

Two configs identical except for the constitutive model, specified through Lamé
constants (`lambda: 80.937`, `mu: 23.7` GPa) rather than `E`/`nu`:

- `plane_stress_2d_linear.yaml` — linear elastic
- `plane_stress_2d_nonlinear.yaml` — `hyperelastic_model: secant_degradation`

Running both and overlaying the stress-strain curves isolates the effect of
secant stiffness degradation from the effect of the heterogeneity itself.

## `with_lambda_and_mu/` — heterogeneity in λ and μ separately

Four configs (128×128, `pixel: 0.7` nm) that specify λ and μ as independent
fields, to separate which modulus the heterogeneity lives in:

| Config | λ | μ |
|---|---|---|
| `plane_stress_2d_homogenous.yaml` | constant 81 | constant 25 |
| `plane_stress_2d_heterogenous_nonaffine.yaml` | constant 81 | generated, mean 25 |
| `plane_stress_2d_heterogenous_affine.yaml` | constant 81 | generated, mean 42 |
| `plane_stress_2d_heterogenous_affine_full.yaml` | generated, mean 91 | generated, mean 42 |

All generated fields use std 5.0 and correlation length 10 voxels.

## Run

```bash
cd examples/3_uniaxial_tension_heterogenous
python ../../run.py plane_stress_2d.yaml
```
