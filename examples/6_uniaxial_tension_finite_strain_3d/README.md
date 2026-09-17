# 6 — Finite strain, fully 3D

The finite-strain solver on a genuine three-dimensional grid — 16×16×16 voxels,
rather than the single-layer grids used in examples 2 and 4. The grid is small
because the finite-strain path solves a non-linear equilibrium problem at every
loading step.

`finite_strain.yaml`: `pixel: 1.0` nm, `E = 70` GPa, `nu = 0.3`,
`strain_assumption: finite_strain`, `xx` driven to `eps_target: 0.08` in steps of
`1e-3`, with **both** `yy` and `zz` relaxed to zero stress — the fully
unconstrained lateral contraction that only the 3D path can represent.

## Run

```bash
cd examples/6_uniaxial_tension_finite_strain_3d
python ../../run.py finite_strain.yaml
```
