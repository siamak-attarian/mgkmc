# 2 — Uniaxial tension, 3D solver on a single layer

The same homogeneous uniaxial-tension problem as example 1, but run through the
**3D** code path (`dimensionality: 3d`) on a grid one voxel deep (`nz: 1`).

This is a consistency check between the 2D and 3D solvers: a 3D grid of thickness
one, with the appropriate lateral stress targets, should reproduce the
corresponding 2D result. Run these alongside example 1 and compare the summary
logs.

Common to all three configs: 128×128×1 voxels at `pixel: 1.0` nm, `E = 70` GPa,
`nu = 0.3`, driven to `eps_target: 0.08` in steps of `1e-4` on `xx`.

| Config | Lateral boundary condition | Compare against |
|---|---|---|
| `plane_strain.yaml` | none — lateral strains held at zero | `1/plane_strain.yaml` |
| `plane_strain_in_zz.yaml` | `yy` relaxed to zero stress | `1/plane_strain_in_zz.yaml` |
| `plane_stress.yaml` | `yy` and `zz` both relaxed to zero stress | `1/plane_stress.yaml` |

## Run

```bash
cd examples/2_uniaxial_tension_2d_3d_homogeneous
python ../../run.py plane_stress.yaml
```
