# 1 — Uniaxial tension, 2D, homogeneous

Elastic-only baseline. A homogeneous 128×128 sheet is pulled along **x** while the
spectral solver enforces mechanical equilibrium; no STZ events occur. This is the
reference case to check the FFT solver against before adding heterogeneity or
plasticity.

Common to all three configs: `simulation_type: linear_elastic`, 128×128 voxels at
`pixel: 1.0` nm, `E = 70` GPa, `nu = 0.3`, driven to `eps_target: 0.08` in steps
of `1e-4` on the `xx` component.

| Config | Plane mode | Lateral boundary condition |
|---|---|---|
| `plane_strain.yaml` | `plane_strain` | none — all lateral strains held at zero |
| `plane_strain_in_zz.yaml` | `plane_strain` | `yy` relaxed to zero stress; `zz` is the constrained direction |
| `plane_stress.yaml` | `plane_stress` | `yy` relaxed to zero stress |

The three differ only in how the non-driven components are treated, so comparing
their stress-strain slopes shows the effect of the plane-mode choice on the
apparent stiffness.

## Run

```bash
cd examples/1_uniaxial_tension_2d_homogeneous
python ../../run.py plane_stress.yaml
```

Running from inside the example directory keeps the output folder
(`output.directory` in the config) next to the config rather than at the repo
root. Results are written as a summary log, a global log, and `.vtu` files.
