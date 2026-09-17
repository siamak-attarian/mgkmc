# 4 — Uniaxial tension with KMC plasticity

The first examples that actually run the Shear Transformation Zone model. Each
voxel carries `M: 20` candidate shear orientations with Gaussian-distributed
activation barriers; the local stress biases those barriers, a Kinetic Monte
Carlo step selects an event, the chosen voxel is sheared by `gamma0`, the FFT
solver re-equilibrates, and softening lowers the barriers near the flipped voxel.

Two parameter sets are provided, each in three variants.

## Variants (present in both `Example1/` and `Example2/`)

| Config | What it runs |
|---|---|
| `kmc_config.yaml` | 2D plane stress, 128×128, exact full-FFT stress update after every event |
| `kmc_config_fast_patching.yaml` | identical physics, but `fast_patching.enabled: true` — the stress update is applied through a precomputed local kernel (radius 3) and resynchronised with a full FFT every 100 steps |
| `kmc_3d.yaml` | the 3D code path on a 128×128×1 grid |

The `kmc_config` / `kmc_config_fast_patching` pair exists to measure what the
fast-patching approximation costs in accuracy and buys in speed — run both and
compare the summary logs.

## Parameter sets

| | `Example1/` | `Example2/` |
|---|---|---|
| voxel size `pixel` | 0.7 nm | 0.763 nm |
| transformation strain `gamma0` | 0.14 | 0.117 |
| plastic softening `jp` | 21.7 | 117.8 |
| transient softening `jt` | 3.52 | 0 (disabled) |
| softening cap | 1.082 eV | 1.013 eV |
| applied strain rate | 1e8 s⁻¹ | 1e9 s⁻¹ |

Both run at `temperature: 10.0` K to `eps_target: 0.12`, with Gaussian barriers
of mean 1.8 eV, and isotropic softening.

`Example2` disables transient (`jt`) softening and compensates with a much larger
plastic coupling, so the two sets bracket rather different softening regimes.

## Run

```bash
cd examples/4-uniaxial_tension_kmc/Example1
python ../../../run.py kmc_config.yaml
```

These are 128×128 KMC runs to 12% strain and take considerably longer than the
elastic examples 1–3.
