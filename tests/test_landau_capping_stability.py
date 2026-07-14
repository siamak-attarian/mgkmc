"""Regression guard: the strain-capped Landau model must stay tangent-stable.

The strain capping only prevents material instability if it engages BEFORE
the tangent stiffness loses positive-definiteness along the loading paths the
simulations visit. That ordering depends on the material calibration
(v1..g4), so it must be re-verified whenever the calibration changes.

If this test fails after refitting the Landau coefficients:
  1. run scratch/diagnose_tangent_stability.py for plots and a cap-limit sweep,
  2. set an explicit strain_capping_limit in your configs that restores the
     margin, and update CAP_LIMITS below.
"""
import pytest

from mgkmc.stability import stability_report

# examples/11-gradualtest calibration (see its source_of_parameters.txt).
# Update together with the material calibration.
MATERIAL = dict(
    lam=80.92e9, mu=23.75e9,
    v1=-236.5e9, v2=-27.4e9, v3=11.8e9,
    g1=-5640.7e9, g2=2207.1e9, g3=-332.5e9, g4=-305.2e9,
)

# Floor on the smallest tangent eigenvalue of the capped model along every
# path (Pa). Positive-definiteness with real margin, and a bound on the
# FFT solver's contrast.
MIN_EIG_FLOOR = 2.0e9

# Production capping settings: keep in sync with the simulation configs.
CAPPING = dict(capping_type="smooth", smooth_power=1.0)

# Recommended explicit strain_capping_limit for production configs with this
# calibration (from the sweep in scratch/diagnose_tangent_stability.py).
# NOTE: the _cap_strain_3d default (0.1248 for this calibration) has ~0%
# margin on the equibiaxial paths — stable in the end, but with no headroom;
# that is why it is only included in the positivity test below, not the
# margin test.
RECOMMENDED_CAP_LIMIT = 0.10

# Fraction of the load-to-instability that must remain when the cap engages.
MIN_MARGIN = 0.10

T_MAX, N = 0.20, 40


@pytest.mark.parametrize("cap_limit", [None, RECOMMENDED_CAP_LIMIT])
def test_capped_tangent_stays_positive(cap_limit):
    rows = stability_report(**MATERIAL, cap_limit=cap_limit,
                            t_max=T_MAX, n=N, **CAPPING)
    for r in rows:
        assert not r["capped_truncated"], (
            f"capped solve diverged on path '{r['path']}' "
            f"(cap_limit={cap_limit})")
        assert r["capped_min_eig"] > MIN_EIG_FLOOR, (
            f"capped tangent too soft on path '{r['path']}': "
            f"min eig = {r['capped_min_eig']/1e9:.2f} GPa "
            f"(cap_limit={cap_limit})")


def test_recommended_cap_engages_with_margin():
    rows = stability_report(**MATERIAL, cap_limit=RECOMMENDED_CAP_LIMIT,
                            t_max=T_MAX, n=N, **CAPPING)
    for r in rows:
        if r["t_unstable"] is None:
            continue  # path never goes unstable in range: nothing to protect
        assert r["t_engage"] is not None, (
            f"path '{r['path']}' goes unstable at t={r['t_unstable']:.3f} "
            f"but the cap never engages")
        assert r["margin"] >= MIN_MARGIN, (
            f"path '{r['path']}': cap engages at t={r['t_engage']:.3f} vs "
            f"instability at t={r['t_unstable']:.3f} — margin "
            f"{r['margin']:.1%} < required {MIN_MARGIN:.0%}")
