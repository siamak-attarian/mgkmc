"""The scalar-field Landau stress must reproduce the tensor reference exactly.

stress_from_strain_landau_2d (scalar-field, production) vs
stress_from_strain_landau_2d_reference (tensor, oracle): same math, different
arithmetic ordering, so agreement must be at floating-point roundoff level —
NOT solver-tolerance level. Any real discrepancy fails loudly.
"""
import numpy as np
import pytest

from mgkmc.elasticity import (stress_from_strain_landau_2d,
                              stress_from_strain_landau_2d_reference)

# examples/11-gradualtest calibration
PARS = (-236.5e9, -27.4e9, 11.8e9, -5640.7e9, 2207.1e9, -332.5e9, -305.2e9)

CAPPING_CASES = [
    None,
    ("piecewise", 1.0),
    ("smooth", 1.0),
    ("smooth", 2.0),
    ("smooth", 4.0),
]


def _fields(seed, scale, nx=32, ny=32, symmetric=True):
    rng = np.random.default_rng(seed)
    lam = 80.92e9 * (1 + 0.10 * rng.standard_normal((nx, ny)))
    mu = 23.75e9 * (1 + 0.10 * rng.standard_normal((nx, ny)))
    eps = scale * rng.standard_normal((nx, ny, 2, 2))
    if symmetric:
        eps[..., 1, 0] = eps[..., 0, 1]
    return lam, mu, eps


def _kw(capping):
    if capping is None:
        return {}
    return dict(strain_capping_enabled=True,
                strain_capping_type=capping[0],
                strain_capping_smooth_power=capping[1],
                strain_capping_limit=0.10,
                strain_capping_tangent_ratio=0.1)


@pytest.mark.parametrize("plane_mode", ["plane_strain", "plane_stress"])
@pytest.mark.parametrize("capping", CAPPING_CASES)
def test_scalar_matches_reference(plane_mode, capping):
    # Uncapped Landau is unstable at large strain: keep the uncapped case in
    # the stable regime; drive the capped cases well past the cap limit.
    scale = 0.008 if capping is None else 0.06
    lam, mu, eps = _fields(seed=3, scale=scale)
    kw = _kw(capping)
    ref = stress_from_strain_landau_2d_reference(
        eps, lam, mu, *PARS, plane_mode=plane_mode, **kw)
    new = stress_from_strain_landau_2d(
        eps, lam, mu, *PARS, plane_mode=plane_mode, **kw)
    rel = np.max(np.abs(new - ref)) / (np.max(np.abs(ref)) + 1.0)
    assert rel < 1e-10, f"rel diff {rel:.3e} ({plane_mode}, capping={capping})"


def test_scalar_matches_reference_asymmetric_input():
    # The reference never symmetrizes its input; the scalar form must not either.
    lam, mu, eps = _fields(seed=7, scale=0.006, symmetric=False)
    ref = stress_from_strain_landau_2d_reference(
        eps, lam, mu, *PARS, plane_mode="plane_stress")
    new = stress_from_strain_landau_2d(
        eps, lam, mu, *PARS, plane_mode="plane_stress")
    rel = np.max(np.abs(new - ref)) / (np.max(np.abs(ref)) + 1.0)
    assert rel < 1e-10, f"rel diff {rel:.3e}"


def test_e33_warm_start_matches_reference():
    lam, mu, eps = _fields(seed=11, scale=0.05)
    kw = _kw(("smooth", 1.0))
    st_ref, st_new = {}, {}
    for eps_k in [eps, eps * 1.002, eps]:   # prime / perturb / return
        ref = stress_from_strain_landau_2d_reference(
            eps_k, lam, mu, *PARS, plane_mode="plane_stress",
            e33_state=st_ref, **kw)
        new = stress_from_strain_landau_2d(
            eps_k, lam, mu, *PARS, plane_mode="plane_stress",
            e33_state=st_new, **kw)
    rel = np.max(np.abs(new - ref)) / (np.max(np.abs(ref)) + 1.0)
    assert rel < 1e-10, f"rel diff {rel:.3e}"
    de33 = np.max(np.abs(st_new["e33"] - st_ref["e33"]))
    assert de33 < 1e-12, f"e33 state diff {de33:.3e}"
