"""
Equivalence tests: Mandel-notation tangent assembly / permutation push-forward
(`tangent_method="mandel"`, default) against the original einsum-dyad path
(`tangent_method="reference"`) in constitutive_hyperelastic_2d / _3d.

The two paths compute the same algebra in different representations, so they
must agree to floating-point reordering error (<< 1e-10 relative).
"""
import numpy as np
import pytest

from mgkmc.finite_strain_simulator import (
    constitutive_hyperelastic_2d, constitutive_hyperelastic_3d,
    _make_identity_tensors_2d, _make_identity_tensors_3d,
    build_C4_2d, build_C4_3d,
)

GPa = 1e9
LANDAU = dict(v1=-97.4 * GPa, v2=-67.0 * GPa, v3=-19.1 * GPa,
              g1=5472.5 * GPa, g2=-1209.0 * GPa, g3=348.7 * GPa,
              g4=-238.3 * GPa)
LAM, MU = 80.9 * GPa, 23.75 * GPa
E_MOD = MU * (3 * LAM + 2 * MU) / (LAM + MU)
NU = LAM / (2 * (LAM + MU))

RTOL = 1e-10


def _rel_diff(a, b):
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1.0)
    return np.max(np.abs(a - b)) / scale


def _setup_2d(nx=8, ny=8, seed=0, amp=0.08, plane_mode="plane_stress"):
    rng = np.random.default_rng(seed)
    E_f = E_MOD * (1.0 + 0.2 * (2 * rng.random((nx, ny)) - 1))
    nu_f = np.full((nx, ny), NU)
    I2, I4, I4rt, I4s, II = _make_identity_tensors_2d(nx, ny)
    C4 = build_C4_2d(E_f, nu_f, I4s, II, plane_mode=plane_mode)
    F = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny))) \
        + amp * (2 * rng.random((2, 2, nx, ny)) - 1)
    return F, C4, I2, I4, I4rt


def _setup_3d(n=5, seed=0, amp=0.08):
    rng = np.random.default_rng(seed)
    E_f = E_MOD * (1.0 + 0.2 * (2 * rng.random((n, n, n)) - 1))
    nu_f = np.full((n, n, n), NU)
    I2, I4, I4rt, I4s, II = _make_identity_tensors_3d(n, n, n)
    C4 = build_C4_3d(E_f, nu_f, I4s, II)
    F = np.einsum('ij,xyz->ijxyz', np.eye(3), np.ones((n, n, n))) \
        + amp * (2 * rng.random((3, 3, n, n, n)) - 1)
    return F, C4, I2, I4, I4rt


CAPPING_CASES = [
    dict(strain_capping_enabled=False),
    dict(strain_capping_enabled=True, strain_capping_limit=0.05,
         strain_capping_type="piecewise"),
    dict(strain_capping_enabled=True, strain_capping_limit=0.05,
         strain_capping_type="smooth", strain_capping_smooth_power=1.0),
]


@pytest.mark.parametrize("plane_mode", ["plane_stress", "plane_strain"])
@pytest.mark.parametrize("capping", CAPPING_CASES,
                         ids=["nocap", "piecewise", "smooth"])
@pytest.mark.parametrize("with_fp", [False, True], ids=["noFp", "Fp"])
def test_landau_2d_mandel_vs_reference(plane_mode, capping, with_fp):
    F, C4, I2, I4, I4rt = _setup_2d(plane_mode=plane_mode)
    Fp = None
    if with_fp:
        rng = np.random.default_rng(7)
        Fp = np.einsum('ij,xy->ijxy', np.eye(2), np.ones(F.shape[2:])) \
             + 0.03 * (2 * rng.random(F.shape) - 1)
    kw = dict(Fp=Fp, model_type="landau", plane_mode=plane_mode,
              **LANDAU, **capping)
    P_m, K4_m, F33_m = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, tangent_method="mandel", **kw)
    P_r, K4_r, F33_r = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, tangent_method="reference", **kw)
    assert _rel_diff(P_m, P_r) < RTOL
    assert _rel_diff(F33_m, F33_r) < RTOL
    assert _rel_diff(K4_m, K4_r) < RTOL


@pytest.mark.parametrize("capping", CAPPING_CASES,
                         ids=["nocap", "piecewise", "smooth"])
@pytest.mark.parametrize("with_fp", [False, True], ids=["noFp", "Fp"])
def test_landau_3d_mandel_vs_reference(capping, with_fp):
    F, C4, I2, I4, I4rt = _setup_3d()
    Fp = None
    if with_fp:
        rng = np.random.default_rng(7)
        Fp = np.einsum('ij,xyz->ijxyz', np.eye(3), np.ones(F.shape[2:])) \
             + 0.03 * (2 * rng.random(F.shape) - 1)
    kw = dict(Fp=Fp, model_type="landau", **LANDAU, **capping)
    P_m, K4_m = constitutive_hyperelastic_3d(
        F, C4, I2, I4, I4rt, tangent_method="mandel", **kw)
    P_r, K4_r = constitutive_hyperelastic_3d(
        F, C4, I2, I4, I4rt, tangent_method="reference", **kw)
    assert _rel_diff(P_m, P_r) < RTOL
    assert _rel_diff(K4_m, K4_r) < RTOL


@pytest.mark.parametrize("model,extra", [
    ("svk", {}),
    ("neo_hookean", {}),
    ("murnaghan", dict(A_m=-100.0 * GPa, B_m=-50.0 * GPa, C_m=-30.0 * GPa)),
], ids=["svk", "neo_hookean", "murnaghan"])
@pytest.mark.parametrize("plane_mode", ["plane_stress", "plane_strain"])
def test_pushforward_permutations_2d(model, extra, plane_mode):
    """The permutation push-forward applies to every model (Fp=None path)."""
    F, C4, I2, I4, I4rt = _setup_2d(plane_mode=plane_mode, amp=0.05)
    kw = dict(model_type=model, plane_mode=plane_mode, **extra)
    P_m, K4_m, _ = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, tangent_method="mandel", **kw)
    P_r, K4_r, _ = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, tangent_method="reference", **kw)
    assert _rel_diff(P_m, P_r) < RTOL
    assert _rel_diff(K4_m, K4_r) < RTOL


def test_pushforward_permutations_3d_svk():
    F, C4, I2, I4, I4rt = _setup_3d(amp=0.05)
    P_m, K4_m = constitutive_hyperelastic_3d(
        F, C4, I2, I4, I4rt, model_type="svk", tangent_method="mandel")
    P_r, K4_r = constitutive_hyperelastic_3d(
        F, C4, I2, I4, I4rt, model_type="svk", tangent_method="reference")
    assert _rel_diff(P_m, P_r) < RTOL
    assert _rel_diff(K4_m, K4_r) < RTOL
