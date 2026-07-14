"""
Plane-stress consistency of the capped finite-strain landau law.

The E33 Newton must satisfy S33 = 0 for the SAME capped law the stress
section evaluates (mirroring the small-strain implementation). The check
evaluates the capped S33 independently, reusing the small-strain capping
map `_cap_strain_3d` from elasticity.py — which also pins the two modules
to the same capping semantics.
"""
import numpy as np
import pytest

from mgkmc.finite_strain_simulator import (
    constitutive_hyperelastic_2d, _make_identity_tensors_2d, build_C4_2d,
)
from mgkmc.elasticity import _cap_strain_3d

GPa = 1e9
LANDAU = dict(v1=-97.4 * GPa, v2=-67.0 * GPa, v3=-19.1 * GPa,
              g1=5472.5 * GPa, g2=-1209.0 * GPa, g3=348.7 * GPa,
              g4=-238.3 * GPa)
LAM, MU = 80.9 * GPa, 23.75 * GPa
E_MOD = MU * (3 * LAM + 2 * MU) / (LAM + MU)
NU = LAM / (2 * (LAM + MU))

# The user's production capping configuration
CAPPING = dict(strain_capping_enabled=True,
               strain_capping_limit=0.10,
               strain_capping_tangent_ratio=0.1,
               strain_capping_smooth_power=1.0)


def _capped_S33(E_GL_2d, E33, capping):
    """Independent evaluation of the capped S33 via elasticity._cap_strain_3d."""
    nx, ny = E33.shape
    E_3d = np.zeros((nx, ny, 3, 3))
    E_3d[..., 0, 0] = E_GL_2d[0, 0]
    E_3d[..., 0, 1] = E_GL_2d[0, 1]
    E_3d[..., 1, 0] = E_GL_2d[1, 0]
    E_3d[..., 1, 1] = E_GL_2d[1, 1]
    E_3d[..., 2, 2] = E33
    Ec, w = _cap_strain_3d(
        E_3d, LAM, MU, LANDAU["g4"],
        capping["strain_capping_enabled"], capping["strain_capping_limit"],
        capping["strain_capping_tangent_ratio"], capping["strain_capping_type"],
        capping["strain_capping_smooth_power"])
    I1 = np.trace(Ec, axis1=-2, axis2=-1)
    Ec2 = np.einsum('...ij,...jk->...ik', Ec, Ec)
    I2 = np.trace(Ec2, axis1=-2, axis2=-1)
    I3 = np.trace(np.einsum('...ij,...jk->...ik', Ec2, Ec),
                  axis1=-2, axis2=-1)
    v1, v2, v3 = LANDAU["v1"], LANDAU["v2"], LANDAU["v3"]
    g1, g2, g3, g4 = LANDAU["g1"], LANDAU["g2"], LANDAU["g3"], LANDAU["g4"]
    A = LAM * I1 + 0.5 * v1 * I1**2 + v2 * I2 + (1/6) * g1 * I1**3 \
        + g2 * I1 * I2 + (4/3) * g3 * I3
    B = 2.0 * (MU + v2 * I1 + 0.5 * g2 * I1**2 + g4 * I2)
    C = 4.0 * (v3 + g3 * I1)
    c33 = Ec[..., 2, 2]
    G_t = capping["strain_capping_tangent_ratio"] * MU
    return A + B * c33 + C * c33**2 + 2.0 * G_t * (E33 - c33) * w


@pytest.mark.parametrize("cap_type", ["smooth", "piecewise"])
@pytest.mark.parametrize("amp", [0.05, 0.15], ids=["below_cap", "above_cap"])
def test_capped_plane_stress_S33_is_zero(cap_type, amp):
    nx = ny = 8
    rng = np.random.default_rng(3)
    E_f = np.full((nx, ny), E_MOD)
    nu_f = np.full((nx, ny), NU)
    I2, I4, I4rt, I4s, II = _make_identity_tensors_2d(nx, ny)
    C4 = build_C4_2d(E_f, nu_f, I4s, II, plane_mode="plane_stress")
    F = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))
    F[0, 0] += amp * rng.random((nx, ny))          # stretch beyond/below cap
    F[1, 1] -= 0.3 * amp * rng.random((nx, ny))
    F[0, 1] += 0.3 * amp * (2 * rng.random((nx, ny)) - 1)

    capping = dict(CAPPING, strain_capping_type=cap_type)
    P, K4, F33 = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, model_type="landau",
        plane_mode="plane_stress", **LANDAU, **capping)

    # Recover the converged E33 and the in-plane Green-Lagrange strain
    E33 = 0.5 * (F33**2 - 1.0)
    Ce = np.einsum('jixy,jkxy->ikxy', F, F)
    E_GL_2d = 0.5 * (Ce - I2)

    S33 = _capped_S33(E_GL_2d, E33, capping)
    # 1e-9 Newton tolerance on E33 x O(100 GPa) stiffness -> O(100 Pa)
    assert np.max(np.abs(S33)) < 1e3, \
        f"capped plane-stress violated: max|S33| = {np.max(np.abs(S33)):.3e} Pa"


def test_uncapped_plane_stress_unchanged():
    """Capping disabled must still use the exact closed-coefficient Newton."""
    nx = ny = 8
    rng = np.random.default_rng(5)
    E_f = np.full((nx, ny), E_MOD)
    nu_f = np.full((nx, ny), NU)
    I2, I4, I4rt, I4s, II = _make_identity_tensors_2d(nx, ny)
    C4 = build_C4_2d(E_f, nu_f, I4s, II, plane_mode="plane_stress")
    F = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny))) \
        + 0.03 * (2 * rng.random((2, 2, nx, ny)) - 1)

    P, K4, F33 = constitutive_hyperelastic_2d(
        F, C4, I2, I4, I4rt, model_type="landau",
        plane_mode="plane_stress", **LANDAU)

    E33 = 0.5 * (F33**2 - 1.0)
    nocap = dict(strain_capping_enabled=False, strain_capping_limit=None,
                 strain_capping_tangent_ratio=0.1,
                 strain_capping_type="piecewise",
                 strain_capping_smooth_power=1.0)
    Ce = np.einsum('jixy,jkxy->ikxy', F, F)
    E_GL_2d = 0.5 * (Ce - I2)
    S33 = _capped_S33(E_GL_2d, E33, nocap)
    assert np.max(np.abs(S33)) < 1e3
