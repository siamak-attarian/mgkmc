"""3-D tangent-stability check for the Landau model with smooth strain capping.

Companion to diagnose_tangent_stability.py (2-D plane stress). In 3-D there is
no plane-stress Poisson channel converting volumetric loading into deviatoric
strain, so paths with small E_eq get no protection from the (deviatoric-only)
cap. This scans mixed triaxial paths and reports whether the capped tangent
stays positive-definite anyway.

Finding for the examples/11-gradualtest calibration (2026-07-12): all paths
capped-stable (min eig 6.2-47 GPa to t=0.2); the volumetric-heavy paths are
intrinsically stable even uncapped (the positive g2 cubic dominates), so the
architectural "volumetric hole" of the cap is not exercised by this
calibration. Re-run when v1..g4 change.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mgkmc.elasticity import stress_from_strain_landau_3d

LAM0, MU0 = 80.92e9, 23.75e9
PARS = (-236.5e9, -27.4e9, 11.8e9, -5640.7e9, 2207.1e9, -332.5e9, -305.2e9)
CAPPING_TYPE = "smooth"
SMOOTH_POWER = 1.0
E_CAP = np.sqrt(0.2 * MU0 / abs(PARS[-1]))
T_MAX, N = 0.20, 50
SQ2 = np.sqrt(2.0)

lam = np.full((1, 1, 1), LAM0)
mu = np.full((1, 1, 1), MU0)


def stress(v, capping):
    """Mandel [e11,e22,e33,sq2*e23,sq2*e13,sq2*e12] -> Mandel stress."""
    eps = np.zeros((1, 1, 1, 3, 3))
    eps[..., 0, 0], eps[..., 1, 1], eps[..., 2, 2] = v[0], v[1], v[2]
    eps[..., 1, 2] = eps[..., 2, 1] = v[3] / SQ2
    eps[..., 0, 2] = eps[..., 2, 0] = v[4] / SQ2
    eps[..., 0, 1] = eps[..., 1, 0] = v[5] / SQ2
    s = stress_from_strain_landau_3d(
        eps, lam, mu, *PARS,
        strain_capping_enabled=capping,
        strain_capping_type=CAPPING_TYPE,
        strain_capping_smooth_power=SMOOTH_POWER)[0, 0, 0]
    return np.array([s[0, 0], s[1, 1], s[2, 2],
                     SQ2 * s[1, 2], SQ2 * s[0, 2], SQ2 * s[0, 1]])


def min_eig(v, capping, h=1e-6):
    C = np.zeros((6, 6))
    for j in range(6):
        dp, dm = v.copy(), v.copy()
        dp[j] += h
        dm[j] -= h
        C[:, j] = (stress(dp, capping) - stress(dm, capping)) / (2 * h)
    return np.linalg.eigvalsh(0.5 * (C + C.T))[0]


def e_eq(v):
    e = np.array([[v[0], v[5]/SQ2, v[4]/SQ2],
                  [v[5]/SQ2, v[1], v[3]/SQ2],
                  [v[4]/SQ2, v[3]/SQ2, v[2]]])
    d = e - np.trace(e) / 3.0 * np.eye(3)
    return np.sqrt(2.0 / 3.0 * np.sum(d * d))


PATHS = {
    "uniaxial strain 3D":      lambda t: np.array([t, 0, 0, 0, 0, 0.0]),
    "triax tension (0.3,0.3)": lambda t: np.array([t, 0.3*t, 0.3*t, 0, 0, 0.0]),
    "uniax-like (-0.3,-0.3)":  lambda t: np.array([t, -0.3*t, -0.3*t, 0, 0, 0.0]),
    "hydro + shear":           lambda t: np.array([t/3, t/3, t/3, 0, 0, 0.5*t]),
    "equitriaxial tension":    lambda t: np.array([t/2, t/2, t/2, 0, 0, 0.0]),
    "pure shear 3D":           lambda t: np.array([0, 0, 0, 0, 0, t]),
}

print(f"{'path':<26} {'t: uncapped mineig<0':<22} "
      f"{'capped min-eig (GPa)':<22} {'E_eq/E_cap at t_max'}")
for name, pf in PATHS.items():
    t_unst, worst = None, np.inf
    for t in np.linspace(T_MAX / N, T_MAX, N):
        try:
            e = min_eig(pf(t), False)
        except FloatingPointError:
            e = -1.0
        if e < 0 and t_unst is None:
            t_unst = t
    for t in np.linspace(T_MAX / N, T_MAX, N):
        try:
            worst = min(worst, min_eig(pf(t), True))
        except FloatingPointError:
            worst = float("-inf")
            break
    ratio = e_eq(pf(T_MAX)) / E_CAP
    print(f"{name:<26} {str(None if t_unst is None else round(t_unst, 3)):<22} "
          f"{worst/1e9:<22.2f} {ratio:.2f}")
