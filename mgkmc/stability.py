"""Tangent-stiffness stability diagnostics for the Landau small-strain model.

The strain capping bounds the deviatoric equivalent strain E_eq, so it only
prevents material instability if it engages *before* the tangent stiffness
loses positive-definiteness along the loading paths the simulation visits.
This module checks that, at a single material point, for a set of
representative plane-stress paths.

Used by scratch/diagnose_tangent_stability.py (plots, cap-limit sweeps) and
tests/test_landau_capping_stability.py (regression guard for new material
calibrations).
"""
import numpy as np

from .elasticity import stress_from_strain_landau_2d


def default_cap_limit(mu, g4):
    """The E_cap used by _cap_strain_3d when strain_capping_limit is unset."""
    mu_val = float(np.mean(mu))
    if g4 < 0:
        return float(np.sqrt(0.2 * mu_val / abs(g4)))
    return 999.0


def _e_eq_3d(e11, e22, e12, e33):
    """Deviatoric equivalent strain of the full 3D tensor (as _cap_strain_3d)."""
    tr = e11 + e22 + e33
    d11, d22, d33 = e11 - tr / 3.0, e22 - tr / 3.0, e33 - tr / 3.0
    return float(np.sqrt(max(0.0, (2.0 / 3.0) * (d11**2 + d22**2 + d33**2
                                                 + 2.0 * e12**2))))


class LandauPoint:
    """Single-material-point Landau model for tangent diagnostics."""

    def __init__(self, lam, mu, v1, v2, v3, g1, g2, g3, g4,
                 plane_mode="plane_stress",
                 capping_enabled=False, cap_limit=None,
                 capping_type="piecewise", tangent_ratio=0.1,
                 smooth_power=1.0):
        self.lam = np.full((1, 1), float(lam))
        self.mu = np.full((1, 1), float(mu))
        self.pars = (v1, v2, v3, g1, g2, g3, g4)
        self.kw = dict(
            plane_mode=plane_mode,
            strain_capping_enabled=capping_enabled,
            strain_capping_limit=cap_limit,
            strain_capping_type=capping_type,
            strain_capping_tangent_ratio=tangent_ratio,
            strain_capping_smooth_power=smooth_power,
        )
        self.cap_limit = (float(cap_limit) if cap_limit is not None and cap_limit > 0
                          else default_cap_limit(mu, g4))

    def stress(self, e11, e22, e12, return_e33=False):
        eps = np.zeros((1, 1, 2, 2))
        eps[0, 0] = [[e11, e12], [e12, e22]]
        state = {}
        sig = stress_from_strain_landau_2d(
            eps, self.lam, self.mu, *self.pars, e33_state=state, **self.kw)
        if return_e33:
            e33 = float(state["e33"][0, 0]) if "e33" in state else 0.0
            return sig[0, 0], e33
        return sig[0, 0]

    def tangent_mandel(self, e11, e22, e12, h=1e-6):
        """3x3 tangent dsigma/deps in Mandel notation [11, 22, sqrt(2)*12],
        by central finite differences."""
        def sig_m(v):
            s = self.stress(v[0], v[1], v[2] / np.sqrt(2.0))
            return np.array([s[0, 0], s[1, 1], np.sqrt(2.0) * s[0, 1]])
        x0 = np.array([e11, e22, np.sqrt(2.0) * e12])
        C = np.zeros((3, 3))
        for j in range(3):
            dp = x0.copy(); dp[j] += h
            dm = x0.copy(); dm[j] -= h
            C[:, j] = (sig_m(dp) - sig_m(dm)) / (2.0 * h)
        return C

    def solve_uniaxial_e22(self, e11, guess=0.0):
        """eps22 such that sig22 = 0 (scalar secant, warm-startable guess)."""
        e22 = guess
        s = self.stress(e11, e22, 0.0)[1, 1]
        e22b = e22 - s / 1.5e11
        for _ in range(60):
            sb = self.stress(e11, e22b, 0.0)[1, 1]
            if abs(sb) < 1.0:      # 1 Pa
                return e22b
            denom = sb - s
            if denom == 0 or not np.isfinite(denom):
                break
            e22, s, e22b = e22b, sb, e22b - sb * (e22b - e22) / denom
        return e22b


# name -> callable t -> (e11, e22, e12); "uniaxial_stress" is special-cased
# (e22 solved so that sig_yy = 0, as in the mixed-BC simulations).
DEFAULT_PATHS = {
    "uniaxial_stress": "uniaxial_stress",
    "equibiaxial_tension": lambda t: (t, t, 0.0),
    "equibiaxial_compression": lambda t: (-t, -t, 0.0),
    "pure_shear": lambda t: (0.0, 0.0, t),
    "uniaxial_strain": lambda t: (t, 0.0, 0.0),
}


def scan_path(point, path, t_max=0.2, n=60):
    """March a loading path, recording min tangent eigenvalue and E_eq.
    Truncates (rather than raises) when the model diverges."""
    ts, mineig, eeq = [], [], []
    guess = 0.0
    truncated = False
    for t in np.linspace(t_max / n, t_max, n):
        try:
            if path == "uniaxial_stress":
                e22 = point.solve_uniaxial_e22(t, guess)
                guess = e22
                e11, e12 = t, 0.0
            else:
                e11, e22, e12 = path(t)
            _, e33 = point.stress(e11, e22, e12, return_e33=True)
            C = point.tangent_mandel(e11, e22, e12)
        except FloatingPointError:
            truncated = True
            break
        if not np.all(np.isfinite(C)):
            truncated = True
            break
        w = np.linalg.eigvalsh(0.5 * (C + C.T))
        ts.append(t)
        mineig.append(w[0])
        eeq.append(_e_eq_3d(e11, e22, e12, e33))
    return {"ts": np.array(ts), "min_eig": np.array(mineig),
            "e_eq": np.array(eeq), "truncated": truncated,
            "t_end": ts[-1] if ts else 0.0}


def stability_report(lam, mu, v1, v2, v3, g1, g2, g3, g4,
                     cap_limit=None, t_max=0.2, n=60,
                     plane_mode="plane_stress", capping_type="piecewise",
                     tangent_ratio=0.1, smooth_power=1.0, paths=None,
                     uncapped_scans=None):
    """Per-path stability summary for a material calibration.

    Returns a list of dicts with keys:
      path            : path name
      t_unstable      : load at which the UNCAPPED tangent first goes
                        indefinite (or the solve diverges); None if stable
                        throughout [0, t_max]
      t_engage        : load at which E_eq first reaches the cap limit
      margin          : 1 - t_engage/t_unstable  (fraction of load-to-instability
                        left when the cap engages); None if no instability
      capped_min_eig  : smallest tangent eigenvalue along the CAPPED path (Pa)
      capped_truncated: True if the capped solve itself diverged (bad!)

    uncapped_scans : optional dict path-name -> scan result to reuse across
        cap-limit sweeps (the uncapped scan does not depend on the limit).
    """
    mat = (lam, mu, v1, v2, v3, g1, g2, g3, g4)
    paths = paths if paths is not None else DEFAULT_PATHS
    rows = []
    for name, path in paths.items():
        if uncapped_scans is not None and name in uncapped_scans:
            so = uncapped_scans[name]
        else:
            p_off = LandauPoint(*mat, plane_mode=plane_mode,
                                capping_enabled=False)
            so = scan_path(p_off, path, t_max, n)
            if uncapped_scans is not None:
                uncapped_scans[name] = so
        p_on = LandauPoint(*mat, plane_mode=plane_mode, capping_enabled=True,
                           cap_limit=cap_limit, capping_type=capping_type,
                           tangent_ratio=tangent_ratio,
                           smooth_power=smooth_power)
        sn = scan_path(p_on, path, t_max, n)

        t_unst = None
        neg = so["min_eig"] < 0.0
        if neg.any():
            t_unst = float(so["ts"][int(np.argmax(neg))])
        elif so["truncated"]:
            t_unst = float(so["t_end"]) if so["t_end"] > 0 else None

        eng = sn["e_eq"] >= p_on.cap_limit
        t_eng = float(sn["ts"][int(np.argmax(eng))]) if eng.any() else None

        margin = None
        if t_eng is not None and t_unst is not None and t_unst > 0:
            margin = 1.0 - t_eng / t_unst

        rows.append({
            "path": name,
            "t_unstable": t_unst,
            "t_engage": t_eng,
            "margin": margin,
            "capped_min_eig": (float(sn["min_eig"].min())
                               if len(sn["min_eig"]) else float("nan")),
            "capped_truncated": sn["truncated"],
        })
    return rows
