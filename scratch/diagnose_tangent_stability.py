"""Single-point tangent-stiffness stability diagnostic for the Landau model.

Question this answers: does the material tangent go soft/unstable through the
VOLUMETRIC channel (v1, g1, and the v2*I1 / g2*I1^2 shear-degradation terms)
before the deviatoric strain reaches the cap E_cap? The strain capping only
bounds the deviatoric equivalent strain, so any instability arriving through
I1 coupling is not prevented by it.

Method: along several 2-D plane-stress loading paths, compute the in-plane
tangent stiffness dsigma/deps of stress_from_strain_landau_2d by central
finite differences in Mandel notation [e11, e22, sqrt(2) e12], with capping
ON and OFF. Track:
  - min eigenvalue of the symmetrized tangent (instability when < 0),
  - E_eq / E_cap (does the cap engage before or after softening?),
  - volumetric fraction of the softest eigenmode (which channel went soft),
  - asymmetry of the tangent (non-conservative stress in capped pixels).

Usage:  python scratch/diagnose_tangent_stability.py
Writes: scratch/tangent_stability.png + console summary.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mgkmc.elasticity import stress_from_strain_landau_2d

# --- material: examples/11-gradualtest parameters -------------------------
LAM0, MU0 = 80.92e9, 23.75e9
V1, V2, V3 = -236.5e9, -27.4e9, 11.8e9
G1, G2, G3, G4 = -5640.7e9, 2207.1e9, -332.5e9, -305.2e9
E_CAP_DEFAULT = np.sqrt(0.2 * MU0 / abs(G4))   # what _cap_strain_3d uses

# Production capping settings (keep in sync with configs / the test).
CAPPING_TYPE = "smooth"
SMOOTH_POWER = 1.0

lam = np.full((1, 1), LAM0)
mu = np.full((1, 1), MU0)


def landau_sig(eps11, eps22, eps12, capping, e33_out=None):
    """Macro stress (2x2) at a single material point, plane stress."""
    eps = np.zeros((1, 1, 2, 2))
    eps[0, 0] = [[eps11, eps12], [eps12, eps22]]
    state = {}
    sig = stress_from_strain_landau_2d(
        eps, lam, mu, V1, V2, V3, G1, G2, G3, G4,
        plane_mode="plane_stress",
        strain_capping_enabled=capping,
        strain_capping_type=CAPPING_TYPE,
        strain_capping_smooth_power=SMOOTH_POWER,
        e33_state=state)
    if e33_out is not None:
        e33_out.append(float(state["e33"][0, 0]))
    return sig[0, 0]


def mandel_tangent(e11, e22, e12, capping, h=1e-6):
    """3x3 tangent dsigma/deps in Mandel notation via central differences."""
    def sig_m(v):
        s = landau_sig(v[0], v[1], v[2] / np.sqrt(2.0), capping)
        return np.array([s[0, 0], s[1, 1], np.sqrt(2.0) * s[0, 1]])
    x0 = np.array([e11, e22, np.sqrt(2.0) * e12])
    C = np.zeros((3, 3))
    for j in range(3):
        dp = x0.copy(); dp[j] += h
        dm = x0.copy(); dm[j] -= h
        C[:, j] = (sig_m(dp) - sig_m(dm)) / (2.0 * h)
    return C


def e_eq_3d(e11, e22, e12, e33):
    """Deviatoric equivalent strain of the full 3D tensor (matches _cap_strain_3d)."""
    tr = e11 + e22 + e33
    d11, d22, d33 = e11 - tr / 3, e22 - tr / 3, e33 - tr / 3
    return np.sqrt(np.maximum(0.0, (2.0 / 3.0) * (d11**2 + d22**2 + d33**2 + 2 * e12**2)))


def solve_uniaxial_eps22(e11, capping, guess):
    """eps22 such that sig22 = 0 (scalar secant)."""
    e22 = guess
    s = landau_sig(e11, e22, 0.0, capping)[1, 1]
    e22b = e22 - s / (1.5e11)          # linear-stiffness first step
    for _ in range(60):
        sb = landau_sig(e11, e22b, 0.0, capping)[1, 1]
        if abs(sb) < 1.0:              # 1 Pa
            return e22b
        denom = (sb - s)
        if denom == 0 or not np.isfinite(denom):
            break
        e22, s, e22b = e22b, sb, e22b - sb * (e22b - e22) / denom
    return e22b


# --- loading paths ---------------------------------------------------------
def path_uniaxial_stress(t, mem={ "e22": 0.0 }, capping=True):
    e22 = solve_uniaxial_eps22(t, capping, mem["e22"])
    mem["e22"] = e22
    return t, e22, 0.0

PATHS = {
    "uniaxial stress (sig_yy=0)": "uniaxial",
    "equibiaxial tension":        lambda t: (t, t, 0.0),
    "equibiaxial compression":    lambda t: (-t, -t, 0.0),
    "pure shear":                 lambda t: (0.0, 0.0, t),
    "uniaxial strain (confined)": lambda t: (t, 0.0, 0.0),
}

T_MAX, N = 0.20, 80
VOL_DIR = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
axes = axes.ravel()
summary = []

for ip, (name, pf) in enumerate(PATHS.items()):
    ax = axes[ip]
    for capping, color in [(False, "tab:red"), (True, "tab:blue")]:
        ts, mineig, eeq_ratio, volfrac, asym = [], [], [], [], []
        mem = {"e22": 0.0}
        for t in np.linspace(1e-4, T_MAX, N):
            try:
                if pf == "uniaxial":
                    e11, e22, e12 = path_uniaxial_stress(t, mem, capping)
                else:
                    e11, e22, e12 = pf(t)
                e33_box = []
                landau_sig(e11, e22, e12, capping, e33_out=e33_box)
                C = mandel_tangent(e11, e22, e12, capping)
            except FloatingPointError:
                break
            Cs = 0.5 * (C + C.T)
            w, v = np.linalg.eigh(Cs)
            n = v[:, 0]
            ts.append(t)
            mineig.append(w[0] / 1e9)
            eeq_ratio.append(e_eq_3d(e11, e22, e12, e33_box[0]) / E_CAP_DEFAULT)
            volfrac.append(float(np.dot(n, VOL_DIR) ** 2))
            asym.append(np.linalg.norm(C - C.T) / max(np.linalg.norm(C), 1.0))
        ts, mineig, eeq_ratio = map(np.array, (ts, mineig, eeq_ratio))
        lbl = "capped" if capping else "uncapped"
        ax.plot(ts, mineig, color=color, label=f"min eig, {lbl}")
        if capping:
            ax2 = ax.twinx()
            ax2.plot(ts, eeq_ratio, "g--", alpha=0.6, label="E_eq/E_cap")
            ax2.axhline(1.0, color="g", lw=0.5, alpha=0.5)
            ax2.set_ylabel("E_eq / E_cap", color="g")
            ax2.set_ylim(0, max(2.0, eeq_ratio.max() * 1.1))
        # summary bookkeeping (both capping states)
        i_neg = np.argmax(mineig < 0) if np.any(mineig < 0) else None
        i_cap = np.argmax(eeq_ratio >= 1.0) if np.any(eeq_ratio >= 1.0) else None
        t_neg = ts[i_neg] if i_neg is not None else None
        t_cap = ts[i_cap] if i_cap is not None else None
        vf = volfrac[i_neg] if i_neg is not None else (volfrac[-1] if volfrac else float("nan"))
        summary.append((f"{name} [{'cap' if capping else 'nocap'}]", t_neg, t_cap, vf,
                        float(np.max(asym)) if asym else float("nan"),
                        float(mineig.min()) if len(mineig) else float("nan"),
                        float(ts[-1]) if len(ts) else float("nan")))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(0.1 * MU0 / 1e9, color="gray", ls=":", lw=0.8)  # residual G_t scale
    ax.set_ylim(-150, 60)   # post-instability uncapped values are garbage; clip
    ax.set_title(name)
    ax.set_xlabel("load parameter t")
    ax.set_ylabel("min tangent eigenvalue (GPa)")
    ax.legend(loc="lower left", fontsize=8)

axes[-1].axis("off")
fig.suptitle(f"Landau tangent stability, plane stress, capping ON (blue, "
             f"{CAPPING_TYPE} p={SMOOTH_POWER}) vs OFF (red); "
             f"E_cap(default) = {E_CAP_DEFAULT:.4f}", fontsize=12)
fig.tight_layout()
out_png = os.path.join(os.path.dirname(__file__), "tangent_stability.png")
fig.savefig(out_png, dpi=110)
print(f"wrote {out_png}\n")

print(f"{'path':<38} {'t: mineig<0':<12} {'t: E_eq=E_cap':<14} "
      f"{'vol frac of soft mode':<22} {'max asym':<10} {'min eig (GPa)':<14} {'t reached'}")
for name, t_neg, t_cap, vf, asym, me, tend in summary:
    print(f"{name:<38} {str(None if t_neg is None else round(t_neg,4)):<12} "
          f"{str(None if t_cap is None else round(t_cap,4)):<14} "
          f"{vf:<22.2f} {asym:<10.2e} {me:<14.3f} {tend:.3f}")

# --- strain_capping_limit sweep: how much margin does each limit buy? ------
from mgkmc.stability import stability_report, default_cap_limit

uncapped_cache = {}
for power in [1.0, 2.0]:
    print(f"\n=== strain_capping_limit sweep, type={CAPPING_TYPE} p={power} "
          f"(default limit = {default_cap_limit(MU0, G4):.4f}) ===")
    print(f"{'E_cap':<10} {'worst margin':<14} "
          f"{'worst capped min-eig (GPa)':<28} {'worst path'}")
    for cand in [None, 0.12, 0.115, 0.11, 0.105, 0.10, 0.09, 0.08]:
        rows = stability_report(LAM0, MU0, V1, V2, V3, G1, G2, G3, G4,
                                cap_limit=cand, t_max=T_MAX, n=N,
                                capping_type=CAPPING_TYPE, smooth_power=power,
                                uncapped_scans=uncapped_cache)
        margins = [(r["margin"], r["path"]) for r in rows if r["margin"] is not None]
        worst_m, worst_path = min(margins) if margins else (None, "-")
        worst_e = min(r["capped_min_eig"] for r in rows)
        bad = any(r["capped_truncated"] for r in rows)
        lbl = "default" if cand is None else f"{cand:.3f}"
        print(f"{lbl:<10} {worst_m:<14.1%} {worst_e/1e9:<28.2f} {worst_path}"
              + ("  [CAPPED SOLVE DIVERGED]" if bad else ""))
