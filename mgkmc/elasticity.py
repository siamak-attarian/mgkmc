import numpy as np

def compute_lame_2d(E, nu, plane_mode="plane_strain"):
    """Lamé parameters for 2D isotropic elasticity."""
    mu = E / (2 * (1 + nu))
    if plane_mode == "plane_stress":
        lam_2d = E * nu / (1 - nu**2)
    else:  # plane_strain
        lam_2d = E * nu / ((1 + nu) * (1 - 2 * nu))
    return lam_2d, mu

def stress_from_strain_2d(eps, E, nu, plane_mode="plane_strain"):
    """σ = 2μ ε + λ* trε I, element-centered. eps is (nx, ny, 2, 2)."""
    lam_2d, mu = compute_lame_2d(E, nu, plane_mode)
    # tr_eps is trace over the last two dimensions
    tr_eps = np.trace(eps, axis1=2, axis2=3)[..., None, None]
    I = np.eye(2)[None,None,:,:]
    return 2*mu[...,None,None]*eps + lam_2d[...,None,None]*tr_eps*I

def green_operator_2d(kx, ky, lam0, mu0):
    """
    Fully-correct 2D isotropic Green operator for strain.
    Γ^0_khij(k), spatial dimensions first: (nx, ny, 2, 2, 2, 2)
    """
    nx, ny = kx.shape
    k2 = kx*kx + ky*ky
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0
    
    q = [kx, ky]
    
    Gamma = np.zeros((nx, ny, 2, 2, 2, 2))
    
    A = 1.0/(4*mu0)
    B = (lam0+mu0)/(mu0*(lam0+2*mu0))
    
    for k in range(2):
        for h in range(2):
            for i in range(2):
                for j in range(2):
                    term1 = 0.0
                    if k==i: term1 += q[h]*q[j]
                    if h==i: term1 += q[k]*q[j]
                    if k==j: term1 += q[h]*q[i]
                    if h==j: term1 += q[k]*q[i]
                    term1 = A * term1 / k2_safe
                    
                    term2 = B * (q[k]*q[h]*q[i]*q[j]) / (k2_safe*k2_safe)
                    
                    Gamma[:, :, k, h, i, j] = term1 - term2
    
    Gamma[0, 0, :, :, :, :] = 0
    return Gamma

def compute_lame_3d(E, nu):
    """Lamé parameters for isotropic elasticity."""
    mu = E / (2*(1+nu))
    lam = E*nu / ((1+nu)*(1-2*nu))
    return lam, mu


def stress_from_strain_3d(eps, E, nu):
    """σ = 2μ ε + λ trε I, element-centered."""
    lam, mu = compute_lame_3d(E, nu)
    tr_eps = np.trace(eps, axis1=3, axis2=4)[..., None, None]
    I = np.eye(3)[None,None,None,:,:]
    return 2*mu[...,None,None]*eps + lam[...,None,None]*tr_eps*I


def green_operator_3d(kx, ky, kz, lam0, mu0):
    """
    Fully-correct 3D isotropic Green operator for strain.
    Γ^0_khij(k), with spatial dimensions first: (nx, ny, nz, 3, 3, 3, 3)
    Gamma[x,y,z,k,h,i,j] operates on tau[x,y,z,i,j] to give eps_tilde[x,y,z,k,h]
    """
    nx, ny, nz = kx.shape
    k2 = kx*kx + ky*ky + kz*kz
    k2_safe = k2.copy()
    k2_safe[0,0,0] = 1.0
    
    q = [kx, ky, kz]
    
    Gamma = np.zeros((nx, ny, nz, 3, 3, 3, 3))
    
    A = 1.0/(4*mu0)
    B = (lam0+mu0)/(mu0*(lam0+2*mu0))
    
    for k in range(3):
        for h in range(3):
            for i in range(3):
                for j in range(3):
                    term1 = 0.0
                    if k==i: term1 += q[h]*q[j]
                    if h==i: term1 += q[k]*q[j]
                    if k==j: term1 += q[h]*q[i]
                    if h==j: term1 += q[k]*q[i]
                    term1 = A * term1 / k2_safe
                    
                    term2 = B * (q[k]*q[h]*q[i]*q[j]) / (k2_safe*k2_safe)
                    
                    Gamma[:, :, :, k, h, i, j] = term1 - term2
    
    Gamma[0, 0, 0, :, :, :, :] = 0
    
    return Gamma


# ---------------------------------------------------------------------------
# Secant Elastic Degradation helpers
# ---------------------------------------------------------------------------

def von_mises_strain_2d(eps):
    """
    Von Mises equivalent strain for a 2-D strain field.

    Parameters
    ----------
    eps : ndarray, shape (nx, ny, 2, 2)
        Small-strain tensor field (symmetric).

    Returns
    -------
    eps_eq : ndarray, shape (nx, ny)
        Equivalent strain  eps_eq = sqrt(2/3 * eps' : eps')
        where  eps' = eps - (1/3) tr(eps) I  is the deviatoric part.

    Notes
    -----
    In 2-D plane-strain the out-of-plane component eps_33 = 0 is included
    implicitly: tr(eps) = eps_11 + eps_22, and the 3-D deviatoric formula
    gives  eps'_33 = -(eps_11 + eps_22)/3.
    """
    e11 = eps[:, :, 0, 0]
    e22 = eps[:, :, 1, 1]
    e12 = eps[:, :, 0, 1]
    e21 = eps[:, :, 1, 0]

    # Deviatoric components (plane-strain: eps_33 = 0 => eps'_33 = -(e11+e22)/3)
    tr3  = (e11 + e22) / 3.0
    ep11 = e11 - tr3
    ep22 = e22 - tr3
    ep33 = -(e11 + e22) / 3.0   # out-of-plane deviatoric

    eps_eq = np.sqrt(
        (2.0 / 3.0) * (ep11**2 + ep22**2 + ep33**2
                        + 0.5 * (e12 + e21)**2)
    )
    return eps_eq


def von_mises_strain_3d(eps):
    """
    Von Mises equivalent strain for a 3-D strain field.

    Parameters
    ----------
    eps : ndarray, shape (nx, ny, nz, 3, 3)
        Small-strain tensor field (symmetric).

    Returns
    -------
    eps_eq : ndarray, shape (nx, ny, nz)
        Equivalent strain  eps_eq = sqrt(2/3 * eps' : eps').
    """
    e11 = eps[:, :, :, 0, 0]
    e22 = eps[:, :, :, 1, 1]
    e33 = eps[:, :, :, 2, 2]
    e12 = eps[:, :, :, 0, 1]
    e21 = eps[:, :, :, 1, 0]
    e13 = eps[:, :, :, 0, 2]
    e31 = eps[:, :, :, 2, 0]
    e23 = eps[:, :, :, 1, 2]
    e32 = eps[:, :, :, 2, 1]

    tr3  = (e11 + e22 + e33) / 3.0
    ep11 = e11 - tr3
    ep22 = e22 - tr3
    ep33 = e33 - tr3

    eps_eq = np.sqrt(
        (2.0 / 3.0) * (
            ep11**2 + ep22**2 + ep33**2
            + 0.5 * ((e12 + e21)**2 + (e13 + e31)**2 + (e23 + e32)**2)
        )
    )
    return eps_eq


def secant_shear_field(eps_eq, mu, d, k):
    """
    Pointwise secant shear modulus from the exponential degradation law:

        mu_sec(eps_eq) = mu * [1 - d * (1 - exp(-k * eps_eq))]

    Parameters
    ----------
    eps_eq : ndarray
        Von Mises equivalent strain, any shape.
    mu : float or ndarray broadcastable to eps_eq
        Undegraded shear modulus (Pa).
    d : float
        Degradation magnitude, 0 <= d <= 1.
        d=0 -> no degradation (linear elastic).
        d=1 -> maximum degradation (mu_sec -> 0 as eps_eq -> inf).
    k : float
        Degradation rate (dimensionless; larger k = faster softening onset).

    Returns
    -------
    mu_sec : ndarray (same shape as eps_eq)
        Secant shear modulus field (Pa).
    """
    return mu * (1.0 - d * (1.0 - np.exp(-k * eps_eq)))


def stress_from_strain_secant_2d(eps, lam, mu, d, k,
                                  plane_mode="plane_strain"):
    """
    Secant-elastic Cauchy stress for a 2-D strain field:

        sigma = lam * tr(eps) * I + 2 * mu_sec(eps_eq) * eps

    where mu_sec follows the exponential degradation law and eps_eq is the
    von Mises equivalent strain computed from eps.

    Parameters
    ----------
    eps : ndarray, shape (nx, ny, 2, 2)
        Small-strain tensor field.
    lam : float or ndarray, shape (nx, ny)
        First Lame parameter lambda (Pa).
        For plane stress, pass the effective lambda* = E*nu/(1-nu^2).
    mu : float or ndarray, shape (nx, ny)
        Undegraded shear modulus mu (Pa).
    d : float
        Degradation magnitude (0 <= d <= 1).
    k : float
        Degradation rate (dimensionless).
    plane_mode : str
        "plane_strain" (default) or "plane_stress".  The caller is
        responsible for passing the correct lam for the chosen mode;
        this function uses lam directly.

    Returns
    -------
    sig : ndarray, shape (nx, ny, 2, 2)
        Cauchy stress tensor field (Pa).
    """
    eps_eq  = von_mises_strain_2d(eps)               # (nx, ny)
    mu_sec  = secant_shear_field(eps_eq, mu, d, k)  # (nx, ny)

    tr_eps  = np.trace(eps, axis1=2, axis2=3)[..., None, None]   # (nx,ny,1,1)
    I       = np.eye(2)[None, None, :, :]

    lam_    = np.asarray(lam)[..., None, None]
    mu_sec_ = mu_sec[..., None, None]

    return lam_ * tr_eps * I + 2.0 * mu_sec_ * eps


def stress_from_strain_secant_3d(eps, lam, mu, d, k):
    """
    Secant-elastic Cauchy stress for a 3-D strain field:

        sigma = lam * tr(eps) * I + 2 * mu_sec(eps_eq) * eps

    Parameters
    ----------
    eps : ndarray, shape (nx, ny, nz, 3, 3)
        Small-strain tensor field.
    lam : float or ndarray, shape (nx, ny, nz)
        First Lame parameter lambda (Pa).
    mu : float or ndarray, shape (nx, ny, nz)
        Undegraded shear modulus mu (Pa).
    d : float
        Degradation magnitude (0 <= d <= 1).
    k : float
        Degradation rate (dimensionless).

    Returns
    -------
    sig : ndarray, shape (nx, ny, nz, 3, 3)
        Cauchy stress tensor field (Pa).
    """
    eps_eq  = von_mises_strain_3d(eps)               # (nx, ny, nz)
    mu_sec  = secant_shear_field(eps_eq, mu, d, k)  # (nx, ny, nz)

    tr_eps  = np.trace(eps, axis1=3, axis2=4)[..., None, None]   # (nx,ny,nz,1,1)
    I       = np.eye(3)[None, None, None, :, :]

    lam_    = np.asarray(lam)[..., None, None]
    mu_sec_ = mu_sec[..., None, None]

    return lam_ * tr_eps * I + 2.0 * mu_sec_ * eps


def _cap_strain_3d(eps, lam, mu, g4, strain_capping_enabled, strain_capping_limit, 
                   strain_capping_tangent_ratio, strain_capping_type, strain_capping_smooth_power):
    if not strain_capping_enabled:
        return eps, np.zeros(eps.shape[:-2])
        
    I1 = np.trace(eps, axis1=-2, axis2=-1)
    I_3d = np.eye(3)
    shape_prefix = eps.shape[:-2]
    I_broadcast = np.broadcast_to(I_3d, shape_prefix + (3, 3))
    
    eps_dev = eps - (1.0/3.0) * I1[..., None, None] * I_broadcast
    eps_dev_double_dot = np.einsum('...ij,...ji->...', eps_dev, eps_dev)
    E_eq = np.sqrt(np.maximum(0.0, (2.0 / 3.0) * eps_dev_double_dot))
    
    mu_val = mu.mean() if isinstance(mu, np.ndarray) else mu
    if strain_capping_limit is None or strain_capping_limit <= 0:
        if g4 < 0:
            E_cap_local = np.sqrt(0.2 * mu_val / np.abs(g4))
        else:
            E_cap_local = 999.0
    else:
        E_cap_local = strain_capping_limit
        
    f = np.ones_like(E_eq)
    nonzero = E_eq > 1e-12
    if strain_capping_type == "smooth":
        p = strain_capping_smooth_power
        ratio = E_eq[nonzero] / E_cap_local
        # Clip ratio**p to 100.0 to prevent float overflow when local strain is extremely high
        ratio_p = np.minimum(100.0, ratio**p)
        f[nonzero] = (E_cap_local / E_eq[nonzero]) * (np.tanh(ratio_p))**(1.0/p)
        capping_weight = np.tanh(np.minimum(100.0, (E_eq / E_cap_local)**p))**2
    else:
        is_capped = E_eq > E_cap_local
        np.divide(E_cap_local, E_eq, out=f, where=is_capped)
        capping_weight = is_capped.astype(float)
        
    eps_capped = eps_dev * f[..., None, None] + (1.0/3.0) * I1[..., None, None] * I_broadcast
    return eps_capped, capping_weight
def stress_from_strain_landau_2d_reference(eps, lam, mu, v1, v2, v3, g1, g2, g3, g4,
                                 plane_mode="plane_strain",
                                 strain_capping_enabled=False,
                                 strain_capping_limit=None,
                                 strain_capping_tangent_ratio=0.1,
                                 strain_capping_type="piecewise",
                                 strain_capping_smooth_power=1.0,
                                 e33_state=None):
    """Cauchy stress for 2-D Landau small-strain model — tensor reference
    implementation.

    Retained as the validation oracle for the scalar-field implementation in
    ``stress_from_strain_landau_2d`` (see tests/test_landau_stress_fast.py);
    production code should call ``stress_from_strain_landau_2d``, which
    reproduces this function to floating-point roundoff at a fraction of the
    cost.

    For ``plane_strain`` the stress is computed directly from the strain invariants.
    For ``plane_stress`` an out‑of‑plane strain component ``e33`` is solved
    iteratively (Newton‑Raphson) such that the out‑of‑plane stress σ₃₃ = 0.

    e33_state : optional dict shared across calls. The converged e33 field is
        stored under key "e33" and reused as the Newton initial guess on the
        next call, which cuts the plane-stress iteration count when successive
        calls see nearly identical strain fields (LS/macro iterations, KMC steps).
    """
    eps_orig = eps

    if plane_mode == "plane_stress":
        nx, ny = eps.shape[0], eps.shape[1]
        # Compute in‑plane invariants to start guess
        I1_in = np.trace(eps, axis1=2, axis2=3)
        e33_warm = e33_state.get("e33") if e33_state is not None else None
        if e33_warm is not None and e33_warm.shape == (nx, ny):
            e33 = e33_warm.copy()
        else:
            e33 = -lam / (lam + 2.0 * mu) * I1_in
        # Absolute Newton tolerance on e33: 1e-9 strain resolves sigma33 to
        # ~1e2 Pa for O(100 GPa) stiffness, far tighter than any macro tolerance.
        tol_e33 = 1e-9

        for _ in range(20):
            # Construct 3D strain tensor
            eps_3d = np.zeros((nx, ny, 3, 3))
            eps_3d[..., 0:2, 0:2] = eps
            eps_3d[..., 2, 2] = e33
            
            eps_3d_capped, capping_weight = _cap_strain_3d(
                eps_3d, lam, mu, g4,
                strain_capping_enabled, strain_capping_limit,
                strain_capping_tangent_ratio, strain_capping_type,
                strain_capping_smooth_power
            )
            
            I1_tot = np.trace(eps_3d_capped, axis1=2, axis2=3)
            eps2_3d = np.einsum('...ij,...jk->...ik', eps_3d_capped, eps_3d_capped)
            I2_tot = np.trace(eps2_3d, axis1=2, axis2=3)
            eps3_3d = np.einsum('...ij,...jk->...ik', eps2_3d, eps_3d_capped)
            I3_tot = np.trace(eps3_3d, axis1=2, axis2=3)
            
            coeff_I = lam * I1_tot + 0.5 * v1 * I1_tot ** 2 + v2 * I2_tot + (1.0/6.0) * g1 * I1_tot ** 3 + g2 * I1_tot * I2_tot + (4.0/3.0) * g3 * I3_tot
            coeff_eps = 2.0 * (mu + v2 * I1_tot + 0.5 * g2 * I1_tot ** 2 + g4 * I2_tot)
            coeff_eps2 = 4.0 * (v3 + g3 * I1_tot)
            
            eps_33_cap = eps_3d_capped[..., 2, 2]
            sigma33_landau = coeff_I + coeff_eps * eps_33_cap + coeff_eps2 * (eps_33_cap ** 2)
            
            dcoeff_I = (lam + v1 * I1_tot + 0.5 * g1 * I1_tot ** 2 + g2 * I2_tot
                        + 2.0 * (v2 + g2 * I1_tot) * eps_33_cap + 4.0 * g3 * (eps_33_cap ** 2))
            dcoeff_eps = 2.0 * (v2 + g2 * I1_tot) + 4.0 * g4 * eps_33_cap
            dcoeff_eps2 = 4.0 * g3
            dsigma33_landau = (dcoeff_I + dcoeff_eps * eps_33_cap + coeff_eps
                              + dcoeff_eps2 * (eps_33_cap ** 2) + 2.0 * coeff_eps2 * eps_33_cap)
            
            if strain_capping_enabled:
                G_tangent = strain_capping_tangent_ratio * mu
                sigma33 = sigma33_landau + 2.0 * G_tangent * (e33 - eps_33_cap) * capping_weight
                dsigma33 = dsigma33_landau + 2.0 * G_tangent * capping_weight
            else:
                sigma33 = sigma33_landau
                dsigma33 = dsigma33_landau
                
            delta = -sigma33 / (dsigma33 + 1e-12)
            if not np.all(np.isfinite(delta)):
                raise FloatingPointError(
                    "stress_from_strain_landau_2d: non-finite e33 Newton update "
                    "(Landau energy likely unstable at this strain; consider "
                    "strain capping or smaller load steps)."
                )
            e33 += delta
            if np.all(np.abs(delta) < tol_e33):
                break
        else:
            # Loop exhausted without meeting tol (linear convergence is expected
            # in capped pixels, where dsigma33 omits the cap-factor chain rule).
            import warnings
            warnings.warn(
                f"stress_from_strain_landau_2d: e33 Newton not converged in 20 "
                f"iterations (max|delta| = {np.max(np.abs(delta)):.2e}); "
                f"sigma33 = 0 is only approximately satisfied.",
                RuntimeWarning
            )

        if e33_state is not None:
            e33_state["e33"] = e33

        eps_capped = eps_3d_capped[..., 0:2, 0:2]
        I1 = I1_tot
        I2 = I2_tot
        I3 = I3_tot
        
    else:
        nx, ny = eps.shape[0], eps.shape[1]
        eps_3d = np.zeros((nx, ny, 3, 3))
        eps_3d[..., 0:2, 0:2] = eps
        
        eps_3d_capped, capping_weight = _cap_strain_3d(
            eps_3d, lam, mu, g4,
            strain_capping_enabled, strain_capping_limit,
            strain_capping_tangent_ratio, strain_capping_type,
            strain_capping_smooth_power
        )
        
        eps_capped = eps_3d_capped[..., 0:2, 0:2]
        I1 = np.trace(eps_3d_capped, axis1=2, axis2=3)
        eps2_3d = np.einsum('...ij,...jk->...ik', eps_3d_capped, eps_3d_capped)
        I2 = np.trace(eps2_3d, axis1=2, axis2=3)
        eps3_3d = np.einsum('...ij,...jk->...ik', eps2_3d, eps_3d_capped)
        I3 = np.trace(eps3_3d, axis1=2, axis2=3)

    # Coefficients for updated invariants
    coeff_I = lam * I1 + 0.5 * v1 * I1 ** 2 + v2 * I2 + (1.0/6.0) * g1 * I1 ** 3 + g2 * I1 * I2 + (4.0/3.0) * g3 * I3
    coeff_eps = 2.0 * (mu + v2 * I1 + 0.5 * g2 * I1 ** 2 + g4 * I2)
    coeff_eps2 = 4.0 * (v3 + g3 * I1)

    # Construct stress tensor using eps_capped
    I = np.eye(2)[None, None, :, :]
    c_I = coeff_I[..., None, None]
    c_eps = coeff_eps[..., None, None]
    c_eps2 = coeff_eps2[..., None, None]
    
    eps2_capped = np.einsum('...ij,...jk->...ik', eps_capped, eps_capped)
    sig = c_I * I + c_eps * eps_capped + c_eps2 * eps2_capped
    
    if strain_capping_enabled:
        G_tangent = strain_capping_tangent_ratio * mu
        sig += 2.0 * G_tangent[..., None, None] * (eps_orig - eps_capped) * capping_weight[..., None, None]

    return sig


def stress_from_strain_landau_2d(eps, lam, mu, v1, v2, v3, g1, g2, g3, g4,
                                 plane_mode="plane_strain",
                                 strain_capping_enabled=False,
                                 strain_capping_limit=None,
                                 strain_capping_tangent_ratio=0.1,
                                 strain_capping_type="piecewise",
                                 strain_capping_smooth_power=1.0,
                                 e33_state=None):
    """Cauchy stress for the 2-D Landau small-strain model (scalar-field form).

    Mathematically identical to ``stress_from_strain_landau_2d_reference``
    (validated to floating-point roundoff in tests/test_landau_stress_fast.py)
    but several times faster: in this 2-D setting the 3D strain tensor is
    always block-diagonal (in-plane 2x2 block plus e33, zero out-of-plane
    shears), and strain capping preserves that structure, so the invariants
    and capped components reduce to closed-form expressions in the component
    fields — tr(B^3) of the in-plane block via Cayley-Hamilton — with no
    (nx, ny, 3, 3) tensor builds or einsum products per evaluation.

    See the reference implementation for parameter documentation.
    """
    a11 = eps[..., 0, 0]
    a22 = eps[..., 1, 1]
    a12 = eps[..., 0, 1]
    a21 = eps[..., 1, 0]
    nx, ny = eps.shape[0], eps.shape[1]

    capping = strain_capping_enabled
    if capping:
        mu_val = mu.mean() if isinstance(mu, np.ndarray) else mu
        if strain_capping_limit is None or strain_capping_limit <= 0:
            E_cap = np.sqrt(0.2 * mu_val / np.abs(g4)) if g4 < 0 else 999.0
        else:
            E_cap = strain_capping_limit
        G_tangent = strain_capping_tangent_ratio * mu

    def _scalars(e33):
        """Capped components, invariants and Landau coefficients at e33."""
        I1_raw = a11 + a22 + e33
        m3 = I1_raw / 3.0
        d11 = a11 - m3
        d22 = a22 - m3
        d33 = e33 - m3
        if capping:
            E_eq = np.sqrt(np.maximum(0.0, (2.0 / 3.0) *
                                      (d11*d11 + d22*d22 + d33*d33
                                       + 2.0 * a12 * a21)))
            f = np.ones_like(E_eq)
            nonzero = E_eq > 1e-12
            if strain_capping_type == "smooth":
                p = strain_capping_smooth_power
                ratio = E_eq[nonzero] / E_cap
                ratio_p = np.minimum(100.0, ratio**p)
                f[nonzero] = (E_cap / E_eq[nonzero]) * (np.tanh(ratio_p))**(1.0/p)
                w = np.tanh(np.minimum(100.0, (E_eq / E_cap)**p))**2
            else:
                is_capped = E_eq > E_cap
                np.divide(E_cap, E_eq, out=f, where=is_capped)
                w = is_capped.astype(float)
            c11 = f * d11 + m3
            c22 = f * d22 + m3
            c12 = f * a12
            c21 = f * a21
            c33 = f * d33 + m3
        else:
            w = None
            c11, c22, c12, c21, c33 = a11, a22, a12, a21, e33

        # invariants of the block-diagonal capped tensor
        I1 = c11 + c22 + c33
        cross = c12 * c21
        tr2_ip = c11*c11 + c22*c22 + 2.0*cross
        I2 = tr2_ip + c33*c33
        t_ip = c11 + c22
        det_ip = c11*c22 - cross
        I3 = t_ip * (tr2_ip - det_ip) + c33**3   # tr(B^3) = t(tr(B^2)-det) per block

        coeff_I = (lam * I1 + 0.5 * v1 * I1**2 + v2 * I2
                   + (1.0/6.0) * g1 * I1**3 + g2 * I1 * I2
                   + (4.0/3.0) * g3 * I3)
        coeff_eps = 2.0 * (mu + v2 * I1 + 0.5 * g2 * I1**2 + g4 * I2)
        coeff_eps2 = 4.0 * (v3 + g3 * I1)
        return I1, I2, c11, c22, c12, c21, c33, w, coeff_I, coeff_eps, coeff_eps2

    if plane_mode == "plane_stress":
        e33_warm = e33_state.get("e33") if e33_state is not None else None
        if e33_warm is not None and e33_warm.shape == (nx, ny):
            e33 = e33_warm.copy()
        else:
            e33 = -lam / (lam + 2.0 * mu) * (a11 + a22)
        # Same tolerance rationale as the reference: 1e-9 strain resolves
        # sigma33 to ~1e2 Pa for O(100 GPa) stiffness.
        tol_e33 = 1e-9

        for _ in range(20):
            (I1, I2, c11, c22, c12, c21, c33, w,
             coeff_I, coeff_eps, coeff_eps2) = _scalars(e33)

            sigma33 = coeff_I + coeff_eps * c33 + coeff_eps2 * c33**2
            dcoeff_I = (lam + v1 * I1 + 0.5 * g1 * I1**2 + g2 * I2
                        + 2.0 * (v2 + g2 * I1) * c33 + 4.0 * g3 * c33**2)
            dcoeff_eps = 2.0 * (v2 + g2 * I1) + 4.0 * g4 * c33
            dsigma33 = (dcoeff_I + dcoeff_eps * c33 + coeff_eps
                        + 4.0 * g3 * c33**2 + 2.0 * coeff_eps2 * c33)
            if capping:
                sigma33 = sigma33 + 2.0 * G_tangent * (e33 - c33) * w
                dsigma33 = dsigma33 + 2.0 * G_tangent * w

            delta = -sigma33 / (dsigma33 + 1e-12)
            if not np.all(np.isfinite(delta)):
                raise FloatingPointError(
                    "stress_from_strain_landau_2d: non-finite e33 Newton update "
                    "(Landau energy likely unstable at this strain; consider "
                    "strain capping or smaller load steps)."
                )
            e33 = e33 + delta
            if np.all(np.abs(delta) < tol_e33):
                break
        else:
            import warnings
            warnings.warn(
                f"stress_from_strain_landau_2d: e33 Newton not converged in 20 "
                f"iterations (max|delta| = {np.max(np.abs(delta)):.2e}); "
                f"sigma33 = 0 is only approximately satisfied.",
                RuntimeWarning
            )

        if e33_state is not None:
            e33_state["e33"] = e33
    else:
        e33 = np.zeros_like(a11)
        (I1, I2, c11, c22, c12, c21, c33, w,
         coeff_I, coeff_eps, coeff_eps2) = _scalars(e33)

    # In-plane stress from the last computed capped state (for plane_stress
    # this is the pre-final-update Newton state, matching the reference).
    cross = c12 * c21
    t_ip = c11 + c22
    sig = np.empty(eps.shape)
    sig[..., 0, 0] = coeff_I + coeff_eps * c11 + coeff_eps2 * (c11*c11 + cross)
    sig[..., 1, 1] = coeff_I + coeff_eps * c22 + coeff_eps2 * (c22*c22 + cross)
    sig[..., 0, 1] = coeff_eps * c12 + coeff_eps2 * (c12 * t_ip)
    sig[..., 1, 0] = coeff_eps * c21 + coeff_eps2 * (c21 * t_ip)

    if capping:
        r = 2.0 * G_tangent * w
        sig[..., 0, 0] += r * (a11 - c11)
        sig[..., 1, 1] += r * (a22 - c22)
        sig[..., 0, 1] += r * (a12 - c12)
        sig[..., 1, 0] += r * (a21 - c21)

    return sig


def von_mises_strain_2d(eps):
    """
    Von Mises equivalent strain for a 2-D strain field.
    Note: For plane-strain, we assume eps_33 = 0, which 
    gives  eps'_33 = -(eps_11 + eps_22)/3.
    """
    e11 = eps[:, :, 0, 0]
    e22 = eps[:, :, 1, 1]
    e12 = eps[:, :, 0, 1]
    e21 = eps[:, :, 1, 0]

    # Deviatoric components (plane-strain: eps_33 = 0 => eps'_33 = -(e11+e22)/3)
    tr3  = (e11 + e22) / 3.0
    ep11 = e11 - tr3
    ep22 = e22 - tr3
    ep33 = -(e11 + e22) / 3.0   # out-of-plane deviatoric

    eps_eq = np.sqrt(
        (2.0 / 3.0) * (ep11**2 + ep22**2 + ep33**2
                        + 0.5 * (e12 + e21)**2)
    )
    return eps_eq


def von_mises_strain_3d(eps):
    """
    Von Mises equivalent strain for a 3-D strain field.

    Parameters
    ----------
    eps : ndarray, shape (nx, ny, nz, 3, 3)
        Small-strain tensor field (symmetric).

    Returns
    -------
    eps_eq : ndarray, shape (nx, ny, nz)
        Equivalent strain  eps_eq = sqrt(2/3 * eps' : eps').
    """
    e11 = eps[:, :, :, 0, 0]
    e22 = eps[:, :, :, 1, 1]
    e33 = eps[:, :, :, 2, 2]
    e12 = eps[:, :, :, 0, 1]
    e21 = eps[:, :, :, 1, 0]
    e13 = eps[:, :, :, 0, 2]
    e31 = eps[:, :, :, 2, 0]
    e23 = eps[:, :, :, 1, 2]
    e32 = eps[:, :, :, 2, 1]

    tr3  = (e11 + e22 + e33) / 3.0
    ep11 = e11 - tr3
    ep22 = e22 - tr3
    ep33 = e33 - tr3

    eps_eq = np.sqrt(
        (2.0 / 3.0) * (
            ep11**2 + ep22**2 + ep33**2
            + 0.5 * ((e12 + e21)**2 + (e13 + e31)**2 + (e23 + e32)**2)
        )
    )
    return eps_eq


def secant_shear_field(eps_eq, mu, d, k):
    """
    Pointwise secant shear modulus from the exponential degradation law:

        mu_sec(eps_eq) = mu * [1 - d * (1 - exp(-k * eps_eq))]

    Parameters
    ----------
    eps_eq : ndarray
        Von Mises equivalent strain, any shape.
    mu : float or ndarray broadcastable to eps_eq
        Undegraded shear modulus (Pa).
    d : float
        Degradation magnitude, 0 <= d <= 1.
        d=0 -> no degradation (linear elastic).
        d=1 -> maximum degradation (mu_sec -> 0 as eps_eq -> inf).
    k : float
        Degradation rate (dimensionless; larger k = faster softening onset).

    Returns
    -------
    mu_sec : ndarray (same shape as eps_eq)
        Secant shear modulus field (Pa).
    """
    return mu * (1.0 - d * (1.0 - np.exp(-k * eps_eq)))


def stress_from_strain_secant_2d(eps, lam, mu, d, k,
                                  plane_mode="plane_strain"):
    """
    Secant-elastic Cauchy stress for a 2-D strain field:

        sigma = lam * tr(eps) * I + 2 * mu_sec(eps_eq) * eps

    where mu_sec follows the exponential degradation law and eps_eq is the
    von Mises equivalent strain computed from eps.

    Parameters
    ----------
    eps : ndarray, shape (nx, ny, 2, 2)
        Small-strain tensor field.
    lam : float or ndarray, shape (nx, ny)
        First Lame parameter lambda (Pa).
        For plane stress, pass the effective lambda* = E*nu/(1-nu^2).
    mu : float or ndarray, shape (nx, ny)
        Undegraded shear modulus mu (Pa).
    d : float
        Degradation magnitude (0 <= d <= 1).
    k : float
        Degradation rate (dimensionless).
    plane_mode : str
        "plane_strain" (default) or "plane_stress".  The caller is
        responsible for passing the correct lam for the chosen mode;
        this function uses lam directly.

    Returns
    -------
    sig : ndarray, shape (nx, ny, 2, 2)
        Cauchy stress tensor field (Pa).
    """
    eps_eq  = von_mises_strain_2d(eps)               # (nx, ny)
    mu_sec  = secant_shear_field(eps_eq, mu, d, k)  # (nx, ny)

    tr_eps  = np.trace(eps, axis1=2, axis2=3)[..., None, None]   # (nx,ny,1,1)
    I       = np.eye(2)[None, None, :, :]

    lam_    = np.asarray(lam)[..., None, None]
    mu_sec_ = mu_sec[..., None, None]

    return lam_ * tr_eps * I + 2.0 * mu_sec_ * eps


def stress_from_strain_secant_3d(eps, lam, mu, d, k):
    """
    Secant-elastic Cauchy stress for a 3-D strain field:

        sigma = lam * tr(eps) * I + 2 * mu_sec(eps_eq) * eps

    Parameters
    ----------
    eps : ndarray, shape (nx, ny, nz, 3, 3)
        Small-strain tensor field.
    lam : float or ndarray, shape (nx, ny, nz)
        First Lame parameter lambda (Pa).
    mu : float or ndarray, shape (nx, ny, nz)
        Undegraded shear modulus mu (Pa).
    d : float
        Degradation magnitude (0 <= d <= 1).
    k : float
        Degradation rate (dimensionless).

    Returns
    -------
    sig : ndarray, shape (nx, ny, nz, 3, 3)
        Cauchy stress tensor field (Pa).
    """
    eps_eq  = von_mises_strain_3d(eps)               # (nx, ny, nz)
    mu_sec  = secant_shear_field(eps_eq, mu, d, k)  # (nx, ny, nz)

    tr_eps  = np.trace(eps, axis1=3, axis2=4)[..., None, None]   # (nx,ny,nz,1,1)
    I       = np.eye(3)[None, None, None, :, :]

    lam_    = np.asarray(lam)[..., None, None]
    mu_sec_ = mu_sec[..., None, None]

    return lam_ * tr_eps * I + 2.0 * mu_sec_ * eps


def stress_from_strain_landau_3d(eps, lam, mu, v1, v2, v3, g1, g2, g3, g4,
                                 strain_capping_enabled=False,
                                 strain_capping_limit=None,
                                 strain_capping_tangent_ratio=0.1,
                                 strain_capping_type="piecewise",
                                 strain_capping_smooth_power=1.0):
    """
    Cauchy stress for 3-D Landau small-strain model:
        sigma = (lam*I1 + 0.5*v1*I1^2 + v2*I2 + 1/6*g1*I1^3 + g2*I1*I2 + 4/3*g3*I3)*I
              + 2*(mu + v2*I1 + 0.5*g2*I1^2 + g4*I2)*eps
              + 4*(v3 + g3*I1)*eps^2
    """
    eps_orig = eps
    eps_capped, capping_weight = _cap_strain_3d(
        eps, lam, mu, g4,
        strain_capping_enabled, strain_capping_limit,
        strain_capping_tangent_ratio, strain_capping_type,
        strain_capping_smooth_power
    )
    eps = eps_capped

    # 1. Compute strain invariants
    I1 = np.trace(eps, axis1=3, axis2=4)  # (nx, ny, nz)
    
    eps2 = np.einsum('...ij,...jk->...ik', eps, eps)  # (nx, ny, nz, 3, 3)
    I2 = np.trace(eps2, axis1=3, axis2=4)  # (nx, ny, nz)
    
    eps3 = np.einsum('...ij,...jk->...ik', eps2, eps)
    I3 = np.trace(eps3, axis1=3, axis2=4)  # (nx, ny, nz)
    
    # 2. Coefficients
    lam_arr = np.asarray(lam)
    mu_arr = np.asarray(mu)
    
    coeff_I = lam_arr * I1 + 0.5 * v1 * I1**2 + v2 * I2 + (1.0/6.0) * g1 * I1**3 + g2 * I1 * I2 + (4.0/3.0) * g3 * I3
    coeff_eps = 2.0 * (mu_arr + v2 * I1 + 0.5 * g2 * I1**2 + g4 * I2)
    coeff_eps2 = 4.0 * (v3 + g3 * I1)
    
    # 3. Construct stress tensor
    I = np.eye(3)[None, None, None, :, :]
    c_I = coeff_I[..., None, None]
    c_eps = coeff_eps[..., None, None]
    c_eps2 = coeff_eps2[..., None, None]
    
    sig = c_I * I + c_eps * eps + c_eps2 * eps2
    
    if strain_capping_enabled:
        G_tangent = strain_capping_tangent_ratio * mu_arr[..., None, None]
        sig += 2.0 * G_tangent * (eps_orig - eps) * capping_weight[..., None, None]
        
    return sig


# ===========================================================================
# Rose Secant Modulus Model helpers
# ===========================================================================

def rose_shear_field(eps_eq, mu, eta, mu_floor_fraction=0.1):
    """
    Pointwise secant shear modulus from the Rose degradation law:
        G_sec(eps_eq) = max(G0 * exp(-eta * eps_eq), mu_floor_fraction * G0)
    """
    G0 = mu
    G_sec = G0 * np.exp(-eta * eps_eq)
    G_floor = mu_floor_fraction * G0
    return np.maximum(G_sec, G_floor)


def stress_from_strain_rose_2d(eps, lam, mu, eta, mu_floor_fraction=0.1,
                               plane_mode="plane_strain"):
    """
    Cauchy stress for a 2-D strain field using the Rose secant model.
    For plane_stress, solves for out-of-plane strain eps_zz self-consistently.
    """
    eps_xx = eps[..., 0, 0]
    eps_yy = eps[..., 1, 1]
    eps_xy = eps[..., 0, 1]

    K = lam + (2.0 / 3.0) * mu

    if plane_mode == "plane_stress":
        # Initial guess using linear Poisson relation: eps_zz = -lam / (lam + 2*mu) * (eps_xx + eps_yy)
        eps_zz = -lam / (lam + 2.0 * mu) * (eps_xx + eps_yy)
        
        for _ in range(5):
            tr = eps_xx + eps_yy + eps_zz
            tr3 = tr / 3.0
            ep11 = eps_xx - tr3
            ep22 = eps_yy - tr3
            ep33 = eps_zz - tr3
            
            eps_eq = np.sqrt(
                (2.0 / 3.0) * (ep11**2 + ep22**2 + ep33**2 + 2.0 * eps_xy**2)
            )
            
            G_sec = rose_shear_field(eps_eq, mu, eta, mu_floor_fraction)
            
            num = 3.0 * K - 2.0 * G_sec
            den = 3.0 * K + 4.0 * G_sec
            eps_zz = - (num / den) * (eps_xx + eps_yy)
            
        # Final evaluation with converged eps_zz
        tr = eps_xx + eps_yy + eps_zz
        tr3 = tr / 3.0
        ep11 = eps_xx - tr3
        ep22 = eps_yy - tr3
        ep33 = eps_zz - tr3
        eps_eq = np.sqrt(
            (2.0 / 3.0) * (ep11**2 + ep22**2 + ep33**2 + 2.0 * eps_xy**2)
        )
        G_sec = rose_shear_field(eps_eq, mu, eta, mu_floor_fraction)
    else:  # plane_strain
        eps_zz = np.zeros_like(eps_xx)
        tr = eps_xx + eps_yy
        tr3 = tr / 3.0
        ep11 = eps_xx - tr3
        ep22 = eps_yy - tr3
        ep33 = eps_zz - tr3
        
        eps_eq = np.sqrt(
            (2.0 / 3.0) * (ep11**2 + ep22**2 + ep33**2 + 2.0 * eps_xy**2)
        )
        G_sec = rose_shear_field(eps_eq, mu, eta, mu_floor_fraction)

    # Compute stress components keeping K constant
    sig = np.zeros_like(eps)
    lam_eff = K - (2.0 / 3.0) * G_sec
    sig[..., 0, 0] = lam_eff * tr + 2.0 * G_sec * eps_xx
    sig[..., 1, 1] = lam_eff * tr + 2.0 * G_sec * eps_yy
    sig[..., 0, 1] = 2.0 * G_sec * eps_xy
    sig[..., 1, 0] = sig[..., 0, 1]

    return sig

