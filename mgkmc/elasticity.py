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


def stress_from_strain_landau_3d(eps, lam, mu, v1, v2, v3, g1, g2, g3, g4):
    """
    Cauchy stress for 3-D Landau small-strain model:
        sigma = (lam*I1 + 0.5*v1*I1^2 + v2*I2 + 1/6*g1*I1^3 + g2*I1*I2 + 4/3*g3*I3)*I
              + 2*(mu + v2*I1 + 0.5*g2*I1^2 + g4*I2)*eps
              + 4*(v3 + g3*I1)*eps^2
    """
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
def stress_from_strain_landau_2d(eps, lam, mu, v1, v2, v3, g1, g2, g3, g4, plane_mode="plane_strain"):
    """Cauchy stress for 2-D Landau small-strain model.

    This implementation follows the original nonlinear formulation.
    For ``plane_strain`` the stress is computed directly from the strain invariants.
    For ``plane_stress`` an out‑of‑plane strain component ``e33`` is solved
    iteratively (Newton‑Raphson) such that the out‑of‑plane stress σ₃₃ = 0.
    """
    # 1. Compute in‑plane strain invariants
    I1 = np.trace(eps, axis1=2, axis2=3)  # (nx, ny)
    eps2 = np.einsum('...ij,...jk->...ik', eps, eps)
    I2 = np.trace(eps2, axis1=2, axis2=3)
    eps3 = np.einsum('...ij,...jk->...ik', eps2, eps)
    I3 = np.trace(eps3, axis1=2, axis2=3)

    if plane_mode == "plane_stress":
        # Initial guess for out‑of‑plane strain e33 (linear plane‑stress solution)
        e33 = -lam / (lam + 2.0 * mu) * I1
        for _ in range(20):
            # Augmented invariants including e33
            I1_tot = I1 + e33
            I2_tot = I2 + e33 ** 2
            I3_tot = I3 + e33 ** 3
            # Coefficients based on augmented invariants
            coeff_I = lam * I1_tot + 0.5 * v1 * I1_tot ** 2 + v2 * I2_tot + (1.0/6.0) * g1 * I1_tot ** 3 + g2 * I1_tot * I2_tot + (4.0/3.0) * g3 * I3_tot
            coeff_eps = 2.0 * (mu + v2 * I1_tot + 0.5 * g2 * I1_tot ** 2 + g4 * I2_tot)
            coeff_eps2 = 4.0 * (v3 + g3 * I1_tot)
            # Out‑of‑plane stress σ₃₃
            sigma33 = coeff_I + coeff_eps * e33 + coeff_eps2 * e33 ** 2
            # Exact derivative dσ₃₃/de33, via chain rule through I1_tot, I2_tot,
            # I3_tot (coeff_I, coeff_eps, coeff_eps2 all depend on e33 through
            # these invariants, not just the explicit e33 and e33**2 factors).
            dcoeff_I = (lam + v1 * I1_tot + 0.5 * g1 * I1_tot ** 2 + g2 * I2_tot
                        + 2.0 * (v2 + g2 * I1_tot) * e33 + 4.0 * g3 * e33 ** 2)
            dcoeff_eps = 2.0 * (v2 + g2 * I1_tot) + 4.0 * g4 * e33
            dcoeff_eps2 = 4.0 * g3
            dsigma33 = (dcoeff_I + dcoeff_eps * e33 + coeff_eps
                        + dcoeff_eps2 * e33 ** 2 + 2.0 * coeff_eps2 * e33)
            # Newton update (guard against division by zero)
            delta = -sigma33 / (dsigma33 + 1e-12)
            e33 += delta
            if np.all(np.abs(delta) < 1e-12):
                break
        # Update invariants with converged e33
        I1 = I1 + e33
        I2 = I2 + e33 ** 2
        I3 = I3 + e33 ** 3

    # 2. Coefficients for (possibly updated) invariants
    coeff_I = lam * I1 + 0.5 * v1 * I1 ** 2 + v2 * I2 + (1.0/6.0) * g1 * I1 ** 3 + g2 * I1 * I2 + (4.0/3.0) * g3 * I3
    coeff_eps = 2.0 * (mu + v2 * I1 + 0.5 * g2 * I1 ** 2 + g4 * I2)
    coeff_eps2 = 4.0 * (v3 + g3 * I1)

    # 3. Construct stress tensor (2‑D)
    I = np.eye(2)[None, None, :, :]
    c_I = coeff_I[..., None, None]
    c_eps = coeff_eps[..., None, None]
    c_eps2 = coeff_eps2[..., None, None]
    sig = c_I * I + c_eps * eps + c_eps2 * eps2
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


def stress_from_strain_landau_3d(eps, lam, mu, v1, v2, v3, g1, g2, g3, g4):
    """
    Cauchy stress for 3-D Landau small-strain model:
        sigma = (lam*I1 + 0.5*v1*I1^2 + v2*I2 + 1/6*g1*I1^3 + g2*I1*I2 + 4/3*g3*I3)*I
              + 2*(mu + v2*I1 + 0.5*g2*I1^2 + g4*I2)*eps
              + 4*(v3 + g3*I1)*eps^2
    """
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
    return sig
