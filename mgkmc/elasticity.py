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


