import numpy as np
import time

def constitutive_hyperelastic_2d_orig(F, C4, I2, I4, I4rt, Fp=None):
    ndim = 2
    if Fp is not None:
        det = Fp[0, 0] * Fp[1, 1] - Fp[0, 1] * Fp[1, 0]
        det_safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
        Fp_inv = np.zeros_like(Fp)
        Fp_inv[0, 0] =  Fp[1, 1] / det_safe
        Fp_inv[1, 1] =  Fp[0, 0] / det_safe
        Fp_inv[0, 1] = -Fp[0, 1] / det_safe
        Fp_inv[1, 0] = -Fp[1, 0] / det_safe
        
        Fe = np.einsum('ijxy,jkxy->ikxy', F, Fp_inv)
    else:
        Fe = F
        
    E_GL = 0.5 * (np.einsum('jixy,jkxy->ikxy', Fe, Fe) - I2)
    S    = np.einsum('ijklxy,lkxy->ijxy', C4, E_GL)
    
    if Fp is not None:
        P = np.einsum('ikxy,klxy,jlxy->ijxy', Fe, S, Fp_inv)
        S_ref = np.einsum('mkxy,klxy,jlxy->mjxy', Fp_inv, S, Fp_inv)
        K4 = np.einsum('bjxy,ia->ijabxy', S_ref, np.eye(ndim)) + \
             np.einsum('ikxy,anxy,klmnxy,jlxy,bmxy->ijabxy', Fe, Fe, C4, Fp_inv, Fp_inv)
    else:
        P = np.einsum('ijxy,jkxy->ikxy', Fe, S)
        # 3D/2D original formula for K4 when Fp is None
        # omitted here for simplicity
        K4 = None

    return P, K4

def constitutive_hyperelastic_2d_opt(F, C4, I2, I4, I4rt, Fp=None):
    ndim = 2
    if Fp is not None:
        det = Fp[0, 0] * Fp[1, 1] - Fp[0, 1] * Fp[1, 0]
        det_safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
        Fp_inv = np.zeros_like(Fp)
        Fp_inv[0, 0] =  Fp[1, 1] / det_safe
        Fp_inv[1, 1] =  Fp[0, 0] / det_safe
        Fp_inv[0, 1] = -Fp[0, 1] / det_safe
        Fp_inv[1, 0] = -Fp[1, 0] / det_safe
        
        Fe = np.einsum('ijxy,jkxy->ikxy', F, Fp_inv, optimize=True)
    else:
        Fe = F
        
    E_GL = 0.5 * (np.einsum('jixy,jkxy->ikxy', Fe, Fe, optimize=True) - I2)
    S    = np.einsum('ijklxy,lkxy->ijxy', C4, E_GL, optimize=True)
    
    if Fp is not None:
        # Use optimize=True for all einsums!
        P = np.einsum('ikxy,klxy,jlxy->ijxy', Fe, S, Fp_inv, optimize=True)
        S_ref = np.einsum('mkxy,klxy,jlxy->mjxy', Fp_inv, S, Fp_inv, optimize=True)
        
        # Decomposed K4 contraction:
        A = np.einsum('klmnxy,jlxy->kjmnxy', C4, Fp_inv, optimize=True)
        B = np.einsum('kjmnxy,bmxy->kjbnxy', A, Fp_inv, optimize=True)
        C = np.einsum('kjbnxy,ikxy->ijbnxy', B, Fe, optimize=True)
        term2 = np.einsum('ijbnxy,anxy->ijabxy', C, Fe, optimize=True)
        
        term1 = np.einsum('bjxy,ia->ijabxy', S_ref, np.eye(ndim), optimize=True)
        K4 = term1 + term2
    else:
        P = np.einsum('ijxy,jkxy->ikxy', Fe, S, optimize=True)
        K4 = None

    return P, K4

def main():
    nx, ny = 128, 128
    ndim = 2
    F = np.random.rand(ndim, ndim, nx, ny)
    C4 = np.random.rand(ndim, ndim, ndim, ndim, nx, ny)
    I2 = np.einsum('ij,xy->ijxy', np.eye(2), np.ones((nx, ny)))
    Fp = np.random.rand(ndim, ndim, nx, ny)
    
    t0 = time.perf_counter()
    for _ in range(50):
        P1, K1 = constitutive_hyperelastic_2d_orig(F, C4, I2, None, None, Fp)
    t1 = time.perf_counter()
    orig_time = (t1 - t0) / 50
    print(f"Original constitutive: {orig_time*1000:.3f} ms")
    
    t0 = time.perf_counter()
    for _ in range(50):
        P2, K2 = constitutive_hyperelastic_2d_opt(F, C4, I2, None, None, Fp)
    t1 = time.perf_counter()
    opt_time = (t1 - t0) / 50
    print(f"Optimized constitutive: {opt_time*1000:.3f} ms")
    
    assert np.allclose(P1, P2)
    assert np.allclose(K1, K2)
    print("All match!")

if __name__ == "__main__":
    main()
