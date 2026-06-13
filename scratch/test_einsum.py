import numpy as np
import time

def test_contraction():
    nx, ny = 128, 128
    ndim = 2
    
    Fe = np.random.rand(ndim, ndim, nx, ny)
    C4 = np.random.rand(ndim, ndim, ndim, ndim, nx, ny)
    Fp_inv = np.random.rand(ndim, ndim, nx, ny)
    
    # 1. Original
    t0 = time.perf_counter()
    for _ in range(20):
        K4_orig = np.einsum('ikxy,anxy,klmnxy,jlxy,bmxy->ijabxy', Fe, Fe, C4, Fp_inv, Fp_inv)
    t1 = time.perf_counter()
    orig_time = (t1 - t0) / 20
    print(f"Original einsum: {orig_time*1000:.3f} ms")
    
    # 2. Optimized einsum
    t0 = time.perf_counter()
    for _ in range(20):
        K4_opt = np.einsum('ikxy,anxy,klmnxy,jlxy,bmxy->ijabxy', Fe, Fe, C4, Fp_inv, Fp_inv, optimize=True)
    t1 = time.perf_counter()
    opt_time = (t1 - t0) / 20
    print(f"Optimized einsum: {opt_time*1000:.3f} ms")
    
    # 3. Manual decomposition
    t0 = time.perf_counter()
    for _ in range(20):
        # A_kjmnxy = C4_klmnxy * Fp_inv_jlxy (sum over l)
        A = np.einsum('klmnxy,jlxy->kjmnxy', C4, Fp_inv)
        # B_kjbnxy = A_kjmnxy * Fp_inv_bmxy (sum over m)
        B = np.einsum('kjmnxy,bmxy->kjbnxy', A, Fp_inv)
        # C_ijbnxy = B_kjbnxy * Fe_ikxy (sum over k)
        C = np.einsum('kjbnxy,ikxy->ijbnxy', B, Fe)
        # D_ijabxy = C_ijbnxy * Fe_anxy (sum over n)
        K4_decomp = np.einsum('ijbnxy,anxy->ijabxy', C, Fe)
    t1 = time.perf_counter()
    decomp_time = (t1 - t0) / 20
    print(f"Decomposed einsum: {decomp_time*1000:.3f} ms")
    
    assert np.allclose(K4_orig, K4_opt)
    assert np.allclose(K4_orig, K4_decomp)
    print("All match!")

if __name__ == "__main__":
    test_contraction()
