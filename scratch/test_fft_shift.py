import numpy as np

def build_ghat4_2d_shifted(nx, ny, Lx, Ly):
    ndim = 2
    fx = np.arange(-nx / 2., nx / 2.) if nx % 2 == 0 else np.arange(-(nx - 1) / 2., (nx + 1) / 2.)
    fy = np.arange(-ny / 2., ny / 2.) if ny % 2 == 0 else np.arange(-(ny - 1) / 2., (ny + 1) / 2.)
    xi_x = fx / Lx
    xi_y = fy / Ly
    Xi_x, Xi_y = np.meshgrid(xi_x, xi_y, indexing='ij')
    xi  = np.stack([Xi_x, Xi_y], axis=0)
    xi2 = Xi_x**2 + Xi_y**2
    nyquist_mask = np.zeros((nx, ny), dtype=bool)
    if nx % 2 == 0:
        nyquist_mask |= (Xi_x == -nx / (2. * Lx))
    if ny % 2 == 0:
        nyquist_mask |= (Xi_y == -ny / (2. * Ly))
    safe_xi2 = xi2.copy()
    safe_xi2[xi2 == 0] = 1.0
    Ghat4 = np.zeros((ndim, ndim, ndim, ndim, nx, ny))
    delta  = np.eye(ndim)
    for i in range(ndim):
        for j in range(ndim):
            for l in range(ndim):
                for m in range(ndim):
                    val = delta[i, m] * xi[j] * xi[l] / safe_xi2
                    val[xi2 == 0] = 0.0
                    val[nyquist_mask] = 0.0
                    Ghat4[i, j, l, m] = val
    return Ghat4

def _fft2_tensor_shifted(A):
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(A, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )

def _ifft2_tensor_shifted(A_hat):
    return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(A_hat, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    ).real

def project_shifted(A2, Ghat4):
    A2_hat = _fft2_tensor_shifted(A2)
    res_hat = np.einsum('ijklxy,lkxy->ijxy', Ghat4, A2_hat)
    return _ifft2_tensor_shifted(res_hat)


# --- UNSHIFTED VERSION ---

def build_ghat4_2d_unshifted(nx, ny, Lx, Ly):
    ndim = 2
    # Use standard FFT frequencies (unshifted)
    xi_x = np.fft.fftfreq(nx, d=Lx/nx)
    xi_y = np.fft.fftfreq(ny, d=Ly/ny)
    Xi_x, Xi_y = np.meshgrid(xi_x, xi_y, indexing='ij')
    xi  = np.stack([Xi_x, Xi_y], axis=0)
    xi2 = Xi_x**2 + Xi_y**2
    nyquist_mask = np.zeros((nx, ny), dtype=bool)
    if nx % 2 == 0:
        nyquist_mask |= (Xi_x == -nx / (2. * Lx))
    if ny % 2 == 0:
        nyquist_mask |= (Xi_y == -ny / (2. * Ly))
    safe_xi2 = xi2.copy()
    safe_xi2[xi2 == 0] = 1.0
    Ghat4 = np.zeros((ndim, ndim, ndim, ndim, nx, ny))
    delta  = np.eye(ndim)
    for i in range(ndim):
        for j in range(ndim):
            for l in range(ndim):
                for m in range(ndim):
                    val = delta[i, m] * xi[j] * xi[l] / safe_xi2
                    val[xi2 == 0] = 0.0
                    val[nyquist_mask] = 0.0
                    Ghat4[i, j, l, m] = val
    return Ghat4

def project_unshifted(A2, Ghat4_unshifted):
    # Standard FFT/IFFT without any shifts
    A2_hat = np.fft.fftn(A2, axes=(-2, -1))
    res_hat = np.einsum('ijklxy,lkxy->ijxy', Ghat4_unshifted, A2_hat)
    return np.fft.ifftn(res_hat, axes=(-2, -1)).real


# --- COMPARISON ---

nx, ny = 128, 128
Lx, Ly = 128.0, 128.0

# Generate a random tensor field
np.random.seed(42)
A2 = np.random.randn(2, 2, nx, ny)

G_shifted = build_ghat4_2d_shifted(nx, ny, Lx, Ly)
G_unshifted = build_ghat4_2d_unshifted(nx, ny, Lx, Ly)

res_shifted = project_shifted(A2, G_shifted)
res_unshifted = project_unshifted(A2, G_unshifted)

diff = np.max(np.abs(res_shifted - res_unshifted))
print(f"Max difference between shifted and unshifted projection: {diff}")
assert np.allclose(res_shifted, res_unshifted, atol=1e-12), "Results differ!"
print("Verification SUCCESS: The projections are identical.")
