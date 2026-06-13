import numpy as np
import time
import pyfftw

# Setup pyfftw interface
pyfftw.interfaces.cache.enable()
import pyfftw.interfaces.numpy_fft as pyfft

# Shifted NumPy (Current implementation)
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

# Unshifted NumPy
def _fft2_tensor_unshifted_np(A):
    return np.fft.fftn(A, axes=(-2, -1))

def _ifft2_tensor_unshifted_np(A_hat):
    return np.fft.ifftn(A_hat, axes=(-2, -1)).real

# Unshifted pyfftw
def _fft2_tensor_unshifted_pyfftw(A, threads=1):
    return pyfft.fftn(A, axes=(-2, -1), threads=threads)

def _ifft2_tensor_unshifted_pyfftw(A_hat, threads=1):
    return pyfft.ifftn(A_hat, axes=(-2, -1), threads=threads).real


# Benchmark configurations
nx, ny = 128, 128
A2 = np.random.randn(2, 2, nx, ny)
A2_hat = _fft2_tensor_shifted(A2)

# Warm up pyfftw
_ = _fft2_tensor_unshifted_pyfftw(A2, threads=1)
_ = _ifft2_tensor_unshifted_pyfftw(A2_hat, threads=1)

n_iters = 1000

# 1. Current Shifted NumPy
t0 = time.perf_counter()
for _ in range(n_iters):
    _ = _fft2_tensor_shifted(A2)
    _ = _ifft2_tensor_shifted(A2_hat)
t1 = time.perf_counter()
t_shifted_np = t1 - t0
print(f"Shifted NumPy: {t_shifted_np:.4f} seconds for {n_iters} iterations")

# 2. Unshifted NumPy
t0 = time.perf_counter()
for _ in range(n_iters):
    _ = _fft2_tensor_unshifted_np(A2)
    _ = _ifft2_tensor_unshifted_np(A2_hat)
t1 = time.perf_counter()
t_unshifted_np = t1 - t0
print(f"Unshifted NumPy: {t_unshifted_np:.4f} seconds for {n_iters} iterations (Speedup: {t_shifted_np/t_unshifted_np:.2f}x)")

# 3. Unshifted pyfftw (1 thread)
t0 = time.perf_counter()
for _ in range(n_iters):
    _ = _fft2_tensor_unshifted_pyfftw(A2, threads=1)
    _ = _ifft2_tensor_unshifted_pyfftw(A2_hat, threads=1)
t1 = time.perf_counter()
t_unshifted_pyfftw1 = t1 - t0
print(f"Unshifted pyfftw (1 thread): {t_unshifted_pyfftw1:.4f} seconds for {n_iters} iterations (Speedup: {t_shifted_np/t_unshifted_pyfftw1:.2f}x)")

# 4. Unshifted pyfftw (4 threads)
t0 = time.perf_counter()
for _ in range(n_iters):
    _ = _fft2_tensor_unshifted_pyfftw(A2, threads=4)
    _ = _ifft2_tensor_unshifted_pyfftw(A2_hat, threads=4)
t1 = time.perf_counter()
t_unshifted_pyfftw4 = t1 - t0
print(f"Unshifted pyfftw (4 threads): {t_unshifted_pyfftw4:.4f} seconds for {n_iters} iterations (Speedup: {t_shifted_np/t_unshifted_pyfftw4:.2f}x)")
