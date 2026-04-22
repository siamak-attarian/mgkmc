import multiprocessing
import numpy as np
import pyfftw

# Setup FFTW threading
pyfftw.interfaces.cache.enable()
import pyfftw.interfaces.numpy_fft as fft

def compute_wave_vectors_3d(nx, ny, nz, Lx, Ly, Lz):
    """3D Fourier-space wavevectors for periodic box of size Lx,Ly,Lz."""
    kx_1d = 2*np.pi*fft.fftfreq(nx, d=Lx/nx)
    ky_1d = 2*np.pi*fft.fftfreq(ny, d=Ly/ny)
    kz_1d = 2*np.pi*fft.fftfreq(nz, d=Lz/nz)
    return np.meshgrid(kx_1d, ky_1d, kz_1d, indexing="ij")

def compute_wave_vectors_2d(nx, ny, Lx, Ly):
    """2D Fourier-space wavevectors for periodic box of size Lx,Ly."""
    kx_1d = 2*np.pi*fft.fftfreq(nx, d=Lx/nx)
    ky_1d = 2*np.pi*fft.fftfreq(ny, d=Ly/ny)
    return np.meshgrid(kx_1d, ky_1d, indexing="ij")

def fft_field(f, threads=None):
    if threads is None:
        threads = pyfftw.config.NUM_THREADS
    return fft.fftn(f, norm="ortho", threads=threads)


def ifft_field(f_hat, threads=None):
    if threads is None:
        threads = pyfftw.config.NUM_THREADS
    return fft.ifftn(f_hat, norm="ortho", threads=threads).real