from setuptools import setup, find_packages

setup(
    name="mgkmc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "meshio",
        "pyfftw",
        "h5py",
        "numba",
    ],
)
