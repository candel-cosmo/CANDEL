# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.

from setuptools import setup, find_packages

setup(
    name="candel",
    version="0.1.0",
    author="Richard Stiskalek",
    author_email="richard.stiskalek@protonmail.com",
    description=(
        "JAX framework for peculiar-velocity inference, distance-ladder "
        "calibration, and megamaser disk modelling"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/candel-cosmo/CANDEL",
    packages=find_packages(include=["candel", "candel.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.10",
    install_requires=[
        "jax==0.9.2",
        "jaxlib==0.9.2",
        "jax-cuda12-plugin==0.9.2; sys_platform == 'linux'",
        "numpyro",
        "numpy",
        "scipy",
        "h5py",
        "tomli; python_version < '3.11'",
        "interpax",
        "astropy",
        "matplotlib",
        "corner",
        "tomli_w",
        "scienceplots",
        "joblib",
        "getdist",
        "healpy",
        "tqdm",
        "numba",
        "mpi4py",
    ],
    package_data={"candel": ["configs/*.toml"]},
    zip_safe=False,
)
