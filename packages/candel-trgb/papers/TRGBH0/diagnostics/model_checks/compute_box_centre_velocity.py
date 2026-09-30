#!/usr/bin/env python
"""Manticore-Local velocity at the box centre (observer) and the LG dipole.

Reads the 80 ManticoreLocalCOLA (CIC) forward fields used in the TRGBH0
inference, trilinearly interpolates the reconstructed velocity at the observer
position (box centre), averages over the realisations, and combines it with the
free-beta baseline external velocity to form the Local-Group-frame dipole for
comparison with the Planck LG value.

Baseline result (80 fields, seed 44):
  observer velocity : 376 km/s toward (l, b) = (230, 47) deg
  observer + Vext   : 588 (+50 -51) km/s toward (262 +5 -7, 26 +7 -9) deg
  Planck LG         : 620 km/s toward (271.9, 29.6) deg
Cross-checked against the SPH fields (384 km/s toward (228, 43) deg).
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import candel

FIELD_ROOT = Path(
    "/mnt/extraspace/rstiskalek/MANTICORE/2MPP_MULTIBIN_N256_DES_V2/"
    "forward_fields/CIC")
NREAL = 80
SEED = 44
NDRAW = 200_000

# Free-beta 48-pixel baseline Vext (galactic), as quoted in the paper.
VEXT_MAG, VEXT_MAG_E = 332.0, 10.0
VEXT_ELL, VEXT_ELL_E = 285.0, 2.0
VEXT_B, VEXT_B_E = -4.0, 3.0

# Planck LG (CMB-frame) reference.
PLANCK = (620.0, 271.9, 29.6)


def observer_velocity_one(path):
    """Trilinearly interpolate the velocity at the observer (box centre)."""
    with h5py.File(path, "r") as h:
        box = float(h.attrs["boxsize"])
        ngrid = int(h["velocity"].shape[0])
        obs = np.asarray(h.attrs["observer_position"], dtype=np.float64)
        cell = box / ngrid
        c = obs / cell - 0.5  # voxel-centre convention obs = (i + 0.5) * cell
        i0, i1 = int(np.floor(c.min())) - 1, int(np.ceil(c.max())) + 2
        sl = slice(i0, i1)
        vel = h["velocity"][sl, sl, sl, :].astype(np.float64)
    axis = (np.arange(i0, i1) + 0.5) * cell
    out = np.empty(3)
    for comp in range(3):
        interp = RegularGridInterpolator(
            (axis, axis, axis), vel[..., comp], method="linear",
            bounds_error=False, fill_value=None)
        out[comp] = interp(obs)[0]
    return out


def gal(vectors):
    mag, ell, b = candel.radec_cartesian_to_galactic(
        vectors[..., 0], vectors[..., 1], vectors[..., 2])
    return np.atleast_1d(mag), np.atleast_1d(ell), np.atleast_1d(b)


def summary(vectors):
    mag, ell, b = gal(vectors)
    mean_vec = np.mean(vectors, axis=0)
    mmag, mell, mb = gal(mean_vec[None, :])
    mell = float(mell[0])
    off = (ell - mell + 180.0) % 360.0 - 180.0
    return dict(mean_mag=float(mmag[0]), mean_ell=mell, mean_b=float(mb[0]),
                mag=np.percentile(mag, [16, 50, 84]),
                ell=(mell + np.percentile(off, [16, 50, 84])) % 360.0,
                b=np.percentile(b, [16, 50, 84]))


def show(tag, s):
    print(f"\n{tag}")
    print(f"  mean vector: |v| = {s['mean_mag']:.1f} km/s toward "
          f"(l, b) = ({s['mean_ell']:.1f}, {s['mean_b']:.1f}) deg")
    print(f"  |v| 16/50/84 = {s['mag'][0]:.0f} / {s['mag'][1]:.0f} / "
          f"{s['mag'][2]:.0f} km/s")
    print(f"  l   16/50/84 = {s['ell'][0]:.0f} / {s['ell'][1]:.0f} / "
          f"{s['ell'][2]:.0f} deg")
    print(f"  b   16/50/84 = {s['b'][0]:.0f} / {s['b'][1]:.0f} / "
          f"{s['b'][2]:.0f} deg")


def main():
    v = np.array([observer_velocity_one(FIELD_ROOT / f"mcmc_{i}.hdf5")
                  for i in range(NREAL)])
    print(f"Read {NREAL} ManticoreLocalCOLA (CIC) fields from {FIELD_ROOT}")
    show("Manticore velocity at observer (box centre):", summary(v))

    rng = np.random.default_rng(SEED)
    mag = rng.normal(VEXT_MAG, VEXT_MAG_E, NDRAW)
    ell = rng.normal(VEXT_ELL, VEXT_ELL_E, NDRAW)
    b = rng.normal(VEXT_B, VEXT_B_E, NDRAW)
    rhat = np.asarray(candel.galactic_to_radec_cartesian(ell, b))
    if rhat.shape[0] == 3:  # (3, N) -> (N, 3)
        rhat = rhat.T
    vext = mag[:, None] * rhat
    combined = v[rng.integers(0, NREAL, size=NDRAW)] + vext
    show("Local-Group-frame dipole = observer velocity + baseline Vext:",
         summary(combined))

    print(f"\nPlanck LG reference: |v| = {PLANCK[0]} km/s toward "
          f"(l, b) = ({PLANCK[1]}, {PLANCK[2]}) deg")


if __name__ == "__main__":
    main()
