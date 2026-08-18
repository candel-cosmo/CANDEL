#!/usr/bin/env python
"""Local Manticore velocity and the implied Local-Group dipole.

Two reconstruction-side quantities are cached per Manticore realisation of the
COLA/PCS ensemble that the TRGBH0 inference uses:

* the velocity interpolated at the observer position;
* the cone-averaged velocity over ``0 < r < rmax``, weighted by the volume
  element ``r^2 dr`` along HEALPix lines of sight and averaged over the sphere.

The Local-Group dipole is then the reconstruction velocity plus the inferred
external flow. The two are combined per realisation, pairing each field's
Vext posterior with that same field's reconstruction, and stacked over the
ensemble both equal-weight and by the single-field evidence.
"""
import argparse
import re
import sys
from pathlib import Path

import healpy as hp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(p for p in SCRIPT_DIR.parents if p.name == "paper_TRGBH0")
for _path in (SCRIPT_DIR, PLOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import candel  # noqa: E402
from candel.field.field_interp import build_regular_interpolator  # noqa: E402
from candel.field.loader import ManticoreLocalCOLA_FieldLoader  # noqa: E402
from stack_fields import (evidence_weights, load_fields,  # noqa: E402
                          stacked_samples, weighted_summary)
from trgbh0_plot_style import OUTPUT_DIR, TRGBH0_RESULTS  # noqa: E402

FIELD_ROOT = ("/mnt/extraspace/rstiskalek/MANTICORE/"
              "2MPP_MULTIBIN_N256_DES_V2/forward_fields")
BASELINE = (TRGBH0_RESULTS / "single_fields_smoothed" /
            ("EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_"
             "bmin10_skyhp_nside2_k192_ManticoreLocalCOLA_field*"
             "_single_smoothed.hdf5"))
CACHE_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-root", default=FIELD_ROOT)
    parser.add_argument("--which-mas", default="PCS")
    parser.add_argument("--baseline", type=Path, default=BASELINE,
                        help="Glob of the per-field baseline chains.")
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--rmax", type=float, default=15.0,
                        help="Cone-average outer radius in Mpc/h.")
    parser.add_argument("--num-radii", type=int, default=16)
    parser.add_argument("--cache", type=Path,
                        default=OUTPUT_DIR / "manticore_cola_velocity.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def realisation_indices(field_root, which_mas):
    folder = Path(field_root) / which_mas
    out = []
    for path in folder.glob("mcmc_*.hdf5"):
        match = re.fullmatch(r"mcmc_(\d+)\.hdf5", path.name)
        if match:
            out.append(int(match.group(1)))
    if not out:
        raise FileNotFoundError(f"No Manticore realisations in {folder}.")
    return np.asarray(sorted(out), dtype=np.int16)


def build_cache(args, realisations):
    """Observer-point and cone-averaged velocity per realisation."""
    radii = np.linspace(0.0, args.rmax, args.num_radii, dtype=np.float32)
    npix = hp.nside2npix(args.nside)
    theta, phi = hp.pix2ang(args.nside, np.arange(npix))
    ell, b = np.rad2deg(phi), 90.0 - np.rad2deg(theta)
    rhat = candel.galactic_to_radec_cartesian(ell, b).astype(np.float32)
    # Volume element r^2 dr along each line of sight.
    wr = radii**2
    wr = wr / np.trapezoid(wr, radii)

    v_obs = np.empty((len(realisations), 3), dtype=np.float32)
    v_cone = np.empty((len(realisations), 3), dtype=np.float32)
    sky_vrad = np.empty((len(realisations), npix), dtype=np.float32)

    for i, nsim in enumerate(realisations):
        loader = ManticoreLocalCOLA_FieldLoader(
            int(nsim), args.field_root, which_MAS=args.which_mas)
        velocity = loader.load_velocity()
        obs = np.asarray(loader.observer_pos, dtype=np.float32)
        points = (obs[None, None, :]
                  + radii[:, None, None] * rhat[None, :, :]).reshape(-1, 3)
        cone = np.empty((len(radii), npix, 3), dtype=np.float32)
        for comp in range(3):
            interp = build_regular_interpolator(
                velocity[comp], loader.boxsize, fill_value=None)
            v_obs[i, comp] = interp(obs[None, :])[0]
            cone[..., comp] = interp(points).reshape(len(radii), npix)
        # r^2 dr average along each line of sight, then over the sphere.
        cone_mean = np.trapezoid(cone * wr[:, None, None], radii, axis=0)
        v_cone[i] = cone_mean.mean(axis=0)
        sky_vrad[i] = np.einsum("pc,pc->p", cone_mean, rhat)
        print(f"realisation {int(nsim):3d} ({i + 1}/{len(realisations)}): "
              f"|v_obs| = {np.linalg.norm(v_obs[i]):6.1f}, "
              f"|v_cone| = {np.linalg.norm(v_cone[i]):6.1f} km/s", flush=True)
        del velocity, cone

    return {
        "cache_version": np.asarray(CACHE_VERSION, dtype=np.int16),
        "field_root": np.asarray(str(Path(args.field_root) / args.which_mas)),
        "realisations": realisations,
        "nside": np.asarray(args.nside, dtype=np.int16),
        "radii": radii,
        "ell": ell.astype(np.float32),
        "b": b.astype(np.float32),
        "rhat_icrs": rhat,
        "v_obs": v_obs,
        "v_cone": v_cone,
        "sky_vrad": sky_vrad,
    }


def load_or_build_cache(args, realisations):
    if args.cache.exists() and not args.rebuild_cache:
        with np.load(args.cache, allow_pickle=False) as data:
            cache = {key: data[key] for key in data.files}
        ok = (int(cache.get("cache_version", -1)) == CACHE_VERSION
              and np.array_equal(cache["realisations"], realisations)
              and int(cache["nside"]) == args.nside
              and len(cache["radii"]) == args.num_radii
              and np.isclose(cache["radii"][-1], args.rmax))
        if ok:
            print(f"Using cached Manticore velocities: {args.cache}")
            return cache
        print("Cache does not match the requested arguments; rebuilding.")
    cache = build_cache(args, realisations)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.cache, **cache)
    print(f"Cached Manticore velocities: {args.cache}")
    return cache


def galactic(vectors):
    """Magnitude and Galactic direction of ICRS Cartesian vectors."""
    mag, ell, b = candel.radec_cartesian_to_galactic(
        vectors[..., 0], vectors[..., 1], vectors[..., 2])
    return np.atleast_1d(mag), np.atleast_1d(ell), np.atleast_1d(b)


def mean_vector_summary(vectors):
    """Mean vector of an ensemble, with the scatter of its members."""
    mean = vectors.mean(axis=0)
    mag, ell, b = galactic(mean[None, :])
    mags, ells, bs = galactic(vectors)
    # Wrap longitudes about the mean direction before taking the scatter.
    dell = (ells - float(ell[0]) + 180.0) % 360.0 - 180.0
    return {
        "mag": float(mag[0]), "ell": float(ell[0]), "b": float(b[0]),
        "mag_std": float(mags.std(ddof=1)),
        "ell_std": float(dell.std(ddof=1)),
        "b_std": float(bs.std(ddof=1)),
    }


def stack_direction(per_field_vectors, weights=None):
    """Stack per-field dipole samples into magnitude and direction summaries.

    Each entry of ``per_field_vectors`` is that field's (n_samples, 3) set of
    Local-Group dipole draws. Magnitudes and Galactic angles are summarised by
    the same weighted median and standard deviation used for the tabulated
    parameters, with longitudes unwrapped about the ensemble mean direction.
    """
    mean_ell = float(galactic(np.concatenate(per_field_vectors).mean(
        axis=0)[None, :])[1][0])
    mags, ells, bs = [], [], []
    for vectors in per_field_vectors:
        mag, ell, b = galactic(vectors)
        mags.append(mag)
        ells.append((ell - mean_ell + 180.0) % 360.0 - 180.0)
        bs.append(b)
    out = {}
    for key, per_field in (("mag", mags), ("ell", ells), ("b", bs)):
        median, std = weighted_summary(*stacked_samples(per_field, weights))
        out[key] = (median + mean_ell if key == "ell" else median, std)
    out["ell"] = (out["ell"][0] % 360.0, out["ell"][1])
    return out


def main():
    args = parse_args()
    realisations = realisation_indices(args.field_root, args.which_mas)
    cache = load_or_build_cache(args, realisations)

    index, samples, lnz = load_fields(
        args.baseline, keys=("Vext", "H0"))
    weights, n_eff, _ = evidence_weights(lnz)
    if not np.array_equal(index, cache["realisations"].astype(int)):
        raise ValueError("Chain field indices and Manticore realisations "
                         "do not match.")
    # load_fields flattens; restore the (n_samples, 3) Vext shape.
    vext = [v.reshape(-1, 3) for v in samples["Vext"]]

    print("")
    print(f"realisations : {len(realisations)}")
    print(f"N_eff        : {n_eff:.2f}, dominant field "
          f"{index[np.argmax(weights)]}")
    print("")

    for name, key in (("observer point", "v_obs"),
                      (f"cone 0-{args.rmax:.0f} Mpc/h", "v_cone")):
        summary = mean_vector_summary(cache[key])
        print(f"Manticore {name:>18s}: |v| = {summary['mag']:5.1f} +/- "
              f"{summary['mag_std']:4.1f} km/s towards "
              f"(l, b) = ({summary['ell']:5.1f} +/- {summary['ell_std']:4.1f},"
              f" {summary['b']:5.1f} +/- {summary['b_std']:4.1f}) deg")
    print("")

    for name, key in (("observer point", "v_obs"),
                      (f"cone 0-{args.rmax:.0f} Mpc/h", "v_cone")):
        dipole = [v + cache[key][i][None, :] for i, v in enumerate(vext)]
        for label, w in (("evidence", weights), ("equal-weight", None)):
            out = stack_direction(dipole, w)
            print(f"LG dipole [{key}, {label:12s}]: "
                  f"|V| = {out['mag'][0]:5.1f} +/- {out['mag'][1]:4.1f} km/s "
                  f"towards (l, b) = ({out['ell'][0]:5.1f} +/- "
                  f"{out['ell'][1]:4.1f}, {out['b'][0]:5.1f} +/- "
                  f"{out['b'][1]:4.1f}) deg")
        print("")

    for label, w in (("evidence", weights), ("equal-weight", None)):
        out = stack_direction(vext, w)
        print(f"Vext only [{label:12s}]: |V| = {out['mag'][0]:5.1f} +/- "
              f"{out['mag'][1]:4.1f} km/s towards (l, b) = "
              f"({out['ell'][0]:5.1f} +/- {out['ell'][1]:4.1f}, "
              f"{out['b'][0]:5.1f} +/- {out['b'][1]:4.1f}) deg")


if __name__ == "__main__":
    main()
