# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Prepare and optionally run WISDOM CO-TFR redshift2real posteriors."""
import argparse
import csv
import sys
from pathlib import Path

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import ICRS, LSRK, SkyCoord

from candel.util import CANDEL_ROOT as ROOT  # noqa: E402
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import candel  # noqa: E402
from notebooks.redshift2real.bureau_gals import (  # noqa: E402
    METHODS,
    radial_grid,
    run_method,
    save_diagnostic_plots,
    save_results,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


PREFIX = "WISDOM_CO_TFR"
COLUMNS = {
    "name": "Galaxy name",
    "RA": "RA",
    "dec": "Dec.",
    "z_lsrk": "Redshift",
}


def lsrk_to_barycentric(z_lsrk, RA, dec):
    """Transform physical redshifts from LSRK to barycentric."""
    z_lsrk = np.asarray(z_lsrk, dtype=float)
    if np.any(z_lsrk <= -1):
        raise ValueError("Redshifts must be greater than -1")
    zeros = np.zeros_like(z_lsrk)
    coordinates = SkyCoord(
        ra=np.asarray(RA) * u.deg, dec=np.asarray(dec) * u.deg,
        distance=np.ones_like(z_lsrk) * u.Mpc,
        pm_ra_cosdec=zeros * u.mas / u.yr,
        pm_dec=zeros * u.mas / u.yr,
        radial_velocity=zeros * u.km / u.s, frame=LSRK())
    correction = coordinates.transform_to(ICRS()).radial_velocity.to_value(
        u.km / u.s)
    beta = correction / candel.SPEED_OF_LIGHT
    factor = np.sqrt((1 + beta) / (1 - beta))
    return (1 + z_lsrk) * factor - 1, correction, factor


def load_wisdom_catalogue(path, velocity_error_kms=None):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS.values()).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("The WISDOM catalogue is empty")

    catalogue = {
        key: np.asarray([row[column].strip() for row in rows])
        if key == "name" else
        np.asarray([float(row[column]) for row in rows])
        for key, column in COLUMNS.items()
    }
    if len(set(catalogue["name"])) != len(catalogue["name"]):
        raise ValueError("Galaxy names in the WISDOM catalogue are not unique")
    if not all(np.all(np.isfinite(value)) for key, value in catalogue.items()
               if key != "name"):
        raise ValueError("The WISDOM catalogue contains non-finite values")
    if np.any((catalogue["RA"] < 0) | (catalogue["RA"] >= 360)):
        raise ValueError("RA must lie in [0, 360) degrees")
    if np.any(np.abs(catalogue["dec"]) > 90):
        raise ValueError("Declination must lie in [-90, 90] degrees")

    (catalogue["zbary"], catalogue["lsrk_to_bary_kms"], factor) = (
        lsrk_to_barycentric(
            catalogue["z_lsrk"], catalogue["RA"], catalogue["dec"]))
    catalogue["v_radio_lsrk"] = (
        candel.SPEED_OF_LIGHT * catalogue["z_lsrk"]
        / (1 + catalogue["z_lsrk"]))

    if velocity_error_kms is None:
        catalogue["zcmb"] = candel.heliocentric_to_cmb(
            catalogue["zbary"], catalogue["RA"], catalogue["dec"])
    else:
        if not np.isfinite(velocity_error_kms) or velocity_error_kms < 0:
            raise ValueError("velocity-error-kms must be finite and non-negative")
        catalogue["e_zbary"] = (
            velocity_error_kms / candel.SPEED_OF_LIGHT
            * (1 + catalogue["z_lsrk"])**2 * factor)
        catalogue["zcmb"], catalogue["e_zcmb"] = (
            candel.heliocentric_to_cmb(
                catalogue["zbary"], catalogue["RA"], catalogue["dec"],
                catalogue["e_zbary"]))
    return catalogue


def save_prepared_catalogue(output_dir, catalogue):
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = [
        "name", "RA", "dec", "z_lsrk", "v_radio_lsrk",
        "lsrk_to_bary_kms", "zbary", "zcmb",
    ]
    keys += [key for key in ("e_zbary", "e_zcmb") if key in catalogue]
    path = output_dir / f"{PREFIX}_prepared.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        writer.writerows(zip(*(catalogue[key] for key in keys)))
    print(f"Saved {path}")


def save_input_plots(output_dir, catalogue):
    output_dir.mkdir(parents=True, exist_ok=True)
    bins = np.histogram_bin_edges(
        np.concatenate((catalogue["z_lsrk"], catalogue["zcmb"])),
        bins="auto")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(catalogue["z_lsrk"], bins=bins, histtype="step",
            linewidth=1.5, label=r"Quoted $z_{\rm LSRK}$")
    ax.hist(catalogue["zcmb"], bins=bins, histtype="step",
            linewidth=1.5, label=r"Converted $z_{\rm CMB}$")
    ax.set_xlabel("Observed redshift")
    ax.set_ylabel("Number of galaxies")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"{PREFIX}_redshift_distribution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")

    ell, latitude = candel.radec_to_galactic(
        catalogue["RA"], catalogue["dec"])
    longitude = -np.deg2rad((ell + 180) % 360 - 180)
    fig, ax = plt.subplots(figsize=(8, 4.5),
                           subplot_kw={"projection": "mollweide"})
    ax.scatter(longitude, np.deg2rad(latitude), s=14, alpha=0.75,
               color="C0")
    ticks = np.arange(-150, 180, 30)
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels([rf"${(-tick) % 360:d}^\circ$" for tick in ticks])
    ax.set_xlabel(r"Galactic longitude $\ell$")
    ax.set_ylabel(r"Galactic latitude $b$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / f"{PREFIX}_sky_distribution.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "redshift2real")
    parser.add_argument("--run", action="store_true",
                        help="Run the Manticore/Carrick posteriors after prep")
    parser.add_argument("--velocity-error-kms", type=float,
                        help="Common 1-sigma systemic-velocity error")
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--calibration-dir", type=Path,
                        default=ROOT / "results" / "VFO")
    parser.add_argument("--num-calibration", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rmin", type=float, default=0.001)
    parser.add_argument("--rmax", type=float, default=320.0)
    parser.add_argument("--dr", type=float, default=0.25)
    parser.add_argument("--Vext-decay-start-z", type=float, default=0.05)
    parser.add_argument("--Vext-decay-scale", type=float, default=25.0)
    parser.add_argument("--overwrite-los", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run and args.velocity_error_kms is None:
        raise ValueError(
            "--run requires --velocity-error-kms (use 0 only to explicitly "
            "ignore measurement uncertainty)")

    catalogue = load_wisdom_catalogue(
        args.input, velocity_error_kms=args.velocity_error_kms)
    save_prepared_catalogue(args.output_dir, catalogue)
    save_input_plots(args.output_dir, catalogue)
    print(
        f"Loaded {len(catalogue['name'])} galaxies; "
        f"z_CMB range {catalogue['zcmb'].min():.6f} to "
        f"{catalogue['zcmb'].max():.6f}.")
    if not args.run:
        print("Posterior inference not requested; add --run and a velocity error.")
        return

    if args.num_calibration < 1 or args.batch_size < 1:
        raise ValueError("num-calibration and batch-size must be positive")
    if not 0 < args.rmin < args.rmax or args.dr <= 0:
        raise ValueError("Require 0 < rmin < rmax and dr > 0")
    if args.Vext_decay_start_z <= 0 or args.Vext_decay_scale <= 0:
        raise ValueError("Vext decay start and scale must be positive")
    missing = [
        args.calibration_dir / METHODS[method]["calibration"]
        for method in args.methods
        if not (args.calibration_dir / METHODS[method]["calibration"]).exists()
    ]
    if missing:
        paths = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing calibration file(s):\n{paths}")

    r = radial_grid(args.rmin, args.rmax, args.dr)
    config = candel.load_config(
        PACKAGE_ROOT / "configs" / "config.toml")
    los_base = args.output_dir / args.input.name
    results = {}
    for method in args.methods:
        print(f"\nRunning {method}")
        results[method] = run_method(
            METHODS[method], catalogue, config, r, args.calibration_dir,
            los_base, args.num_calibration, args.batch_size,
            args.overwrite_los, args.Vext_decay_start_z,
            args.Vext_decay_scale)

    output = args.output_dir / f"{PREFIX}_zcosmo.hdf5"
    save_results(
        output, catalogue, results, args.input,
        row_order="Original WISDOM CSV order",
        catalogue_keys=(
            "RA", "dec", "z_lsrk", "v_radio_lsrk",
            "lsrk_to_bary_kms", "zbary", "e_zbary", "zcmb", "e_zcmb"),
        catalogue_attrs={
            "input_velocity_frame": "LSRK",
            "input_redshift_provenance": (
                "Derived by source team from radio-convention LSRK velocity"),
            "barycentric_frame": "Astropy LSRK to ICRS",
            "velocity_error_kms": args.velocity_error_kms,
        })
    save_diagnostic_plots(
        args.output_dir, catalogue, results, prefix=PREFIX)


if __name__ == "__main__":
    main()
