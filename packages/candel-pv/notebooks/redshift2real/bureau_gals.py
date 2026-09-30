# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Derive Bureau-galaxy cosmological-redshift posteriors."""
import argparse
import sys
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import matplotlib
import numpy as np
from h5py import File, string_dtype
from scipy.stats import binned_statistic

from candel.util import CANDEL_ROOT as ROOT  # noqa: E402
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import candel  # noqa: E402
import candel_pv  # noqa: E402
from candel.cosmo.cosmography import Distance2Redshift  # noqa: E402
from candel.field.los import load_los  # noqa: E402
from candel.field.los_prep import (  # noqa: E402
    compute_los_file_from_coordinates,
    reconstruction_field_indices,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


METHODS = {
    "Manticore": {
        "reconstruction": "ManticoreLocalCOLA",
        "which_MAS": "PCS",
        "calibration": ROOT / "results" / "test" / (
            "precomputed_los_ManticoreLocalCOLA_CF4_W1_double_powerlaw_"
            "MAS-PCS_beta_1.0.hdf5"),
    },
    "Carrick": {
        "reconstruction": "Carrick2015",
        "calibration": (
            "precomputed_los_Carrick2015_CF4_W1_"
            "double_powerlaw_paper.hdf5"),
    },
}
XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
CATALOGUE_COLUMNS = ("name", "RA (deg)", "DEC (deg)", "z", "z_err")
CALIBRATION_KEYS = (
    "sigma_v", "Vext", "beta", "b1", "alpha_low", "alpha_high",
    "alpha_high_frac", "log_rho_t", "log_rho_width",
)


def load_bureau_catalogue(path):
    """Read the fixed Bureau workbook without adding an Excel dependency."""
    ns = {"x": XLSX_NS}
    with ZipFile(path) as archive:
        shared = ElementTree.fromstring(
            archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(node.text or "" for node in item.iterfind(".//x:t", ns))
            for item in shared.findall("x:si", ns)
        ]
        sheet = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet.findall(".//x:sheetData/x:row", ns):
        values = {}
        for cell in row.findall("x:c", ns):
            column = "".join(x for x in cell.attrib["r"] if x.isalpha())
            value = cell.findtext("x:v", default="", namespaces=ns)
            if cell.attrib.get("t") == "s" and value:
                value = strings[int(value)]
            values[column] = value
        rows.append(values)

    header_index = next(
        i for i, row in enumerate(rows)
        if set(CATALOGUE_COLUMNS).issubset(row.values()))
    columns = {value: key for key, value in rows[header_index].items()}
    records = [
        row for row in rows[header_index + 1:]
        if row.get(columns["name"], "").strip()
    ]
    catalogue = {
        "name": np.asarray([
            row[columns["name"]].strip() for row in records]),
        "RA": np.asarray([
            float(row[columns["RA (deg)"]]) for row in records]),
        "dec": np.asarray([
            float(row[columns["DEC (deg)"]]) for row in records]),
        "zhelio": np.asarray([
            float(row[columns["z"]]) for row in records]),
        "e_zhelio": np.asarray([
            float(row[columns["z_err"]]) for row in records]),
    }
    if len(set(catalogue["name"])) != len(catalogue["name"]):
        raise ValueError("Galaxy names in the Bureau workbook are not unique")
    if not all(np.all(np.isfinite(x)) for key, x in catalogue.items()
               if key != "name"):
        raise ValueError("The Bureau workbook contains non-finite values")
    return catalogue


def load_calibration_samples(path, max_samples, seed=42):
    with File(path, "r") as handle:
        group = handle["samples"]
        samples = {
            key: group[key][...]
            for key in CALIBRATION_KEYS if key in group
        }
    if "alpha_high" not in samples and "alpha_high_frac" in samples:
        samples["alpha_high"] = (
            samples["alpha_low"] * samples.pop("alpha_high_frac"))

    required = {"sigma_v", "alpha_low", "alpha_high", "log_rho_t"}
    missing = required.difference(samples)
    if missing:
        raise KeyError(f"Missing calibration samples in {path}: {missing}")

    nsamples = len(samples["sigma_v"])
    if max_samples < nsamples:
        index = np.sort(
            np.random.default_rng(seed).choice(
                nsamples, size=max_samples, replace=False))
        samples = {key: value[index] for key, value in samples.items()}
    return samples


def radial_grid(rmin, rmax, dr):
    ninterval = int(np.ceil((rmax - rmin) / dr))
    if ninterval % 2:
        ninterval += 1
    return np.linspace(rmin, rmax, ninterval + 1)


def reconstruction_omega_m(config, reconstruction):
    kwargs = config["io"]["reconstruction_main"][reconstruction]
    if "Om0" in kwargs:
        return kwargs["Om0"]
    nsim = reconstruction_field_indices(config, reconstruction)[0]
    loader = candel.field.name2field_loader(reconstruction)(
        nsim=nsim, **kwargs)
    return loader.Omega_m


def run_method(settings, catalogue, config, r, calibration_dir, input_path,
               max_calibration, batch_size, overwrite_los,
               Vext_decay_start_z, Vext_decay_scale):
    reconstruction = settings["reconstruction"]
    if "which_MAS" in settings:
        config["io"]["reconstruction_main"][reconstruction]["which_MAS"] = (
            settings["which_MAS"])
    calibration_path = calibration_dir / settings["calibration"]
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing {calibration_path}")
    calibration = load_calibration_samples(
        calibration_path, max_calibration, seed=42)

    los_path = input_path.with_name(
        f"{input_path.stem}_LOS_{reconstruction}.hdf5")
    compute_los_file_from_coordinates(
        "generic", reconstruction, config, catalogue["RA"], catalogue["dec"],
        r=r, output_path=los_path, overwrite=overwrite_los)
    los = load_los(str(los_path), {})

    Om0 = reconstruction_omega_m(config, reconstruction)
    z_of_r = np.asarray(Distance2Redshift(Om0=Om0)(r))
    if not z_of_r[0] <= Vext_decay_start_z <= z_of_r[-1]:
        raise ValueError("Vext decay redshift lies outside the radial grid")
    Vext_decay_start = float(np.interp(
        Vext_decay_start_z, z_of_r, r))

    model = candel_pv.redshift2real.Redshift2Real(
        RA=catalogue["RA"], dec=catalogue["dec"],
        zcmb=catalogue["zcmb"], e_zcmb=catalogue["e_zcmb"],
        los_r=r, los_density=los["los_density"],
        los_velocity=los["los_velocity"], which_bias="double_powerlaw",
        calibration_samples=calibration, Rmin=r[0], Rmax=r[-1],
        num_rgrid=len(r), Om0=Om0,
        Vext_decay_start=Vext_decay_start,
        Vext_decay_scale=Vext_decay_scale)
    z_grid, log_posterior = model(batch_size=batch_size)
    return {
        "z_grid": z_grid,
        "log_posterior": log_posterior,
        "summary": model.posterior_summary(z_grid, log_posterior),
        "los_path": los_path,
        "field_indices": los["los_field_indices"],
        "calibration_path": calibration_path,
        "num_calibration": len(calibration["sigma_v"]),
        "Vext_decay_start_z": Vext_decay_start_z,
        "Vext_decay_start": Vext_decay_start,
        "Vext_decay_scale": Vext_decay_scale,
    }


def save_results(path, catalogue, results, source_path,
                 row_order="Original Bureau workbook order",
                 catalogue_keys=(
                     "RA", "dec", "zhelio", "e_zhelio", "zcmb", "e_zcmb"),
                 catalogue_attrs=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with File(path, "w") as handle:
        handle.attrs["source"] = str(source_path)
        handle.attrs["row_order"] = row_order
        handle.attrs["posterior_measure"] = "p(z_cosmo) dz_cosmo"
        inputs = handle.create_group("catalogue")
        inputs.create_dataset(
            "name", data=catalogue["name"].astype(object),
            dtype=string_dtype("utf-8"))
        for key in catalogue_keys:
            inputs.create_dataset(key, data=catalogue[key])
        for key, value in (catalogue_attrs or {}).items():
            inputs.attrs[key] = value

        for method, result in results.items():
            group = handle.create_group(method)
            group.attrs["los_path"] = str(result["los_path"])
            group.attrs["calibration_path"] = str(
                result["calibration_path"])
            group.attrs["num_calibration"] = result["num_calibration"]
            group.attrs["Vext_decay_start_z"] = (
                result["Vext_decay_start_z"])
            group.attrs["Vext_decay_start_mpc_h"] = (
                result["Vext_decay_start"])
            group.attrs["Vext_decay_scale_mpc_h"] = (
                result["Vext_decay_scale"])
            group.attrs["Vext_decay_form"] = (
                "1 below start; exp(-(r-r_start)/scale) above start")
            group.create_dataset("field_indices", data=result["field_indices"])
            z_grid = group.create_dataset("z_grid", data=result["z_grid"])
            z_grid.attrs["description"] = "Cosmological-redshift grid"
            log_posterior = group.create_dataset(
                "log_posterior", data=result["log_posterior"],
                compression="gzip")
            log_posterior.attrs["description"] = (
                "Normalized log p(z_cosmo | z_CMB); exponentiate for PDF")
            summary_names = {
                "mean": "zcosmo_mean", "std": "zcosmo_std",
                "map": "zcosmo_map", "median": "zcosmo_p50",
                "ci_low": "zcosmo_p16", "ci_high": "zcosmo_p84",
            }
            for key, name in summary_names.items():
                dataset = group.create_dataset(
                    name, data=result["summary"][key])
                dataset.attrs["description"] = (
                    f"Marginal posterior {key} of z_cosmo")
    print(f"Saved {path}")


def save_diagnostic_plots(output_dir, catalogue, results,
                          prefix="Bureau_gals"):
    colours = {"Manticore": "C0", "Carrick": "C3"}
    order = np.argsort(catalogue["zcmb"])
    shown = order[np.linspace(0, len(order) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for ax, index in zip(axes.flat, shown):
        for method, result in results.items():
            z_grid = result["z_grid"]
            posterior = np.exp(result["log_posterior"][index])
            ax.plot(candel.SPEED_OF_LIGHT * z_grid,
                    posterior / candel.SPEED_OF_LIGHT,
                    color=colours[method], label=method)
        means = np.asarray([
            x["summary"]["mean"][index] for x in results.values()])
        stds = np.asarray([
            x["summary"]["std"][index] for x in results.values()])
        low = min(catalogue["zcmb"][index], *(means - 5 * stds))
        high = max(catalogue["zcmb"][index], *(means + 5 * stds))
        ax.set_xlim(candel.SPEED_OF_LIGHT * low,
                    candel.SPEED_OF_LIGHT * high)
        ax.axvline(candel.SPEED_OF_LIGHT * catalogue["zcmb"][index],
                   color="k", linestyle="--", linewidth=1)
        ax.text(0.03, 0.92, catalogue["name"][index],
                transform=ax.transAxes, va="top")
        ax.set_xlabel(r"$cz_{\rm cosmo}$ [km s$^{-1}$]")
    axes[0, 0].set_ylabel(r"$p(cz_{\rm cosmo}\mid cz_{\rm CMB})$")
    axes[1, 0].set_ylabel(r"$p(cz_{\rm cosmo}\mid cz_{\rm CMB})$")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"{prefix}_posterior_examples.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.linspace(
        catalogue["zcmb"].min(), catalogue["zcmb"].max(), 8)
    centres = 0.5 * (bins[:-1] + bins[1:])
    for method, result in results.items():
        summary = result["summary"]
        residual = candel.SPEED_OF_LIGHT * (
            catalogue["zcmb"] - summary["mean"])
        error = candel.SPEED_OF_LIGHT * summary["std"]
        colour = colours[method]
        ax.errorbar(catalogue["zcmb"], residual, yerr=error, fmt="o",
                    markersize=2, linewidth=0.5, alpha=0.25, color=colour)
        mean = binned_statistic(
            catalogue["zcmb"], residual, statistic="mean", bins=bins
        ).statistic
        std = binned_statistic(
            catalogue["zcmb"], residual, statistic="std", bins=bins
        ).statistic
        mask = np.isfinite(mean) & np.isfinite(std)
        ax.errorbar(centres[mask], mean[mask], yerr=std[mask], fmt="s-",
                    markersize=4, linewidth=1.2, capsize=3, color=colour,
                    label=method)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Observed redshift $z_{\rm CMB}$")
    ax.set_ylabel(
        r"$c\,(z_{\rm CMB} - \langle z_{\rm cosmo}\rangle)$ "
        r"[km s$^{-1}$]")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"{prefix}_redshift_residuals.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for method, result in results.items():
        summary = result["summary"]
        median = summary["median"]
        residual = candel.SPEED_OF_LIGHT * (catalogue["zcmb"] - median)
        ax.errorbar(
            catalogue["zcmb"], residual,
            yerr=candel.SPEED_OF_LIGHT * np.asarray([
                summary["ci_high"] - median,
                median - summary["ci_low"],
            ]),
            fmt="o", markersize=2, linewidth=0.5, alpha=0.3,
            color=colours[method], label=method)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Observed redshift $z_{\rm CMB}$")
    ax.set_ylabel(
        r"$c\,(z_{\rm CMB} - z_{\rm cosmo})$ [km s$^{-1}$]")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / f"{prefix}_observed_vs_cosmological.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    ell, latitude = candel.radec_to_galactic(
        catalogue["RA"], catalogue["dec"])
    longitude = -np.deg2rad((ell + 180) % 360 - 180)
    fig, ax = plt.subplots(figsize=(8, 4.5),
                           subplot_kw={"projection": "mollweide"})
    ax.scatter(longitude, np.deg2rad(latitude), s=10, alpha=0.7, color="C0")
    ticks = np.arange(-150, 180, 30)
    ax.set_xticks(np.deg2rad(ticks))
    ax.set_xticklabels([rf"${(-tick) % 360:d}^\circ$" for tick in ticks])
    ax.set_xlabel(r"Galactic longitude $\ell$")
    ax.set_ylabel(r"Galactic latitude $b$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = output_dir / f"{prefix}_sky_map.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    if {"Manticore", "Carrick"}.issubset(results):
        manticore = results["Manticore"]["summary"]
        carrick = results["Carrick"]["summary"]
        speed = candel.SPEED_OF_LIGHT
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 8))
        axes[0].errorbar(
            speed * carrick["median"], speed * manticore["median"],
            xerr=speed * np.asarray([
                carrick["median"] - carrick["ci_low"],
                carrick["ci_high"] - carrick["median"],
            ]),
            yerr=speed * np.asarray([
                manticore["median"] - manticore["ci_low"],
                manticore["ci_high"] - manticore["median"],
            ]),
            fmt="o", markersize=2, linewidth=0.5, alpha=0.25, color="C4")
        limits = np.array([
            speed * min(carrick["ci_low"].min(),
                        manticore["ci_low"].min()),
            speed * max(carrick["ci_high"].max(),
                        manticore["ci_high"].max()),
        ])
        axes[0].plot(limits, limits, "k--", linewidth=1)
        axes[0].set(xlim=limits, ylim=limits,
                    xlabel=r"Carrick $cz_{\rm cosmo}$ [km s$^{-1}$]",
                    ylabel=r"Manticore $cz_{\rm cosmo}$ [km s$^{-1}$]")
        axes[0].set_aspect("equal", adjustable="box")

        difference = speed * (
            manticore["median"] - carrick["median"])
        axes[1].scatter(catalogue["zcmb"], difference, s=8,
                        alpha=0.5, color="C4")
        axes[1].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[1].set_xlabel(r"Observed redshift $z_{\rm CMB}$")
        axes[1].set_ylabel(
            r"$c\,(z_{\rm cosmo}^{\rm Manticore} - "
            r"z_{\rm cosmo}^{\rm Carrick})$ [km s$^{-1}$]")
        fig.tight_layout()
        path = output_dir / f"{prefix}_manticore_vs_carrick.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for method, result in results.items():
            summary = result["summary"]
            median = summary["median"]
            axes[0, 0].errorbar(
                catalogue["zcmb"], speed * (catalogue["zcmb"] - median),
                yerr=speed * np.asarray([
                    summary["ci_high"] - median,
                    median - summary["ci_low"],
                ]), fmt="o", markersize=2, linewidth=0.4, alpha=0.3,
                color=colours[method], label=method)
        axes[0, 0].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[0, 0].set_xlabel(r"Observed redshift $z_{\rm CMB}$")
        axes[0, 0].set_ylabel(
            r"$c\,(z_{\rm CMB}-z_{\rm cosmo})$ [km s$^{-1}$]")
        axes[0, 0].legend(frameon=False)

        axes[0, 1].scatter(
            speed * carrick["median"], speed * manticore["median"],
            s=8, alpha=0.4, color="C4")
        limits = np.asarray([
            speed * min(carrick["median"].min(),
                        manticore["median"].min()),
            speed * max(carrick["median"].max(),
                        manticore["median"].max()),
        ])
        axes[0, 1].plot(limits, limits, "k--", linewidth=1)
        axes[0, 1].set(
            xlim=limits, ylim=limits,
            xlabel=r"Carrick $cz_{\rm cosmo}$ [km s$^{-1}$]",
            ylabel=r"Manticore $cz_{\rm cosmo}$ [km s$^{-1}$]")
        axes[0, 1].set_aspect("equal", adjustable="box")

        axes[1, 0].scatter(
            catalogue["zcmb"], difference, s=8, alpha=0.5, color="C4")
        axes[1, 0].axhline(0, color="k", linestyle="--", linewidth=1)
        axes[1, 0].set_xlabel(r"Observed redshift $z_{\rm CMB}$")
        axes[1, 0].set_ylabel(
            r"$c\,(z_{\rm cosmo}^{\rm Manticore}-"
            r"z_{\rm cosmo}^{\rm Carrick})$ [km s$^{-1}$]")

        axes[1, 1].hist(difference, bins=25, color="C4", alpha=0.75)
        axes[1, 1].axvline(
            np.median(difference), color="k", linestyle="--", linewidth=1,
            label=rf"Median ${np.median(difference):.0f}$ km s$^{{-1}}$")
        axes[1, 1].set_xlabel(
            r"$c\,(z_{\rm cosmo}^{\rm Manticore}-"
            r"z_{\rm cosmo}^{\rm Carrick})$ [km s$^{-1}$]")
        axes[1, 1].set_ylabel("Number of galaxies")
        axes[1, 1].legend(frameon=False)
        fig.tight_layout()
        path = output_dir / f"{prefix}_summary.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
    print(f"Saved diagnostics to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--calibration-dir", type=Path,
                        default=ROOT / "results" / "VFO")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "redshift2real")
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
        raise FileNotFoundError(
            "Missing calibration file(s); copy them from glamdring or pass "
            f"--calibration-dir:\n{paths}")

    input_path = ROOT / "data" / "others" / "Bureau_gals.xlsx"
    catalogue = load_bureau_catalogue(input_path)
    catalogue["zcmb"], catalogue["e_zcmb"] = candel.heliocentric_to_cmb(
        catalogue["zhelio"], catalogue["RA"], catalogue["dec"],
        catalogue["e_zhelio"])
    r = radial_grid(args.rmin, args.rmax, args.dr)
    config = candel.load_config(
        PACKAGE_ROOT / "configs" / "config.toml")

    results = {}
    for method in args.methods:
        print(f"\nRunning {method}")
        results[method] = run_method(
            METHODS[method], catalogue, config, r, args.calibration_dir,
            input_path, args.num_calibration,
            args.batch_size, args.overwrite_los,
            args.Vext_decay_start_z, args.Vext_decay_scale)

    output = args.output_dir / "Bureau_gals_zcosmo.hdf5"
    save_results(output, catalogue, results, input_path)
    save_diagnostic_plots(args.output_dir, catalogue, results)


if __name__ == "__main__":
    main()
