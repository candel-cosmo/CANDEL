#!/usr/bin/env python
"""Write the TRGBH0 variants table from the single-field posterior ensembles.

Every variant is run once per Manticore realisation, so both footings of the
table come from the same chains: the evidence-stacked posterior, which is the
exact field marginalisation, and the equal-weight stack, which carries the
field-to-field scatter. The two evidence columns are the ensemble evidence
Z_ens and the matched-field paired difference; see stack_fields.py.

The no-reconstruction rows have no field ensemble and come from the single
chains under results/TRGBH0_paper/table/.
"""
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stack_fields import (LN10, evidence_weights,  # noqa: E402
                          field_median_scatter, load_fields,
                          matched_field_dlogz, stacked_samples,
                          weighted_summary)
from trgbh0_plot_style import (PAPER_DIR, TRGBH0_RESULTS,  # noqa: E402
                               TRGBH0_TABLE_RESULTS)

SMOOTHED = TRGBH0_RESULTS / "single_fields_smoothed"
UNSMOOTHED = TRGBH0_RESULTS / "single_fields"
# PAPER_DIR assumes the cluster layout (CANDEL and Papers as siblings), so fall
# back to the local output directory and let --output point at the paper.
TABLE = (PAPER_DIR if PAPER_DIR.is_dir() else PLOT_DIR / "output") \
    / "TRGBH0_variants_table.tex"

ST = "cz-student_t_"
BMIN = "sel-TRGB_magnitude_bmin10"
MANTICORE = "ManticoreLocalCOLA"
DASH = "--"


def unsmoothed(likelihood, sky):
    stem = (f"EDD_TRGB_{likelihood}MAS-PCS_{BMIN}"
            f"{sky}_{MANTICORE}_field*_single.hdf5")
    return UNSMOOTHED / stem


K192 = "_skyhp_nside2_k192"
K48 = "_skyhp_nside1_k48"

# (label, glob or single-chain path, is_ensemble). The Student-t runs carry the
# `cz-student_t_` token before MAS-PCS, so they are spelled out rather than
# threaded through another argument. Every row fixes beta=1; the baseline is
# therefore st(4, K192), the same ensemble the redshift-likelihood, smoothing
# and sky-exposure blocks quote as their Student-t reference.


def st(smooth, sky, suffix="", extra=""):
    return SMOOTHED / (
        f"EDD_TRGB_rhoSmoothR{smooth}_{extra}{ST}MAS-PCS_{BMIN}"
        f"{sky}_{MANTICORE}{suffix}_field*_single_smoothed.hdf5")


def ga(smooth, sky, suffix="", extra=""):
    return SMOOTHED / (
        f"EDD_TRGB_rhoSmoothR{smooth}_{extra}MAS-PCS_{BMIN}"
        f"{sky}_{MANTICORE}{suffix}_field*_single_smoothed.hdf5")


BLOCKS = [
    ("Redshift likelihood", [
        (r"{\bfseries\boldmath Student-$t$}", st(4, K192), True),
        ("Gaussian", ga(4, K192), False),
    ]),
    ("Density smoothing", [
        (r"$4\Mpch$, Student-$t$", st(4, K192), False),
        (r"$4\Mpch$, Gaussian", ga(4, K192), False),
        (r"$8\Mpch$, Student-$t$", st(8, K192), False),
        (r"$8\Mpch$, Gaussian", ga(8, K192), False),
        (r"No smoothing, Student-$t$", unsmoothed(ST, K192), False),
        (r"No smoothing, Gaussian", unsmoothed("", K192), False),
    ]),
    ("Angular sky exposure", [
        (r"$48$-pixel ($N_{\rm side}=2$), Student-$t$", st(4, K192), False),
        (r"$48$-pixel ($N_{\rm side}=2$), Gaussian", ga(4, K192), False),
        (r"$12$-pixel ($N_{\rm side}=1$), Student-$t$", st(4, K48), False),
        (r"$12$-pixel ($N_{\rm side}=1$), Gaussian", ga(4, K48), False),
        (r"No sky exposure, Student-$t$", st(4, ""), False),
        (r"No sky exposure, Gaussian", ga(4, ""), False),
    ]),
    ("Flow sector", [
        (r"Velocity monopole $V_{\rm mono}$, Student-$t$",
         st(4, K192, extra="Vmono_"), False),
        (r"Velocity monopole $V_{\rm mono}$, Gaussian",
         ga(4, K192, extra="Vmono_"), False),
        (r"No reconstruction, free $\Vext$, Student-$t$",
         TRGBH0_TABLE_RESULTS / f"EDD_TRGB_{ST}{BMIN}_Vext_main.hdf5", False),
        (r"No reconstruction, free $\Vext$, Gaussian",
         TRGBH0_TABLE_RESULTS / f"EDD_TRGB_{BMIN}_Vext_main.hdf5", False),
    ]),
]

KEYS = ("H0", "sigma_v")


def read_ensemble(pattern):
    index, samples, lnz = load_fields(pattern, KEYS)
    weights, n_eff, log_z_ens = evidence_weights(lnz)
    row = {"index": index, "lnz": lnz, "n_eff": n_eff,
           "log_z_ens": log_z_ens, "ensemble": True,
           "scatter": field_median_scatter(samples["H0"])}
    for key in KEYS:
        row[f"ev_{key}"] = weighted_summary(
            *stacked_samples(samples[key], weights))
        row[f"eq_{key}"] = weighted_summary(*stacked_samples(samples[key]))
    return row


def read_single(path):
    with h5py.File(path, "r") as handle:
        row = {"ensemble": False,
               "log_z_ens": float(handle["gof/lnZ_harmonic"][()]) / LN10}
        for key in KEYS:
            values = np.asarray(handle[f"samples/{key}"]).reshape(-1)
            row[f"ev_{key}"] = (float(np.median(values)),
                                float(np.std(values, ddof=1)))
            row[f"eq_{key}"] = None
    return row


def marked(text, track):
    return rf"\rvs{{{text}}}" if track else text


def pair(value, digits, track=False):
    if value is None:
        return DASH
    median, std = value
    text = f"${median:.{digits}f}\\pm{std:.{digits}f}$"
    return rf"\rvs{{{text}}}" if track else text


def sigma_pair(value, track=False):
    if value is None:
        return DASH
    median, std = value
    text = f"${median:.0f}\\pm{std:.0f}$"
    return rf"\rvs{{{text}}}" if track else text


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--no-track-changes", action="store_true",
                        help="Emit clean cells instead of \\rvs{} markers.")
    parser.add_argument("--output", type=Path, default=TABLE)
    args = parser.parse_args()
    track = not args.no_track_changes

    rows = []
    for block, entries in BLOCKS:
        for label, source, is_baseline in entries:
            source = Path(source)
            if "field*" in source.name:
                row = read_ensemble(source)
            else:
                row = read_single(source)
            row.update(label=label, block=block, baseline=is_baseline)
            rows.append(row)

    baseline = next(row for row in rows if row["baseline"])
    for row in rows:
        row["d_ens"] = row["log_z_ens"] - baseline["log_z_ens"]

    # The matched-field difference is pairwise, so both its mean and its
    # scatter depend on which row it is paired against. Find the reference row
    # in a first pass against the baseline, then recompute every pair against
    # that reference so the quoted scatter matches the quoted mean.
    def paired(row, reference):
        if not (row["ensemble"] and reference["ensemble"]):
            return None
        return matched_field_dlogz(row["index"], row["lnz"],
                                   reference["index"], reference["lnz"])

    ranked = [(paired(row, baseline), row) for row in rows]
    reference = max((r for r in ranked if r[0]), key=lambda r: r[0][0])[1]
    for row in rows:
        # The reference row is paired against itself, so it is exactly zero
        # with no scatter; print it without a spurious uncertainty.
        row["d_field"] = None if row is reference else paired(row, reference)
        row["is_reference"] = row is reference

    # The field-median scatter is the number the prose quotes as the spread
    # between realisations, so print it here rather than leaving it uncomputed.
    print(f"{'row':46s} {'Neff':>5s} {'dZens':>8s} {'dZfield':>14s} "
          f"{'H0 scatter':>10s}")
    for row in rows:
        field = ("--" if row["d_field"] is None
                 else f"{row['d_field'][0]:+.1f} +- {row['d_field'][1]:.1f}")
        n_eff = f"{row['n_eff']:.2f}" if row["ensemble"] else "--"
        scatter = f"{row['scatter']:.2f}" if row["ensemble"] else "--"
        print(f"{row['label'][:46]:46s} {n_eff:>5s} "
              f"{row['d_ens']:8.1f} {field:>14s} {scatter:>10s}")

    lines = []
    for block, entries in BLOCKS:
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{block}}}}} \\")
        for label, _, _ in entries:
            row = next(r for r in rows
                       if r["label"] == label and r["block"] == block)
            bold = row["baseline"]
            wrap = (lambda text: rf"{{\boldmath {text}}}") if bold else str
            d_ens = wrap("$0.0$" if bold
                         else marked(f"${row['d_ens']:.1f}$", track))
            if row["is_reference"]:
                d_field = "$0.0$"
            elif row["d_field"] is None:
                d_field = DASH
            else:
                d_field = (f"${row['d_field'][0]:.1f}\\pm"
                           f"{row['d_field'][1]:.1f}$")
            lines.append(
                " & ".join([
                    label,
                    wrap(pair(row["ev_H0"], 2, track)),
                    wrap(sigma_pair(row["ev_sigma_v"], track)),
                    d_ens,
                    wrap(pair(row["eq_H0"], 1)),
                    wrap(sigma_pair(row["eq_sigma_v"])),
                    wrap(d_field),
                ]) + r" \\")
        lines.append(r"\addlinespace")
    body = "\n".join(lines[:-1])

    print(f"\nRegenerated body for {args.output} "
          f"({len(rows)} rows); merge it into the table environment there.")
    args.output.with_suffix(".body.tex").write_text(body + "\n")
    print(f"Wrote {args.output.with_suffix('.body.tex')}")


if __name__ == "__main__":
    main()
