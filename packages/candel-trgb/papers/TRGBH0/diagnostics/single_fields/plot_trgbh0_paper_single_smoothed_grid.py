#!/usr/bin/env python
"""Paper figure: TRGBH0 single-field smoothed model-variant grid.

Two stacked panels sharing the variant axis: field-median H0 distributions
(top) and matched-field harmonic evidence relative to the best variant
(bottom), over the 80 Manticore realisations of each variant.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(p for p in SCRIPT_DIR.parents if p.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: F401,E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from plot_trgbh0_single_smoothed_sets_diagnostics import (  # noqa: E402
    by_field_lnz, grouped_by_set, load_rows, matched_fields)
from trgbh0_plot_style import (PAPER_FIGURE_DIR, ROOT,  # noqa: E402
                               save_pdf_png, set_paper_rc)

DEFAULT_RESULTS_DIR = (
    ROOT / "results" / "TRGBH0_paper" / "single_fields_smoothed")
# Realisation-marginalised counterparts of the same variants (one chain each,
# the field marginalised inside the likelihood) live alongside the table runs.
DEFAULT_MARGINALISED_DIR = ROOT / "results" / "TRGBH0_paper" / "table"
DEFAULT_OUT = (
    SCRIPT_DIR.parents[1] / "output" / "trgbh0_single_smoothed_sets"
    / "trgbh0_single_smoothed_grid.pdf")
LN10 = np.log(10.0)
H0_LABEL = r"$H_0~[\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$"
# Variants grouped by the model axis under test, so competing variants sit
# next to each other. The group header carries the axis (sky resolution,
# monopole, smoothing), so each column label only needs its redshift
# likelihood and whether beta is free. Keys must match those built by
# `parse_run` ("sky"=48-pixel nside=2, "sky12"=12-pixel nside=1); cells with
# no run on disk are drawn as empty placeholders.
GAUSS = "Gaussian"
STUD = "Student-$t$"
GAUSS_B = "Gaussian,\nfree $\\beta$"
STUD_B = "Student-$t$,\nfree $\\beta$"
GROUPS = [
    ("No sky", [
        ("R4 Gauss", GAUSS),
        ("R4 Stud", STUD),
    ]),
    ("12-pixel sky", [
        ("R4 Gauss sky12", GAUSS),
        ("R4 Stud sky12", STUD),
    ]),
    ("48-pixel sky", [
        ("R4 Gauss sky", GAUSS),
        ("R4 Stud sky", STUD),
        ("R4 Gauss sky beta", GAUSS_B),
        ("R4 Stud sky beta", STUD_B),
    ]),
    ("Velocity monopole", [
        ("R4 Gauss Vmono sky", GAUSS),
        ("R4 Stud Vmono sky", STUD),
    ]),
    ("8 Mpc/h smoothing", [
        ("R8 Gauss sky", GAUSS),
        ("R8 Stud sky", STUD),
    ]),
]
# Colour the competing likelihoods distinctly.
LIKELIHOOD_COLOUR = {"Gauss": "#473198", "Stud": "#fe9000"}
# Realisation-marginalised overlay. Bright red, deliberately close in hue to
# the SH0ES band; the band is a pale unsaturated fill, so the saturated marker
# still reads against it.
MARGINALISED_COLOUR = "#e8000b"
MARGINALISED_DX = 0.26
GROUP_GAP = 0.9
VIOLIN_W = 0.72
PLANCK_H0 = 67.4
SHOES_H0 = 73.0
# Drawn at MNRAS \textwidth (504pt) so the figure is included at width=\textwidth
# and no down-scaling shrinks the type below the sizes set here.
FIG_WIDTH = 6.97
FIG_HEIGHT = 5.9
# Type and rule weights, chosen for a 1:1 include rather than the shrunk default.
RC = {"font.size": 9.0, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
      "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
      "axes.linewidth": 0.9, "xtick.major.width": 0.9,
      "ytick.major.width": 0.9, "xtick.minor.width": 0.7,
      "ytick.minor.width": 0.7}
LABEL_FS = 8.5


def _column_colour(key):
    return LIKELIHOOD_COLOUR["Stud" if "Stud" in key else "Gauss"]


def _marginalised_stem(key):
    """Rebuild the marginalised run stem from a grid column key.

    The single-field and marginalised runs share one naming scheme, so the
    column features (smoothing, monopole, likelihood, sky partition, free
    beta) map onto the stem directly rather than through a lookup table.
    """
    tokens = key.split()
    stem = f"EDD_TRGB_rhoSmooth{tokens[0]}"
    if "Vmono" in tokens:
        stem += "_Vmono"
    if "Stud" in tokens:
        stem += "_cz-student_t"
    stem += "_MAS-PCS_sel-TRGB_magnitude_bmin10"
    if "sky" in tokens:
        stem += "_skyhp_nside2_k192"
    elif "sky12" in tokens:
        stem += "_skyhp_nside1_k48"
    stem += "_ManticoreLocalCOLA"
    if "beta" in tokens:
        stem += "_beta_free"
    return stem + "_main"


def _marginalised_h0(key, results_dir):
    """(q16, q50, q84) of the marginalised H0 posterior, or None if absent."""
    path = Path(results_dir) / f"{_marginalised_stem(key)}.hdf5"
    if not path.is_file():
        return None
    with h5py.File(path, "r") as handle:
        samples = np.asarray(handle["samples/H0"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return None
    return np.percentile(samples, [16.0, 50.0, 84.0])


def _layout():
    """Flatten GROUPS into columns with x-positions and group separators."""
    cols, seps, centres = [], [], []
    x = 0.0
    for gi, (gname, items) in enumerate(GROUPS):
        if gi > 0:
            seps.append(x - 0.5 * GROUP_GAP)
        start = x
        for key, disp in items:
            cols.append({"key": key, "disp": disp, "x": x,
                         "colour": _column_colour(key)})
            x += 1.0
        centres.append((gname, 0.5 * (start + x - 1.0)))
        x += GROUP_GAP
    return cols, seps, centres


def _draw_violin(ax, x, vals, colour):
    """Violin (where the sample varies) plus jittered points and median."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if vals.size > 1 and np.std(vals) > 1e-9:
        parts = ax.violinplot([vals], positions=[x], widths=VIOLIN_W,
                              showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_linewidth(0.9)
            body.set_alpha(0.32)
    jitter = np.linspace(-0.15, 0.15, vals.size)
    ax.scatter(x + jitter, vals, s=7, color=colour, alpha=0.34,
               edgecolor="none")
    q16, q50, q84 = np.percentile(vals, [16.0, 50.0, 84.0])
    ax.errorbar(x, q50, yerr=[[q50 - q16], [q84 - q50]], fmt="o",
                color="black", ms=4.4, capsize=3.0, elinewidth=1.3,
                capthick=1.3, zorder=5)


def _mark_empty(ax, x):
    """Shade an absent variant column and label it `empty`."""
    ax.axvspan(x - 0.42, x + 0.42, color="0.85", alpha=0.35, lw=0, zorder=0)
    ax.text(x, 0.5, "empty", transform=ax.get_xaxis_transform(),
            rotation=90.0, ha="center", va="center", fontsize=LABEL_FS - 1.0,
            color="0.5", fontstyle="italic")


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--marginalised-dir", type=Path,
                        default=DEFAULT_MARGINALISED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--paper-figdir", type=Path, default=None,
        help="If set, also copy the PDF into this paper Figures directory.")
    return parser.parse_args()


def build_figure(groups, out_pdf, marginalised_dir=DEFAULT_MARGINALISED_DIR):
    cols, seps, centres = _layout()
    present = [c for c in cols if c["key"] in groups]
    fields = matched_fields({c["key"]: groups[c["key"]] for c in present})
    lnz_map = by_field_lnz({c["key"]: groups[c["key"]] for c in present})
    mean_lnz = {c["key"]: np.mean([lnz_map[c["key"]][f] for f in fields])
                for c in present}
    best = max(mean_lnz, key=mean_lnz.get)

    with plt.style.context(["science", "no-latex"]):
        set_paper_rc(RC)
        fig, (ax_h0, ax_z) = plt.subplots(
            2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True,
            constrained_layout=True, height_ratios=(1.3, 1.0))

        # --- Top: field-median H0 distributions ---
        ax_h0.axhspan(PLANCK_H0 - 0.5, PLANCK_H0 + 0.5, color="0.7",
                      alpha=0.30, lw=0)
        ax_h0.axhspan(SHOES_H0 - 1.0, SHOES_H0 + 1.0, color="#ef476f",
                      alpha=0.16, lw=0)
        ax_h0.axhline(PLANCK_H0, color="0.45", lw=1.1, ls="--")
        ax_h0.axhline(SHOES_H0, color="#ef476f", lw=1.1, ls="--")
        for c in cols:
            if c["key"] in groups:
                medians = np.asarray(
                    [r["H0_q50"] for r in groups[c["key"]]], float)
                _draw_violin(ax_h0, c["x"], medians, c["colour"])
            else:
                _mark_empty(ax_h0, c["x"])
            marg = _marginalised_h0(c["key"], marginalised_dir)
            if marg is not None:
                q16, q50, q84 = marg
                ax_h0.errorbar(c["x"] + MARGINALISED_DX, q50,
                               yerr=[[q50 - q16], [q84 - q50]], fmt="s",
                               color=MARGINALISED_COLOUR, ms=4.0, capsize=3.0,
                               lw=1.5, capthick=1.5, zorder=6)
        ax_h0.text(0.008, 0.965, "Planck", transform=ax_h0.transAxes,
                   ha="left", va="top", fontsize=LABEL_FS, color="0.4",
                   fontstyle="italic")
        ax_h0.text(0.008, 0.035, "SH0ES", transform=ax_h0.transAxes,
                   ha="left", va="bottom", fontsize=LABEL_FS, color="#ef476f")
        ax_h0.set_ylabel(H0_LABEL)
        # Headroom for the legend, and a little foot room so the violin tails
        # are not clipped by the autoscaled limit.
        y0, y1 = ax_h0.get_ylim()
        ax_h0.set_ylim(y0 - 0.03 * (y1 - y0), y1 + 0.15 * (y1 - y0))
        legend_handles = [
            Patch(facecolor=LIKELIHOOD_COLOUR["Gauss"], alpha=0.5,
                  label="Gaussian"),
            Patch(facecolor=LIKELIHOOD_COLOUR["Stud"], alpha=0.5,
                  label="Student-$t$"),
            Line2D([], [], color=MARGINALISED_COLOUR, marker="s", ms=4.0,
                   lw=1.5, label="Field-marginalised")]
        ax_h0.legend(handles=legend_handles, loc="upper right", frameon=False,
                     fontsize=LABEL_FS, handlelength=1.4, ncol=3,
                     columnspacing=1.4, borderaxespad=0.3)

        # --- Bottom: matched-field harmonic evidence, per-realisation ---
        for c in cols:
            if c["key"] in groups:
                dz = np.asarray(
                    [(lnz_map[c["key"]][f] - lnz_map[best][f]) / LN10
                     for f in fields], float)
                _draw_violin(ax_z, c["x"], dz, c["colour"])
            else:
                _mark_empty(ax_z, c["x"])
        ax_z.axhline(0.0, color="0.35", lw=1.1, ls="--")
        ax_z.set_ylabel(r"$\Delta\log_{10} Z_{\rm harm}$")

        # --- Group separators, headers, and variant tick labels ---
        for sx in seps:
            for ax in (ax_h0, ax_z):
                ax.axvline(sx, color="0.75", lw=0.9, ls=":", zorder=0)
        for gname, cx in centres:
            ax_h0.text(cx, 1.02, gname, transform=ax_h0.get_xaxis_transform(),
                       ha="center", va="bottom", fontsize=LABEL_FS + 0.5,
                       color="0.25")
        ax_z.set_xticks([c["x"] for c in cols])
        # Columns sit ~0.4 in apart, too narrow for horizontal labels at this
        # type size, so rotate them and anchor the right end under the tick.
        ax_z.set_xticklabels([c["disp"].replace("\n", " ") for c in cols],
                             fontsize=LABEL_FS, rotation=40.0, ha="right",
                             rotation_mode="anchor")
        ax_z.set_xlim(cols[0]["x"] - 0.7, cols[-1]["x"] + 0.7)
        return save_pdf_png(fig, out_pdf)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results_dir)
    groups = grouped_by_set(rows)
    pdf, png = build_figure(groups, args.out, args.marginalised_dir)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    figdir = args.paper_figdir or PAPER_FIGURE_DIR
    if figdir is not None and Path(figdir).is_dir():
        dest = Path(figdir) / pdf.name
        dest.write_bytes(pdf.read_bytes())
        print(f"Copied {dest}")


if __name__ == "__main__":
    main()
