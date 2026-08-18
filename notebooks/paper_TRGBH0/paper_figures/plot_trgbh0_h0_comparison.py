#!/usr/bin/env python
"""Plot TRGBH0 H0 posteriors against SH0ES and Planck bands."""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from stack_fields import (evidence_weights, load_fields,  # noqa: E402
                          stacked_samples, weighted_summary)
from trgbh0_plot_style import (FIGURE_DPI, OUTPUT_DIR,  # noqa: E402
                               TRGBH0_COLOURS, TRGBH0_RESULTS, paper_style,
                               save_figure)


matplotlib.use("Agg")
import scienceplots  # noqa: E402,F401

OUTDIR = OUTPUT_DIR
OUTNAME = "trgbh0_h0_comparison.pdf"

H0_COLOURS = {
    "density_sigv": TRGBH0_COLOURS[0],
    "student_t": TRGBH0_COLOURS[1],
    "planck": TRGBH0_COLOURS[2],
    "shoes": TRGBH0_COLOURS[3],
}

# Fiducial model: Student-t, beta=1, 48-pixel sky exposure. Both curves come
# from the same 80 single-field chains, stacked by evidence and with equal
# weight; there is no on-the-fly marginalised chain any more.
FIDUCIAL_GLOB = (
    TRGBH0_RESULTS / "single_fields_smoothed"
    / "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_bmin10_skyhp_nside2_k192_ManticoreLocalCOLA_field*_single_smoothed.hdf5"  # noqa: E501
)


REFERENCE_BANDS = [
    ("Planck", 67.4, 0.5, H0_COLOURS["planck"]),
    ("SH0ES", 73.04, 1.04, H0_COLOURS["shoes"]),
]


def kde_line(ax, samples, weights, label, color, fill=False, ls="-", bw=1.0):
    """Weighted KDE, so the evidence stack needs no resampling."""
    samples = np.asarray(samples).reshape(-1)
    lo, hi = weighted_quantiles(samples, weights, (0.001, 0.999))
    x = np.linspace(lo, hi, 500)
    kde = gaussian_kde(samples, weights=weights)
    kde.set_bandwidth(kde.factor * bw)
    y = kde(x)
    ax.plot(x, y, color=color, ls=ls, label=label)
    if fill:
        ax.fill_between(x, 0, y, color=color, alpha=0.20)


def weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values)
    x, w = values[order], np.asarray(weights)[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(quantiles, cdf, x)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    with paper_style(styles=("science",)):
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        for label, mean, sigma, color in REFERENCE_BANDS:
            ax.axvspan(
                mean - sigma, mean + sigma,
                color=color,
                alpha=0.18,
                lw=0,
                label=rf"{label} $\pm1\sigma$",
                zorder=0,
            )

        index, samples, lnz = load_fields(FIDUCIAL_GLOB)
        field_weights, n_eff, _ = evidence_weights(lnz)
        n_fields = index.size
        evidence = stacked_samples(samples["H0"], field_weights)
        equal = stacked_samples(samples["H0"])
        for name, stack in (("evidence", evidence), ("equal", equal)):
            median, std = weighted_summary(*stack)
            print(f"  {name}-stacked H0 = {median:.2f} +- {std:.2f}")
        print(f"  N_eff = {n_eff:.2f} of {n_fields}")

        curves = [
            (rf"Evidence-stacked ($N_{{\rm eff}}={n_eff:.2f}$)", evidence,
             H0_COLOURS["student_t"], "-", True),
            (rf"Equal-weight stacked (${n_fields}$ fields)", equal,
             H0_COLOURS["density_sigv"], "--", False),
        ]
        for label, (values, weights), color, ls, fill in curves:
            kde_line(ax, values, weights, label, color, fill=fill, ls=ls,
                     bw=1.5)

        ax.set_xlabel(
            r"$H_0 ~ [\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}]$")
        ax.set_ylabel("Normalised PDF")
        ax.set_xlim(56.0, 82.0)
        ax.set_ylim(bottom=0)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=6.5,
            handlelength=1.6,
            columnspacing=1.0,
            frameon=False,
        )

        fig.tight_layout()
        save_figure(fig, OUTNAME, output_dir=OUTDIR, dpi=FIGURE_DPI,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUTDIR / OUTNAME}")


if __name__ == "__main__":
    main()
