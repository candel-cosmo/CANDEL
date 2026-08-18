#!/usr/bin/env python
"""Compare the Gaussian and Student-t redshift-residual likelihoods.

Illustrates why the two inferred sigma_v values are not directly comparable:
the Student-t (fiducial) has a taller, narrower core but much heavier tails
than the Gaussian, so the likelihood peaks at a smaller scale under the
Student-t.
Both densities are evaluated with the exact candel likelihood kernels
(``normal_logpdf_var`` / ``student_t_logpdf_var``), not a re-implementation.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from candel.model.utils import normal_logpdf_var, student_t_logpdf_var

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trgbh0_plot_style import (FIGURE_DPI, OUTPUT_DIR,  # noqa: E402
                               TRGBH0_COLOURS, paper_style, save_figure)

matplotlib.use("Agg")
import scienceplots  # noqa: E402,F401

OUTNAME = "trgbh0_cz_likelihood_pdfs.pdf"

# Equal-weight stacked fixed-beta redshift-likelihood posteriors, which is the
# fiducial footing (paper Sec. 4.3). Gaussian sigma_v is the residual std;
# Student-t sigma_v is the *scale* (var = sigma_v^2), so its distribution std
# is sigma_v*sqrt(nu/(nu-2)).
GAUSS_SIGMA_V = 132.0   # km/s
STUDENT_SIGMA_V = 67.0  # km/s (scale)
STUDENT_NU = 2.59


def main():
    x = np.linspace(-600.0, 600.0, 2001)

    pdf_gauss = np.exp(np.asarray(
        normal_logpdf_var(x, 0.0, GAUSS_SIGMA_V**2)))
    pdf_student = np.exp(np.asarray(
        student_t_logpdf_var(x, 0.0, STUDENT_SIGMA_V**2, STUDENT_NU)))

    with paper_style(styles=("science",)):
        fig, ax = plt.subplots(figsize=(3.45, 2.55))

        # sigma_v and nu are quoted in the caption, not repeated in the legend.
        ax.plot(x, pdf_gauss, color=TRGBH0_COLOURS[0], ls="-",
                label="Gaussian")
        ax.plot(x, pdf_student, color=TRGBH0_COLOURS[1], ls="--",
                label=r"Student-$t$")

        ax.set_yscale("log")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(1e-6, 1e-2)
        ax.set_xlabel(
            r"$c z_{\rm obs} - c z_{\rm pred} ~ "
            r"[\mathrm{km}\,\mathrm{s}^{-1}]$")
        ax.set_ylabel("Redshift-residual PDF")
        ax.legend(loc="upper left", frameon=False, handlelength=1.6,
                  fontsize=6.5)

        fig.tight_layout()
        save_figure(fig, OUTNAME, output_dir=OUTPUT_DIR, dpi=FIGURE_DPI,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUTPUT_DIR / OUTNAME}")


if __name__ == "__main__":
    main()
