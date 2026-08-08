#!/usr/bin/env python
"""Histogram of the H0 standardised bias across the TRGB mock suite."""
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trgbh0_plot_style import (FIGURE_DPI, OUTPUT_DIR, ROOT,  # noqa: E402
                               TRGBH0_COLOURS, paper_style, save_figure)

matplotlib.use("Agg")
import scienceplots  # noqa: E402,F401

MOCK_NPZ = (
    ROOT / "results" / "mocks_TRGB"
    / "mock_TRGB_biases_TRGB_magnitude_field_ManticoreLocalCOLA42"
    "_infersel_student_t_bmin10_smooth4_0_gpu_merged.npz"
)
OUTNAME = "trgbh0_mock_h0_bias.pdf"


def main():
    with np.load(MOCK_NPZ, allow_pickle=True) as d:
        bias = np.asarray(d["H0"], float)  # (post. mean - truth) / post. std
    n = bias.size
    ks = stats.kstest(bias, "norm")
    print(f"n={n}  mean={bias.mean():+.3f}  std={bias.std(ddof=1):.3f}  "
          f"KS p={ks.pvalue:.3f}")

    with paper_style(styles=("science",)):
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        ax.hist(bias, bins=np.linspace(-4, 4, 33), density=True,
                color=TRGBH0_COLOURS[1], alpha=0.55,
                label=f"{n} mocks")
        x = np.linspace(-4, 4, 400)
        ax.plot(x, stats.norm.pdf(x), color="k", lw=1.1,
                label=r"$\mathcal{N}(0,\,1)$")
        ax.set_xlabel(r"$(\langle H_0\rangle - H_0^{\rm true})/\sigma_{H_0}$")
        ax.set_ylabel("Probability density function")
        ax.set_xlim(-4, 4)
        ax.legend(frameon=False)
        fig.tight_layout()
        save_figure(fig, OUTNAME, output_dir=OUTPUT_DIR, dpi=FIGURE_DPI,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {OUTPUT_DIR / OUTNAME}")


if __name__ == "__main__":
    main()
