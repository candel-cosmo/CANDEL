"""Plot the CH0 mock forecast: fractional reduction in sigma(H0) relative to
the 35-host HST mock (M0), paired seed by seed."""
import os
import sys
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

sys.path.insert(0, os.path.dirname(__file__))
from mock_CH0 import OUT  # noqa: E402

SEEDS = range(10)


def _sigma(name):
    return np.array([np.load(join(OUT, f"{name}_seed{i:03d}.npz"))["H0"].std()
                     for i in SEEDS])


def gain(name, base):
    """Mean and standard error of 1 - sigma / sigma_M0 in per cent."""
    r = 100 * (1 - _sigma(name) / base)
    return r.mean(), r.std() / np.sqrt(len(r))


def main(fout):
    base = _sigma("M0")
    ks = (5, 10, 20)
    plt.style.use("science")
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for prefix, col, lab in [("M3", "C3", r"$3300$--$5000~\mathrm{km\,s^{-1}}$"),
                             ("M2", "C0", r"$< 3300~\mathrm{km\,s^{-1}}$")]:
        for tag, ls, q in [("", "-", "JWST"), ("h", "--", "HST")]:
            y = np.array([gain(f"{prefix}{tag}_k{k}", base) for k in ks])
            ax.errorbar(ks, y[:, 0], y[:, 1], color=col, ls=ls, marker="o",
                        ms=3, capsize=2, label=f"{q}, {lab}")
    m1, e1 = gain("M1", base)
    ax.axhspan(m1 - e1, m1 + e1, color="grey", alpha=0.3, lw=0,
               label="Current 35, errors halved")
    ax.set_xlabel("Number of added hosts")
    ax.set_ylabel(r"Reduction in $\sigma(H_0)$ [per cent]")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, None)
    # Absolute uncertainty, scaling the median 35-host HST mock uncertainty.
    s0 = np.median(base)
    sec = ax.secondary_yaxis("right", functions=(
        lambda g: s0 * (1 - g / 100), lambda s: 100 * (1 - s / s0)))
    sec.set_ylabel(r"$\sigma(H_0)~[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.savefig(fout, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
