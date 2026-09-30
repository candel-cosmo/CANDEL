# Copyright (C) 2026 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Plot Q (reddening-free index) against mW, logP and [O/H]."""
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401

from candel.util import data_path

plt.style.use(['science', 'no-latex'])

data_dir = data_path("data", "MWCepheids")
df = pd.read_csv(join(data_dir, "Riess2021_Table1_with_coords.csv"))
df["OH"] = df["FeH"] + 0.06

c22 = df[df["set"] == "Cycle22"]
c27 = df[df["set"] == "Cycle27"]

outliers = ["CP-CEP", "DR-VEL", "GQ-ORI"]

fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

xkeys = ["mW_H", "logP", "OH"]
xerr_keys = ["mW_H_err", None, None]
xlabels = [r"$m_W^H$ [mag]", r"$\log P$ [days]", r"$[\mathrm{O/H}]$ [dex]"]

for ax, xk, xek, xlab in zip(axes, xkeys, xerr_keys, xlabels):
    for subset, marker, col, label in [
        (c22, 'o', '#3c91e6', 'C22'),
        (c27, 's', '#c52233', 'C27'),
    ]:
        xerr = subset[xek] if xek else None
        ax.errorbar(subset[xk], subset["Q"], xerr=xerr, yerr=subset["Q_err"],
                    fmt=marker, ms=3, color=col, elinewidth=0.6, capsize=0,
                    alpha=0.85, label=label)

    for _, row in df[df["Cepheid"].isin(outliers)].iterrows():
        xval = row[xk] if xk != "OH" else row["FeH"] + 0.06
        ax.annotate(row["Cepheid"], (xval, row["Q"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=5.5)

    ax.set_xlabel(xlab)

axes[0].set_ylabel(r"$Q$")
axes[2].legend(frameon=True)

fig.tight_layout()
fig.savefig(join(data_dir, "..", "Q_panel.png"), dpi=300)
print("Saved Q_panel.png")
plt.show()
