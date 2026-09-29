"""Tabulate H0 mean and std from the `CH0_JWST_forecast` task outputs."""
from os.path import exists, join

import h5py
import numpy as np

ROOT = "results/CH0_paper/jwst_forecast"

ROWS = [
    ("CH0_noVext_sel-{sel}{v}_jwst", "No pec. vel., sigma_v"),
    ("CH0_sel-{sel}{v}_Vext_jwst", "Constant Vext, sigma_v"),
    ("CH0_noVext_sel-{sel}{v}_PV_covmat{nf}_jwst", "LCDM cov., sigma_v"),
    ("CH0_noVext_sel-{sel}{v}_PV_covmat_PV_covmat_scaling{nf}_jwst",
     "LCDM cov., A, sigma_v"),
    ("CH0_sel-{sel}{v}_Carrick2015_jwst", "Carrick2015, Vext, sigma_v"),
    ("CH0_MAS-PCS_sel-{sel}{v}_ManticoreLocalCOLA_jwst",
     "Manticore, Vext, sigma_v"),
    ("CH0_MAS-PCS_sel-{sel}{v}_ManticoreLocalCOLA_sigv_rho_jwst",
     "Manticore, Vext, sigma_v(delta)"),
    ("CH0_MAS-PCS_sel-{sel}{v}_ManticoreLocalCOLA_beta_free_jwst",
     "Manticore, Vext, sigma_v, beta"),
]

VARIANTS = [
    ("", "HST (as is)"),
    ("_ceperr0p5-35", "all 35 SN hosts x0.5"),
    ("_ceperr0p5-all", "SN hosts + anchors x0.5"),
    ("_ceperr0p5-19", "19 hosts NOT on list x0.5"),
    ("_keep16hosts", "list only, HST"),
    ("_keep16hosts_ceperr0p5-16", "list only, x0.5"),
]

SELECTIONS = [("SN_magnitude", "SN mag."), ("redshift", "Redshift")]


def load_H0(stem):
    fname = join(ROOT, f"{stem}.hdf5")
    if not exists(fname):
        return None
    with h5py.File(fname, "r") as f:
        return np.asarray(f["samples/H0"])


def main():
    for sel, sel_label in SELECTIONS:
        print(f"\n### Selection: {sel_label} -- H0 mean +- std "
              "[km/s/Mpc]\n")
        print("| PV model | " + " | ".join(v[1] for v in VARIANTS) + " |")
        print("|---" * (len(VARIANTS) + 1) + "|")
        for pattern, label in ROWS:
            nf = "_weight_by_Neff" if sel == "redshift" else ""
            cells = []
            for v, _ in VARIANTS:
                stem = pattern.format(sel=sel, v=v, nf=nf)
                H0 = load_H0(stem)
                cells.append("--" if H0 is None
                             else f"{H0.mean():.2f} +- {H0.std():.2f}")
            print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
