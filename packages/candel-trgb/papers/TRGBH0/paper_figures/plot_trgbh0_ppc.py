#!/usr/bin/env python
"""Posterior predictive check for the fiducial TRGBH0 model.

Draws the fiducial Student-t Manticore-Local posterior (4 Mpc/h density
smoothing, beta=1) through the forward model and compares the predicted
tip-magnitude, redshift, distance, and sky distributions with the observed
EDD TRGB hosts. Writes the magnitude-redshift PPC (main text), the sky PPC
(appendix), and the distance PPC.

Three variants are supported: ``fiducial`` is the baseline model (48-pixel
HEALPix angular sky-exposure), evaluated on its dominant realisation (field
66); ``noexp`` is the same realisation without the sky-exposure term; and
``skyexp`` is the superseded 12-pixel run. The sky-exposure term is required
for the angular PPC to reproduce the observed host concentration.
"""
import argparse
import copy
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(p for p in SCRIPT_DIR.parents if p.name == "paper_TRGBH0")
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import tomli_w  # noqa: E402
from trgbh0_plot_style import (OUTPUT_DIR, ROOT,  # noqa: E402
                               TRGBH0_RESULTS, TRGBH0_TABLE_RESULTS)

import candel  # noqa: E402
import candel_trgb  # noqa: E402
from candel_trgb import (generate_trgb_ppc, plot_trgb_ppc,  # noqa: E402
                         plot_trgb_ppc_distance, plot_trgb_ppc_sky)

matplotlib.use("Agg")

# Baseline run: Student-t, 4 Mpc/h smoothing, beta=1, 48-pixel angular
# sky-exposure. The evidence weights over the 80 realisations have an effective
# sample of ~1 field, so the ensemble collapses onto its single dominant
# realisation; the PPC therefore uses that field's single-field posterior and
# realisation. Field 66 dominates (highest harmonic evidence, median H0=72.3
# and Vext toward (285, -3), matching the evidence-stacked result). The
# no-exposure check uses the same realisation so that the comparison isolates
# the sky-exposure term; the 12-pixel run is kept for reference.
GENERATED = ROOT / "scripts/runs/generated_configs/TRGBH0_main"
SINGLE_SMOOTHED = TRGBH0_RESULTS / "single_fields_smoothed"
_STEM = "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_bmin10_"
FIELD = 66

# variant -> (config stem, posterior path, Manticore realisation index).
VARIANTS = {
    "fiducial": (
        _STEM + "skyhp_nside2_k192_ManticoreLocalCOLA_main",
        SINGLE_SMOOTHED / (_STEM + "skyhp_nside2_k192_ManticoreLocalCOLA"
                           f"_field{FIELD}_single_smoothed.hdf5"),
        FIELD),
    "noexp": (
        _STEM + "ManticoreLocalCOLA_main",
        SINGLE_SMOOTHED / (_STEM + "ManticoreLocalCOLA"
                           f"_field{FIELD}_single_smoothed.hdf5"),
        FIELD),
    "skyexp": (
        _STEM + "skyhp_nside1_k48_ManticoreLocalCOLA_main",
        TRGBH0_TABLE_RESULTS / (_STEM + "skyhp_nside1_k48"
                                "_ManticoreLocalCOLA_main.hdf5"),
        0),
}
SEED = 42
N_WORKERS = 6


def load_observed_data(config):
    """Load only catalogue observables, without reconstruction products."""
    data_config = copy.deepcopy(config)
    data_config.setdefault("model", {})["use_reconstruction"] = False
    data_config.setdefault("io", {})["load_host_los"] = False
    with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".toml", delete=False) as handle:
        tmp_path = Path(handle.name)
        tomli_w.dump(data_config, handle)
    try:
        return candel_trgb.load_EDD_TRGB_from_config(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS),
                        default="fiducial")
    parser.add_argument("--figdir", type=Path, default=None,
                        help="Extra directory to copy the paper PDFs into.")
    parser.add_argument("--replot", action="store_true",
                        help="Re-plot from the cached PPC without regenerating.")
    args = parser.parse_args()

    config_stem, posterior, field_index = VARIANTS[args.variant]
    cache = OUTPUT_DIR / f"trgbh0_ppc_{args.variant}_cache.npz"
    _keys = ("mag_sim", "cz_sim", "r_sim", "ra_sim", "dec_sim",
             "mag_obs", "cz_obs", "ra_obs", "dec_obs")

    if args.replot and cache.exists():
        print(f"variant   : {args.variant} (replot from {cache})", flush=True)
        ppc = dict(np.load(cache))
    else:
        config = candel.load_config(str(GENERATED / f"{config_stem}.toml"),
                                    replace_los_prior=False)
        data = load_observed_data(config)
        samples = candel.read_samples("", str(posterior))
        print(f"variant   : {args.variant}", flush=True)
        print(f"posterior : {posterior}", flush=True)
        print(f"field     : {field_index}", flush=True)
        print(f"n_hosts   : {len(data['mag_obs'])}", flush=True)
        ppc = generate_trgb_ppc(
            samples, data, config, seed=SEED, field_index=field_index,
            n_workers=N_WORKERS)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{k: ppc[k] for k in _keys if k in ppc})

    # Only the baseline carries the paper figure names; the checks are tagged
    # so that they cannot overwrite them.
    tag = "" if args.variant == "fiducial" else f"_{args.variant}"
    outputs = {
        f"trgbh0_ppc_magnitude_redshift{tag}.pdf": plot_trgb_ppc,
        f"trgbh0_ppc_sky{tag}.pdf": plot_trgb_ppc_sky,
        f"trgbh0_ppc_distance{tag}.pdf": plot_trgb_ppc_distance,
    }
    figdirs = [OUTPUT_DIR] + ([args.figdir] if args.figdir else [])
    stats = {}
    for name, plotter in outputs.items():
        for figdir in figdirs:
            figdir.mkdir(parents=True, exist_ok=True)
            stats[name] = plotter(ppc, str(figdir / name), mnras=True)

    mag_cz = stats[f"trgbh0_ppc_magnitude_redshift{tag}.pdf"]
    dist = stats[f"trgbh0_ppc_distance{tag}.pdf"]
    print(f"n_ppc            : {len(ppc['mag_sim'])}", flush=True)
    print(f"KS mag p         : {mag_cz['ks_mag_pvalue']:.4g}", flush=True)
    print(f"KS cz  p         : {mag_cz['ks_cz_pvalue']:.4g}", flush=True)
    print(f"PPC distance med : {dist['r_median']:.3g} Mpc "
          f"[{dist['r_p16']:.3g}, {dist['r_p84']:.3g}]", flush=True)


if __name__ == "__main__":
    main()
