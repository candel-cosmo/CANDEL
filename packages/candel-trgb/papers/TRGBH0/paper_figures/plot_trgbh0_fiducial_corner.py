#!/usr/bin/env python
"""Plot the fiducial TRGBH0 posterior corner."""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = next(path for path in SCRIPT_DIR.parents
                if path.name == "paper_TRGBH0")
for path in (SCRIPT_DIR, PLOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from candel.plotting.corner import plot_corner_from_hdf5  # noqa: E402
from stack_fields import (evidence_weights, field_paths,  # noqa: E402
                          load_fields)
from trgbh0_plot_style import OUTPUT_DIR, TRGBH0_RESULTS  # noqa: E402

OUTDIR = OUTPUT_DIR

# Fiducial: beta=1 Student-t, R4 smoothing, 48-pixel sky exposure, taken
# from the single-field ensemble rather than an on-the-fly marginalised chain.
FIDUCIAL_GLOB = (
    TRGBH0_RESULTS / "single_fields_smoothed"
    / "EDD_TRGB_rhoSmoothR4_cz-student_t_MAS-PCS_sel-TRGB_magnitude_bmin10_skyhp_nside2_k192_ManticoreLocalCOLA_field*_single_smoothed.hdf5"  # noqa: E501
)


def fiducial_chain():
    """The evidence-stacked posterior, which at N_eff = 1 is one chain.

    The stack puts essentially all its weight on the highest-evidence
    realisation, so plotting that chain is the stack. The guard fails loudly if
    the weights ever spread far enough for that to stop being true, in which
    case the corner needs a genuinely weighted stack instead.
    """
    index, _, lnz = load_fields(FIDUCIAL_GLOB)
    weights, n_eff, _ = evidence_weights(lnz)
    if n_eff > 1.05:
        raise RuntimeError(
            f"N_eff = {n_eff:.2f}: the evidence stack is no longer a single "
            "realisation, so this corner would misrepresent it.")
    dominant = int(np.argmax(weights))
    print(f"Evidence-stacked corner: field {index[dominant]}, "
          f"weight {weights[dominant]:.6f}, N_eff = {n_eff:.2f}")
    return Path(field_paths(FIDUCIAL_GLOB)[dominant])


CORNER_KEYS = [
    "H0",
    "M_TRGB",
    "c_bar",
    "w_c",
    "c_star",
    "mu_LMC",
    "mu_N4258",
    "sigma_int",
    "sigma_v",
    "mag_lim_TRGB",
    "mag_lim_TRGB_width",
    "Vext_mag",
    "Vext_ell",
    "Vext_b",
    "alpha_low",
    "alpha_high",
    "log_rho_t",
    "log_rho_width",
]


def main():
    fiducial = fiducial_chain()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_corner_from_hdf5(
        fiducial,
        keys=CORNER_KEYS,
        # Single posterior, so no legend; the model is named in the caption.
        filled=False,
        # 19 panels wide, so the canvas is ~38 in and shrinks by ~5x at
        # \textwidth; the font size has to be scaled up to match.
        fontsize=42,
        ranges={
            "alpha_low": [0.0, None],
            "alpha_high": [0.0, None],
            "sigma_v": [0.0, None],
        },
        filename=str(OUTDIR / "trgbh0_manticore_density_sigma_v_corner.pdf"),
        show_fig=False,
    )


if __name__ == "__main__":
    main()
