# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""EDD TRGB data loaders."""
from os.path import join

import numpy as np

from candel.field.angular_scatter import (angular_position_scatter_from_config,
                                          scatter_data_coordinates)
from candel.field.field_products import (
    field_smoothing_scale_from_config, resolve_or_build_los_data_path,
    velocity_field_smoothing_scale_from_config)
from candel.field.los import _filter_data, _zcmb_blat_mask
from candel.field.volume_density import _load_h0_volume_data_from_config
from candel.util import SPEED_OF_LIGHT, fprint, get_nested, load_config


def _parse_edd_trgb_txt(fpath):
    """Parse an EDD TRGB text file (5 header lines, comma-delimited)."""
    with open(fpath) as f:
        lines = f.readlines()
    header = [c.strip() for c in lines[1].strip().split(",")]
    ncol = len(header)
    rows = []
    for line in lines[5:]:
        row = [c.strip().strip('"') for c in line.strip().split(",")]
        if len(row) == ncol:
            rows.append(row)
    return rows


def _edd_col_float(rows, idx):
    """Extract a float column, returning NaN for empty/missing cells."""
    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        try:
            out[i] = float(row[idx])
        except (ValueError, IndexError):
            pass
    return out


def _edd_col_str(rows, idx):
    return np.array([row[idx].strip() for row in rows])


def load_EDD_TRGB(root, zcmb_min=None, zcmb_max=None, b_min=None,
                  los_data_path=None, return_all=False, return_mask=False,
                  e_czcmb_default=20.0, mag_min_TRGB=22.1, field_indices=None):
    """Load EDD TRGB data (``EDD_TRGB.txt``)."""
    rows = _parse_edd_trgb_txt(join(root, "EDD_TRGB.txt"))
    n_orig = len(rows)
    fprint(f"initially loaded {n_orig} galaxies from EDD TRGB data.")

    RA = _edd_col_float(rows, 7)        # RAJ
    dec = _edd_col_float(rows, 8)        # DeJ
    czcmb = _edd_col_float(rows, 20)     # individual Vcmb
    T814 = _edd_col_float(rows, 45)
    T8_lo = _edd_col_float(rows, 46)
    T8_hi = _edd_col_float(rows, 47)
    colour_606_814 = _edd_col_float(rows, 48)
    colour_lo = _edd_col_float(rows, 49)
    colour_hi = _edd_col_float(rows, 50)
    A_814 = _edd_col_float(rows, 62)
    M_TRGB_Anand = _edd_col_float(rows, 63)
    names = _edd_col_str(rows, 35)

    zcmb_arr = czcmb / SPEED_OF_LIGHT
    colour_edd = 1.23 + (M_TRGB_Anand + 4.06) / 0.20
    e_colour_edd = np.abs(colour_hi - colour_lo) / 2
    e_colour_edd = np.where(np.isfinite(e_colour_edd), e_colour_edd, 0.0)

    data = dict(
        RA=RA,
        dec=dec,
        zcmb=zcmb_arr,
        e_zcmb=np.full(n_orig, e_czcmb_default / SPEED_OF_LIGHT),
        mag=T814 - A_814,
        e_mag=(T8_hi - T8_lo) / 2,
        colour_dered=colour_edd,
        colour_606_814=colour_606_814,
        e_colour_dered=e_colour_edd,
        host_names=names,
    )

    if return_all:
        return data

    keep = np.ones(n_orig, dtype=bool)

    # Drop anchor and satellite galaxies (treated separately in the model).
    drop = np.isin(names, ["LMC", "SMC", "NGC4258", "NGC4258-DF6"])
    if np.any(drop):
        fprint(f"dropping {np.sum(drop)} anchor/satellite galaxies: "
               f"{', '.join(names[drop])}")
    keep &= ~drop

    # Drop galaxies with missing TRGB magnitudes.
    bad_mag = keep & ~np.isfinite(data["mag"])
    if np.any(bad_mag):
        fprint(f"dropping {np.sum(bad_mag)} galaxies with missing TRGB "
               f"magnitudes.")
    keep &= ~bad_mag

    if mag_min_TRGB is not None:
        bright_mag = keep & (data["mag"] < mag_min_TRGB)
        if np.any(bright_mag):
            fprint(
                f"dropping {np.sum(bright_mag)} galaxies brighter than "
                f"mag_min_TRGB={mag_min_TRGB}.")
        keep &= ~bright_mag

    # Drop galaxies without the EDD/Rizzi colour-standardization term.
    bad_colour = keep & (
        ~np.isfinite(data["colour_dered"])
        | ~np.isfinite(data["e_colour_dered"])
    )
    if np.any(bad_colour):
        fprint(f"dropping {np.sum(bad_colour)} galaxies with missing "
               f"EDD/Rizzi colour-standardization term.")
    keep &= ~bad_colour

    # Drop galaxies with fill-value Vcmb (9999 = no measured velocity).
    bad_vcmb = keep & (np.abs(czcmb) >= 9999)
    if np.any(bad_vcmb):
        fprint(f"dropping {np.sum(bad_vcmb)} galaxies with fill-value Vcmb.")
    keep &= ~bad_vcmb

    # Apply zcmb / galactic latitude cuts on the kept subset.
    sub_mask = _zcmb_blat_mask(
        zcmb_arr[keep], RA[keep], dec[keep], zcmb_min, zcmb_max, b_min)
    keep[np.where(keep)[0][~sub_mask]] = False

    data = _filter_data(
        data, keep, los_data_path, field_indices=field_indices)

    if return_mask:
        return data, keep
    return data


def load_EDD_TRGB_from_config(config_path):
    """Load EDD TRGB data from config."""
    config_key = "EDD_TRGB"
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    d = config["io"]["PV_main"][config_key]
    root = d["root"]

    zcmb_min = get_nested(config, f"io/PV_main/{config_key}/zcmb_min", None)
    zcmb_max = get_nested(config, f"io/PV_main/{config_key}/zcmb_max", None)
    b_min = get_nested(config, f"io/PV_main/{config_key}/b_min", None)

    mag_min_TRGB = get_nested(config, "model/mag_min_TRGB", 22.1)
    reconstruction = get_nested(
        config, f"io/PV_main/{config_key}/reconstruction", None)
    field_indices = get_nested(config, "io/field_indices", None)
    field_smoothing_scale = field_smoothing_scale_from_config(config)
    velocity_field_smoothing_scale = (
        velocity_field_smoothing_scale_from_config(config))

    los_data_path = None
    if get_nested(config, "io/load_host_los", False):
        los_data_path = resolve_or_build_los_data_path(
            config, config_key, reconstruction, d.get("los_file", None),
            field_smoothing_scale=field_smoothing_scale,
            velocity_field_smoothing_scale=velocity_field_smoothing_scale,
            config_path=config_path, field_indices=field_indices)

    data, mask = load_EDD_TRGB(root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
                               b_min=b_min, return_mask=True,
                               mag_min_TRGB=mag_min_TRGB,
                               los_data_path=los_data_path,
                               field_indices=field_indices)
    if los_data_path is None:
        scatter = angular_position_scatter_from_config(config)
        if scatter is not None:
            scatter_data_coordinates(data, scatter, label=config_key)

    data["RA_host"] = data.pop("RA")
    data["dec_host"] = data.pop("dec")
    data["mag_obs"] = data.pop("mag")
    data["e_mag_obs"] = data.pop("e_mag")
    data["czcmb"] = data.pop("zcmb") * SPEED_OF_LIGHT
    data["e_czcmb"] = data.pop("e_zcmb") * SPEED_OF_LIGHT
    data["e_mag_median"] = float(np.median(data["e_mag_obs"]))

    fprint(f"reconstruction: {reconstruction or 'none'}")
    if los_data_path is not None:
        los_data_path = getattr(los_data_path, "resolved_path", los_data_path)
        fprint(f"  host LOS path: {los_data_path}")
        data["host_los_density"] = data.pop("los_density")
        data["host_los_velocity"] = data.pop("los_velocity")
        data["host_los_r"] = data.pop("los_r")
        data["host_los_field_indices"] = data.pop("los_field_indices")
        fprint(f"  host LOS shape: {data['host_los_density'].shape}")

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, reconstruction, config_key,
        velocity_selections=("TRGB_magnitude_redshift",),
        field_indices=data.get("host_los_field_indices", None))

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

    anchors = get_nested(config, "model/anchors", {})
    # Fallbacks mirror `config_EDD_TRGB.toml`; see its TRGB Calibration block
    # for the LMC error budget and why the Hoyt (2023) systematic is not used
    # as-is.
    data["mu_LMC_anchor"] = anchors.get("mu_LMC", 18.477)
    data["e_mu_LMC_anchor"] = anchors.get("e_mu_LMC", 0.026)
    data["mag_LMC_TRGB"] = anchors.get("mag_LMC_TRGB", 14.439)
    data["e_mag_LMC_TRGB"] = anchors.get("e_mag_LMC_TRGB", 0.030)
    data["mu_N4258_anchor"] = anchors.get("mu_N4258", 29.397)
    data["e_mu_N4258_anchor"] = anchors.get("e_mu_N4258", 0.032)
    data["mag_N4258_TRGB"] = anchors.get("mag_N4258_TRGB", 25.347)
    data["e_mag_N4258_TRGB"] = anchors.get("e_mag_N4258_TRGB", 0.0443)

    return data
