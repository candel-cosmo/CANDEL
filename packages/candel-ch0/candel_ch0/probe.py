# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""CANDEL probe for the Cepheid-calibrated H0 model (`which_run = CH0`)."""
from pathlib import Path

import numpy as np

import candel
from candel import Probe, get_nested
from candel.field.los_prep import pv_main_los_config
from candel.tasks import (is_active, is_delta_prior, is_manticore_los,
                          tag_number)
from candel.util import SPEED_OF_LIGHT

from .data import load_SH0ES_from_config, load_SH0ES_separated
from .model import CH0Model
from .specs import TASK_SPECS


def _num_hosts(config):
    """Number of Cepheid hosts inside the configured cz cut, if readable."""
    root = get_nested(config, "io/SH0ES/root", None)
    if root is None:
        return None
    cz_max = get_nested(config, "io/SH0ES/cepheid_host_cz_cmb_max", None)
    root = Path(root)
    if not root.is_absolute():
        root = Path(candel.__file__).resolve().parents[1] / root
    try:
        redshifts = np.load(
            root / "processed" / "Cepheid_anchors_redshifts.npy")
    except Exception:
        return None
    czcmb = redshifts["zCMB"] * SPEED_OF_LIGHT
    return int(np.sum(czcmb < cz_max)) if cz_max is not None else len(czcmb)


class CH0Probe(Probe):
    which_run = "CH0"
    uses_h0_volume = True
    los_catalogue = "SH0ES"
    reconstruction_key = "io/SH0ES/reconstruction"
    los_file_key = "io/PV_main/SH0ES/los_file"
    task_specs = TASK_SPECS

    def load_data(self, config_path):
        return load_SH0ES_from_config(config_path)

    def build_model(self, config_path, data):
        return CH0Model(config_path, data)

    def sky_positions(self, catalogue, config):
        if catalogue != "SH0ES":
            return None
        los_file, kwargs = pv_main_los_config(config, catalogue)
        data = load_SH0ES_separated(**kwargs)
        return data["RA_host"], data["dec_host"], los_file

    def h0_volume_field_key(self, config):
        which_sel = get_nested(config, "model/which_selection", None)
        if which_sel in ("redshift", "SN_magnitude_redshift",
                         "TRGB_magnitude_redshift"):
            return "velocity"
        if which_sel == "SN_magnitude_or_redshift_Nmag":
            n_mag = get_nested(config, "model/num_hosts_selection_mag", None)
            n_hosts = _num_hosts(config)
            if type(n_mag) is int and n_hosts is not None:
                return "velocity" if n_mag < n_hosts else "density"
            return f"mixed_Nmag={n_mag}"
        return "density"

    def task_tag_parts(self, config):
        parts = []
        which_sel = get_nested(config, "model/which_selection", None)
        if is_active(which_sel):
            parts.append(f"sel-{which_sel}")
            if which_sel == "SN_magnitude_or_redshift_Nmag":
                parts.append(
                    f"Nmag{get_nested(config, 'model/num_hosts_selection_mag', None)}")  # noqa

        if get_nested(config, "model/use_uniform_mu_host_priors", False):
            parts.append("uniform_mu_host")

        drop_observation = get_nested(
            config, "io/SH0ES/drop_observation", None)
        if is_active(drop_observation):
            if type(drop_observation) is not int:
                raise TypeError(
                    "`io/SH0ES/drop_observation` must be an integer active "
                    "host index for CH0 generated tasks.")
            parts.append(f"drop{drop_observation:02d}")

        cz_max = get_nested(config, "io/SH0ES/cepheid_host_cz_cmb_max", 3300)
        if cz_max != 3300:
            parts.append(f"czmax{tag_number(cz_max)}")

        keep_hosts = get_nested(config, "io/SH0ES/keep_hosts", "all")
        if keep_hosts != "all":
            parts.append(f"keep{len(keep_hosts)}hosts")

        cep_scale = get_nested(config, "io/SH0ES/cepheid_error_scale", 1.0)
        if float(cep_scale) != 1.0:
            cep_hosts = get_nested(
                config, "io/SH0ES/cepheid_error_scale_hosts", "all")
            n_scaled = "all" if cep_hosts == "all" else len(cep_hosts)
            parts.append(f"ceperr{tag_number(cep_scale)}-{n_scaled}")

        r_prior = get_nested(config, "model/which_distance_prior", "volume")
        if r_prior != "volume":
            parts.append(r_prior)

        if not get_nested(config, "model/use_Cepheid_host_redshift", True):
            parts.append("no_Cepheid_redshift")

        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        use_pv_covmat = get_nested(
            config, "model/use_fiducial_Cepheid_host_PV_covariance", False)
        Vext_prior = get_nested(config, "model/priors/Vext", None)
        if not use_reconstruction and not use_pv_covmat \
                and not is_delta_prior(Vext_prior):
            parts.append("Vext")

        if use_reconstruction:
            reconstruction = get_nested(
                config, "io/SH0ES/reconstruction", None)
            parts.append(reconstruction)
            which_bias = get_nested(config, "model/which_bias", None)
            if which_bias == "uniform":
                parts.append(which_bias)
            beta_prior = get_nested(config, "model/priors/beta", None)
            if is_manticore_los(reconstruction):
                if not is_delta_prior(beta_prior):
                    parts.append("beta_free")
            if get_nested(config, "model/use_density_dependent_sigma_v", False):  # noqa
                parts.append("sigv_rho")

        if use_pv_covmat:
            parts.append("PV_covmat")

        if get_nested(config, "model/use_PV_covmat_scaling", False):
            parts.append("PV_covmat_scaling")

        if get_nested(config, "model/weight_selection_by_covmat_Neff", False):
            parts.append("weight_by_Neff")

        return parts
