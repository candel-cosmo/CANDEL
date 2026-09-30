# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""CANDEL probe for the EDD TRGB H0 model (`which_run = EDD_TRGB`)."""
from candel import Probe, fprint, get_nested, read_samples
from candel.field.los_prep import pv_main_los_config
from candel.inference import run_inference
from candel.tasks import is_active, is_delta_prior, tag_number

from .data import load_EDD_TRGB, load_EDD_TRGB_from_config
from .model import TRGBModel
from .specs import TASK_SPECS

# Default `model/mag_min_TRGB`; other values are tagged in task filenames.
DEFAULT_MAG_MIN_TRGB = 22.1


class TRGBProbe(Probe):
    which_run = "EDD_TRGB"
    uses_h0_volume = True
    los_catalogue = "EDD_TRGB"
    reconstruction_key = "io/PV_main/EDD_TRGB/reconstruction"
    los_file_key = "io/PV_main/EDD_TRGB/los_file"
    task_specs = TASK_SPECS

    def load_data(self, config_path):
        return load_EDD_TRGB_from_config(config_path)

    def build_model(self, config_path, data):
        return TRGBModel(config_path, data)

    def run(self, config_path):
        """Run inference, then the posterior predictive check if enabled."""
        data = self.load_data(config_path)
        model = self.build_model(config_path, data)
        run_inference(model)

        if model.num_fields > 1:
            fprint(f"skipping posterior predictive check: marginalizing "
                   f"over {model.num_fields} field realizations.")
            return
        if not get_nested(model.config, "model/run_ppc", True):
            return
        from .ppc import (generate_trgb_ppc, plot_trgb_ppc, plot_trgb_ppc_sky,
                          plot_trgb_ppc_sky_exposure)
        fprint("running posterior predictive check...")
        fname_out = model.config["io"]["fname_output"]
        samples = read_samples("", fname_out)
        ppc = generate_trgb_ppc(samples, data, config_path)
        ppc_fname = fname_out.rsplit(".", 1)[0] + "_ppc.png"
        plot_trgb_ppc(ppc, ppc_fname)
        ppc_root = ppc_fname.rsplit(".", 1)[0]
        if all(k in ppc for k in ("ra_sim", "dec_sim", "ra_obs", "dec_obs")):
            plot_trgb_ppc_sky(ppc, ppc_root + "_sky.png")
        if "sky_exposure" in ppc:
            plot_trgb_ppc_sky_exposure(ppc, ppc_root + "_sky_exposure.png")

    def sky_positions(self, catalogue, config):
        if catalogue != "EDD_TRGB":
            return None
        los_file, kwargs = pv_main_los_config(config, catalogue)
        data = load_EDD_TRGB(return_all=True, **kwargs)
        return data["RA"], data["dec"], los_file

    def h0_volume_field_key(self, config):
        which_sel = get_nested(config, "model/which_selection", None)
        return ("velocity" if which_sel == "TRGB_magnitude_redshift"
                else "density")

    def task_tag_parts(self, config):
        parts = []
        which_sel = get_nested(config, "model/which_selection", None)
        if is_active(which_sel):
            parts.append(f"sel-{which_sel}")
        b_min = get_nested(config, "io/PV_main/EDD_TRGB/b_min", None)
        if is_active(b_min):
            parts.append(f"bmin{tag_number(b_min)}")
        mag_min = get_nested(config, "model/mag_min_TRGB", None)
        if (is_active(mag_min)
                and float(mag_min) != DEFAULT_MAG_MIN_TRGB):
            parts.append(f"magmin{tag_number(mag_min)}")
        sky_exposure = get_nested(config, "model/TRGB_sky_exposure", {})
        if (isinstance(sky_exposure, dict)
                and sky_exposure.get("enabled", False)):
            if "n_pix" in sky_exposure:
                raise ValueError(
                    "`model/TRGB_sky_exposure/n_pix` is no longer "
                    "supported; use `model/TRGB_sky_exposure/nside`.")
            if sky_exposure.get("nside", None) is None:
                raise ValueError(
                    "Enabled TRGB sky exposure requires "
                    "`model/TRGB_sky_exposure/nside`.")
            nside = int(sky_exposure["nside"])
            if nside <= 0 or nside & (nside - 1):
                raise ValueError(
                    "`model/TRGB_sky_exposure/nside` must be a positive "
                    "power of two.")
            kappa = sky_exposure.get("kappa", 48.0)
            parts.append(
                f"skyhp_nside{nside}_k{tag_number(kappa)}")
        if not get_nested(config, "model/use_TRGB_host_redshift", True):
            parts.append("no_TRGB_redshift")
        use_reconstruction = get_nested(
            config, "model/use_reconstruction", False)
        Vext_prior = get_nested(config, "model/priors/Vext", None)
        if not use_reconstruction and not is_delta_prior(Vext_prior):
            parts.append("Vext")
        if use_reconstruction:
            reconstruction = get_nested(
                config, "io/PV_main/EDD_TRGB/reconstruction", None)
            parts.append(reconstruction)
            which_bias = get_nested(config, "model/which_bias", "linear")
            if reconstruction == "Carrick2015" and which_bias != "linear":
                parts.append(which_bias)
            if get_nested(config, "model/use_density_dependent_sigma_v", False):  # noqa
                parts.append("sigv_rho")
            beta_prior = get_nested(config, "model/priors/beta", None)
            if (reconstruction == "Carrick2015"
                    and is_delta_prior(beta_prior)):
                parts.append(f"beta_{tag_number(beta_prior.get('value'))}")
            elif (isinstance(beta_prior, dict)
                    and not is_delta_prior(beta_prior)):
                beta_loc = beta_prior.get("loc", beta_prior.get("mean"))
                beta_scale = beta_prior.get("scale", beta_prior.get("std"))
                if not (beta_prior.get("dist") == "normal"
                        and beta_loc == 0.461 and beta_scale == 0.013):
                    parts.append("beta_free")

        return parts
