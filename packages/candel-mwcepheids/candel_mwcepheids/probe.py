# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""CANDEL probe for the MW Cepheid model (`which_run = MWCepheids`)."""
from candel import Probe, get_nested, load_config
from candel.tasks import is_active

from .data import load_MWCepheids_from_config
from .model import MWCepheidModel
from .specs import TASK_SPECS


class MWCepheidsProbe(Probe):
    which_run = "MWCepheids"
    task_specs = TASK_SPECS

    def load_data(self, config_path):
        return load_MWCepheids_from_config(config_path)

    def build_model(self, config_path, data):
        return MWCepheidModel(
            load_config(config_path, replace_los_prior=False), data)

    def task_tag_parts(self, config):
        parts = []
        model_type = get_nested(config, "model/model_type", "forward")
        parts.append(model_type)
        if model_type == "R21" and get_nested(config, "model/use_Q", False):
            parts.append("Q")
        if model_type == "forward":
            if get_nested(config, "model/marginalise_distance", False):
                parts.append("margd")
            else:
                parts.append("sampled")

            distance_prior = get_nested(
                config, "model/distance_prior", "disk")
            if distance_prior != "disk":
                parts.append(distance_prior)

            if not get_nested(config, "model/shared_scatter", True):
                parts.append("split_scatter")

            if get_nested(config, "model/use_Q", False):
                parts.append("Q")

            anchors = get_nested(config, "model/anchors", [])
            if not anchors:
                parts.append("no_anchors")
            elif tuple(anchors) != ("NGC4258", "LMC"):
                parts.append("anchors-" + "+".join(anchors))

            if is_active(get_nested(
                    config, "model/anchor_scatter_correction", None)):
                parts.append("anchor_scatter")

            if get_nested(config, "model/spiral_arms/apply", False):
                parts.append("spiral")

            c22 = []
            for key, label in (
                    ("apply_mW", "mW"), ("apply_AH", "AH"),
                    ("apply_pi", "pi"), ("apply_logP", "logP")):
                if get_nested(config, f"model/C22/selection/{key}", False):
                    c22.append(label)
            if c22:
                parts.append("C22-" + "+".join(c22))
                if get_nested(
                        config, "model/C22/selection/mW_width", None) \
                        == "infer":
                    parts.append("C22_mW_width_infer")
                if get_nested(
                        config, "model/C22/selection/logP_width", None) \
                        == "infer":
                    parts.append("C22_logP_width_infer")
                if not get_nested(
                        config, "model/C22/selection/pi_smooth", True):
                    parts.append("C22_pi_hard")

            c27 = []
            for key, label in (("apply_pi", "pi"), ("apply_mW", "mW")):
                if get_nested(config, f"model/C27/selection/{key}", False):
                    c27.append(label)
            if c27:
                parts.append("C27-" + "+".join(c27))
                if not get_nested(
                        config, "model/C27/selection/pi_smooth", True):
                    parts.append("C27_pi_hard")

        return parts
