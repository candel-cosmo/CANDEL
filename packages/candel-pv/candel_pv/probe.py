# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""CANDEL probe for the peculiar-velocity models (no `model/which_run`)."""
from candel import Probe, fprint, get_nested, load_config
from candel.field.los_prep import pv_main_los_config

from . import catalogues
from .base_pv import JointPVModel
from .frame import load_PV_dataframes
from .specs import TASK_SPECS

# Catalogues whose sky positions come from `io/PV_main/<name>`.
_SKY_LOADERS = {
    "CF4": catalogues.load_CF4_data,
    "2MTF": catalogues.load_2MTF,
    "SFI": catalogues.load_SFI,
    "PantheonPlus": catalogues.load_PantheonPlus,
    "PantheonPlusLane": catalogues.load_PantheonPlus_Lane,
    "Foundation": catalogues.load_Foundation,
    "LOSS": catalogues.load_LOSS,
    "SDSS_FP": catalogues.load_SDSS_FP,
    "6dF_FP": catalogues.load_6dF_FP,
}


class PVProbe(Probe):
    which_run = None
    task_specs = TASK_SPECS

    def load_data(self, config_path):
        return load_PV_dataframes(config_path)

    def build_model(self, config_path, data):
        from . import name2model
        config = load_config(config_path, replace_los_prior=False)
        model_name = config["inference"]["model"]
        fprint(f"Loading model `{model_name}` from `{config_path}` for data "
               f"`{config['io']['catalogue_name']}`")
        model = name2model(
            model_name, get_nested(config, "inference/shared_params", None),
            config_path)
        if isinstance(data, list):
            if not isinstance(model, JointPVModel):
                raise TypeError(
                    "You provided multiple datasets, but the selected model "
                    f"`{model.__class__.__name__}` is not JointPVModel.")
            if len(data) != len(model.submodels):
                raise ValueError(
                    f"Number of datasets ({len(data)}) does not match number "
                    f"of submodels ({len(model.submodels)}) in the joint "
                    "model.")
        return model

    def model_kwargs(self, model, data):
        return {"data": data}

    def sky_positions(self, catalogue, config):
        loader = _SKY_LOADERS.get(catalogue)
        if loader is None:
            return None
        los_file, kwargs = pv_main_los_config(config, catalogue)
        data = loader(return_all=True, **kwargs)
        return data["RA"], data["dec"], los_file
