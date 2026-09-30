# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Peculiar-velocity forward models (TFR, SN, FP) and their catalogues."""
from candel.util import fprint, fsection, get_nested, load_config

from .base_pv import BasePVModel, JointPVModel                                  # noqa
from .catalogues import (                                                       # noqa
    load_2MTF,                                                                  # noqa
    load_6dF_FP,                                                                # noqa
    load_CF4_data,                                                              # noqa
    load_CF4_mock,                                                              # noqa
    load_Foundation,                                                            # noqa
    load_generic,                                                               # noqa
    load_LOSS,                                                                  # noqa
    load_PantheonPlus,                                                          # noqa
    load_PantheonPlus_Lane,                                                     # noqa
    load_SDSS_FP,                                                               # noqa
    load_SFI,                                                                   # noqa
    )
from .frame import PVDataFrame, load_PV_dataframes                             # noqa
from .growth_rate import Beta2Cosmology                                         # noqa
from .mock import gen_TFR_mock                                                  # noqa
from .model_PV_FP import FPModel                                                # noqa
from .model_PV_PantheonPlus import PantheonPlusModel                            # noqa
from .model_PV_SN import SNModel                                                # noqa
from .model_PV_TFR import TFRModel                                              # noqa
from .redshift2real import Redshift2Real                                        # noqa


def _model_section_title(name, cat):
    return f"Model: {name} ({cat})" if cat else f"Model: {name}"


def name2model(name, shared_param=None, config=None):
    mapping = {
        "TFRModel": TFRModel,
        "SNModel": SNModel,
        "PantheonPlusModel": PantheonPlusModel,
        "FPModel": FPModel,
        }

    cats = None
    if config is not None:
        cfg = load_config(config, replace_none=False, replace_los_prior=False,
                          fill_paths=False)
        cats = get_nested(cfg, "io/catalogue_name", None)

    if isinstance(name, str):
        if name not in mapping:
            raise ValueError(f"Model name `{name}` not recognized.\n"
                             f"Available models: {list(mapping.keys())}")
        cat = cats if isinstance(cats, str) else None
        fsection(_model_section_title(name, cat))
        return mapping[name](config)

    if isinstance(name, list):
        unknown = [n for n in name if n not in mapping]
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}\n"
                             f"Available models: {list(mapping.keys())}")
        if shared_param is None:
            shared_param = []
        else:
            fprint(f"using shared parameters: `{shared_param}`")

        cat_list = (cats if isinstance(cats, list)
                    else [None] * len(name))
        submodels = []
        for n, cat in zip(name, cat_list):
            fsection(_model_section_title(n, cat))
            submodels.append(mapping[n](config))
        return JointPVModel(submodels, shared_param)

    raise TypeError("`name` must be a string or a list of strings.")
