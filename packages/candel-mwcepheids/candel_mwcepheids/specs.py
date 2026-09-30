# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Named MW Cepheid task specs for generate_tasks.py."""
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


MWCEPHEIDS_ROOT = "results/MWCepheids"


MWCEPHEIDS_DENSE_MASS_BLOCKS = (
    ("M_H_1", "b_W", "Z_W", "delta_pi"),
    ("mu_logP_C22", "sigma_logP_C22"),
    ("mu_logP_C27", "sigma_logP_C27"),
    ("mu_logP_NGC4258", "sigma_logP_NGC4258"),
    ("mu_logP_LMC", "sigma_logP_LMC"),
)


MWCEPHEIDS_RUN_COMMON = {
    "inference/num_chains": 4,
    "inference/chain_method": "sequential",
    "inference/init_maxiter": 500,
    "inference/target_accept_prob": 0.95,
    "inference/compute_log_density": False,
    "inference/dense_mass_blocks": MWCEPHEIDS_DENSE_MASS_BLOCKS,
}


def _mwcepheids_forward_base():
    return {
        "model/model_type": "forward",
        "model/distance_prior": "disk",
        "model/shared_scatter": True,
        "model/marginalise_distance": True,
        "model/use_Q": False,
        "model/Q_err_median": False,
        "model/anchors": ("NGC4258", "LMC"),
        "model/anchor_scatter_correction": "none",
        "model/spiral_arms/apply": False,
        "model/C22/selection/apply_mW": False,
        "model/C22/selection/apply_AH": False,
        "model/C22/selection/apply_pi": False,
        "model/C22/selection/apply_logP": False,
        "model/C27/selection/apply_pi": False,
        "model/C27/selection/apply_mW": False,
    }


def _mwcepheids_selection_no_ah():
    return {
        **_mwcepheids_forward_base(),
        "model/C22/selection/apply_mW": True,
        "model/C22/selection/apply_pi": True,
        "model/C22/selection/apply_logP": True,
        "model/C22/selection/mW_max": "infer",
        "model/C22/selection/mW_width": 0.5,
        "model/C22/selection/pi_min": "infer",
        "model/C22/selection/pi_smooth": True,
        "model/C22/selection/pi_width": 0.1,
        "model/C22/selection/logP_min": 0.903,
        "model/C22/selection/logP_width": 0.01,
        "model/C27/selection/apply_pi": True,
        "model/C27/selection/apply_mW": True,
        "model/C27/selection/pi_min": 0.8,
        "model/C27/selection/pi_smooth": True,
        "model/C27/selection/pi_width": 0.05,
        "model/C27/selection/mW_min": "infer",
        "model/C27/selection/mW_width": 0.1,
    }


def _mwcepheids_selection_ah():
    return {
        **_mwcepheids_selection_no_ah(),
        "model/C22/selection/apply_AH": True,
        "model/C22/selection/AH_max": 0.4,
        "model/C22/selection/AH_width": 0.1,
        "model/C22/selection/dust_map": "bayestar+marshall",
    }


def _mwcepheids_smoke_datasets():
    return [
        {
            "model/model_type": "R21",
            "model/marginalise_distance": False,
        },
        _mwcepheids_forward_base(),
        _mwcepheids_selection_no_ah(),
        _mwcepheids_selection_ah(),
    ]


def _mwcepheids_main_datasets():
    return [
        {
            "model/model_type": "R21",
            "model/marginalise_distance": False,
        },
        {
            "model/model_type": "R21",
            "model/marginalise_distance": False,
            "model/use_Q": True,
        },
        _mwcepheids_forward_base(),
        {
            **_mwcepheids_forward_base(),
            "model/use_Q": True,
        },
        {
            **_mwcepheids_forward_base(),
            "model/shared_scatter": False,
        },
        {
            **_mwcepheids_forward_base(),
            "model/shared_scatter": False,
            "model/anchor_scatter_correction": 0.06,
        },
        _mwcepheids_selection_no_ah(),
        _mwcepheids_selection_ah(),
        {
            **_mwcepheids_forward_base(),
            "model/marginalise_distance": False,
            "model/spiral_arms/apply": True,
        },
        {
            **_mwcepheids_forward_base(),
            "model/spiral_arms/apply": True,
        },
        {
            **_mwcepheids_selection_no_ah(),
            "model/spiral_arms/apply": True,
        },
        {
            **_mwcepheids_selection_ah(),
            "model/spiral_arms/apply": True,
        },
    ]


def _mwcepheids_selection_datasets():
    return [
        {
            **_mwcepheids_forward_base(),
            "model/C22/selection/apply_mW": True,
            "model/C22/selection/mW_max": "infer",
            "model/C22/selection/mW_width": 0.5,
        },
        {
            **_mwcepheids_forward_base(),
            "model/C22/selection/apply_mW": True,
            "model/C22/selection/mW_max": "infer",
            "model/C22/selection/mW_width": "infer",
        },
        {
            **_mwcepheids_forward_base(),
            "model/C22/selection/apply_pi": True,
            "model/C22/selection/pi_min": "infer",
            "model/C22/selection/pi_smooth": True,
            "model/C22/selection/pi_width": 0.1,
        },
        {
            **_mwcepheids_forward_base(),
            "model/C22/selection/apply_pi": True,
            "model/C22/selection/pi_min": 0.35,
            "model/C22/selection/pi_smooth": False,
        },
        {
            **_mwcepheids_forward_base(),
            "model/C22/selection/apply_logP": True,
            "model/C22/selection/logP_min": 0.903,
            "model/C22/selection/logP_width": 0.01,
        },
        {
            **_mwcepheids_forward_base(),
            "model/C22/selection/apply_AH": True,
            "model/C22/selection/AH_max": 0.4,
            "model/C22/selection/AH_width": 0.1,
            "model/C22/selection/dust_map": "bayestar+marshall",
        },
        {
            **_mwcepheids_forward_base(),
            "model/C27/selection/apply_pi": True,
            "model/C27/selection/pi_min": 0.8,
            "model/C27/selection/pi_smooth": True,
            "model/C27/selection/pi_width": 0.05,
        },
        {
            **_mwcepheids_forward_base(),
            "model/C27/selection/apply_pi": True,
            "model/C27/selection/apply_mW": True,
            "model/C27/selection/pi_min": 0.8,
            "model/C27/selection/mW_min": "infer",
            "model/C27/selection/mW_width": 0.1,
        },
        _mwcepheids_selection_no_ah(),
        _mwcepheids_selection_ah(),
    ]


TASK_SPECS = {
    "MWCepheids_smoke": {
        "description": "Short MW-Cepheid task-generator smoke runs.",
        "config_path": str(CONFIG_DIR / "config_MWCepheids.toml"),
        "tag": "smoke",
        "common": {
            **MWCEPHEIDS_RUN_COMMON,
            "inference/num_warmup": 20,
            "inference/num_samples": 20,
            "inference/num_chains": 1,
            "inference/init_maxiter": 0,
            "io/root_output": f"{MWCEPHEIDS_ROOT}/smoke",
        },
        "datasets": _mwcepheids_smoke_datasets(),
        "expected_tasks": 4,
    },
    "MWCepheids_main": {
        "description": (
            "MW-Cepheid calibration production grid covering R21, "
            "forward, Q, scatter, selection, AH, and spiral variants."),
        "config_path": str(CONFIG_DIR / "config_MWCepheids.toml"),
        "tag": "main",
        "common": {
            **MWCEPHEIDS_RUN_COMMON,
            "inference/num_warmup": 1000,
            "inference/num_samples": 4000,
            "io/root_output": f"{MWCEPHEIDS_ROOT}/main",
        },
        "datasets": _mwcepheids_main_datasets(),
        "expected_tasks": 12,
    },
    "MWCepheids_selection": {
        "description": (
            "MW-Cepheid selection-function grid for C22/C27 selection "
            "branches."),
        "config_path": str(CONFIG_DIR / "config_MWCepheids.toml"),
        "tag": "selection",
        "common": {
            **MWCEPHEIDS_RUN_COMMON,
            "inference/num_warmup": 1000,
            "inference/num_samples": 4000,
            "io/root_output": f"{MWCEPHEIDS_ROOT}/selection",
        },
        "datasets": _mwcepheids_selection_datasets(),
        "expected_tasks": 10,
    },
}
