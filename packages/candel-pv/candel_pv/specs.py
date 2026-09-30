# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Named peculiar-velocity task specs for generate_tasks.py."""
from pathlib import Path

from candel.tasks import delta, normal, nu_cz_student_t_prior

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


S8_ROOT = "results/S8"


S8_PV_KIND = "precomputed_los_Carrick2015"


S8_BIAS_MODELS = ["linear", "quadratic", "double_powerlaw"]


VEXT_RAD_ROOT = "results/Vext_rad"


VEXT_RAD_SDSS_FP_VEXT_KNOTS = [0, 100, 200, 300, 400]


VEXT_RAD_SDSS_FP_CARRICK_KNOTS = [0, 20, 40, 60, 80, 100, 120, 140]


VEXT_RAD_SDSS_FP_VEXT_PRIOR = {
    "dist": "vector_radial_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_VEXT_KNOTS,
    "method": "cubic",
}


VEXT_RADMAG_SDSS_FP_VEXT_PRIOR = {
    "dist": "vector_radialmag_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_VEXT_KNOTS,
    "method": "cubic",
}


VEXT_RAD_SDSS_FP_CARRICK_PRIOR = {
    "dist": "vector_radial_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_CARRICK_KNOTS,
    "method": "cubic",
}


VEXT_RADMAG_SDSS_FP_CARRICK_PRIOR = {
    "dist": "vector_radialmag_uniform",
    "low": 0.0,
    "high": 500,
    "rknot": VEXT_RAD_SDSS_FP_CARRICK_KNOTS,
    "method": "cubic",
}


VFO_ROOT = "results/VFO"


VFO_MANTICORE_LOS = "ManticoreLocalSWIFT"


VFO_MANTICORE_COLA_LOS = "ManticoreLocalCOLA"


def _s8_production_datasets():
    return [
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_W1",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_i",
            "pv_model/galaxy_bias": ["linear", "double_powerlaw"],
        },
        {
            # quadratic split off with its own seed: seed 44 drove L-BFGS init
            # into a NaN region (ABNORMAL line-search), then NUTS stalled.
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_i",
            "pv_model/galaxy_bias": "quadratic",
            "inference/seed": 45,
        },
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "6dF_FP",
            "pv_model/galaxy_bias": [*S8_BIAS_MODELS, "cubic"],
        },
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "SDSS_FP",
            "pv_model/galaxy_bias": [*S8_BIAS_MODELS, "cubic"],
        },
        {
            "inference/model": "PantheonPlusModel",
            "io/catalogue_name": "PantheonPlus",
            "inference/init_maxiter": 0,
        },
        {
            "inference/model": [
                "TFRModel", "TFRModel", "PantheonPlusModel"],
            "io/catalogue_name": ["CF4_i", "CF4_W1", "PantheonPlus"],
            "inference/shared_params": "beta,sigma_v",
            "inference/init_maxiter": 0,
        },
    ]


def _vfo_datasets():
    catalogues = [
        {
            "inference/model": "SNModel",
            "io/catalogue_name": "LOSS",
        },
        {
            "inference/model": "SNModel",
            "io/catalogue_name": "Foundation",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_W1",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_i",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "2MTF",
        },
        {
            "inference/model": "TFRModel",
            "io/catalogue_name": "SFI",
        },
    ]
    pv_models = [
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": delta(1.0),
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": delta(1.0),
        },
    ]
    manticore_linear_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": delta(1.0),
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "quadratic",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": delta(1.0),
        },
    ]
    manticore_beta_free_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": normal(1.0, 0.1),
        },
    ]
    carrick_double_powerlaw_models = [
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
    ]
    fp_catalogues = [
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "SDSS_FP",
        },
        {
            "inference/model": "FPModel",
            "io/catalogue_name": "6dF_FP",
        },
    ]
    fp_pv_models = [
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": delta(1.0),
        },
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 2.0,
            },
        },
    ]
    student_t_pv_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "model/priors/beta": delta(1.0),
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": nu_cz_student_t_prior(),
        },
        {
            "pv_model/kind": "precomputed_los_Carrick2015",
            "pv_model/galaxy_bias": "linear",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 1.0,
            },
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": nu_cz_student_t_prior(),
        },
    ]
    cola_manticore_pv_models = [
        {
            "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_COLA_LOS}",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/priors/beta": delta(1.0),
            "model/cz_likelihood": "gaussian",
        },
    ]
    return [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in pv_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in manticore_linear_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in manticore_beta_free_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues
        for pv_model in carrick_double_powerlaw_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in fp_catalogues
        for pv_model in fp_pv_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues + fp_catalogues
        for pv_model in student_t_pv_models
    ] + [
        {
            **catalogue,
            **pv_model,
        }
        for catalogue in catalogues + fp_catalogues
        for pv_model in cola_manticore_pv_models
    ]


def _vfo_single_datasets():
    datasets = []
    for los, n_fields, subsample_fraction in (
            (VFO_MANTICORE_LOS, 30, 0.1),
            (VFO_MANTICORE_COLA_LOS, 50, 0.5),
    ):
        for field in range(n_fields):
            dataset = {
                "pv_model/kind": f"precomputed_los_{los}",
                "io/field_indices": field,
            }
            if subsample_fraction != 0.1:
                dataset[
                    "pv_model/density_3d_subsample_fraction"
                ] = subsample_fraction
            datasets.append(dataset)
    return datasets


TASK_SPECS = {
    "test": {
        "description": (
            "Foundation SN Carrick/COLA field tests plus a fiducial CF4 W1 "
            "Manticore/COLA PCS run."),
        "config_path": str(CONFIG_DIR / "config.toml"),
        "common": {
            "io/field_cache_project": "TEST",
            "inference/model": "SNModel",
            "io/catalogue_name": "Foundation",
            "inference/num_chains": 1,
            "inference/chain_method": "sequential",
            "inference/num_warmup": 500,
            "inference/num_samples": 500,
            "inference/compute_evidence": False,
            "inference/compute_log_density": False,
            "inference/target_accept_prob": 0.9,
            "pv_model/density_3d_geometry": "sphere",
            "pv_model/density_3d_radius": 150.0,
            "pv_model/density_3d_downsample": 1,
            "pv_model/density_3d_subsample_fraction": 0.5,
            "model/field_3d_smoothing_scale": [0.0, 8.0],
            "io/root_output": "results/test",
        },
        "datasets": [
            {
                "pv_model/kind": "precomputed_los_Carrick2015",
                "pv_model/galaxy_bias": "linear",
                "model/priors/beta": {
                    "dist": "uniform",
                    "low": 0.0,
                    "high": 1.0,
                },
            },
            {
                "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_COLA_LOS}",
                "pv_model/galaxy_bias": "double_powerlaw",
                "model/priors/beta": delta(1.0),
                "model/cz_likelihood": "gaussian",
                "io/field_indices": 0,
            },
            {
                "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_COLA_LOS}",
                "pv_model/galaxy_bias": "double_powerlaw",
                "model/priors/beta": delta(1.0),
                "model/cz_likelihood": "gaussian",
            },
            {
                "inference/model": "TFRModel",
                "io/catalogue_name": "CF4_W1",
                "pv_model/kind": f"precomputed_los_{VFO_MANTICORE_COLA_LOS}",
                "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
                "pv_model/galaxy_bias": "double_powerlaw",
                "model/priors/beta": delta(1.0),
                "model/cz_likelihood": "gaussian",
                "model/field_3d_smoothing_scale": 0.0,
                "pv_model/density_3d_subsample_fraction": 0.05,
                "inference/num_warmup": 1500,
                "inference/num_samples": 3000,
            },
        ],
        "expected_tasks": 7,
    },
    "Vext_rad": {
        "description": (
            "CF4 W1 and SDSS FP radial Vext comparison."),
        "config_path": str(CONFIG_DIR / "config.toml"),
        "common": {
            "io/field_cache_project": "VextRad",
            "inference/model": "TFRModel",
            "inference/num_chains": 1,
            "inference/chain_method": "sequential",
            "inference/compute_log_density": False,
            "inference/compute_evidence": False,
            "pv_model/which_Vext": [
                "constant", "radial", "radial_magnitude"],
            "io/root_output": VEXT_RAD_ROOT,
        },
        "datasets": [
            {
                "io/catalogue_name": "CF4_W1",
                "pv_model/kind": "Vext",
            },
            {
                "io/catalogue_name": "CF4_W1",
                "pv_model/kind": "precomputed_los_Carrick2015",
                "pv_model/galaxy_bias": "linear",
                "model/priors/beta": {
                    "dist": "uniform",
                    "low": 0.0,
                    "high": 2.0,
                },
            },
            {
                "inference/model": "FPModel",
                "io/catalogue_name": "SDSS_FP",
                "io/SDSS_FP/zcmb_max": 0.1,
                "pv_model/kind": "Vext",
                "model/priors/Vext_radial": VEXT_RAD_SDSS_FP_VEXT_PRIOR,
                "model/priors/Vext_radial_magnitude": (
                    VEXT_RADMAG_SDSS_FP_VEXT_PRIOR),
            },
            {
                "inference/model": "FPModel",
                "io/catalogue_name": "SDSS_FP",
                "pv_model/kind": "precomputed_los_Carrick2015",
                "pv_model/galaxy_bias": "linear",
                "model/priors/Vext_radial": VEXT_RAD_SDSS_FP_CARRICK_PRIOR,
                "model/priors/Vext_radial_magnitude": (
                    VEXT_RADMAG_SDSS_FP_CARRICK_PRIOR),
                "model/priors/beta": {
                    "dist": "uniform",
                    "low": 0.0,
                    "high": 2.0,
                },
            },
        ],
        "expected_tasks": 12,
    },
    "S8_production": {
        "description": (
            "S8 production PV sweep for individual and joint catalogues."),
        "config_path": str(CONFIG_DIR / "config.toml"),
        "tag": "default",
        "common": {
            "io/field_cache_project": "S8",
            "pv_model/kind": S8_PV_KIND,
            "pv_model/galaxy_bias": S8_BIAS_MODELS,
            "pv_model/density_3d_downsample": 1,
            "model/priors/beta": {
                "dist": "uniform",
                "low": 0.0,
                "high": 2.0,
            },
            "inference/num_chains": 1,
            "inference/num_warmup": 2000,
            "inference/num_samples": 10000,
            "io/root_output": S8_ROOT,
        },
        "datasets": _s8_production_datasets(),
        "expected_tasks": 20,
    },
    "VFO": {
        "description": (
            "VFO PV catalogue Carrick2015/Manticore/COLA comparison."),
        "config_path": str(CONFIG_DIR / "config.toml"),
        "tag": "paper",
        "common": {
            "io/field_cache_project": "VFO",
            "pv_model/density_3d_geometry": "sphere",
            "pv_model/density_3d_radius": 150.0,
            "pv_model/density_3d_downsample": 1,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            "io/root_output": VFO_ROOT,
        },
        "datasets": _vfo_datasets(),
        "expected_tasks": 74,
    },
    "VFO_single": {
        "description": (
            "CF4 W1 TFR one-field Manticore/COLA runs for the VFO setup."),
        "config_path": str(CONFIG_DIR / "config.toml"),
        "tag": "single",
        "common": {
            "io/field_cache_project": "VFO",
            "inference/model": "TFRModel",
            "io/catalogue_name": "CF4_W1",
            "pv_model/galaxy_bias": "double_powerlaw",
            "pv_model/density_3d_subsample_fraction": 0.1,
            "pv_model/density_3d_geometry": "sphere",
            "pv_model/density_3d_radius": 150.0,
            "pv_model/density_3d_downsample": 1,
            "model/priors/beta": delta(1.0),
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            "io/root_output": f"{VFO_ROOT}/single_fields",
        },
        "datasets": _vfo_single_datasets(),
        "expected_tasks": 80,
    },
}
