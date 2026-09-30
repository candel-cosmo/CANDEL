# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Named CH0 task specs for generate_tasks.py."""
from pathlib import Path

from candel.tasks import delta, normal, with_root

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


CH0_PAPER_ROOT = "results/CH0_paper"


CH0_MANTICORE_LOS = "ManticoreLocalSWIFT"


CH0_MANTICORE_COLA_LOS = "ManticoreLocalCOLA"


CH0_MANTICORE_BIAS = "double_powerlaw"


CH0_PAPER_COMMON = {
    "inference/compute_log_density": False,
    "inference/compute_evidence": False,
    "inference/num_chains": 12,
    "inference/chain_method": "sequential",
    "inference/num_warmup": 1000,
    "inference/num_samples": 6000,
    "model/use_uniform_mu_host_priors": False,
    "model/selection_integral_geometry": "sphere",
    "model/selection_integral_grid_radius": 60.0,
    "model/density_3d_subsample_fraction": 1.0,
    "model/priors/M_B": {"dist": "uniform", "low": -22.0, "high": -18.0},
    "model/priors/Vext": {
        "dist": "vector_uniform_fixed",
        "low": 0.0,
        "high": 1000.0,
    },
}


CH0_FIXED_BIAS_PRIORS = {
    "model/priors/alpha_low": delta(1.835),
    "model/priors/alpha_high": delta(0.343),
    "model/priors/log_rho_t": delta(0.313),
    "model/priors/log_rho_width": delta(0.879),
}


CH0_SWIFT_FIXED_BIAS_PRIORS = {
    "model/priors/alpha_low": delta(1.542),
    "model/priors/alpha_high": delta(0.286),
    "model/priors/log_rho_t": delta(-0.027),
    "model/priors/log_rho_width": delta(0.954),
}


def _ch0_selection(selection):
    return {"model/which_selection": selection}


def _ch0_main_datasets():
    selections = ("none", "SN_magnitude", "redshift")
    pv_models = [
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/priors/Vext": delta([0.0, 0.0, 0.0]),
        },
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
        },
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": True,
            "model/use_PV_covmat_scaling": False,
            "model/priors/Vext": delta([0.0, 0.0, 0.0]),
        },
        {
            "model/use_reconstruction": False,
            "model/use_fiducial_Cepheid_host_PV_covariance": True,
            "model/use_PV_covmat_scaling": True,
            "model/priors/Vext": delta([0.0, 0.0, 0.0]),
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": "Carrick2015",
            "model/priors/beta": normal(0.43, 0.02),
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "model/field_3d_smoothing_scale": 0.0,
            "model/velocity_3d_smoothing_scale": 0.0,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
            "model/which_bias": CH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": True,
            "model/field_3d_smoothing_scale": 0.0,
            "model/velocity_3d_smoothing_scale": 0.0,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
            "model/which_bias": CH0_MANTICORE_BIAS,
        },
        {
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "model/field_3d_smoothing_scale": 0.0,
            "model/velocity_3d_smoothing_scale": 0.0,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
            "model/which_bias": CH0_MANTICORE_BIAS,
            "model/priors/beta": normal(1.0, 0.5),
        },
    ]

    datasets = []
    for pv_model in pv_models:
        for selection in selections:
            dataset = {
                **pv_model,
                **_ch0_selection(selection),
            }
            if dataset.get("model/use_fiducial_Cepheid_host_PV_covariance") \
                    and selection == "redshift":
                dataset["model/weight_selection_by_covmat_Neff"] = True
            else:
                dataset.setdefault(
                    "model/weight_selection_by_covmat_Neff", False)
            datasets.append(dataset)
    return datasets


def _ch0_distance_only_datasets():
    base = {
        "model/use_Cepheid_host_redshift": False,
        "model/use_reconstruction": False,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,
        "model/priors/Vext": delta([0.0, 0.0, 0.0]),
        **with_root(f"{CH0_PAPER_ROOT}/distances"),
    }
    return [
        {
            **base,
            **_ch0_selection("none"),
            "model/use_uniform_mu_host_priors": True,
        },
        {
            **base,
            **_ch0_selection("none"),
            "model/use_uniform_mu_host_priors": False,
        },
        {
            **base,
            **_ch0_selection("SN_magnitude"),
            "model/use_uniform_mu_host_priors": False,
        },
    ]


def _ch0_imb_geometric_datasets():
    base = {
        "model/which_selection": "SN_magnitude",
        "model/use_uniform_mu_host_priors": False,
        "model/use_Cepheid_host_redshift": False,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,
        "model/use_density_dependent_sigma_v": False,
        "model/field_3d_smoothing_scale": 0.0,
        "model/velocity_3d_smoothing_scale": 0.0,
    }
    no_density = {
        **base,
        "model/use_reconstruction": False,
        "model/priors/Vext": delta([0.0, 0.0, 0.0]),
        "model/priors/alpha_high": delta(1.0),
    }
    carrick = {
        **base,
        "model/use_reconstruction": True,
        "io/SH0ES/reconstruction": "Carrick2015",
        "model/which_bias": "linear",
        "model/priors/alpha_high": delta(1.0),
    }
    manticore = [
        {
            **base,
            "model/use_reconstruction": True,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
        }
        for field in range(80)
    ]
    return [no_density, carrick, *manticore]


def _ch0_mixed_selection_datasets():
    return [
        {
            "model/which_selection": "SN_magnitude_or_redshift_Nmag",
            "model/num_hosts_selection_mag": n_mag,
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "model/field_3d_smoothing_scale": 0.0,
            "model/velocity_3d_smoothing_scale": 0.0,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
            "model/which_bias": CH0_MANTICORE_BIAS,
        }
        for n_mag in range(36)
    ]


def _ch0_manticore_field_datasets():
    datasets = []
    for field in range(30):
        datasets.append({
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
        })

    for mas in ("CIC", "PCS", "SPH"):
        for field in range(80):
            datasets.append({
                "model/which_selection": "SN_magnitude",
                "model/use_reconstruction": True,
                "model/use_fiducial_Cepheid_host_PV_covariance": False,
                "model/use_PV_covmat_scaling": False,
                "model/weight_selection_by_covmat_Neff": False,
                "model/use_density_dependent_sigma_v": False,
                "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
                "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": mas,
                "model/which_bias": CH0_MANTICORE_BIAS,
                "io/field_indices": field,
            })

    for field in range(80):
        datasets.append({
            "model/which_selection": "redshift",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
        })

    return datasets


def _ch0_manticore_cola_cic_field_datasets():
    return [
        {
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
            "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "CIC",
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
        }
        for field in range(80)
    ]


def _ch0_manticore_cola_cic_fixed_bias_datasets():
    return [
        {
            **dataset,
            "model/which_bias": CH0_MANTICORE_BIAS,
            **CH0_FIXED_BIAS_PRIORS,
        }
        for dataset in _ch0_manticore_cola_cic_field_datasets()
    ]


def _ch0_manticore_swift_fixed_bias_datasets():
    return [
        {
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": CH0_MANTICORE_BIAS,
            "io/field_indices": field,
            **CH0_SWIFT_FIXED_BIAS_PRIORS,
        }
        for field in range(30)
    ]


def _ch0_manticore_swift_cola_cic_uniform_bias_datasets():
    swift_datasets = [
        {
            "model/which_selection": "SN_magnitude",
            "model/use_reconstruction": True,
            "model/use_fiducial_Cepheid_host_PV_covariance": False,
            "model/use_PV_covmat_scaling": False,
            "model/weight_selection_by_covmat_Neff": False,
            "model/use_density_dependent_sigma_v": False,
            "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
            "model/which_bias": "uniform",
            "model/field_3d_smoothing_scale": 0.0,
            "io/field_indices": field,
        }
        for field in range(30)
    ]
    cola_datasets = [
        {
            **dataset,
            "model/which_bias": "uniform",
            "model/field_3d_smoothing_scale": 0.0,
        }
        for dataset in _ch0_manticore_cola_cic_field_datasets()
    ]
    return swift_datasets + cola_datasets


# SN hosts proposed for JWST Cepheid re-observation (Riess, 2026-09).
CH0_JWST_HOSTS = [
    "N3982", "N3254", "N5917", "N3021", "N7541", "N1309", "N4680", "N2608",
    "N1015", "N3583", "N0691", "U9391", "M1337", "N5728", "N7678", "N7329"]


# The 35 SN hosts with cz_cmb < 3300 km/s (anchors excluded).
CH0_SN_HOSTS = [
    "M101", "M1337", "N0691", "N1015", "N1309", "N1365", "N1448", "N1559",
    "N2442", "N2525", "N2608", "N3021", "N3147", "N3254", "N3370", "N3447",
    "N3583", "N3972", "N3982", "N4038", "N4424", "N4536", "N4639", "N4680",
    "N5468", "N5584", "N5643", "N5728", "N5861", "N5917", "N7250", "N7329",
    "N7541", "N7678", "U9391"]


CH0_JWST_ERROR_SCALE = 0.5


def _ch0_jwst_forecast_datasets(distances_only=False):
    """Paper Table 3 grid with Cepheid errors rescaled as for JWST."""
    # Host lists are wrapped in a one-element list so the generator treats
    # them as a single value rather than a sweep axis.
    scale = {"io/SH0ES/cepheid_error_scale": CH0_JWST_ERROR_SCALE}
    not_jwst = [h for h in CH0_SN_HOSTS if h not in CH0_JWST_HOSTS]
    variants = [
        {},
        {**scale, "io/SH0ES/cepheid_error_scale_hosts": [CH0_SN_HOSTS]},
        {**scale, "io/SH0ES/cepheid_error_scale_hosts": "all"},
        {**scale, "io/SH0ES/cepheid_error_scale_hosts": [not_jwst]},
        {"io/SH0ES/keep_hosts": [CH0_JWST_HOSTS]},
        {"io/SH0ES/keep_hosts": [CH0_JWST_HOSTS], **scale,
         "io/SH0ES/cepheid_error_scale_hosts": [CH0_JWST_HOSTS]},
    ]
    if distances_only:
        # Cepheid-only distance covariance for the error-budget toy.
        base = _ch0_distance_only_datasets()[0]
        root = with_root(f"{CH0_PAPER_ROOT}/jwst_forecast/distances")
        return [{**base, **v, **root} for v in variants[1:]]
    table = [d for d in _ch0_main_datasets()
             if d["model/which_selection"] != "none"]
    return [{**d, **v} for v in variants for d in table]


def _ch0_jwst_far_hosts_datasets():
    """Add the two SH0ES hosts beyond 3300 km/s (NGC 976, NGC 105)."""
    table = [d for d in _ch0_main_datasets()
             if d["model/which_selection"] == "SN_magnitude"]
    fiducial, no_pv = table[5], table[0]
    far = {"io/SH0ES/cepheid_host_cz_cmb_max": 5500}
    halved = {"io/SH0ES/cepheid_error_scale": CH0_JWST_ERROR_SCALE,
              "io/SH0ES/cepheid_error_scale_hosts": [["N0976", "N0105"]]}
    return [{**no_pv, **far}, {**fiducial, **far},
            {**fiducial, **far, **halved}]


def _ch0_jwst_sigv150_datasets():
    """Carrick and fiducial Manticore forecast runs, sigma_v = 150 km/s."""
    fixed = {"model/priors/sigma_v": delta(150.0)}

    def keep(d):
        recon = d.get("io/SH0ES/reconstruction")
        if not d.get("model/use_reconstruction"):
            return False
        if recon == "Carrick2015":
            return True
        return (recon == CH0_MANTICORE_COLA_LOS
                and not d.get("model/use_density_dependent_sigma_v")
                and "model/priors/beta" not in d)

    # Scenario 3 halves NGC 4258 but not the LMC or M31 Cepheids, whose
    # errors are already at the intrinsic period-luminosity width.
    hosts_n4258 = {"io/SH0ES/cepheid_error_scale_hosts":
                   [CH0_SN_HOSTS + ["N4258"]]}
    root = with_root(f"{CH0_PAPER_ROOT}/jwst_forecast/sigv150")
    out = []
    for d in _ch0_jwst_forecast_datasets():
        if not keep(d):
            continue
        if d.get("io/SH0ES/cepheid_error_scale_hosts") == "all":
            d = {**d, **hosts_n4258}
        out.append({**d, **fixed, **root})
    return out


def _ch0_leaveoneout_datasets():
    return [{
        "model/which_selection": "SN_magnitude",
        "model/use_reconstruction": True,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,
        "model/use_density_dependent_sigma_v": False,
        "io/SH0ES/reconstruction": CH0_MANTICORE_LOS,
        "model/which_bias": CH0_MANTICORE_BIAS,
        "io/field_indices": 21,
        "io/SH0ES/drop_observation": list(range(35)),
    }]


def _ch0_angular_scatter_datasets():
    return [{
        "model/which_selection": "SN_magnitude",
        "model/use_reconstruction": True,
        "model/use_fiducial_Cepheid_host_PV_covariance": False,
        "model/use_PV_covmat_scaling": False,
        "model/weight_selection_by_covmat_Neff": False,
        "model/use_density_dependent_sigma_v": False,
        "io/SH0ES/reconstruction": CH0_MANTICORE_COLA_LOS,
        "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "CIC",
        "model/which_bias": CH0_MANTICORE_BIAS,
        "io/field_indices": list(range(30)),
        "io/angular_position_scatter_deg": [2.0, 4.0, 8.0, 16.0],
        "io/angular_position_scatter_seed": 42,
    }]


TASK_SPECS = {
    "CH0_main": {
        "description": "CH0 paper H0 grid plus redshift-free distance runs.",
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "paper",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            **with_root(f"{CH0_PAPER_ROOT}/table"),
        },
        "datasets": _ch0_main_datasets() + _ch0_distance_only_datasets(),
        "expected_tasks": 27,
    },
    "CH0_JWST_forecast": {
        "description": (
            "CH0 Table 3 grid with SH0ES Cepheid errors rescaled to "
            "forecast JWST re-observation of the SN hosts."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "jwst",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            **with_root(f"{CH0_PAPER_ROOT}/jwst_forecast"),
        },
        "datasets": _ch0_jwst_forecast_datasets(),
        "expected_tasks": 96,
    },
    "CH0_JWST_forecast_distances": {
        "description": (
            "Redshift-free CH0 Cepheid distances with JWST-rescaled Cepheid "
            "errors, for the error-budget toy."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "jwst",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
        },
        "datasets": _ch0_jwst_forecast_datasets(distances_only=True),
        "expected_tasks": 5,
    },
    "CH0_JWST_forecast_far_hosts": {
        "description": (
            "CH0 with the two SH0ES hosts beyond 3300 km/s added, to test "
            "the error-budget forecast for new distant hosts."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "jwst",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            **with_root(f"{CH0_PAPER_ROOT}/jwst_forecast/far_hosts"),
        },
        "datasets": _ch0_jwst_far_hosts_datasets(),
        "expected_tasks": 3,
    },
    "CH0_JWST_forecast_sigv150": {
        "description": (
            "CH0 JWST forecast with Carrick2015 and Manticore-Local, "
            "residual velocity scatter fixed at 150 km/s."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "jwst",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
        },
        "datasets": _ch0_jwst_sigv150_datasets(),
        "expected_tasks": 24,
    },
    "CH0_mixed_selection": {
        "description": "CH0 paper mixed SN-magnitude/redshift split.",
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "paper_mixed",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 5,
            "inference/chain_method": "sequential",
            "inference/num_warmup": 1000,
            "inference/num_samples": 5000,
            **with_root(f"{CH0_PAPER_ROOT}/mixed_selection"),
            "model/density_3d_subsample_fraction": 0.5,
        },
        "datasets": _ch0_mixed_selection_datasets(),
        "expected_tasks": 36,
    },
    "CH0_imb_geometric": {
        "description": (
            "CH0 redshift-free geometric Cepheid-calibration runs testing "
            "no density field, Carrick linear bias, and Manticore COLA/PCS "
            "double-power-law density priors."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "imb_geometric",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            **with_root(f"{CH0_PAPER_ROOT}/imb_geometric"),
        },
        "datasets": _ch0_imb_geometric_datasets(),
        "expected_tasks": 82,
    },
    "CH0_single": {
        "description": (
            "CH0 one-Manticore-field runs with evidence."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "single",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **with_root(f"{CH0_PAPER_ROOT}/single_fields"),
        },
        "datasets": _ch0_manticore_field_datasets(),
        "expected_tasks": 350,
    },
    "CH0_single_smoothed": {
        "description": (
            "CH0 CIC COLA one-field runs with density-field smoothing."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "single_smoothed",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            "model/field_3d_smoothing_scale": [4.0, 8.0, 16.0, 32.0],
            "model/velocity_3d_smoothing_scale": 0.0,
            **with_root(f"{CH0_PAPER_ROOT}/single_fields_smoothed"),
        },
        "datasets": (
            _ch0_manticore_cola_cic_field_datasets()
            + _ch0_manticore_swift_cola_cic_uniform_bias_datasets()
        ),
        "expected_tasks": 430,
    },
    "CH0_single_fixed_bias": {
        "description": (
            "CH0 CIC COLA and SWIFT one-field runs with fixed "
            "double-power-law bias."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "single_fixed_bias",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **with_root(f"{CH0_PAPER_ROOT}/single_fields_fixed_bias"),
        },
        "datasets": (
            _ch0_manticore_cola_cic_fixed_bias_datasets()
            + _ch0_manticore_swift_fixed_bias_datasets()
        ),
        "expected_tasks": 110,
    },
    "CH0_leaveoneout": {
        "description": (
            "CH0 SN-magnitude leave-one-out runs for ManticoreLocalSWIFT "
            "field 21."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "leaveoneout",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **with_root(f"{CH0_PAPER_ROOT}/leaveoneout"),
        },
        "datasets": _ch0_leaveoneout_datasets(),
        "expected_tasks": 35,
    },
    "CH0_angular_scatter": {
        "description": (
            "CH0 SN-magnitude angular-position scatter runs for CIC "
            "ManticoreLocalCOLA fields 0-29."),
        "config_path": str(CONFIG_DIR / "config_CH0.toml"),
        "tag": "angular_scatter",
        "common": {
            **CH0_PAPER_COMMON,
            "inference/compute_log_density": True,
            "inference/compute_evidence": True,
            "inference/num_chains": 1,
            "inference/num_warmup": 1000,
            "inference/num_samples": 2000,
            "inference/save_log_likelihood_per_galaxy": True,
            **with_root(f"{CH0_PAPER_ROOT}/angular_scatter"),
        },
        "datasets": _ch0_angular_scatter_datasets(),
        "expected_tasks": 120,
    },
}
