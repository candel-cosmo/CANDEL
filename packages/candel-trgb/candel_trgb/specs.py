# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Named TRGB H0 task specs for generate_tasks.py."""
from pathlib import Path

from candel.tasks import delta, nu_cz_student_t_prior, with_root

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


TRGBH0_ROOT = "results/TRGBH0_paper"


TRGBH0_MANTICORE_COLA_LOS = "ManticoreLocalCOLA"


TRGBH0_MANTICORE_BIAS = "double_powerlaw"


TRGBH0_MAIN_MANTICORE_BIAS = TRGBH0_MANTICORE_BIAS


TRGBH0_EDD_B_MIN = 10.0


TRGBH0_EDD_MAG_MIN = 22.1


TRGBH0_EDD_MAG_LIM_LOW = 22.101


TRGBH0_EDD_MAG_LIM_HIGH = 29.0


TRGBH0_SELECTION_SUPERSAMPLE_RADIUS = 15.0


TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX = 0.325


# 48-pixel (Nside=2) baseline sky-exposure map; kappa is the total Dirichlet
# concentration, so alpha_q = kappa / N_pix = 4 per pixel at both resolutions.
TRGBH0_SKY_EXPOSURE_NSIDE = 2


TRGBH0_SKY_EXPOSURE_KAPPA = 192.0


TRGBH0_SKY_EXPOSURE_NSIDE_12PIX = 1


TRGBH0_SKY_EXPOSURE_KAPPA_12PIX = 48.0


TRGBH0_COMMON = {
    "inference/compute_log_density": True,
    "inference/compute_evidence": True,
    "inference/num_chains": 1,
    "inference/chain_method": "sequential",
    "inference/num_warmup": 1000,
    "inference/num_samples": 1000,
    "inference/num_chains_harmonic": 10,
    "model/selection_integral_geometry": "sphere",
    "model/selection_integral_grid_radius": 50.0,
    "model/density_3d_subsample_fraction": 1.0,
    "model/priors/Vext": {
        "dist": "vector_uniform_fixed",
        "low": 0.0,
        "high": 1000.0,
    },
    "model/priors/sigma_int": {
        "dist": "truncated_normal",
        "mean": 0.1,
        "scale": 0.01,
        "low": 0.01,
    },
    "model/priors/alpha_c": delta(0.2),
}


def _trgbh0_edd_mag_lim_uninformative_prior():
    return {
        "dist": "uniform",
        "low": TRGBH0_EDD_MAG_LIM_LOW,
        "high": TRGBH0_EDD_MAG_LIM_HIGH,
    }


def _trgbh0_selection(selection):
    return {"model/which_selection": selection}


def _trgbh0_edd_b_cut():
    return {
        "io/PV_main/EDD_TRGB/b_min": TRGBH0_EDD_B_MIN,
    }


def _trgbh0_sky_exposure(
        nside=TRGBH0_SKY_EXPOSURE_NSIDE,
        kappa=TRGBH0_SKY_EXPOSURE_KAPPA):
    return {
        "model/TRGB_sky_exposure/enabled": True,
        "model/TRGB_sky_exposure/nside": nside,
        "model/TRGB_sky_exposure/kappa": kappa,
    }


def _trgbh0_edd_selection_datasets(
        pv_models, selections=("TRGB_magnitude",)):
    return [
        {
            **_trgbh0_edd_b_cut(),
            **pv_model,
            **_trgbh0_selection(selection),
        }
        for pv_model in pv_models
        for selection in selections
    ]


def _trgbh0_main_datasets():
    """Realisation-marginalised grid, one dataset per tab:trgb_h0_variants row.

    The baseline is the Manticore COLA PCS reconstruction with 4 Mpc/h
    source-density smoothing, a Student-t redshift likelihood, the 48-pixel
    angular sky-exposure term, and the velocity amplitude fixed at beta=1
    (enforced for Manticore by the generator). Variants change one axis off
    this baseline: redshift likelihood, smoothing scale, sky exposure, and
    the coherent-flow monopole; a no-reconstruction free-Vext control and the
    redshift-free distance run complete the set.
    """
    selections = ("TRGB_magnitude",)
    base = {
        "model/use_reconstruction": True,
        "model/use_density_dependent_sigma_v": False,
        "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
        "model/priors/mag_lim_TRGB": (
            _trgbh0_edd_mag_lim_uninformative_prior()),
        "io/PV_main/EDD_TRGB/reconstruction": TRGBH0_MANTICORE_COLA_LOS,
        "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": "PCS",
        "model/which_bias": TRGBH0_MAIN_MANTICORE_BIAS,
        "model/field_3d_smoothing_scale": 4.0,
        "model/velocity_3d_smoothing_scale": 0.0,
    }
    sky = _trgbh0_sky_exposure()
    sky_12pix = _trgbh0_sky_exposure(
        nside=TRGBH0_SKY_EXPOSURE_NSIDE_12PIX,
        kappa=TRGBH0_SKY_EXPOSURE_KAPPA_12PIX)

    def student_t(cfg):
        return {
            **cfg,
            "model/cz_likelihood": "student_t",
            "model/priors/nu_cz": nu_cz_student_t_prior(),
        }

    def gaussian(cfg):
        return {**cfg, "model/cz_likelihood": "gaussian"}

    manticore_variants = [
        # Baseline: Student-t, 48-pixel sky, R_rho=4 Mpc/h, beta=1.
        student_t({**base, **sky}),
        # Redshift likelihood: Gaussian at the baseline configuration.
        gaussian({**base, **sky}),
        # Source-density smoothing R_rho=8 Mpc/h.
        student_t({**base, **sky, "model/field_3d_smoothing_scale": 8.0}),
        gaussian({**base, **sky, "model/field_3d_smoothing_scale": 8.0}),
        # Source-density smoothing removed entirely (R_rho=0).
        student_t({**base, **sky, "model/field_3d_smoothing_scale": 0.0}),
        gaussian({**base, **sky, "model/field_3d_smoothing_scale": 0.0}),
        # Angular sky exposure off (Galactic-plane mask only).
        student_t(base),
        gaussian(base),
        # Angular sky exposure at the 12-pixel (Nside=1) resolution.
        student_t({**base, **sky_12pix}),
        gaussian({**base, **sky_12pix}),
        # Coherent-flow sector: constant velocity monopole.
        student_t({**base, **sky, "model/which_Vext_monopole": "constant"}),
        gaussian({**base, **sky, "model/which_Vext_monopole": "constant"}),
    ]
    no_reconstruction = {
        "model/use_reconstruction": False,
        "model/use_density_dependent_sigma_v": False,
        "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
        "model/priors/mag_lim_TRGB": (
            _trgbh0_edd_mag_lim_uninformative_prior()),
    }
    single_field_check = [
        {
            **student_t({**base, **sky}),
            "io/field_indices": 66,
            "inference/seed": seed,
            "io/run_label": f"seed{seed}",
        }
        for seed in range(44, 54)
    ]

    return (
        _trgbh0_edd_selection_datasets(manticore_variants, selections)
        + _trgbh0_edd_selection_datasets(
            [student_t(no_reconstruction), gaussian(no_reconstruction)],
            selections)
        + _trgbh0_distance_only_datasets()
        + _trgbh0_edd_selection_datasets(single_field_check, selections)
    )


def _trgbh0_distance_only_datasets():
    return [
        {
            **_trgbh0_edd_b_cut(),
            "model/use_TRGB_host_redshift": False,
            "model/use_reconstruction": False,
            "model/use_density_dependent_sigma_v": False,
            "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
            "model/priors/mag_lim_TRGB": (
                _trgbh0_edd_mag_lim_uninformative_prior()),
            "model/priors/H0": delta(73.04),
            "model/priors/Vext": delta([0.0, 0.0, 0.0]),
            "model/priors/sigma_v": delta(100.0),
            **_trgbh0_selection("TRGB_magnitude"),
            **with_root(f"{TRGBH0_ROOT}/distances"),
        },
    ]


def _trgbh0_manticore_cola_mas_field_datasets(mas_values):
    datasets = []
    for mas in mas_values:
        for field in range(80):
            datasets.append({
                **_trgbh0_edd_b_cut(),
                "model/use_reconstruction": True,
                "model/use_density_dependent_sigma_v": False,
                "model/cz_likelihood": "gaussian",
                "model/mag_min_TRGB": TRGBH0_EDD_MAG_MIN,
                "model/priors/mag_lim_TRGB": (
                    _trgbh0_edd_mag_lim_uninformative_prior()),
                "io/PV_main/EDD_TRGB/reconstruction": (
                    TRGBH0_MANTICORE_COLA_LOS),
                "io/reconstruction_main/ManticoreLocalCOLA/which_MAS": mas,
                "model/which_bias": TRGBH0_MANTICORE_BIAS,
                "io/field_indices": field,
                **_trgbh0_selection("TRGB_magnitude"),
            })
    return datasets


def _trgbh0_manticore_cola_single_datasets():
    return (
        _trgbh0_manticore_cola_pcs_field_datasets()
        + _trgbh0_manticore_cola_pcs_student_t_field_datasets()
    )


def _trgbh0_manticore_cola_pcs_student_t_field_datasets():
    datasets = _trgbh0_manticore_cola_mas_field_datasets(("PCS",))
    for dataset in datasets:
        dataset["model/cz_likelihood"] = "student_t"
        dataset["model/priors/nu_cz"] = nu_cz_student_t_prior()
    return datasets


def _trgbh0_manticore_cola_pcs_field_datasets():
    return _trgbh0_manticore_cola_mas_field_datasets(("PCS",))


def _trgbh0_manticore_cola_pcs_monopole_smoothed_field_datasets():
    datasets = _trgbh0_manticore_cola_pcs_field_datasets()
    for dataset in datasets:
        dataset["model/which_Vext_monopole"] = "constant"
        dataset["model/field_3d_smoothing_scale"] = 4.0
    return datasets


def _trgbh0_manticore_cola_pcs_monopole_smoothed_nosky_field_datasets():
    datasets = _trgbh0_manticore_cola_pcs_monopole_smoothed_field_datasets()
    for dataset in datasets:
        dataset["model/TRGB_sky_exposure/enabled"] = False
    return datasets


def _trgbh0_manticore_cola_pcs_monopole_student_t_smoothed_field_datasets():
    datasets = _trgbh0_manticore_cola_pcs_monopole_smoothed_field_datasets()
    for dataset in datasets:
        dataset["model/cz_likelihood"] = "student_t"
        dataset["model/priors/nu_cz"] = nu_cz_student_t_prior()
    return datasets


def _trgbh0_manticore_cola_pcs_smoothed_nosky_field_datasets():
    datasets = _trgbh0_manticore_cola_pcs_field_datasets()
    for dataset in datasets:
        dataset["model/field_3d_smoothing_scale"] = 4.0
        dataset["model/TRGB_sky_exposure/enabled"] = False
    return datasets


def _trgbh0_manticore_cola_pcs_student_t_smoothed_nosky_field_datasets():
    datasets = _trgbh0_manticore_cola_pcs_smoothed_nosky_field_datasets()
    for dataset in datasets:
        dataset["model/cz_likelihood"] = "student_t"
        dataset["model/priors/nu_cz"] = nu_cz_student_t_prior()
    return datasets


def _trgbh0_manticore_cola_pcs_student_t_smoothed_12pix_field_datasets():
    # Field-stacked partner of the 12-pixel (Nside=1) Student-t TRGBH0_main
    # variant: R4 Student-t at the coarse sky-exposure resolution.
    datasets = _trgbh0_manticore_cola_pcs_student_t_field_datasets()
    for dataset in datasets:
        dataset["model/field_3d_smoothing_scale"] = 4.0
        dataset["model/TRGB_sky_exposure/nside"] = (
            TRGBH0_SKY_EXPOSURE_NSIDE_12PIX)
        dataset["model/TRGB_sky_exposure/kappa"] = (
            TRGBH0_SKY_EXPOSURE_KAPPA_12PIX)
    return datasets


_pcs_mono_smoothed_nosky = (
    _trgbh0_manticore_cola_pcs_monopole_smoothed_nosky_field_datasets)


_pcs_mono_student_t_smoothed = (
    _trgbh0_manticore_cola_pcs_monopole_student_t_smoothed_field_datasets)


_pcs_student_t_smoothed_12pix = (
    _trgbh0_manticore_cola_pcs_student_t_smoothed_12pix_field_datasets)


TASK_SPECS = {
    "TRGBH0_main": {
        "description": (
            "TRGB H0 realisation-marginalised grid (one row per "
            "tab:trgb_h0_variants entry): Manticore COLA PCS baseline with "
            "redshift-likelihood, source-density smoothing, angular "
            "sky-exposure, and coherent-flow (Vmono) variants, plus a "
            "no-reconstruction free-Vext control, the redshift-free "
            "distance run, and ten field-66 seed repeats."),
        "config_path": str(CONFIG_DIR / "config_EDD_TRGB.toml"),
        "tag": "main",
        "common": {
            **TRGBH0_COMMON,
            "inference/init_maxiter": 500,
            "inference/init_num_starts": 4,
            "inference/init_median_num_samples": 100,
            "inference/num_warmup": 2000,
            "inference/num_samples": 5000,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/H0/low": 40,
            "model/priors/H0/high": 100,
            "model/priors/mag_lim_TRGB": (
                _trgbh0_edd_mag_lim_uninformative_prior()),
            "model/priors/mag_lim_TRGB_width/low": 0.15,
            **with_root(f"{TRGBH0_ROOT}/table"),
        },
        "datasets": _trgbh0_main_datasets(),
        "expected_tasks": 25,
    },
    "TRGBH0_single": {
        "description": (
            "TRGB H0 Manticore COLA PCS one-field runs (unsmoothed) with "
            "Gaussian and Student-t redshift likelihoods, providing the "
            "field-stacked no-smoothing baseline."),
        "config_path": str(CONFIG_DIR / "config_EDD_TRGB.toml"),
        "tag": "single",
        "common": {
            **TRGBH0_COMMON,
            "inference/num_warmup": 1000,
            "inference/num_samples": 1000,
            "inference/save_log_likelihood_per_galaxy": True,
            "model/priors/H0/low": 40,
            "model/priors/H0/high": 100,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/mag_lim_TRGB_width/low": 0.15,
            **_trgbh0_edd_b_cut(),
            **_trgbh0_sky_exposure(),
            **with_root(f"{TRGBH0_ROOT}/single_fields"),
        },
        "datasets": _trgbh0_manticore_cola_single_datasets(),
        "expected_tasks": 160,
    },
    "TRGBH0_single_smoothed": {
        "description": (
            "TRGB H0 PCS COLA one-field runs with density-field smoothing, "
            "including Gaussian and Student-t redshift likelihoods."),
        "config_path": str(CONFIG_DIR / "config_EDD_TRGB.toml"),
        "tag": "single_smoothed",
        "common": {
            **TRGBH0_COMMON,
            "inference/num_warmup": 1000,
            "inference/num_samples": 1000,
            "inference/save_log_likelihood_per_galaxy": True,
            "model/priors/H0/low": 40,
            "model/priors/H0/high": 100,
            "model/selection_integral_supersample_radius": (
                TRGBH0_SELECTION_SUPERSAMPLE_RADIUS),
            "model/selection_integral_supersample_target_dx": (
                TRGBH0_SELECTION_SUPERSAMPLE_TARGET_DX),
            "model/priors/mag_lim_TRGB_width/low": 0.15,
            **_trgbh0_edd_b_cut(),
            **_trgbh0_sky_exposure(),
            "model/field_3d_smoothing_scale": [4.0, 8.0],
            "model/velocity_3d_smoothing_scale": 0.0,
            **with_root(f"{TRGBH0_ROOT}/single_fields_smoothed"),
        },
        "datasets": (
            _trgbh0_manticore_cola_pcs_field_datasets()
            + _trgbh0_manticore_cola_pcs_student_t_field_datasets()
            + _trgbh0_manticore_cola_pcs_monopole_smoothed_field_datasets()
            + _pcs_mono_smoothed_nosky()
            + _pcs_mono_student_t_smoothed()
            + _trgbh0_manticore_cola_pcs_smoothed_nosky_field_datasets()
            + _trgbh0_manticore_cola_pcs_student_t_smoothed_nosky_field_datasets()  # noqa: E501
            + _pcs_student_t_smoothed_12pix()
        ),
        "expected_tasks": 800,
    },
}
