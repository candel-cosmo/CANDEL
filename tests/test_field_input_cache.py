from scripts.preprocess import field_input_cache as cache_mod
from scripts.preprocess.field_input_cache import (_cache_group_key,
                                                  _h0_velocity_key,
                                                  _variant_action,
                                                  _variant_info)


def test_reconstructed_no_selection_ch0_still_warms_3d_cache():
    config = {
        "model": {
            "which_run": "CH0",
            "which_selection": None,
            "use_reconstruction": True,
        },
    }

    assert _variant_action(config) == "check/cache"


def test_reconstructed_no_selection_cchp_skips_3d_cache():
    config = {
        "model": {
            "which_run": "CCHP",
            "which_selection": None,
            "use_reconstruction": True,
        },
    }

    assert _variant_action(config) == "skip 3D"


def test_reconstructed_no_selection_trgb_string_none_warms_3d_cache():
    config = {
        "model": {
            "which_run": "EDD_TRGB",
            "which_selection": "none",
            "use_reconstruction": True,
        },
    }

    assert _variant_action(config) == "check/cache"


def test_edd_trgb_redshift_selection_does_not_request_velocity_cache():
    config = {
        "model": {
            "which_run": "EDD_TRGB",
            "which_selection": "redshift",
        },
    }

    assert _h0_velocity_key(config) == "density"


def test_cache_off_plan_does_not_require_project(tmp_path):
    config_path = tmp_path / "cache_off.toml"
    config_path.write_text(
        "[io]\n"
        "field_cache_enabled = false\n"
        "[model]\n"
        'which_run = "CH0"\n'
        "use_reconstruction = true\n",
        encoding="utf-8")

    info = _variant_info(config_path)

    assert info["action"] == "cache off"
    assert info["cache_dir"] is None
    assert info["cache_project"] is None


def test_h0_cache_group_distinguishes_loader_settings(tmp_path, monkeypatch):
    config = {
        "io": {
            "field_cache_dir": str(tmp_path),
            "field_cache_project": "TRGBH0",
            "reconstruction_main": {
                "Recon": {"which_MAS": "CIC"},
            },
        },
        "model": {
            "which_run": "EDD_TRGB",
            "which_selection": "TRGB_magnitude",
            "use_reconstruction": True,
        },
    }
    monkeypatch.setattr(
        cache_mod, "_h0_los_config", lambda config: ("Recon", "los.hdf5"))
    monkeypatch.setattr(
        cache_mod, "_resolve_los_path", lambda *args, **kwargs: "los.hdf5")
    monkeypatch.setattr(
        cache_mod, "_h0_field_indices_for_plan",
        lambda *args, **kwargs: [0])
    monkeypatch.setattr(
        cache_mod, "_h0_supersampling_payload",
        lambda *args, **kwargs: {})

    cic = _cache_group_key(config)
    config["io"]["reconstruction_main"]["Recon"]["which_MAS"] = "PCS"

    assert cic != _cache_group_key(config)
