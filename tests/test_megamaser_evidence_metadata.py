"""Metadata regressions for single-galaxy megamaser evidence."""
import os
import sys
from types import SimpleNamespace

from h5py import File as H5File
import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEGAMASER_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser")
if MEGAMASER_DIR not in sys.path:
    sys.path.insert(0, MEGAMASER_DIR)

import evidence_single_galaxy as ev  # noqa: E402
import run_maser as rm  # noqa: E402


class DummyTarget:
    def __init__(self, names):
        self.names = tuple(names)


def test_chain_attrs_restore_uniform_da_prior():
    cfg = {
        "model": {
            "D_c_prior": "uniform",
            "galaxies": {
                "NGC6264": {
                    "mass_parameterization": "log_mbh",
                    "use_ecc": False,
                    "use_quadratic_warp": True,
                },
            },
        },
    }
    attrs = {
        "uniform_da_prior": True,
        "mass_parameterization": b"eta",
        "use_ecc": np.bool_(True),
        "use_quadratic_warp": "false",
    }

    gblk = ev._apply_chain_attrs_to_config(cfg, "NGC6264", attrs)

    assert cfg["model"]["D_c_prior"] == "uniform_D_A"
    assert gblk["mass_parameterization"] == "eta"
    assert gblk["use_ecc"] is True
    assert gblk["use_quadratic_warp"] is False


def test_chain_attrs_restore_da2_prior():
    cfg = {"model": {"D_c_prior": "uniform_D_A",
                     "galaxies": {"NGC4258": {}}}}
    ev._apply_chain_attrs_to_config(
        cfg, "NGC4258", {"D_c_prior": "volume_D_A"})
    assert cfg["model"]["D_c_prior"] == "volume_D_A"


def test_da2_prior_is_tagged_in_chain_suffix():
    model = SimpleNamespace(
        use_ecc=False, use_quadratic_warp=True, galaxy_name="NGC4258")
    args = SimpleNamespace(da2_prior=True)
    assert rm._variant_suffix(model, args, "config") == \
        "_qw_da2_initconfig"


def test_ngc5765b_chain_attrs_restore_error_floor_mode():
    cfg = {
        "model": {
            "use_ngc5765b_clump2_floors": True,
            "galaxies": {"NGC5765b": {}},
        },
    }
    ev._apply_chain_attrs_to_config(
        cfg, "NGC5765b",
        {"error_floor_policy": "sampled_ngc5765b_single_floor"})
    assert cfg["model"]["use_ngc5765b_clump2_floors"] is False
    assert cfg["model"]["ngc5765b_clump2_acceleration_only"] is False

    ev._apply_chain_attrs_to_config(
        cfg, "NGC5765b",
        {"error_floor_policy":
         "sampled_ngc5765b_clump2_acceleration_only"})
    assert cfg["model"]["use_ngc5765b_clump2_floors"] is True
    assert cfg["model"]["ngc5765b_clump2_acceleration_only"] is True

    ev._apply_chain_attrs_to_config(
        cfg, "NGC5765b",
        {"theta_sites": "D_A,sigma_a_floor_clump2"})
    assert cfg["model"]["use_ngc5765b_clump2_floors"] is True
    assert cfg["model"]["ngc5765b_clump2_acceleration_only"] is True


def test_sampled_global_names_ignore_deterministic_extras():
    target = DummyTarget(("D_A", "eta", "x0"))
    samples = {
        "D_A": np.array([1.0, 2.0]),
        "eta": np.array([3.0, 4.0]),
        "x0": np.array([5.0, 6.0]),
        "D_c": np.array([7.0, 8.0]),
        "log_MBH": np.array([9.0, 10.0]),
    }

    names = ev._sampled_global_names(target, samples)
    X = ev._stack_globals(samples, names)

    assert names == ("D_A", "eta", "x0")
    np.testing.assert_allclose(X, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])


def test_sampled_global_names_report_missing_sampled_site():
    target = DummyTarget(("D_A", "eta"))
    samples = {
        "D_c": np.array([1.0]),
        "eta": np.array([2.0]),
    }

    with pytest.raises(KeyError, match="uniform_da_prior"):
        ev._sampled_global_names(target, samples)


def test_chain_writer_preserves_sample_precision_and_skips_latents(tmp_path):
    result = SimpleNamespace(
        samples={
            "D_A": np.array([7.1, 7.2], dtype=np.float64),
            "r_ang": np.array([[1.0], [1.1]], dtype=np.float64),
        },
        log_density=np.array([-1.0, -0.5], dtype=np.float64),
        info={},
        parameters={},
        theta_sites=("D_A",),
    )
    path = tmp_path / "chain.hdf5"
    rm._save_hdf5(
        path, result, {"precision": "float64"}, save_latents=False)

    with H5File(path, "r") as f:
        assert f["samples/D_A"].dtype == np.dtype("float64")
        assert "samples/r_ang" not in f
        assert f.attrs["precision"] == "float64"


def test_evidence_precision_prefers_metadata_then_force_f64():
    f32 = {"x": np.array([1.0], dtype=np.float32)}
    assert ev._production_dtype(
        {"precision": "float32"}, {"force_f64": True}, f32) is ev.jnp.float32
    assert ev._production_dtype(
        {}, {"force_f64": True}, f32) is ev.jnp.float64
    assert ev._production_dtype(
        {}, {}, {"x": np.array([1.0], dtype=np.float64)}) is ev.jnp.float64

    X = ev._stack_globals(
        {"x": np.array([1.0], dtype=np.float64)}, ("x",), dtype=np.float64)
    assert X.dtype == np.dtype("float64")

    with pytest.raises(ValueError, match="unsupported precision"):
        ev._production_dtype({"precision": "float16"}, {}, f32)


def test_mcmc_defaults_to_float64_without_changing_de_policy():
    assert rm._f64_reason_from_argv(["NGC6264"]) == "MCMC default"
    assert rm._f64_reason_from_argv(
        ["NGC6264", "--sampler", "de"]) is None
    assert rm._f64_reason_from_argv(
        ["NGC4258", "--sampler", "de"]) == "forced for NGC4258"
