from pathlib import Path

import numpy as np
import pytest
from h5py import File

from candel.field.field_cache import (
    _VOLUME_FIELD_CACHE_PREFIX, _field_cache_path)
from candel.field.field_products import los_field_cache_path
from candel.field.los import _los_reconstruction_name, load_los
from candel.field.volume_density import _density_unit_normalization


def test_volume_cache_paths_are_portable_and_science_keyed(tmp_path):
    payload = {
        "kind": "volume_field_data",
        "project": "TRGBH0",
        "product": "h0_volume",
        "loader_name": "ManticoreLocalCOLA",
        "loader_kwargs": {
            "fpath_root": "/machine-a/fields",
            "which_MAS": "CIC",
        },
        "field_indices": [0],
        "subcube_radius": 50.0,
        "downsample": 1,
        "geometry": "sphere",
        "load_velocity": False,
        "sources": [{"path": "/machine-a/fields"}],
    }

    path = _field_cache_path(tmp_path, _VOLUME_FIELD_CACHE_PREFIX, payload)
    assert path == _field_cache_path(
        tmp_path, _VOLUME_FIELD_CACHE_PREFIX,
        {
            **payload,
            "loader_kwargs": {
                **payload["loader_kwargs"],
                "fpath_root": "/machine-b/fields",
            },
            "sources": [{"path": "/machine-b/fields"}],
        })
    assert path != _field_cache_path(
        tmp_path, _VOLUME_FIELD_CACHE_PREFIX,
        {
            **payload,
            "loader_kwargs": {
                **payload["loader_kwargs"],
                "which_MAS": "PCS",
            },
        })
    assert path != _field_cache_path(
        tmp_path, _VOLUME_FIELD_CACHE_PREFIX,
        {**payload, "subcube_radius": 55.0})
    assert path != _field_cache_path(
        tmp_path, _VOLUME_FIELD_CACHE_PREFIX,
        {**payload, "field_indices": [1]})


def test_los_field_cache_path_separates_density_and_velocity_smoothing(
        tmp_path):
    config = {
        "io": {
            "field_cache_dir": str(tmp_path),
            "field_cache_project": "TRGBH0",
            "reconstruction_main": {
                "rmin": 0.001,
                "rmax": 201,
                "num_steps": 251,
                "ManticoreLocalCOLA": {"which_MAS": "CIC"},
            },
        },
    }

    path = los_field_cache_path(
        config, "CF4", "ManticoreLocalCOLA", "data/los_<X>.hdf5",
        field_smoothing_scale=8.0,
        velocity_field_smoothing_scale=16.0,
        field_indices=0)

    name = Path(path).name
    assert "density-smooth-R8" in name
    assert "velocity-smooth-R16" in name
    assert "field-smooth-R8" not in name


def test_load_los_rejects_cache_built_for_other_positions(tmp_path):
    path = tmp_path / "los.hdf5"
    with File(path, "w") as f:
        f["RA"], f["dec"] = np.array([10.0, 20.0]), np.array([0.0, 5.0])
        f["r"] = np.linspace(1, 10, 3)
        f["los_density"] = np.ones((1, 2, 3))
        f["los_velocity"] = np.zeros((1, 2, 3))

    load_los(str(path), {"RA": np.array([10.0, 20.0]),
                         "dec": np.array([0.0, 5.0])}, verbose=False)
    with pytest.raises(ValueError, match="stale"):
        load_los(str(path), {"RA": np.array([20.0, 10.0]),
                             "dec": np.array([5.0, 0.0])}, verbose=False)


def test_los_normalisation_ignores_unrelated_directory_names(tmp_path):
    path = (tmp_path / "Manticore_project" / "CH0" / "Carrick2015" / "los"
            / "field-0" / "los__SH0ES.hdf5")
    path.parent.mkdir(parents=True)
    with File(path, "w"):
        pass
    with File(path, "r") as f:
        name = _los_reconstruction_name(str(path), f)
    assert name == "Carrick2015"
    assert _density_unit_normalization(name) is None
