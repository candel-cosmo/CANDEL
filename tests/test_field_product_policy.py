import math
from pathlib import Path

import pytest

from candel.field import (field_allows_raw_product_reads, field_metadata,
                          field_requires_cached_products,
                          supported_field_names)
from candel.pvdata.field_cache import (
    _VOLUME_FIELD_CACHE_PREFIX, _field_cache_dir_from_config,
    _field_cache_path, _field_cache_product_path,
    _field_cache_project_from_config)
from candel.pvdata.field_products import los_field_cache_path
from scripts.megamaser.joint_H0_helpers import _toy_vlos_cache_path


def test_raw_reads_are_allowed_for_cheap_fields():
    assert field_allows_raw_product_reads("Carrick2015")
    assert not field_requires_cached_products("Carrick2015")

    assert field_allows_raw_product_reads("ManticoreLocalCOLA")
    assert not field_requires_cached_products("ManticoreLocalCOLA")
    assert field_metadata("ManticoreLocalCOLA").cache_group == (
        "ManticoreLocalCOLA")
    assert field_metadata("ManticoreLocalCOLA").ngrid is None
    assert field_metadata("ManticoreLocalCOLA").Omega_m is None
    assert not math.isfinite(field_metadata("ManticoreLocalCOLA").boxsize)
    assert field_metadata("ManticoreLocalCOLA").storage_schema == (
        "overdensity_velocity")


def test_non_cola_manticore_requires_cached_products():
    name = "ManticoreLocalSWIFT"

    assert field_requires_cached_products(name)
    assert not field_allows_raw_product_reads(name)
    assert field_metadata(name).cache_group == "ManticoreLocalSWIFT"
    assert field_metadata(name).ngrid == 1024
    assert field_metadata(name).production_method == "nbody_mas_sph"
    assert field_metadata(name).storage_schema == "density_momentum"


def test_unknown_fields_default_to_require_cached_products():
    name = "experimental_large_field"

    assert field_requires_cached_products(name)
    assert not field_allows_raw_product_reads(name)
    assert field_metadata(name).cache_group == "unknown"


def test_supported_field_names_include_loader_families():
    names = supported_field_names()

    assert "Carrick2015" in names
    assert "Lilow2024" in names
    assert "CF4" in names
    assert "CLONES" in names
    assert "HAMLET_V0" in names
    assert "HAMLET_V1" in names
    assert "CB1" in names
    assert "CB2" in names
    assert "ManticoreLocalCOLA" in names
    assert "ManticoreLocalSWIFT" in names


def test_old_manticore_names_are_not_supported():
    assert field_metadata("COLA_manticore_2MPP_MULTIBIN_N256_DES_V2").name == (
        "unknown")
    assert field_metadata("manticore_2MPP_MULTIBIN_N256_DES_V2").name == (
        "unknown")


def test_los_field_cache_path_uses_field_cache_dir(tmp_path):
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
        field_indices=0)

    path = Path(path)
    assert path.parent == (
        tmp_path / "TRGBH0" / "ManticoreLocalCOLA" / "los" / "field-0")
    assert path.name == "los__CF4__field-0__r-0p001-201-n251__mas-CIC.hdf5"
    assert "v1" not in path.name
    assert "field-0" in path.name
    assert path.suffix == ".hdf5"


@pytest.mark.parametrize("reconstruction", [
    "Carrick2015", "ManticoreLocalCOLA"])
def test_maser_vlos_cache_path_uses_field_cache_dir(
        tmp_path, reconstruction):
    config = {"io": {"field_cache_dir": str(tmp_path)}}
    payload = {"reconstruction": reconstruction}

    path = Path(_toy_vlos_cache_path(payload, config))

    assert path.parent == (
        tmp_path / "MMH0" / reconstruction / "maser_los" / "field-0")
    assert path.suffix == ".npz"


def test_field_cache_dir_requires_explicit_path(tmp_path):
    config = {"root_data": str(tmp_path), "io": {}}

    with pytest.raises(ValueError, match="explicit `field_cache_dir`"):
        _field_cache_dir_from_config(config)


@pytest.mark.parametrize("project", [None, "", ".", "..", "bad/name"])
def test_field_cache_project_is_required_and_valid(project):
    config = {"io": {}}
    if project is not None:
        config["io"]["field_cache_project"] = project

    with pytest.raises(ValueError, match="field_cache_project|component"):
        _field_cache_project_from_config(config)


@pytest.mark.parametrize("component", [".", "..", "bad/name"])
def test_field_cache_subpath_components_are_rejected(tmp_path, component):
    with pytest.raises(ValueError, match="component"):
        _field_cache_product_path(
            tmp_path, "TRGBH0", component, "los", "field-0", "cache.hdf5")


def test_volume_products_use_project_product_and_field_scope(tmp_path):
    base = {
        "kind": "volume_field_data",
        "project": "TRGBH0",
        "loader_name": "ManticoreLocalCOLA",
        "field_indices": [66],
        "subcube_radius": 50.0,
        "downsample": 1,
    }
    payloads = {
        "selection_volume": {
            **base, "product": "h0_volume", "geometry": "sphere",
            "load_velocity": False,
        },
        "pv_volume_density": {
            **base, "product": "pv_volume_density",
            "pad_subcube_boundary": True,
        },
        "density_cube": {
            **base, "product": "pv_density_cube", "nsim": 66,
            "pad_subcube_boundary": True,
        },
    }

    for product, payload in payloads.items():
        path = Path(_field_cache_path(
            tmp_path, _VOLUME_FIELD_CACHE_PREFIX, payload))
        assert path.parents[1] == (
            tmp_path / "TRGBH0" / "ManticoreLocalCOLA" / product)
        assert path.parent.name == "field-66"


def test_maser_vlos_cache_path_compresses_multiple_fields(tmp_path):
    config = {"io": {"field_cache_dir": str(tmp_path)}}
    payload = {
        "reconstruction": "ManticoreLocalCOLA",
        "field_indices": list(range(80)),
    }

    path = Path(_toy_vlos_cache_path(payload, config))

    assert path.parent == (
        tmp_path / "MMH0" / "ManticoreLocalCOLA" / "maser_los"
        / "fields-0-79")


def test_trgbh0_and_mmh0_selection_volumes_have_separate_roots(tmp_path):
    payload = {
        "kind": "volume_field_data",
        "product": "h0_volume",
        "loader_name": "Carrick2015",
        "field_indices": [0],
        "subcube_radius": 50.0,
        "downsample": 1,
        "geometry": "sphere",
        "load_velocity": False,
    }

    trgb = Path(_field_cache_path(
        tmp_path, _VOLUME_FIELD_CACHE_PREFIX,
        {**payload, "project": "TRGBH0"}))
    maser = Path(_field_cache_path(
        tmp_path, _VOLUME_FIELD_CACHE_PREFIX,
        {**payload, "project": "MMH0"}))

    assert trgb.relative_to(tmp_path).parts[0] == "TRGBH0"
    assert maser.relative_to(tmp_path).parts[0] == "MMH0"
    assert trgb.relative_to(tmp_path / "TRGBH0") == maser.relative_to(
        tmp_path / "MMH0")


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


def test_maser_vlos_cache_path_ignores_machine_paths(tmp_path):
    config = {"io": {"field_cache_dir": str(tmp_path)}}
    payload = {
        "reconstruction": "ManticoreLocalCOLA",
        "field_kwargs": {
            "fpath_root": "/machine-a/fields",
            "which_MAS": "CIC",
        },
    }
    other_machine = {
        **payload,
        "field_kwargs": {
            **payload["field_kwargs"],
            "fpath_root": "/machine-b/fields",
        },
    }
    other_mas = {
        **payload,
        "field_kwargs": {**payload["field_kwargs"], "which_MAS": "PCS"},
    }

    assert (_toy_vlos_cache_path(payload, config)
            == _toy_vlos_cache_path(other_machine, config))
    assert (_toy_vlos_cache_path(payload, config)
            != _toy_vlos_cache_path(other_mas, config))


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
