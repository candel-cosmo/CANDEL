from pathlib import Path
import tomllib

import numpy as np
import pytest

import candel
from scripts.megamaser.joint_H0_helpers import (
    _toy_vlos_cache_path, _toy_vlos_cache_payload)
from scripts.preprocess.migrate_field_cache import (
    _apply, _entry, _legacy_path, _maser_entries, _preflight)


def test_legacy_path_mapping_is_product_specific(tmp_path):
    root = tmp_path / "field_cache"
    project = "TRGBH0"

    assert _legacy_path(
        root / project / "Recon" / "los" / "field-2" / "los.hdf5",
        root, project) == root / "Recon" / "los" / "los.hdf5"
    assert _legacy_path(
        root / project / "Recon" / "selection_volume" / "field-2"
        / "volume.npz", root, project) == root / "Recon" / "volume.npz"
    assert _legacy_path(
        root / project / "Recon" / "selection_volume" / "field-2"
        / "cache__loader-0123456789ab__density.npz",
        root, project) == root / "Recon" / "cache__density.npz"
    assert _legacy_path(
        root / "MMH0" / "Recon" / "maser_los" / "fields-0-79"
        / "maser.npz", root, "MMH0") == (
            root / "toy_maser_vlos" / "Recon" / "maser.npz")


def test_legacy_path_accepts_explicit_separate_source_root(tmp_path):
    root = tmp_path / "data" / "field_cache"
    legacy_root = tmp_path / "field_cache"
    destination = (
        root / "TRGBH0" / "Recon" / "selection_volume" / "field-0"
        / "volume.npz")

    assert _legacy_path(
        destination, root, "TRGBH0", legacy_root=legacy_root) == (
            legacy_root / "Recon" / "volume.npz")


def test_migration_dry_run_then_link_never_removes_source(tmp_path):
    root = tmp_path / "field_cache"
    source = root / "Recon" / "volume.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cache")
    destination = (
        root / "TRGBH0" / "Recon" / "selection_volume" / "field-0"
        / "volume.npz")
    entries = [_entry(destination, root, "TRGBH0")]

    manifest = _preflight(entries, root, "dry-run")
    _apply(entries, manifest, "dry-run")
    assert manifest[0]["status"] == "ready"
    assert not destination.exists()

    manifest = _preflight(entries, root, "link")
    _apply(entries, manifest, "link")
    assert manifest[0]["status"] == "linked"
    assert source.exists()
    assert destination.samefile(source)


def test_migration_refuses_conflicting_destination(tmp_path):
    root = tmp_path / "field_cache"
    source = root / "Recon" / "volume.npz"
    destination = (
        root / "TRGBH0" / "Recon" / "selection_volume" / "field-0"
        / "volume.npz")
    source.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    destination.write_bytes(b"different")

    manifest = _preflight(
        [_entry(destination, root, "TRGBH0")], root, "copy")

    assert manifest[0]["status"] == "conflicting_destination"
    with pytest.raises(RuntimeError, match="not ready"):
        _apply([_entry(destination, root, "TRGBH0")], manifest, "copy")
    assert destination.read_bytes() == b"different"


def test_migration_copy_is_explicit_and_preserves_source(tmp_path):
    root = tmp_path / "field_cache"
    source = root / "Recon" / "volume.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cache")
    destination = (
        root / "TRGBH0" / "Recon" / "selection_volume" / "field-0"
        / "volume.npz")
    entries = [_entry(destination, root, "TRGBH0")]
    manifest = _preflight(entries, root, "copy")

    _apply(entries, manifest, "copy")

    assert manifest[0]["status"] == "copied"
    assert source.read_bytes() == destination.read_bytes() == b"cache"
    assert not source.samefile(destination)


def test_migration_copy_race_preserves_foreign_destination(tmp_path):
    root = tmp_path / "field_cache"
    source = root / "Recon" / "volume.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    destination = (
        root / "TRGBH0" / "Recon" / "selection_volume" / "field-0"
        / "volume.npz")
    entries = [_entry(destination, root, "TRGBH0")]
    manifest = _preflight(entries, root, "copy")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"concurrent")

    with pytest.raises(FileExistsError):
        _apply(entries, manifest, "copy")

    assert destination.read_bytes() == b"concurrent"
    assert manifest[0]["status"] == "failed"


def test_link_preflight_allows_missing_destination_root(tmp_path):
    root = tmp_path / "new" / "field_cache"
    legacy_root = tmp_path / "old" / "field_cache"
    source = legacy_root / "Recon" / "volume.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cache")
    destination = (
        root / "TRGBH0" / "Recon" / "selection_volume" / "field-0"
        / "volume.npz")
    entries = [_entry(
        destination, root, "TRGBH0", legacy_root=legacy_root)]

    manifest = _preflight(entries, root, "link")
    _apply(entries, manifest, "link")

    assert manifest[0]["status"] == "linked"
    assert destination.samefile(source)


def test_maser_migration_rekeys_explicit_legacy_source(tmp_path):
    root = tmp_path / "field_cache"
    field_config_path = tmp_path / "field.toml"
    field_config_path.write_text(
        "[io]\n"
        f'field_cache_dir = "{root}"\n'
        "[io.reconstruction_main.Recon]\n"
        f'fpath_root = "{tmp_path / "raw"}"\n'
        'which_MAS = "PCS"\n',
        encoding="utf-8")
    field_config = candel.load_config(field_config_path)

    with open("scripts/megamaser/config_maser.toml", "rb") as f:
        master = tomllib.load(f)
    galaxy = "NGC6264"
    galaxy_config = master["model"]["galaxies"][galaxy]
    galaxy_data = [{
        "name": galaxy,
        "RA": float(galaxy_config["ra"]),
        "dec": float(galaxy_config["dec"]),
    }]
    r = np.asarray([0.1, 0.6], dtype=np.float32)
    field_indices = [2]
    payload = _toy_vlos_cache_payload(
        "Recon", field_config["io"]["reconstruction_main"]["Recon"],
        field_indices, r, galaxy_data)
    source = (
        tmp_path / "legacy" / "toy_maser_vlos" / "Recon"
        / "legacy.npz")
    source.parent.mkdir(parents=True)
    np.savez(
        source, r=r, los_velocity=np.zeros((1, 1, len(r))),
        rhat=np.zeros((1, 3)), field_indices=np.asarray(field_indices),
        galaxy_names=np.asarray([galaxy]),
        coordinate_frame=np.asarray("icrs"))

    cache_root, entries = _maser_entries(
        field_config_path, [source], "MMH0", 0.0)

    assert cache_root == root
    assert entries == [{
        "source": source,
        "destination": Path(_toy_vlos_cache_path(payload, field_config)),
    }]
