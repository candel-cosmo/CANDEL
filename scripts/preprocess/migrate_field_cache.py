#!/usr/bin/env python
"""Plan or create project-first links/copies for expected field caches."""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_USER = os.environ.get("USER") or "user"
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), f"candel_mpl_{_USER}"))
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), f"candel_numba_{_USER}"))

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import candel  # noqa: E402
import numpy as np  # noqa: E402
from candel import get_nested  # noqa: E402
from candel.pvdata.field_cache import (  # noqa: E402
    _VOLUME_FIELD_CACHE_PREFIX, _field_cache_dir_from_config,
    _field_cache_path, _field_cache_project_from_config,
    _field_source_metadata, _jsonable)
from candel.pvdata.field_products import (  # noqa: E402
    field_smoothing_cache_payload, field_smoothing_scale_from_config,
    los_field_cache_paths, los_radial_grid_payload_from_array,
    velocity_field_smoothing_cache_payload,
    velocity_field_smoothing_scale_from_config)
from candel.pvdata.volume_density import (  # noqa: E402
    _expected_h0_volume_max_radius_from_loader,
    _expected_pv_volume_max_radius_from_loader,
    _h0_volume_cache_sampling_payload,
    _h0_volume_cache_supersampling_payload,
    _h0_volume_resolved_supersample_factor,
    _h0_volume_supersampling_from_config, _field_loader_native_dx)
from scripts.preprocess import field_input_cache as cache_mod  # noqa: E402
from scripts.preprocess import field_input_los as los_mod  # noqa: E402
from scripts.preprocess.prepare_field_inputs import (  # noqa: E402
    _infer_los_jobs, _plan_los_metadata)
from scripts.megamaser.joint_H0_helpers import (  # noqa: E402
    FIELD_CACHE_PROJECT, _toy_vlos_cache_path, _toy_vlos_cache_payload)


def _config_paths(inputs, task_file=None, tasks=None):
    paths = [cache_mod._resolve_cli_path(path) for path in inputs]
    if task_file is None and paths and cache_mod._looks_like_task_file(paths[0]):
        task_file = paths.pop(0)
    if task_file is not None:
        paths = cache_mod._read_task_file(
            cache_mod._resolve_cli_path(task_file), tasks) + paths
    if not paths:
        raise ValueError("provide a task file or at least one config")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def _legacy_path(destination, cache_root, project, legacy_root=None):
    """Map one canonical destination to its exact former runtime path."""
    legacy_root = cache_root if legacy_root is None else Path(legacy_root)
    relative = destination.relative_to(cache_root / project)
    if len(relative.parts) != 4:
        raise ValueError(f"Unexpected canonical cache path `{destination}`.")
    reconstruction, product, _scope, filename = relative.parts
    if product == "los":
        return legacy_root / reconstruction / "los" / filename
    if product == "maser_los":
        return legacy_root / "toy_maser_vlos" / reconstruction / filename
    filename = re.sub(
        r"__loader-[0-9a-f]{12}(?=__(?:density|vel)\.npz$)", "", filename)
    return legacy_root / reconstruction / filename


def _entry(destination, cache_root, project, legacy_root=None):
    destination = Path(destination)
    return {
        "source": _legacy_path(
            destination, cache_root, project, legacy_root=legacy_root),
        "destination": destination,
    }


def _los_entries(config, cache_root, project, legacy_root=None):
    entries = []
    for job in _infer_los_jobs(config):
        los_template, _ = _plan_los_metadata(config, job["catalogue"])
        indices = los_mod._selected_field_indices(
            config, job["reconstruction"],
            get_nested(config, "io/field_indices", None))
        radial_grid = los_radial_grid_payload_from_array(
            los_mod.radial_los_grid(
                config, job["catalogue"], job["reconstruction"],
                verbose=False))
        paths = los_field_cache_paths(
            config, job["catalogue"], job["reconstruction"], los_template,
            field_smoothing_scale=field_smoothing_scale_from_config(config),
            velocity_field_smoothing_scale=(
                velocity_field_smoothing_scale_from_config(config)),
            field_indices=indices, radial_grid=radial_grid)
        entries.extend(
            _entry(path, cache_root, project, legacy_root)
            for path in paths or [])
    return entries


def _h0_volume_entries(config, cache_root, project, legacy_root=None):
    if cache_mod._variant_action(config) != "check/cache":
        return []
    reconstruction, _ = cache_mod._h0_los_config(config)
    field_kwargs = get_nested(
        config, f"io/reconstruction_main/{reconstruction}", None)
    indices = cache_mod._configured_or_available_field_indices(
        config, reconstruction)
    if reconstruction is None or field_kwargs is None or indices is None:
        return []
    indices = [int(index) for index in indices]

    loaders = []
    sources = []
    for index in indices:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = index
        loader = cache_mod.pvdata_mod.name2field_loader(
            reconstruction)(**kwargs)
        loaders.append(loader)
        sources.append(_field_source_metadata(loader))

    geometry = get_nested(
        config, "model/selection_integral_geometry", "sphere")
    radius = get_nested(config, "model/selection_integral_grid_radius", None)
    factor, supersample_radius, target_dx = (
        _h0_volume_supersampling_from_config(config))
    if target_dx is not None and supersample_radius > 0.0:
        factor = _h0_volume_resolved_supersample_factor(
            _field_loader_native_dx(loaders[0]), factor, target_dx)
    max_radius = _expected_h0_volume_max_radius_from_loader(
        loaders[0], geometry, radius, factor, supersample_radius)
    base = {
        "kind": "volume_field_data",
        "project": project,
        "product": "h0_volume",
        "loader_name": reconstruction,
        "loader_kwargs": _jsonable(field_kwargs),
        "field_indices": indices,
        "subcube_radius": radius,
        "max_radius": max_radius,
        "geometry": geometry,
        "sources": sources,
        **_h0_volume_cache_sampling_payload(),
        **_h0_volume_cache_supersampling_payload(
            factor, supersample_radius),
    }
    payloads = [{
        **base,
        **field_smoothing_cache_payload(
            field_smoothing_scale_from_config(config)),
        "load_velocity": False,
    }]
    if cache_mod._h0_velocity_key(config) == "velocity":
        payloads.append({
            **base,
            **velocity_field_smoothing_cache_payload(
                velocity_field_smoothing_scale_from_config(config)),
            "load_velocity": True,
        })

    entries = []
    for payload in payloads:
        for i, index in enumerate(indices):
            field_payload = {
                **payload, "field_indices": [index],
                "sources": [sources[i]],
            }
            path = _field_cache_path(
                cache_root, _VOLUME_FIELD_CACHE_PREFIX, field_payload)
            entries.append(_entry(path, cache_root, project, legacy_root))
    return entries


def _maser_entries(field_config_path, sources, project,
                   velocity_smoothing_scale):
    """Map explicitly named legacy MMH0 velocity-LOS caches."""
    if project != FIELD_CACHE_PROJECT:
        raise ValueError(
            f"Maser caches belong to project {FIELD_CACHE_PROJECT!r}.")
    config = candel.load_config(field_config_path, replace_los_prior=False)
    cache_root = Path(_field_cache_dir_from_config(config))
    with open(ROOT / "scripts/megamaser/config_maser.toml", "rb") as f:
        galaxy_configs = tomllib.load(f)["model"]["galaxies"]

    entries = []
    for source in sources:
        source = Path(source).resolve()
        if (source.suffix != ".npz"
                or source.parent.parent.name != "toy_maser_vlos"):
            raise ValueError(
                f"Unexpected legacy maser cache path `{source}`.")
        reconstruction = source.parent.name
        field_kwargs = get_nested(
            config, f"io/reconstruction_main/{reconstruction}", None)
        if field_kwargs is None:
            raise ValueError(
                f"Missing reconstruction configuration `{reconstruction}`.")
        with np.load(source, allow_pickle=False) as cached:
            required = (
                "r", "los_velocity", "rhat", "field_indices",
                "galaxy_names", "coordinate_frame")
            missing = [key for key in required if key not in cached.files]
            if missing:
                raise ValueError(
                    f"Legacy maser cache `{source}` is missing {missing}.")
            r = cached["r"]
            field_indices = [int(index) for index in cached["field_indices"]]
            names = [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in cached["galaxy_names"]
            ]
        unknown = [name for name in names if name not in galaxy_configs]
        if unknown:
            raise ValueError(
                f"Unknown maser galaxies in `{source}`: {unknown}.")
        galaxy_data = [{
            "name": name,
            "RA": float(galaxy_configs[name]["ra"]),
            "dec": float(galaxy_configs[name]["dec"]),
        } for name in names]
        payload = _toy_vlos_cache_payload(
            reconstruction, field_kwargs, field_indices, r, galaxy_data,
            velocity_smoothing_scale)
        entries.append({
            "source": source,
            "destination": Path(_toy_vlos_cache_path(payload, config)),
        })
    return cache_root, entries


def _pv_volume_entries(config, cache_root, project, legacy_root=None):
    kind = get_nested(config, "pv_model/kind", "")
    if (not isinstance(kind, str)
            or not kind.startswith("precomputed_los_")
            or get_nested(config, "pv_model/which_distance_prior",
                          "empirical") != "empirical"):
        return []
    reconstruction = kind.replace("precomputed_los_", "")
    field_kwargs = get_nested(
        config, f"io/reconstruction_main/{reconstruction}", None)
    indices = cache_mod._configured_or_available_field_indices(
        config, reconstruction)
    if field_kwargs is None or indices is None:
        return []
    indices = [int(index) for index in indices]
    downsample = int(get_nested(config, "pv_model/density_3d_downsample", 1))
    geometry = get_nested(config, "pv_model/density_3d_geometry", "cube")
    radius = get_nested(config, "pv_model/density_3d_radius", None)
    pad_boundary = geometry == "sphere"
    fraction = get_nested(
        config, "pv_model/density_3d_subsample_fraction", 1.0)
    seed = get_nested(config, "pv_model/density_3d_subsample_seed", 42)
    store_rhat = bool(get_nested(config, "pv_model/use_Mmiss", False))
    smoothing = field_smoothing_cache_payload(
        field_smoothing_scale_from_config(config))

    entries = []
    for index in indices:
        kwargs = dict(field_kwargs)
        kwargs["nsim"] = index
        loader = cache_mod.pvdata_mod.name2field_loader(
            reconstruction)(**kwargs)
        source = _field_source_metadata(loader)
        max_radius = _expected_pv_volume_max_radius_from_loader(
            loader, downsample, radius, pad_boundary, geometry, radius,
            fraction, seed)
        payload = {
            "kind": "volume_field_data",
            "project": project,
            "product": "pv_volume_density",
            "loader_name": reconstruction,
            "loader_kwargs": _jsonable(field_kwargs),
            "field_indices": [index],
            "downsample": downsample,
            "subcube_radius": radius,
            "max_radius": max_radius,
            "pad_subcube_boundary": pad_boundary,
            "voxel_subsample_fraction": float(fraction),
            "voxel_subsample_seed": int(seed),
            "store_rhat_3d": store_rhat,
            **smoothing,
            "sources": [source],
        }
        path = _field_cache_path(
            cache_root, _VOLUME_FIELD_CACHE_PREFIX, payload)
        entries.append(_entry(path, cache_root, project, legacy_root))
    return entries


def _enumerate(config_paths, project, legacy_root=None):
    entries = {}
    cache_root = None
    for config_path in config_paths:
        config = candel.load_config(config_path, replace_los_prior=False)
        configured_project = _field_cache_project_from_config(config)
        if configured_project != project:
            raise ValueError(
                f"Config `{config_path}` owns project {configured_project!r}, "
                f"not {project!r}.")
        root = Path(_field_cache_dir_from_config(config))
        if cache_root is None:
            cache_root = root
        elif root != cache_root:
            raise ValueError("All migration configs must use one cache root.")
        expected = (
            _los_entries(config, root, project, legacy_root)
            + _h0_volume_entries(config, root, project, legacy_root)
            + _pv_volume_entries(config, root, project, legacy_root))
        for entry in expected:
            entries[str(entry["destination"])] = entry
    if not entries:
        raise ValueError("No expected field-cache products found.")
    return cache_root, list(entries.values())


def _preflight(entries, cache_root, mode):
    manifest = []
    for entry in entries:
        source = entry["source"]
        destination = entry["destination"]
        size = source.stat().st_size if source.is_file() else None
        parent = destination.parent
        while not parent.exists():
            parent = parent.parent
        unsafe_parent = any(
            path.is_symlink()
            for path in (destination.parent, *destination.parents)
            if path == cache_root or cache_root in path.parents)
        if source.is_symlink():
            status = "source_symlink"
        elif not source.is_file():
            status = "missing_source"
        elif unsafe_parent:
            status = "unsafe_destination_parent"
        elif destination.exists() or destination.is_symlink():
            status = (
                "already_linked"
                if not destination.is_symlink()
                and os.path.samefile(source, destination)
                else "conflicting_destination")
        elif mode == "link" and source.stat().st_dev != parent.stat().st_dev:
            status = "different_filesystem"
        else:
            status = "ready"
        manifest.append({
            "source": str(source),
            "destination": str(destination),
            "size": size,
            "status": status,
        })
    return manifest


def _apply(entries, manifest, mode):
    if mode == "dry-run":
        return
    for entry, row in zip(entries, manifest):
        if row["status"] == "already_linked":
            continue
        if row["status"] != "ready":
            raise RuntimeError(
                f"Migration entry is not ready: {row['status']}.")
        destination = entry["destination"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = None
        try:
            if mode == "link":
                os.link(entry["source"], destination)
                row["status"] = "linked"
            else:
                with open(entry["source"], "rb") as source:
                    with open(destination, "xb") as target:
                        stat = os.fstat(target.fileno())
                        created = (stat.st_dev, stat.st_ino)
                        shutil.copyfileobj(source, target)
                current = os.lstat(destination)
                if (current.st_dev, current.st_ino) != created:
                    raise FileExistsError(
                        f"Destination changed while copying: {destination}")
                row["status"] = "copied"
        except Exception:
            row["status"] = "failed"
            if created is not None:
                try:
                    current = os.lstat(destination)
                except FileNotFoundError:
                    current = None
                if (current is not None
                        and (current.st_dev, current.st_ino) == created):
                    destination.unlink()
            raise


def _write_manifest(path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--tasks")
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--legacy-root", type=Path,
        help="Explicit source root when legacy files are outside the cache root.")
    parser.add_argument(
        "--maser-source", action="append", default=[], type=Path,
        help="Explicit legacy toy_maser_vlos NPZ; may be repeated.")
    parser.add_argument(
        "--maser-field-config", type=Path,
        help="Field config used to create explicit --maser-source files.")
    parser.add_argument(
        "--maser-velocity-smoothing-scale", type=float,
        help="Explicit smoothing scale used by --maser-source files.")
    parser.add_argument(
        "--mode", choices=("dry-run", "link", "copy"), default="dry-run")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("field_cache_migration_manifest.json"))
    args = parser.parse_args()

    manifest = None
    error = None
    try:
        paths = (
            _config_paths(args.inputs, args.task_file, args.tasks)
            if args.inputs or args.task_file is not None else [])
        cache_root = None
        entries = []
        if paths:
            cache_root, entries = _enumerate(
                paths, args.project, legacy_root=args.legacy_root)
        if args.maser_source:
            if args.maser_field_config is None:
                raise ValueError(
                    "--maser-source requires --maser-field-config.")
            if args.maser_velocity_smoothing_scale is None:
                raise ValueError(
                    "--maser-source requires an explicit "
                    "--maser-velocity-smoothing-scale.")
            maser_root, maser_entries = _maser_entries(
                cache_mod._resolve_cli_path(args.maser_field_config),
                args.maser_source, args.project,
                args.maser_velocity_smoothing_scale)
            if cache_root is not None and maser_root != cache_root:
                raise ValueError(
                    "All migration inputs must use one cache root.")
            cache_root = maser_root
            by_destination = {
                str(entry["destination"]): entry
                for entry in (*entries, *maser_entries)
            }
            entries = list(by_destination.values())
        if not entries:
            raise ValueError("No expected field-cache products found.")
        manifest = _preflight(entries, cache_root, args.mode)
        blocked = [row for row in manifest if row["status"] not in (
            "ready", "already_linked")]
        if not blocked:
            _apply(entries, manifest, args.mode)
    except Exception as exc:
        error = exc

    if manifest is not None:
        _write_manifest(args.manifest, manifest)
    if error is not None:
        parser.error(str(error))

    counts = {}
    for row in manifest:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"manifest: {args.manifest}")
    print("status: " + ", ".join(
        f"{key}={value}" for key, value in sorted(counts.items())))
    if blocked:
        print("migration refused; resolve manifest blockers first.")
        return 2
    if args.mode == "dry-run":
        print("dry run only; no cache files were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
