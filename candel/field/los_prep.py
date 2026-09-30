# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Helpers for preparing reconstruction LOS density and velocity products."""
from os import getpid, makedirs, remove, replace
from os.path import dirname, exists, join

import numpy as np
from h5py import File

import candel
from ..probe import probes
from ..util import fprint
from .field_products import (field_smoothed_los_path, los_radial_grid_payload,
                             reconstruction_los_label,
                             validate_field_smoothing_scale)


def pv_main_los_config(config, catalogue):
    """Return the LOS template and raw catalogue-loader kwargs."""
    d = config["io"]["PV_main"][catalogue].copy()
    los_file = d.pop("los_file")
    d.pop("reconstruction", None)
    d.pop("return_all", None)
    return los_file, d


def load_los(catalogue, config):
    """Return RA, dec (deg) and the LOS-file template of `catalogue`."""
    for probe in probes().values():
        out = probe.sky_positions(catalogue, config)
        if out is not None:
            return out
    raise ValueError(
        f"No installed CANDEL package provides catalogue `{catalogue}`.")


def reconstruction_field_indices(config, reconstruction):
    """Return the reconstruction realisation indices to interpolate."""
    nreal_map = {
        "Carrick2015": 1,
        "Lilow2024": 1,
        "CLONES": 1,
        "CB1": 100,
        "CB2": 20,
        "CF4": 100,
        "HAMLET_V0": 20,
        "HAMLET_V1": 20,
        }

    if reconstruction in nreal_map:
        return list(range(nreal_map[reconstruction]))
    if str(reconstruction).lower().startswith("manticorelocal"):
        candel.field.name2field_loader(reconstruction)
        field_kwargs = config["io"]["reconstruction_main"][reconstruction]
        fpath_root = field_kwargs["fpath_root"]
        if reconstruction == "ManticoreLocalCOLA":
            fpath_root = join(
                fpath_root,
                candel.field.field_mas_directory(
                    field_kwargs.get("which_MAS", "CIC")))
        return candel.field.available_mcmc_field_indices(
            fpath_root)
    raise ValueError(f"Reconstruction `{reconstruction}` not supported.")


def radial_los_grid(config, catalogue, reconstruction, verbose=True):
    """Return the radial grid for a catalogue/reconstruction LOS product."""
    radial_grid = los_radial_grid_payload(config, catalogue, reconstruction)
    if radial_grid is None:
        raise ValueError(
            f"No configured LOS radial grid for `{catalogue}`/"
            f"`{reconstruction}`.")
    rmin = radial_grid["rmin"]
    rmax = radial_grid["rmax"]
    num_steps = radial_grid["num_steps"]
    fprint(f"setting the radial grid from {rmin} to {rmax} "
           f"with {num_steps} steps.", verbose=verbose)
    return np.linspace(rmin, rmax, num_steps)


def resolve_los_output_path(los_file, reconstruction,
                            field_smoothing_scale=None, config=None,
                            velocity_field_smoothing_scale=None):
    """Resolve a LOS template to the output path for this smoothing variant."""
    los_file = los_file.replace(
        "<X>", reconstruction_los_label(config, reconstruction))
    return field_smoothed_los_path(
        los_file, field_smoothing_scale,
        velocity_field_smoothing_scale=velocity_field_smoothing_scale)


def los_file_matches_grid(los_file, r, verbose=True):
    """Return whether an existing LOS file stores the requested radial grid."""
    if not exists(los_file):
        return False
    try:
        with File(los_file, "r") as f:
            if "r" not in f:
                fprint(f"LOS cache `{los_file}` has no `r`; rebuilding.",
                       verbose=verbose)
                return False
            stored = f["r"][...]
    except Exception as exc:
        fprint(f"LOS cache `{los_file}` could not be read ({exc}); "
               "rebuilding.", verbose=verbose)
        return False
    requested = np.asarray(r)
    if stored.shape != requested.shape or not np.allclose(stored, requested):
        fprint(f"LOS cache `{los_file}` has a stale radial grid; "
               "rebuilding.", verbose=verbose)
        return False
    return True


def _selected_field_indices(config, reconstruction, field_indices=None):
    """Return requested reconstruction field indices for one LOS product."""
    available = np.asarray(
        reconstruction_field_indices(config, reconstruction), dtype=np.int32)
    if field_indices is None:
        field_indices = config.get("io", {}).get("field_indices", None)
    if field_indices is None:
        return available.tolist()

    requested = np.asarray(field_indices, dtype=np.int32)
    if requested.ndim == 0:
        requested = requested[None]
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError(
            "`io.field_indices` must be an int or non-empty 1D list.")
    missing = [int(nsim) for nsim in requested if nsim not in available]
    if missing:
        raise ValueError(
            f"Requested field indices {missing} are not available for "
            f"`{reconstruction}`. Available: {available.tolist()}.")
    return requested.tolist()


def compute_los_file_from_coordinates(
        catalogue, reconstruction, config, RA, dec, los_template=None,
        filepath=None, field_smoothing_scale=None, overwrite=False,
        output_path=None, field_indices=None, r=None, verbose=True,
        metadata=None, velocity_field_smoothing_scale=None):
    """Compute one LOS product from already-loaded sky coordinates."""
    field_smoothing_scale = validate_field_smoothing_scale(
        field_smoothing_scale)
    velocity_field_smoothing_scale = validate_field_smoothing_scale(
        velocity_field_smoothing_scale,
        label="model.velocity_3d_smoothing_scale")

    nsims = _selected_field_indices(config, reconstruction, field_indices)
    if r is None:
        r = radial_los_grid(
            config, catalogue, reconstruction, verbose=verbose)
    RA = np.asarray(RA)
    dec = np.asarray(dec)
    if len(RA) != len(dec):
        raise ValueError(
            f"`RA` and `dec` must have the same length, got "
            f"{len(RA)} and {len(dec)}.")

    fprint("LOS build:", verbose=verbose)
    fprint(f"  catalogue: `{catalogue}`", verbose=verbose)
    fprint(f"  reconstruction: `{reconstruction}`", verbose=verbose)
    fprint(f"  catalogue objects: {len(RA)}", verbose=verbose)

    los_file = (
        output_path if output_path is not None
        else resolve_los_output_path(
            los_template, reconstruction, field_smoothing_scale,
            config=config,
            velocity_field_smoothing_scale=velocity_field_smoothing_scale))
    if los_file is None:
        raise ValueError(
            "A LOS output path is required. Provide either `output_path` "
            "or `los_template`.")
    if exists(los_file) and not overwrite:
        if los_file_matches_grid(los_file, r, verbose=verbose):
            fprint(f"  status: cached at `{los_file}`; skipping.",
                   verbose=verbose)
            return {"path": los_file, "status": "exists"}
        fprint(f"  status: stale file at `{los_file}`; overwriting.",
               verbose=verbose)

    out_dir = dirname(los_file)
    if out_dir:
        makedirs(out_dir, exist_ok=True)

    n_sims = len(nsims)
    n_gal = len(RA)
    n_r = len(r)

    loader_cls = candel.field.name2field_loader(reconstruction)
    loader_kwargs = config["io"]["reconstruction_main"][reconstruction]
    geometry_loader = loader_cls(nsim=nsims[0], **loader_kwargs)
    los_geometry = candel.field.prepare_los_geometry(
        geometry_loader, r, RA, dec)

    fprint(f"  output: `{los_file}`", verbose=verbose)
    los_tmp_file = f"{los_file}.tmp.{getpid()}"
    dt = np.dtype(np.float32)
    dt16 = np.dtype(np.float16)
    try:
        with File(los_tmp_file, "w") as fout:
            fout.attrs["reconstruction"] = reconstruction
            if metadata is not None:
                for key, value in metadata.items():
                    fout.attrs[key] = value
            fout.create_dataset("RA", data=RA, dtype=dt)
            fout.create_dataset("dec", data=dec, dtype=dt)
            fout.create_dataset("r", data=r, dtype=dt)
            density_dataset = fout.create_dataset(
                "los_density", shape=(n_sims, n_gal, n_r), dtype=dt,
                fillvalue=np.nan)
            velocity_dataset = fout.create_dataset(
                "los_velocity", shape=(n_sims, n_gal, n_r), dtype=dt16,
                fillvalue=np.nan)
            fout.create_dataset(
                "field_indices", data=np.asarray(nsims, dtype=np.int32))

            for i, nsim in enumerate(nsims):
                fprint(f"  field {i + 1}/{n_sims}: nsim={int(nsim)}",
                       verbose=verbose)
                loader = loader_cls(nsim=nsim, **loader_kwargs)
                dens_i, vel_i = candel.field.interpolate_los_density_velocity(
                    loader, r, RA, dec,
                    field_smoothing_scale=field_smoothing_scale,
                    velocity_field_smoothing_scale=(
                        velocity_field_smoothing_scale),
                    verbose=verbose, los_geometry=los_geometry)
                density_dataset[i] = dens_i.astype(np.float32)
                velocity_dataset[i] = vel_i.astype(np.float16)
        replace(los_tmp_file, los_file)
    finally:
        if exists(los_tmp_file):
            remove(los_tmp_file)
    return {"path": los_file, "status": "computed"}
