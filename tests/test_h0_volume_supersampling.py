from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.special import ndtr

import candel.field.volume_density as volume_density_mod
from candel.cosmo.cosmography import Distance2Distmod
from candel.field.field_cache import (_VOLUME_FIELD_CACHE_PREFIX,
                                      _field_cache_path)
from candel.field.volume_density import (
    _expected_h0_volume_grid_from_loader, _h0_volume_apply_quadrature,
    _h0_volume_quadrature_geometry, _h0_volume_supersampling_cache_arrays,
    _load_volume_data_for_H0,
    _supersample_offsets_3d, _volume_density_geometry)


def _toy_spherical_volume(radius, dx):
    extract_radius = radius + 0.5 * np.sqrt(3.0) * dx
    n = int(np.ceil(2.0 * extract_radius / dx))
    shape = (n, n, n)
    observer = np.array([0.5 * n * dx] * 3, dtype=np.float32)
    log_r, log_dv = _volume_density_geometry(shape, observer, dx)

    axes = [
        (np.arange(shape[i], dtype=np.float32) + 0.5) * dx
        for i in range(3)
    ]
    disp = [
        axes[0][:, None, None] - observer[0],
        axes[1][None, :, None] - observer[1],
        axes[2][None, None, :] - observer[2],
    ]
    r_grid = np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2)
    r_grid = np.maximum(r_grid, 0.25 * dx)
    return log_r, log_dv, disp, r_grid


def _radial_mag_selection_integral(radius, mag_lim):
    distance2distmod = Distance2Distmod(Om0=0.3)
    r = np.linspace(0.01, radius, 20_000)
    mu = np.asarray(distance2distmod(r, h=1.0))
    sigma = np.sqrt(0.05**2 + 0.15**2)
    p_sel = ndtr((mag_lim - (mu - 4.05)) / sigma)
    return simpson(4.0 * np.pi * r**2 * p_sel, x=r)


def _volume_mag_selection_integral(
        radius, dx, mag_lim, supersample_factor, supersample_radius):
    distance2distmod = Distance2Distmod(Om0=0.3)
    log_r, log_dv, disp, r_grid = _toy_spherical_volume(radius, dx)
    quad = _h0_volume_quadrature_geometry(
        log_r, disp, r_grid, dx, "sphere", radius,
        supersample_factor=supersample_factor,
        supersample_radius=supersample_radius)

    r = np.exp(np.asarray(quad["log_r_3d"]))
    mu = np.asarray(distance2distmod(r, h=1.0))
    sigma = np.sqrt(0.05**2 + 0.15**2)
    p_sel = ndtr((mag_lim - (mu - 4.05)) / sigma)
    weights = np.exp(float(log_dv) + quad["log_volume_weight_3d"])
    return np.sum(weights * p_sel)


def test_h0_volume_supersampling_trilinearly_interpolates_subcells():
    dx = 1.0
    shape = (9, 9, 9)
    observer = np.array([4.5, 4.5, 4.5], dtype=np.float32)
    log_r, _ = _volume_density_geometry(shape, observer, dx)
    axes = [
        (np.arange(shape[i], dtype=np.float32) + 0.5) * dx
        for i in range(3)
    ]
    disp = [
        axes[0][:, None, None] - observer[0],
        axes[1][None, :, None] - observer[1],
        axes[2][None, None, :] - observer[2],
    ]
    r_grid = np.maximum(
        np.sqrt(disp[0]**2 + disp[1]**2 + disp[2]**2), 0.25 * dx)
    i, j, k = np.indices(shape, dtype=np.float32)
    field = 2.0 * i - 3.0 * j + 0.5 * k + 7.0

    factor = 4
    quad = _h0_volume_quadrature_geometry(
        log_r, disp, r_grid, dx, "cube", None,
        supersample_factor=factor, supersample_radius=1.5)
    got = _h0_volume_apply_quadrature(field, quad)

    expected = [field.reshape(-1)[quad["unsup_flat"]]]
    parents = np.array(np.unravel_index(quad["sup_flat"], shape)).T
    offsets = _supersample_offsets_3d(factor, dx) / dx
    xyz = parents[:, None, :] + offsets[None, :, :]
    expected.append((
        2.0 * xyz[..., 0] - 3.0 * xyz[..., 1]
        + 0.5 * xyz[..., 2] + 7.0).reshape(-1))
    expected = np.concatenate(expected).astype(np.float32)

    assert np.max(np.abs(got - expected)) < 2e-6
    assert not np.allclose(
        got[-len(expected) + len(quad["unsup_flat"]):],
        np.repeat(
            field.reshape(-1)[quad["sup_flat"]], quad["n_subcells"]))


def test_h0_volume_supersampling_matches_homogeneous_radial_integral():
    radius = 50.0
    dx = 5.0
    mag_lim = 24.0

    radial = _radial_mag_selection_integral(radius, mag_lim)
    coarse = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=1, supersample_radius=0.0)
    supersampled = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=8, supersample_radius=15.0)

    coarse_relerr = abs(coarse - radial) / radial
    supersampled_relerr = abs(supersampled - radial) / radial

    assert supersampled_relerr < 5e-4
    assert supersampled_relerr < coarse_relerr / 100.0


def test_h0_volume_supersampling_handles_bright_homogeneous_limit():
    radius = 50.0
    dx = 681.1 / 256.0
    mag_lim = 22.0

    radial = _radial_mag_selection_integral(radius, mag_lim)
    coarse = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=1, supersample_radius=0.0)
    supersampled = _volume_mag_selection_integral(
        radius, dx, mag_lim,
        supersample_factor=8, supersample_radius=15.0)

    coarse_relerr = abs(coarse - radial) / radial
    supersampled_relerr = abs(supersampled - radial) / radial

    assert coarse_relerr > 0.99
    assert supersampled_relerr < 2e-3


def test_h0_volume_target_dx_warmed_superset_matches_resolved_factor(
        tmp_path, monkeypatch):
    class FakeLoader:
        def __init__(self, nsim, **kwargs):
            self.nsim = nsim
            self.boxsize = 16.0
            self.ngrid = 4
            self.coordinate_frame = "icrs"
            self.observer_pos = np.array([8.0, 8.0, 8.0],
                                         dtype=np.float32)

        def load_density(self):
            raise AssertionError("raw field should not be loaded")

    payload = {
        "kind": "volume_field_data",
        "project": "TRGBH0",
        "product": "h0_volume",
        "loader_name": "fake_manticore",
        "loader_kwargs": {"Om0": 0.3},
        "field_indices": [1],
        "subcube_radius": 50.0,
        "geometry": "sphere",
        "sources": [],
        "downsample": 1,
        "load_velocity": False,
    }
    r_3d, _ = _expected_h0_volume_grid_from_loader(
        FakeLoader(1), "sphere", 50.0, 4, 15.0)
    for factor, value in ((4, 40.0), (8, 80.0)):
        factor_r_3d = (
            r_3d if factor == 4 else
            _expected_h0_volume_grid_from_loader(
                FakeLoader(1), "sphere", 50.0, factor, 15.0)[0])
        cache_payload = {
            **payload,
            "max_radius": float(np.max(factor_r_3d)),
            "supersample": {
                "factor": factor,
                "radius": 15.0,
                "method": "linear",
            },
        }
        cache_path = Path(_field_cache_path(
            tmp_path, _VOLUME_FIELD_CACHE_PREFIX, cache_payload))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            rho_3d_fields=np.full(
                (1, len(factor_r_3d)), value, dtype=np.float32),
            r_3d=factor_r_3d,
            log_dV_3d=np.asarray(0.0, dtype=np.float32),
            log_volume_weight_3d=np.zeros(
                len(factor_r_3d), dtype=np.float32),
            **_h0_volume_supersampling_cache_arrays(factor, 15.0))

    monkeypatch.setattr(
        volume_density_mod, "name2field_loader", lambda name: FakeLoader)
    loaded = _load_volume_data_for_H0(
        "fake_manticore", {"Om0": 0.3}, [1], "linear", 0.3,
        subcube_radius=50.0, geometry="sphere", cache_dir=tmp_path,
        cache_project="TRGBH0",
        cache_enabled=True, supersample_radius=15.0,
        supersample_target_dx=1.0)

    np.testing.assert_allclose(
        np.asarray(loaded["density_3d_fields"]),
        np.full((1, len(r_3d)), 39.0, dtype=np.float32))
