# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Mock generator for the CH0 JWST forecast in a single reconstruction.

Each host has a Cepheid distance modulus known up to a common offset, which
the CH0 model absorbs into `M_W`:

    mag_i = mu_true,i + M_W + eps_i,  eps_i ~ N(0, sigma_mu(d_true,i)^2),

with one such "effective Cepheid" row per host and per anchor (NGC 4258 and
the LMC). The anchors also have geometric distances, and `M_W` is calibrated
by the MW zero points, as in the real data. Hosts are drawn from a density-
weighted pool built once from one field realisation, and each sample is
selected on its observed redshift with a known smoothed window.
"""
import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm

from ..cosmo.cosmography import Distance2Distmod, Distance2Redshift
from ..util import SPEED_OF_LIGHT, fprint, galactic_to_radec_cartesian
from ._field_utils import (build_field_pool, build_field_pool_evaluator,
                           galaxy_bias_log_weight)

# sigma_mu(d) = A + B d / Mpc, fitted to the Cepheid-only distance-modulus
# errors of the 35 SH0ES hosts against their Cepheid distances.
SIGMA_MU_A = 0.019
SIGMA_MU_B = 0.00267
# Lower edge standing in for "no lower limit" in a redshift window.
CZ_NO_LOWER_LIMIT = -1e5


def sigma_mu_law(d_Mpc, factor=1.0):
    """Cepheid distance-modulus error (mag) of a host at distance `d_Mpc`."""
    return factor * (SIGMA_MU_A + SIGMA_MU_B * np.asarray(d_Mpc))


def build_CH0_host_pool(field_loader, r_sphere_h, pool_size, bias_params,
                        which_bias, r_grid_h, gen, rmin_h=3.0,
                        density_divisor=None, batch_size=2_000_000,
                        verbose=True):
    """Draw density-weighted host positions with their LOS profiles.

    Positions follow p(x) ∝ b[rho(x)] within `r_sphere_h` (Mpc/h). Returns
    positions, the field radial velocity at each position, and the density
    and velocity along each host's line of sight on `r_grid_h`.
    """
    evaluator = build_field_pool_evaluator(
        field_loader, density_divisor=density_divisor,
        max_radius_h=r_sphere_h, verbose=verbose)
    # As in the TRGB mock, the global maximum bounds the interpolated density
    # near the sphere edge.
    rho_grid = np.linspace(evaluator["eps"],
                           max(1.0 + evaluator["delta_max"], 1.0), 4096)
    log_w_max = float(np.max(galaxy_bias_log_weight(
        rho_grid, bias_params, which_bias)))

    keys = ("r_h", "v_los", "RA", "dec")
    out = {k: [] for k in keys}
    n = 0
    while n < pool_size:
        p = build_field_pool(field_loader, r_sphere_h, batch_size, gen,
                             rmin_h=rmin_h, field_evaluator=evaluator,
                             verbose=False)
        log_w = galaxy_bias_log_weight(p["rho"], bias_params, which_bias)
        keep = gen.random(len(log_w)) < np.exp(np.minimum(log_w - log_w_max,
                                                          0.0))
        for k in keys:
            out[k].append(p[k][keep])
        n += int(keep.sum())
        fprint(f"host pool: {n} / {pool_size} density-accepted.",
               verbose=verbose)
    del evaluator
    pool = {k: np.concatenate(v)[:pool_size] for k, v in out.items()}

    from ..field import interpolate_los_density_velocity
    los_density, los_velocity = interpolate_los_density_velocity(
        field_loader, r_grid_h, pool["RA"], pool["dec"], verbose=verbose)
    if density_divisor is not None:
        los_density = los_density / density_divisor
    pool["los_density"] = los_density.astype(np.float32)
    pool["los_velocity"] = los_velocity.astype(np.float32)
    pool["los_r"] = np.asarray(r_grid_h)
    return pool


def _window_prob(cz, low, high, width):
    return norm.cdf((high - cz) / width) - norm.cdf((low - cz) / width)


def draw_CH0_hosts(pool, samples, truth, cz_width, sigma_v, e_cz, gen,
                   Om=0.3):
    """Draw selected hosts for each sample from a density-weighted pool.

    `samples` is a list of dicts with keys `n`, `cz_low`, `cz_high` and
    `sigma_factor` (1 for HST, 0.5 for JWST). A candidate is kept with
    probability Phi((cz_high - cz)/w) - Phi((cz_low - cz)/w) in its observed
    redshift. The Cepheid error depends on the true distance only.
    """
    h = truth["H0"] / 100
    r2mu = Distance2Distmod(Om0=Om)
    r2z = Distance2Redshift(Om0=Om)
    Vext = truth["Vext_mag"] * np.asarray(galactic_to_radec_cartesian(
        truth["Vext_ell"], truth["Vext_b"])).reshape(3)

    order = gen.permutation(len(pool["r_h"]))
    r_Mpc = pool["r_h"][order] / h
    rhat = np.stack([np.cos(np.deg2rad(pool["dec"][order]))
                     * np.cos(np.deg2rad(pool["RA"][order])),
                     np.cos(np.deg2rad(pool["dec"][order]))
                     * np.sin(np.deg2rad(pool["RA"][order])),
                     np.sin(np.deg2rad(pool["dec"][order]))], axis=1)
    Vpec = truth["beta"] * pool["v_los"][order] + rhat @ Vext
    z_cosmo = np.asarray(r2z(r_Mpc, h=h))
    cz_true = SPEED_OF_LIGHT * ((1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT)
                                - 1)
    cz_obs = gen.normal(cz_true, np.sqrt(sigma_v**2 + e_cz**2))
    mu_true = np.asarray(r2mu(r_Mpc, h=h))

    used = np.zeros(len(order), dtype=bool)
    chosen = []
    for s in samples:
        p = _window_prob(cz_obs, s["cz_low"], s["cz_high"], cz_width)
        accept = (~used) & (gen.random(len(p)) < p)
        idx = np.flatnonzero(accept)[:s["n"]]
        if len(idx) < s["n"]:
            raise RuntimeError(
                f"Host pool too small: {len(idx)} of {s['n']} hosts in "
                f"[{s['cz_low']}, {s['cz_high']}] km/s.")
        used[idx] = True
        chosen.append((idx, s))

    hosts = {k: [] for k in ("pool_index", "cz_obs", "mu_true", "sigma_mu",
                             "cz_low", "cz_high")}
    for idx, s in chosen:
        d_L = 10**((mu_true[idx] - 25) / 5)
        hosts["pool_index"].append(order[idx])
        hosts["cz_obs"].append(cz_obs[idx])
        hosts["mu_true"].append(mu_true[idx])
        hosts["sigma_mu"].append(sigma_mu_law(d_L, s["sigma_factor"]))
        hosts["cz_low"].append(np.full(len(idx), float(s["cz_low"])))
        hosts["cz_high"].append(np.full(len(idx), float(s["cz_high"])))
    return {k: np.concatenate(v) for k, v in hosts.items()}


def anchor_effective_errors(template):
    """GLS distance-modulus error of the N4258 and LMC Cepheid blocks.

    The template is the real SH0ES data dict; columns of
    `L_Cepheid_host_dist` are ordered as [hosts, N4258, LMC, M31].
    """
    C = np.asarray(template["C_Cepheid"])
    col = np.asarray(template["L_Cepheid_host_dist"]).argmax(axis=1)
    n = int(template["num_hosts"])
    out = []
    for j in (n, n + 1):
        m = col == j
        one = np.ones(int(m.sum()))
        out.append(1 / np.sqrt(one @ np.linalg.solve(C[np.ix_(m, m)], one)))
    return tuple(out)


def make_CH0_mock_data(template, pool, hosts, truth, e_cz, field_index, gen):
    """Assemble a CH0Model data dict from drawn hosts and a real template.

    The template supplies the geometric-anchor errors, the MW zero-point
    errors and the 3D selection volume of the same field.
    """
    data = dict(template)
    n = len(hosts["cz_obs"])
    e_N4258_ceph, e_LMC_ceph = anchor_effective_errors(template)

    mu_N4258, mu_LMC = truth["mu_N4258"], truth["mu_LMC"]
    M_W = truth["M_W"]
    mu_rows = np.concatenate([hosts["mu_true"], [mu_N4258, mu_LMC]])
    sig_rows = np.concatenate([hosts["sigma_mu"], [e_N4258_ceph, e_LMC_ceph]])
    mag = gen.normal(mu_rows + M_W, sig_rows)

    L_dist = np.zeros((n + 2, n + 3))
    L_dist[np.arange(n + 2), np.arange(n + 2)] = 1.0
    C = np.diag(sig_rows**2)

    p = hosts["pool_index"]
    data.update({
        "mag_cepheid": mag,
        "logP": np.zeros(n + 2),
        "OH": np.zeros(n + 2),
        "C_Cepheid": C,
        "L_Cepheid": cholesky(C, lower=True),
        "L_Cepheid_host_dist": L_dist,
        "idx_dZP": np.zeros(0, dtype=int),
        "num_cepheids": n + 2,
        "num_hosts": n,
        # The redshift-selected mock has no SN term; one dummy row keeps the
        # SN arrays well formed.
        "mag_SN_unique_Cepheid_host": np.zeros(1),
        "C_SN_unique_Cepheid_host": np.eye(1),
        "L_SN_unique_Cepheid_host": np.eye(1),
        "L_SN_unique_Cepheid_host_dist": L_dist[:1],
        "mean_std_mag_SN_unique_Cepheid_host": 1.0,
        "mu_N4258_anchor": gen.normal(mu_N4258, template["e_mu_N4258_anchor"]),
        "mu_LMC_anchor": gen.normal(mu_LMC, template["e_mu_LMC_anchor"]),
        "M_HST": gen.normal(M_W, template["e_M_HST"]),
        "M_Gaia": gen.normal(M_W, template["e_M_Gaia"]),
        "host_names": np.array([f"mock_{i}" for i in range(n)]),
        "czcmb_cepheid_host": hosts["cz_obs"],
        "e_czcmb_cepheid_host": np.full(n, float(e_cz)),
        "cz_sel_low_host": hosts["cz_low"],
        "cz_sel_high_host": hosts["cz_high"],
        "RA_host": pool["RA"][p],
        "dec_host": pool["dec"][p],
        "PV_covmat_cepheid_host": np.eye(n),
        "host_los_density": pool["los_density"][p][None, ...],
        "host_los_velocity": pool["los_velocity"][p][None, ...],
        "host_los_r": pool["los_r"],
        "host_los_field_indices": np.array([int(field_index)]),
        "mask_host": np.ones(n, dtype=bool),
        "Neff_C_SN_unique_Cepheid_host": 1.0,
        "Neff_PV_covmat_cepheid_host": float(n),
        "Neff_C_Cepheid": float(n + 2),
    })
    for key in ("q_names", "dropped_observation",
                "dropped_observation_active_index"):
        data.pop(key, None)
    return data
