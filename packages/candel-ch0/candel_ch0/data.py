# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""SH0ES Cepheid-host data loaders."""
from os.path import join

import numpy as np
from astropy.io import fits
from jax import numpy as jnp
from scipy import linalg
from scipy.linalg import cholesky

from candel.field.angular_scatter import (angular_position_scatter_from_config,
                                          scatter_data_coordinates)
from candel.field.field_products import (
    field_smoothing_scale_from_config, resolve_or_build_los_data_path,
    velocity_field_smoothing_scale_from_config)
from candel.field.los import (effective_rank_entropy, load_los,
                              resolve_los_cache_request)
from candel.field.volume_density import _load_h0_volume_data_from_config
from candel.util import SPEED_OF_LIGHT, fprint, get_nested, load_config


def load_SH0ES(root):
    """
    Load the SH0ES data which can be used to sample distances.

    NOTE: Set the zero-width prior to a delta prior so it is not sampled.
    """
    lstsq_results_path = join(root, 'lstsq_results.txt')
    Y_fits_path = join(root, 'ally_shoes_ceph_topantheonwt6.0_112221.fits')
    L_fits_path = join(root, 'alll_shoes_ceph_topantheonwt6.0_112221.fits')
    C_fits_path = join(root, 'allc_shoes_ceph_topantheonwt6.0_112221.fits')

    Y = fits.open(Y_fits_path)[0].data
    L = fits.open(L_fits_path)[0].data
    C = fits.open(C_fits_path)[0].data

    C_inv_cho = linalg.cho_solve(linalg.cho_factor(C), np.identity(C.shape[0]))
    q_lstsq, sigma_lstsq = np.loadtxt(lstsq_results_path, unpack=True)
    mu_list = q_lstsq
    width_list = sigma_lstsq * 10

    ks = np.where(width_list == 0)[0]
    if len(ks) > 0:
        fprint("warning: zero width found in the priors. Setting it to 1e-5.")
        fprint(f"indices of zero width: {ks}")

    if len(ks) != 1:
        raise ValueError("At most one zero width is allowed.")

    k = ks[0]
    fprint(f"found zero-width prior at index {k}. Setting it to 0.")
    width_list[k] = 1e-5
    fixed_idx = k
    fixed_value = 0.

    mu_list = jnp.asarray(mu_list)
    width_list = jnp.asarray(width_list)
    theta_min, theta_max = mu_list - width_list / 2, mu_list + width_list / 2

    data = {
        "Y": Y,
        "L": L,
        "C_inv_cho": C_inv_cho,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "fixed_idx": fixed_idx,
        "fixed_value": fixed_value,
        "C": C,
        }

    for key in data:
        if not key.startswith("fixed_"):
            data[key] = jnp.asarray(data[key], dtype=jnp.float32)

    return data


def _remap_indices_after_mask(indices, mask):
    old_to_new = np.full(len(mask), -1, dtype=int)
    old_to_new[mask] = np.arange(mask.sum())
    indices = old_to_new[indices]
    return indices[indices >= 0]


def _resolve_sh0es_drop_observation(drop_observation, host_names):
    """Resolve a CH0 leave-one-out host selector to an active host index."""
    if drop_observation is None:
        return None

    if isinstance(drop_observation, str):
        value = drop_observation.strip()
        if value == "" or value.lower() == "none":
            return None
    if (isinstance(drop_observation, (int, np.integer))
            and not isinstance(drop_observation, (bool, np.bool_))):
        drop_index = int(drop_observation)
    else:
        raise TypeError(
            "`io.SH0ES.drop_observation` must be 'none' or an integer "
            "active host index.")

    host_names = np.asarray(host_names, dtype=str)
    if not (0 <= drop_index < len(host_names)):
        raise ValueError(
            "`io.SH0ES.drop_observation` index out of range: "
            f"{drop_index}. Active Cepheid host indices run from 0 to "
            f"{len(host_names) - 1}.")
    return drop_index


def _drop_sh0es_observation(data, drop_observation):
    """Drop one active Cepheid host and all observations tied to it."""
    drop_index = _resolve_sh0es_drop_observation(
        drop_observation, data["host_names"])
    if drop_index is None:
        return data

    num_hosts = int(data["num_hosts"])
    host_names = np.asarray(data["host_names"], dtype=str)
    dropped_name = host_names[drop_index]

    keep_host = np.ones(num_hosts, dtype=bool)
    keep_host[drop_index] = False
    keep_host_all = np.concatenate([keep_host, np.ones(3, dtype=bool)])

    keep_cepheid = data["L_Cepheid_host_dist"][:, drop_index] == 0
    keep_sn = data["L_SN_unique_Cepheid_host_dist"][:, drop_index] == 0
    fprint("Leaving out SH0ES Cepheid host observation "
           f"{dropped_name} (active index {drop_index}): dropping "
           f"{np.sum(~keep_cepheid)} Cepheids and {np.sum(~keep_sn)} "
           "SN calibrator row(s).")
    fprint("  dropped host details: "
           f"czcmb={data['czcmb_cepheid_host'][drop_index]:.3f} km/s, "
           f"RA={data['RA_host'][drop_index]:.6f} deg, "
           f"dec={data['dec_host'][drop_index]:.6f} deg.")

    for key in ("OH", "logP", "mag_cepheid"):
        data[key] = data[key][keep_cepheid]
    data["idx_dZP"] = _remap_indices_after_mask(
        data["idx_dZP"], keep_cepheid)
    data["C_Cepheid"] = data["C_Cepheid"][keep_cepheid][:, keep_cepheid]
    data["L_Cepheid"] = cholesky(data["C_Cepheid"], lower=True)
    data["L_Cepheid_host_dist"] = (
        data["L_Cepheid_host_dist"][keep_cepheid][:, keep_host_all])

    data["mag_SN_unique_Cepheid_host"] = (
        data["mag_SN_unique_Cepheid_host"][keep_sn])
    data["C_SN_unique_Cepheid_host"] = (
        data["C_SN_unique_Cepheid_host"][keep_sn][:, keep_sn])
    data["L_SN_unique_Cepheid_host"] = cholesky(
        data["C_SN_unique_Cepheid_host"], lower=True)
    data["L_SN_unique_Cepheid_host_dist"] = (
        data["L_SN_unique_Cepheid_host_dist"][keep_sn][:, keep_host_all])
    data["mean_std_mag_SN_unique_Cepheid_host"] = np.mean(
        np.sqrt(np.diag(data["C_SN_unique_Cepheid_host"])))

    for key in ("czcmb_cepheid_host", "e_czcmb_cepheid_host",
                "RA_host", "dec_host", "host_names"):
        data[key] = data[key][keep_host]
    data["PV_covmat_cepheid_host"] = (
        data["PV_covmat_cepheid_host"][keep_host][:, keep_host])

    if "mask_host" in data:
        mask_host = np.array(data["mask_host"], copy=True)
        active_original = np.flatnonzero(mask_host)
        if len(active_original) == num_hosts:
            mask_host[active_original[drop_index]] = False
        elif len(mask_host) == num_hosts:
            mask_host[drop_index] = False
        else:
            raise ValueError(
                "`mask_host` is inconsistent with active SH0ES hosts: "
                f"{len(mask_host)} mask entries for {num_hosts} hosts.")
        data["mask_host"] = mask_host
    else:
        data["mask_host"] = keep_host

    data["num_hosts"] = int(np.sum(keep_host))
    data["num_cepheids"] = int(np.sum(keep_cepheid))
    data["dropped_observation"] = str(dropped_name)
    data["dropped_observation_active_index"] = int(drop_index)

    return data


def _sh0es_host_short_names(data):
    """Short names of `L_Cepheid_host_dist` columns: hosts, then anchors."""
    return np.array([h.removeprefix("mu_") for h in data["host_names"]]
                    + ["N4258", "LMC", "M31"])


def _check_sh0es_host_names(names, known, key):
    unknown = sorted(set(names) - set(known))
    if unknown:
        raise ValueError(f"`io.SH0ES.{key}` has unknown hosts {unknown}; "
                         f"known: {list(known)}.")


def _keep_sh0es_hosts(data, keep_hosts):
    """Keep only the listed Cepheid hosts (anchors are always kept)."""
    if keep_hosts is None or keep_hosts == "all":
        return data
    _check_sh0es_host_names(
        keep_hosts, _sh0es_host_short_names(data)[:-3], "keep_hosts")
    while True:
        names = _sh0es_host_short_names(data)[:-3]
        drop = np.flatnonzero(~np.isin(names, keep_hosts))
        if len(drop) == 0:
            return data
        data = _drop_sh0es_observation(data, int(drop[0]))


def _scale_sh0es_cepheid_errors(data, scale, hosts):
    """
    Multiply the Cepheid magnitude errors of the listed hosts by `scale`,
    i.e. `C -> D C D` with `D` the per-Cepheid factor, so that covariances
    between a scaled and an unscaled Cepheid scale by `scale` only.
    """
    if scale is None or float(scale) == 1.0:
        return data
    names = _sh0es_host_short_names(data)
    hosts = names if hosts is None or hosts == "all" else np.asarray(hosts)
    _check_sh0es_host_names(hosts, names, "cepheid_error_scale_hosts")
    f_host = np.where(np.isin(names, hosts), float(scale), 1.0)
    f = data["L_Cepheid_host_dist"] @ f_host
    fprint(f"Scaling Cepheid errors by {scale} for {np.sum(f_host != 1)} "
           f"hosts ({int(np.sum(f != 1))} Cepheids).")
    data["C_Cepheid"] = data["C_Cepheid"] * np.outer(f, f)
    data["L_Cepheid"] = cholesky(data["C_Cepheid"], lower=True)
    return data


def load_SH0ES_separated(root, cepheid_host_cz_cmb_max=None,
                         los_data_path=None,
                         volume_data=None, field_indices=None,
                         drop_observation=None, keep_hosts=None,
                         cepheid_error_scale=None,
                         cepheid_error_scale_hosts=None):
    """
    Load the separated SH0ES data, separating the Cepheid and supernovae and
    covariance matrices.

    Structure of the covariance matrix indices:
    ------------------------------------------
    - Indices < 2150: Cepheid hosts without geometric anchors.
    - Index 2150: Start of NGC 4258 Cepheid hosts.
    - Index 2593: Start of M31 Cepheid hosts.
    - Index 2648: Start of LMC Cepheid hosts.

    - Index 3207: Uncertainty on HST zeropoint (sigma_HST).
    - Index 3208: Uncertainty on Gaia zeropoint (sigma_Gaia).
    - Index 3209: Prior on metallicity coefficient Z_W.
    - Index 3210: Unused term (likely placeholder).
    - Index 3211: Ground-based photometry systematic uncertainty (sigma_grnd).
    - Index 3212: Prior on P–L relation slope b_W.
    - Index 3213: Constraint on NGC 4258 anchor offset (delta_mu_N4258).
    - Index 3214: Constraint on LMC anchor offset (delta_mu_LMC).
    """

    # Unpack the SH0ES data.
    data = load_SH0ES(root)
    Y = np.array(data['Y'], copy=True)
    C = np.array(data['C'], copy=True)
    L = np.array(data['L'].T, copy=True)

    # Cepheid data and covariance matrix.
    OH = L[:, -4][:3130]
    logP = L[:, -6][:3130]
    mag_cepheid = Y[:3130]
    C_Cepheid = C[:3130, :3130]

    # Undo the removal of a slope of -3.285
    mag_cepheid += -3.285 * logP

    # This will organise the host distances as
    # `[Host with Cepheids but no geometric anchors, NGC4258, LMC, M31]`. There
    # are 37 of the former, so in total there are 40 distances to be inferred.
    L_dist = np.hstack([L[:, :37], L[:, [37, 39, 40]]])
    L_Cepheid_host_dist = L_dist[:3130]

    # N4258 and LMC anchors.
    mu_N4258_anchor = 29.398
    e_mu_N4258_anchor = 0.032

    mu_LMC_anchor = 18.477
    e_mu_LMC_anchor = 0.0263

    # Undo the anchor offsets.
    mag_cepheid[2150:2593] += mu_N4258_anchor
    mag_cepheid[2648:] += mu_LMC_anchor

    C_SN_Cepheid = C[3130:3207, 3130:3207]
    Y_SN_Cepheid = Y[3130:3207]

    # HST and Gaia zero-points
    M_HST = Y[3207]
    e_M_HST = C[3207, 3207]**0.5

    M_Gaia = Y[3208]
    e_M_Gaia = C[3208, 3208]**0.5

    # Systematic uncertainties btw ground-based and HST photometry.
    sigma_grnd = C[3211, 3211]**0.5
    # Indices of Cepheids needing ground-based dZP correction (column 45 of
    # the original L matrix is Delta_zp).
    idx_dZP = np.where(L[:3130, 45] == 1)[0]

    q_names = np.asanyarray(
        ['mu_M101', 'mu_M1337', 'mu_N0691', 'mu_N1015', 'mu_N0105',
         'mu_N1309', 'mu_N1365', 'mu_N1448', 'mu_N1559', 'mu_N2442',
         'mu_N2525', 'mu_N2608', 'mu_N3021', 'mu_N3147', 'mu_N3254',
         'mu_N3370', 'mu_N3447', 'mu_N3583', 'mu_N3972', 'mu_N3982',
         'mu_N4038', 'mu_N4424', 'mu_N4536', 'mu_N4639', 'mu_N4680',
         'mu_N5468', 'mu_N5584', 'mu_N5643', 'mu_N5728', 'mu_N5861',
         'mu_N5917', 'mu_N7250', 'mu_N7329', 'mu_N7541', 'mu_N7678',
         'mu_N0976', 'mu_U9391', 'Delta_mu_N4258', 'M_H1_W',
         'Delta_mu_LMC', 'mu_M31', 'b_W', 'MB0', 'Z_W', 'undefined',
         'Delta_zp', 'log10_H0'])

    L_SN_Cepheid_dist = L_dist[3130:3207]

    num_hosts = L_Cepheid_host_dist.shape[1] - 3
    num_cepheids = len(mag_cepheid)
    host_names = q_names[np.char.startswith(
        q_names.astype(str), "mu_")]
    host_names = host_names[:num_hosts]

    # Cepheid host redshifts and the PV covariance matrix.
    data_cepheid_host_redshift = np.load(
        join(root, "processed", "Cepheid_anchors_redshifts.npy"),
        allow_pickle=True)
    PV_covmat_cepheid_host = np.load(
        join(root, "processed", "PV_covmat_cepheid_hosts_fiducial.npy"),
        allow_pickle=True)

    host_coords = {
        "RA": np.array(data_cepheid_host_redshift["RA"], copy=True),
        "dec": np.array(data_cepheid_host_redshift["DEC"], copy=True),
    }
    if los_data_path is not None:
        los_data_path = resolve_los_cache_request(
            los_data_path, host_coords)

    host_los_keys = [
        "los_density", "los_velocity", "los_r", "los_field_indices"]
    if los_data_path is None:
        host_los = dict.fromkeys(host_los_keys)
    else:
        host_los = load_los(los_data_path, dict(host_coords),
                            field_indices=field_indices)

    # Keep the brightest (lowest magnitude) SN per Cepheid host galaxy
    n_hosts = L_SN_Cepheid_dist.shape[1]
    best_mag = np.full(n_hosts, np.inf)
    best_idx = np.full(n_hosts, -1, dtype=int)

    for i, y in enumerate(Y_SN_Cepheid):
        # Assuming one-hot host assignment per SN
        j = np.where(L_SN_Cepheid_dist[i] == 1)[0][0]
        if y < best_mag[j]:    # use '>' if working in flux (higher = brighter)
            best_mag[j] = y
            best_idx[j] = i

    valid = best_idx >= 0
    unique_ks = best_idx[valid]

    mag_SN_unique_Cepheid_host = Y_SN_Cepheid[unique_ks]
    C_SN_unique_Cepheid_host = C_SN_Cepheid[np.ix_(unique_ks, unique_ks)]
    L_SN_unique_Cepheid_host_dist = L_SN_Cepheid_dist[unique_ks]

    data = {
        # Individual Cepheid data, covariance matrix and host association.
        "mag_cepheid": mag_cepheid,
        "logP": logP,
        "OH": OH,
        "C_Cepheid": C_Cepheid,
        "L_Cepheid": cholesky(C_Cepheid, lower=True),
        "L_Cepheid_host_dist": L_Cepheid_host_dist,
        "Cepheids_only": False,
        "num_cepheids": num_cepheids,
        "num_hosts": num_hosts,
        # Unique SNe in Cepheid host galaxies.
        "mag_SN_unique_Cepheid_host": mag_SN_unique_Cepheid_host,
        "C_SN_unique_Cepheid_host": C_SN_unique_Cepheid_host,
        "mean_std_mag_SN_unique_Cepheid_host": np.mean(np.sqrt(np.diag(C_SN_unique_Cepheid_host))),  # noqa
        "L_SN_unique_Cepheid_host": cholesky(C_SN_unique_Cepheid_host,
                                             lower=True),
        "L_SN_unique_Cepheid_host_dist": L_SN_unique_Cepheid_host_dist,
        # External constraints/priors.
        "mu_N4258_anchor": mu_N4258_anchor,
        "e_mu_N4258_anchor": e_mu_N4258_anchor,
        "mu_LMC_anchor": mu_LMC_anchor,
        "e_mu_LMC_anchor": e_mu_LMC_anchor,
        "M_HST": M_HST,
        "e_M_HST": e_M_HST,
        "M_Gaia": M_Gaia,
        "e_M_Gaia": e_M_Gaia,
        "sigma_grnd": sigma_grnd,
        "idx_dZP": idx_dZP,
        # Cepheid host galaxy information.
        "q_names": q_names,
        "host_names": host_names,
        "czcmb_cepheid_host": data_cepheid_host_redshift["zCMB"] * SPEED_OF_LIGHT,  # noqa
        "e_czcmb_cepheid_host": data_cepheid_host_redshift["zCMBERR"],
        "RA_host": host_coords["RA"],
        "dec_host": host_coords["dec"],
        "PV_covmat_cepheid_host": PV_covmat_cepheid_host,
        "host_los_density": host_los["los_density"],
        "host_los_velocity": host_los["los_velocity"],
        "host_los_r": host_los["los_r"],
        "host_los_field_indices": host_los["los_field_indices"],
        }

    if cepheid_host_cz_cmb_max is not None:
        if cepheid_host_cz_cmb_max < 1000:
            raise ValueError(
                f"`cz_cmb_max` must be larger than 1000 km/s, got "
                f"{cepheid_host_cz_cmb_max} km/s. Otherwise could eliminate "
                "some geometric anchors.")

        # Switch this flag so that these runs cannot be done jointly with SNe
        # since some shapes might not be correct.
        data["Cepheids_only"] = True

        cz_host = data["czcmb_cepheid_host"]
        cz_host_all = np.hstack([data["czcmb_cepheid_host"], [667, 327, -582]])
        cz_cepheid = data["L_Cepheid_host_dist"] @ cz_host_all
        cz_unique_SN_Cepheid_host = data["L_SN_unique_Cepheid_host_dist"] @ cz_host_all  # noqa

        mask_host = cz_host < cepheid_host_cz_cmb_max
        mask_host_all = cz_host_all < cepheid_host_cz_cmb_max
        mask_cepheid = cz_cepheid < cepheid_host_cz_cmb_max
        mask_cz_unique_SN_Cepheid_host = (
            cz_unique_SN_Cepheid_host < cepheid_host_cz_cmb_max)

        fprint(f"Masking Cepheids with cz_cmb > {cepheid_host_cz_cmb_max} "
               f"km/s: Keeping {np.sum(mask_host)} out of {len(mask_host)}.")

        data["OH"] = data["OH"][mask_cepheid]
        data["logP"] = data["logP"][mask_cepheid]
        data["mag_cepheid"] = data["mag_cepheid"][mask_cepheid]

        # Remap idx_dZP: keep only indices that survive the mask, then
        # convert to new positions in the masked array.
        data["idx_dZP"] = _remap_indices_after_mask(
            data["idx_dZP"], mask_cepheid)
        data["C_Cepheid"] = data["C_Cepheid"][mask_cepheid][:, mask_cepheid]
        data["L_Cepheid"] = cholesky(data["C_Cepheid"], lower=True)

        data["L_Cepheid_host_dist"] = data["L_Cepheid_host_dist"][mask_cepheid][:, mask_host_all]  # noqa
        data["czcmb_cepheid_host"] = data["czcmb_cepheid_host"][mask_host]
        data["e_czcmb_cepheid_host"] = data["e_czcmb_cepheid_host"][mask_host]
        data["RA_host"] = data["RA_host"][mask_host]
        data["dec_host"] = data["dec_host"][mask_host]
        data["PV_covmat_cepheid_host"] = data["PV_covmat_cepheid_host"][mask_host][:, mask_host]  # noqa

        data["L_SN_unique_Cepheid_host_dist"] = data["L_SN_unique_Cepheid_host_dist"][mask_cz_unique_SN_Cepheid_host][:, mask_host_all]  # noqa
        data["mag_SN_unique_Cepheid_host"] = data["mag_SN_unique_Cepheid_host"][mask_cz_unique_SN_Cepheid_host]  # noqa
        data["C_SN_unique_Cepheid_host"] = data["C_SN_unique_Cepheid_host"][mask_cz_unique_SN_Cepheid_host][:, mask_cz_unique_SN_Cepheid_host]  # noqa
        data["L_SN_unique_Cepheid_host"] = cholesky(data["C_SN_unique_Cepheid_host"], lower=True)  # noqa

        data["num_hosts"] = np.sum(mask_host)
        data["num_cepheids"] = np.sum(mask_cepheid)
        data["host_names"] = data["host_names"][mask_host]

        data["mask_host"] = mask_host

    data = _drop_sh0es_observation(data, drop_observation)
    data = _keep_sh0es_hosts(data, keep_hosts)
    data = _scale_sh0es_cepheid_errors(
        data, cepheid_error_scale, cepheid_error_scale_hosts)

    data["Neff_C_SN_unique_Cepheid_host"] = effective_rank_entropy(data["C_SN_unique_Cepheid_host"])  # noqa
    data["Neff_PV_covmat_cepheid_host"] = effective_rank_entropy(data["PV_covmat_cepheid_host"])     # noqa
    data["Neff_C_Cepheid"] = effective_rank_entropy(data["C_Cepheid"])

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True
    else:
        data["has_volume_density_3d"] = False

    return data


def load_SH0ES_from_config(config_path):
    config = load_config(config_path, replace_los_prior=False)
    use_recon = get_nested(config, "model/use_reconstruction", False)
    config["io"]["load_host_los"] = use_recon
    d = config["io"]["SH0ES"]
    root = d["root"]
    cepheid_host_cz_cmb_max = d.get("cepheid_host_cz_cmb_max", None)
    drop_observation = d.get("drop_observation", None)
    reconstruction = d.get("reconstruction", None)
    field_indices = get_nested(config, "io/field_indices", None)
    field_smoothing_scale = field_smoothing_scale_from_config(config)
    velocity_field_smoothing_scale = (
        velocity_field_smoothing_scale_from_config(config))
    if reconstruction is not None:
        if config["io"]["load_host_los"]:
            los_data_path = resolve_or_build_los_data_path(
                config, "SH0ES", reconstruction,
                config["io"]["PV_main"]["SH0ES"]["los_file"],
                field_smoothing_scale=field_smoothing_scale,
                velocity_field_smoothing_scale=velocity_field_smoothing_scale,
                config_path=config_path, field_indices=field_indices)
        else:
            los_data_path = None
    else:
        los_data_path = None

    data = load_SH0ES_separated(
        root, cepheid_host_cz_cmb_max,
        los_data_path=los_data_path,
        field_indices=field_indices, drop_observation=drop_observation,
        keep_hosts=d.get("keep_hosts", None),
        cepheid_error_scale=d.get("cepheid_error_scale", None),
        cepheid_error_scale_hosts=d.get("cepheid_error_scale_hosts", None))
    if los_data_path is None:
        scatter = angular_position_scatter_from_config(config)
        if scatter is not None:
            scatter_data_coordinates(
                data, scatter, keys=("RA_host", "dec_host"),
                label="SH0ES")
    if los_data_path is not None:
        los_data_path = getattr(los_data_path, "resolved_path", los_data_path)

    velocity_selections = ["redshift", "SN_magnitude_redshift"]
    if get_nested(config, "model/which_selection", None) \
            == "SN_magnitude_or_redshift_Nmag":
        n_mag = get_nested(config, "model/num_hosts_selection_mag", None)
        if type(n_mag) is int and n_mag < data["num_hosts"]:
            velocity_selections.append("SN_magnitude_or_redshift_Nmag")

    volume_data = _load_h0_volume_data_from_config(
        config, los_data_path, reconstruction, "SH0ES",
        velocity_selections=tuple(velocity_selections),
        field_indices=data.get("host_los_field_indices", None))

    if volume_data is not None:
        data.update(volume_data)
        data["has_volume_density_3d"] = True

    return data
