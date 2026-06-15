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
"""Posterior predictive check for the EDD TRGB H0 model."""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import ks_2samp, norm
from tqdm.auto import tqdm

from ..field import name2field_loader
from ..pvdata.field_products import (
    field_smoothing_scale_from_config,
    velocity_field_smoothing_scale_from_config)
from ..pvdata.volume_density import _density_unit_normalization
from ..util import (SPEED_OF_LIGHT, fprint, galactic_to_radec, get_nested,
                    load_config, radec_to_cartesian, radec_to_galactic)
from ._field_utils import (build_field_pool, build_field_pool_evaluator,
                           compute_r_max_selection,
                           galaxy_bias_params_from_values, galaxy_bias_weight)


def _flat(x):
    """Flatten scalar samples while preserving vector-valued samples."""
    x = np.asarray(x)
    if x.ndim > 1 and x.shape[-1] != 3:
        return x.reshape(-1)
    if x.ndim > 2:
        return x.reshape(-1, x.shape[-1])
    return x


def _sample_or_default(samples, key, n, default):
    if key in samples:
        return _flat(samples[key])
    return np.full(n, default)


def _sample_empirical(gen, values, size):
    """Draw from an empirical 1D array using integer indexing."""
    values = np.asarray(values)
    return values[gen.integers(0, len(values), size)]


def _prior_reference_value(config, name, default):
    spec = get_nested(config, f"model/priors/{name}", None)
    if isinstance(spec, dict):
        for key in ("value", "loc", "mean"):
            if key in spec:
                return spec[key]
    return default


def _check_vext_dipole_only(config):
    """Raise if the config requests Vext terms the PPC cannot reproduce.

    The PPC forward model only builds the Vext dipole (see ``_vext_samples``);
    the monopole, quadrupole and octupole terms used by the inference model are
    not implemented here. Mirror the model's config parsing so an active term
    fails loudly instead of being silently dropped.
    """
    mono = get_nested(config, "model/which_Vext_monopole", "none")
    if mono is True:
        mono = "constant"
    elif mono is False or mono is None:
        mono = "none"
    if mono == "none" and get_nested(config, "model/use_Vext_monopole", False):
        mono = "constant"

    unsupported = []
    if mono != "none":
        unsupported.append(f"which_Vext_monopole='{mono}'")
    if get_nested(config, "model/use_Vext_quadrupole", False):
        unsupported.append("use_Vext_quadrupole=True")
    if get_nested(config, "model/use_Vext_octupole", False):
        unsupported.append("use_Vext_octupole=True")
    if unsupported:
        raise NotImplementedError(
            "TRGB PPC only supports the Vext dipole, but the config requests "
            + ", ".join(unsupported)
            + ". These higher-order Vext terms are not reproduced by the PPC "
            "forward model.")


def _vext_samples(samples, n_post):
    """Return Cartesian external-velocity samples."""
    if "Vext" in samples:
        return _flat(samples["Vext"])
    if all(k in samples for k in ("Vext_mag", "Vext_ell", "Vext_b")):
        mag = _flat(samples["Vext_mag"])
        ell = _flat(samples["Vext_ell"])
        b = _flat(samples["Vext_b"])
        ra, dec = galactic_to_radec(ell, b)
        return mag[:, None] * radec_to_cartesian(ra, dec)
    return np.zeros((n_post, 3))


def _draw_cz(gen, mean, sigma, nu=None):
    """Draw Gaussian or Student-t redshift residuals."""
    if nu is None:
        return gen.normal(mean, sigma)
    return mean + sigma * gen.standard_t(nu, size=np.shape(mean))


class _FastDistanceConversions:
    """NumPy distance-modulus/redshift interpolation for PPC draws."""

    def __init__(self, Om0=0.3, zmin=1e-8, zmax=0.5, npoints=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        self.z_grid = np.logspace(np.log10(zmin), np.log10(zmax), npoints)
        self.r_grid = cosmo.comoving_distance(self.z_grid).value
        self.log_r_grid = np.log(self.r_grid)
        self.mu_grid = cosmo.distmod(self.z_grid).value

    def distmod(self, r, h=1):
        return (
            np.interp(np.log(np.asarray(r) * h),
                      self.log_r_grid, self.mu_grid)
            - 5 * np.log10(h))

    def redshift(self, r, h=1):
        return np.interp(np.asarray(r) * h, self.r_grid, self.z_grid)


def _sigma_v_from_density(rho, sigma_v_low, sigma_v_high, log_rho_t, k):
    rho = np.clip(rho, 1e-6, None)
    return sigma_v_low + (sigma_v_high - sigma_v_low) / (
        1.0 + np.exp(-k * (np.log(rho) - log_rho_t)))


def _available_field_indices(data):
    """Choose one field realization for a reconstruction PPC."""
    if "los_field_indices" in data:
        return np.asarray(data["los_field_indices"], dtype=int)
    elif "host_los_density" in data:
        n = np.asarray(data["host_los_density"]).shape[0]
        return np.arange(n, dtype=int)
    return np.array([0], dtype=int)


def _selected_field_index(data, gen, field_index=None):
    """Choose one field realization for a reconstruction PPC."""
    if field_index is not None:
        return int(field_index)
    field_indices = _available_field_indices(data)
    return int(gen.choice(field_indices))


def _field_name_config(config):
    field_name = get_nested(
        config, "io/PV_main/EDD_TRGB/reconstruction", None)
    if field_name is None:
        raise ValueError(
            "use_reconstruction=True but no reconstruction specified")
    field_config = dict(config["io"]["reconstruction_main"][field_name])
    return field_name, field_config


def _density_divisor(field_name):
    norm_info = _density_unit_normalization(field_name)
    if norm_info is None:
        return None
    return norm_info[0]


def _smoothing_label(scale):
    """Format an optional smoothing scale for status messages."""
    if scale is None:
        return "none"
    return f"{scale:g} Mpc/h"


def _posterior_param_label(name, values):
    """Format a compact posterior summary for a scalar parameter."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return f"{name}=nan"
    if values.size == 1 or np.allclose(values, values[0]):
        return f"{name}={values[0]:.3g}"
    q16, q50, q84 = np.percentile(values, [16, 50, 84])
    return f"{name}={q50:.3g} [{q16:.3g}, {q84:.3g}]"


def _bias_parameter_labels(bias):
    """Return printable summaries for the active galaxy-bias parameters."""
    which_bias = bias["which_bias"]
    if which_bias == "uniform":
        return ["none"]
    if which_bias == "unity":
        return ["fixed unity"]
    if which_bias in ("linear", "linear_from_beta",
                      "linear_from_beta_stochastic"):
        return [_posterior_param_label("b1", bias["b1"])]
    if which_bias == "powerlaw":
        return [_posterior_param_label("alpha", bias["alpha"])]
    if which_bias == "double_powerlaw":
        labels = [_posterior_param_label("alpha_low", bias["alpha_low"])]
        if "alpha_high" in bias:
            labels.append(_posterior_param_label(
                "alpha_high", bias["alpha_high"]))
        else:
            alpha_high = bias["alpha_low"] * bias["alpha_high_frac"]
            labels.append(_posterior_param_label(
                "alpha_high", alpha_high))
            labels.append(_posterior_param_label(
                "alpha_high_frac", bias["alpha_high_frac"]))
        labels.append(_posterior_param_label(
            "log_rho_t", bias["log_rho_t"]))
        labels.append(_posterior_param_label(
            "log_rho_width", bias["log_rho_width"]))
        return labels
    if which_bias == "quadratic":
        return [
            _posterior_param_label("b1", bias["b1"]),
            _posterior_param_label("b2", bias["b2"]),
        ]
    if which_bias == "cubic":
        return [
            _posterior_param_label("b1", bias["b1"]),
            _posterior_param_label("b2", bias["b2"]),
            _posterior_param_label("b3", bias["b3"]),
        ]
    return ["unknown"]


def _bias_samples(samples, config, beta, n_post):
    which_bias = get_nested(config, "model/which_bias", "linear")
    Om = get_nested(config, "model/Om", get_nested(config, "model/Om0", 0.3))
    out = {"which_bias": which_bias}
    if which_bias in ("uniform", "unity"):
        return out
    if which_bias == "linear_from_beta":
        out["b1"] = Om**0.55 / beta
    elif which_bias == "linear_from_beta_stochastic":
        if "b1" in samples:
            out["b1"] = _flat(samples["b1"])
        else:
            key = (
                "delta_b1_skipZ" if "delta_b1_skipZ" in samples
                else "delta_b1")
            out["b1"] = Om**0.55 / beta + _sample_or_default(
                samples, key, n_post, 0.0)
    elif which_bias == "linear":
        out["b1"] = _sample_or_default(samples, "b1", n_post, 1.0)
    elif which_bias == "powerlaw":
        out["alpha"] = _sample_or_default(samples, "alpha", n_post, 1.0)
    elif which_bias == "double_powerlaw":
        out["alpha_low"] = _flat(samples["alpha_low"])
        if "alpha_high" in samples:
            out["alpha_high"] = _flat(samples["alpha_high"])
        elif "alpha_high_skipZ" in samples:
            out["alpha_high"] = _flat(samples["alpha_high_skipZ"])
        else:
            out["alpha_high_frac"] = _flat(samples["alpha_high_frac"])
        out["log_rho_t"] = _flat(samples["log_rho_t"])
        out["log_rho_width"] = _sample_or_default(
            samples, "log_rho_width", n_post, 1.0)
    elif which_bias == "quadratic":
        out["b1"] = _sample_or_default(samples, "b1", n_post, 1.0)
        out["b2"] = _sample_or_default(samples, "b2", n_post, 0.0)
    elif which_bias == "cubic":
        out["b1"] = _sample_or_default(samples, "b1", n_post, 1.0)
        out["b2"] = _sample_or_default(samples, "b2", n_post, 0.0)
        out["b3"] = _sample_or_default(samples, "b3", n_post, 0.0)
    else:
        raise ValueError(f"Unsupported PPC galaxy-bias model: {which_bias}")
    return out


def _bias_values(rho, bias, idx_post):
    which_bias = bias["which_bias"]
    if which_bias == "uniform":
        return np.ones_like(rho, dtype=float)
    params = galaxy_bias_params_from_values(bias, which_bias, idx=idx_post)
    return galaxy_bias_weight(rho, params, which_bias)


def _bias_upper_bound(rho_support, bias, idx_post):
    """Return a conservative bias-weight bound for rejection sampling."""
    which_bias = bias["which_bias"]
    idx_post = np.asarray(idx_post)
    rho_min = float(np.min(rho_support))
    rho_max = float(np.max(rho_support))

    if which_bias == "uniform":
        return np.ones_like(idx_post, dtype=float)
    if which_bias == "unity":
        bound = np.max(_bias_values(rho_support, bias, 0))
        return np.full_like(idx_post, bound, dtype=float)
    if "linear" in which_bias:
        b1 = bias["b1"][idx_post]
        rho_extreme = np.where(b1 >= 0.0, rho_max, rho_min)
        return _bias_values(rho_extreme, bias, idx_post)
    if which_bias == "double_powerlaw":
        rho_t = np.exp(bias["log_rho_t"][idx_post])
        rho_eval = np.vstack([
            np.full_like(rho_t, rho_min),
            np.clip(rho_t, rho_min, rho_max),
            np.full_like(rho_t, rho_max),
        ])
        idx = idx_post[None, :]
        return np.max(_bias_values(rho_eval, bias, idx), axis=0)
    rho = rho_support[:, None]
    idx = idx_post[None, :]
    return np.max(_bias_values(rho, bias, idx), axis=0)


def _rho_support(rho, max_size=4096):
    rho = np.asarray(rho)
    if len(rho) <= max_size:
        return rho
    q = np.linspace(0.0, 1.0, max_size)
    return np.quantile(rho, q)


def _uniform_density_pool(gen, r_sphere, pool_size):
    """Build a unit-density, zero-velocity pool in Mpc/h coordinates."""
    r_h = r_sphere * gen.random(pool_size)**(1.0 / 3.0)
    phi = gen.uniform(0.0, 2.0 * np.pi, pool_size)
    sin_dec = gen.uniform(-1.0, 1.0, pool_size)
    cos_dec = np.sqrt(1.0 - sin_dec**2)
    rhat = np.column_stack([
        cos_dec * np.cos(phi),
        cos_dec * np.sin(phi),
        sin_dec,
    ])
    return {
        "r_h": r_h,
        "rho": np.ones(pool_size, dtype=np.float64),
        "v_los": np.zeros(pool_size, dtype=np.float64),
        "RA": np.rad2deg(phi),
        "dec": np.rad2deg(np.arcsin(sin_dec)),
        "rhat_icrs": rhat,
        "base_weight": None,
        "rho_support": np.array([1.0], dtype=np.float64),
    }


def _galactic_latitude_cut_from_config(config):
    b_min = get_nested(config, "io/PV_main/EDD_TRGB/b_min", None)
    if b_min is None:
        return None
    b_min = float(b_min)
    if b_min < 0.0 or b_min > 90.0:
        raise ValueError(
            "`io/PV_main/EDD_TRGB/b_min` must be in [0, 90] deg.")
    return b_min


def _apply_galactic_latitude_cut_to_pool(pool, b_min):
    """Keep only random sky positions satisfying |b| >= b_min."""
    if b_min is None or b_min <= 0.0:
        return pool
    _, b = radec_to_galactic(pool["RA"], pool["dec"])
    keep = np.abs(b) >= b_min
    n = len(pool["r_h"])
    if not np.any(keep):
        raise RuntimeError(
            f"PPC pool has no positions satisfying |b| >= {b_min:g} deg.")

    out = {}
    for key, value in pool.items():
        if value is None:
            out[key] = value
            continue
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[keep]
        else:
            out[key] = value
    out["rho_support"] = _rho_support(out["rho"])
    return out

###############################################################################
#                         PPC generation                                      #
###############################################################################


def generate_trgb_ppc(samples, data, config, n_ppc=None, seed=42,
                      field_index=None, progress=True):
    """Generate posterior predictive samples for the EDD TRGB model.

    When a reconstruction field is available, galaxies are sampled from
    the full 3D density field (matching the mock generator). A large pool
    of random 3D positions is pre-evaluated once with interpolated
    density/velocity values for efficiency.

    Parameters
    ----------
    samples : dict
        Posterior samples loaded from HDF5.
    data : dict
        Data dict from ``load_EDD_TRGB_from_config``.
    config : str or dict
        Path to config TOML or loaded config dict.
    n_ppc : int or None
        Number of PPC galaxies to generate.
    seed : int
        Random seed.
    progress : bool
        Whether to show a tqdm progress bar for accepted PPC galaxies.

    Returns
    -------
    dict with keys: mag_sim, cz_sim, r_sim, mag_obs, cz_obs.
    """
    if isinstance(config, str):
        config = load_config(config, replace_los_prior=False)
    _check_vext_dipole_only(config)
    gen = np.random.default_rng(seed)

    H0 = _flat(samples["H0"])
    M_TRGB = _flat(samples["M_TRGB"])
    sigma_int = _flat(samples["sigma_int"])
    n_post = len(H0)
    c_star = _sample_or_default(
        samples, "c_star", n_post,
        _prior_reference_value(config, "c_star", 1.23))
    alpha_c = _sample_or_default(
        samples, "alpha_c", n_post,
        _prior_reference_value(config, "alpha_c", 0.2))
    Vext = _vext_samples(samples, n_post)
    beta = _sample_or_default(
        samples, "beta", n_post,
        _prior_reference_value(config, "beta", 0.0))
    use_reconstruction = get_nested(config, "model/use_reconstruction", False)
    bias = (_bias_samples(samples, config, beta, n_post)
            if use_reconstruction else {"which_bias": "uniform"})
    if use_reconstruction:
        field_name = get_nested(
            config, "io/PV_main/EDD_TRGB/reconstruction", None)
        mas = None
        if field_name is not None:
            mas = get_nested(
                config,
                f"io/reconstruction_main/{field_name}/which_MAS",
                None)
        density_smoothing = _smoothing_label(
            field_smoothing_scale_from_config(config))
        velocity_smoothing = _smoothing_label(
            velocity_field_smoothing_scale_from_config(config))
        status = [
            f"reconstruction={field_name}",
            f"bias={bias['which_bias']}",
            f"density_smoothing={density_smoothing}",
            f"velocity_smoothing={velocity_smoothing}",
        ]
        if mas is not None:
            status.append(f"MAS={mas}")
    else:
        status = ["reconstruction=none", f"bias={bias['which_bias']}"]
    fprint("PPC config: " + ", ".join(status))
    fprint("PPC bias parameters: "
           + ", ".join(_bias_parameter_labels(bias)))

    use_density_sigma_v = get_nested(
        config, "model/use_density_dependent_sigma_v", False)
    if use_density_sigma_v:
        sigma_v = (
            _flat(samples["sigma_v_low"]),
            _flat(samples["sigma_v_high"]),
            _flat(samples["log_sigma_v_rho_t"]),
            _flat(samples["sigma_v_k"]),
        )
    else:
        sigma_v = _flat(samples["sigma_v"])

    cz_likelihood = get_nested(config, "model/cz_likelihood", "gaussian")
    if cz_likelihood == "student_t":
        if "nu_cz" not in samples:
            raise ValueError(
                "PPC config requests Student-t cz likelihood, but posterior "
                "samples do not contain `nu_cz`.")
        nu_cz = _flat(samples["nu_cz"])
    else:
        nu_cz = None
        if "nu_cz" in samples:
            fprint("PPC warning: posterior contains `nu_cz`, but "
                   "model/cz_likelihood is not 'student_t'; using Gaussian "
                   "cz draws.")

    cz_labels = [f"likelihood={cz_likelihood}"]
    if use_reconstruction:
        cz_labels.append(_posterior_param_label("beta", beta))
    if use_density_sigma_v:
        cz_labels.extend([
            _posterior_param_label("sigma_v_low", sigma_v[0]),
            _posterior_param_label("sigma_v_high", sigma_v[1]),
            _posterior_param_label("log_sigma_v_rho_t", sigma_v[2]),
            _posterior_param_label("sigma_v_k", sigma_v[3]),
        ])
    else:
        cz_labels.append(_posterior_param_label("sigma_v", sigma_v))
    cz_labels.append(_posterior_param_label(
        "|Vext|", np.linalg.norm(Vext, axis=1)))
    if nu_cz is not None:
        cz_labels.append(_posterior_param_label("nu_cz", nu_cz))
    fprint("PPC cz parameters: " + ", ".join(cz_labels))

    # Selection parameters
    which_sel = get_nested(config, "model/which_selection", None)
    mag_lim_samples = _flat(samples["mag_lim_TRGB"]) \
        if "mag_lim_TRGB" in samples else None
    mag_width_samples = _flat(samples["mag_lim_TRGB_width"]) \
        if "mag_lim_TRGB_width" in samples else None
    mag_min_fixed = get_nested(config, "model/mag_min_TRGB", None)
    mag_lim_fixed = get_nested(config, "model/mag_lim_TRGB", None)
    mag_width_fixed = get_nested(config, "model/mag_lim_TRGB_width", None)
    cz_lim_samples = _flat(samples["cz_lim_selection"]) \
        if "cz_lim_selection" in samples else None
    cz_width_samples = _flat(samples["cz_lim_selection_width"]) \
        if "cz_lim_selection_width" in samples else None
    cz_lim_fixed = get_nested(config, "model/cz_lim_selection", None)
    cz_width_fixed = get_nested(config, "model/cz_lim_selection_width", None)

    # ---- Data ----
    mag_obs = np.asarray(data["mag_obs"])
    cz_obs = np.asarray(data["czcmb"])
    e_mag_obs_all = np.asarray(data["e_mag_obs"])
    e_czcmb_all = np.asarray(data["e_czcmb"])
    colour_dered_all = np.asarray(data["colour_dered"])
    e_colour_dered_all = np.asarray(
        data.get("e_colour_dered", np.zeros_like(colour_dered_all)))
    colour_dered_mean = np.mean(colour_dered_all)
    colour_dered_std = np.std(colour_dered_all)
    n_hosts = len(mag_obs)
    has_trgb_colour = "colour_dered" in data and "e_colour_dered" in data
    c_bar = _sample_or_default(
        samples, "c_bar", n_post,
        _prior_reference_value(config, "c_bar", colour_dered_mean))
    w_c = _sample_or_default(
        samples, "w_c", n_post,
        _prior_reference_value(config, "w_c",
                               max(float(colour_dered_std), 1e-3)))
    colour_model = {
        "has_colour": has_trgb_colour,
        "alpha_c": alpha_c,
        "c_bar": c_bar,
        "w_c": w_c,
        "e_colour": e_colour_dered_all,
    }

    if n_ppc is None:
        ppc_factor = get_nested(config, "model/ppc_factor", 10)
        n_ppc = ppc_factor * n_hosts

    # ---- Cosmography ----
    Om = get_nested(config, "model/Om", 0.3)
    dist = _FastDistanceConversions(Om0=Om)
    r2mu = dist.distmod
    r2z = dist.redshift

    # ---- Distance limits ----
    r_limits = get_nested(config, "model/r_limits_malmquist", [0.01, 150])
    r_min, r_max = r_limits[0], r_limits[1]

    # ---- Effective sphere radius (selection-aware) ----
    r_max_eff = compute_r_max_selection(
        mag_lim=mag_lim_samples if mag_lim_samples is not None else mag_lim_fixed,  # noqa
        M_abs=M_TRGB, sigma_int=sigma_int, e_mag=e_mag_obs_all,
        mag_lim_width=mag_width_samples if mag_width_samples is not None else mag_width_fixed,  # noqa
        cz_lim=None,
        h=H0 / 100, r_max=r_max,
        colour_mean=c_bar if has_trgb_colour else None,
        c_star=c_star if has_trgb_colour else None,
        colour_std=w_c if has_trgb_colour else None,
        alpha_c=alpha_c if has_trgb_colour else 0.0)
    fprint(f"PPC: effective r_max = {r_max_eff:.1f} Mpc "
           f"(config r_max = {r_max:.1f} Mpc)")

    mag_sim, cz_sim, r_sim, colour_sim = _ppc_field_path(
        gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
        beta, bias, nu_cz, use_density_sigma_v, data, field_index,
        colour_model,
        which_sel, mag_min_fixed, mag_lim_samples, mag_lim_fixed,
        mag_width_samples, mag_width_fixed,
        e_mag_obs_all, e_czcmb_all,
        r_min, r_max_eff, r2mu, r2z, n_ppc, n_hosts, progress,
        use_reconstruction=use_reconstruction,
        cz_lim_samples=cz_lim_samples, cz_lim_fixed=cz_lim_fixed,
        cz_width_samples=cz_width_samples, cz_width_fixed=cz_width_fixed)

    out = {
        "mag_sim": mag_sim,
        "cz_sim": cz_sim,
        "r_sim": r_sim,
        "mag_obs": mag_obs,
        "cz_obs": cz_obs,
    }
    if colour_sim is not None:
        out["colour_sim"] = colour_sim
        out["colour_obs"] = colour_dered_all
    return out


###############################################################################
#                    Field-based PPC path (3D density)                        #
###############################################################################


def _ppc_field_path(gen, config, H0, M_TRGB, c_star, sigma_int, sigma_v, Vext,
                    beta, bias, nu_cz, use_density_sigma_v, data,
                    field_index, colour_model,
                    which_sel, mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                    mag_width_samples, mag_width_fixed,
                    e_mag_obs_all, e_czcmb_all,
                    r_min, r_max, r2mu, r2z, n_ppc, n_hosts, progress,
                    use_reconstruction=True, pool=None, pool_label=None,
                    cz_lim_samples=None, cz_lim_fixed=None,
                    cz_width_samples=None, cz_width_fixed=None):
    """PPC using a 3D pool; use rho=1 and v_los=0 for homogeneous PPC."""
    n_post = len(H0)
    pool_size = max(n_ppc * 100, 500_000)
    field_loader = None
    field_evaluator = None
    b_min = _galactic_latitude_cut_from_config(config)
    if use_reconstruction:
        field_name, base_field_config = _field_name_config(config)
        selected_field_index = _selected_field_index(
            data, gen, field_index=field_index)
        field_pool_label = (
            f"field={field_name}[{selected_field_index}], full field pool")
    else:
        field_name = None
        base_field_config = None
        selected_field_index = None
        field_pool_label = "homogeneous"
    if b_min is not None and b_min > 0.0:
        field_pool_label += f", |b| >= {b_min:g} deg"

    def build_pool():
        nonlocal field_loader, field_evaluator
        if not use_reconstruction:
            r_sphere = r_max * float(np.max(H0) / 100)
            new_pool = _uniform_density_pool(gen, r_sphere, pool_size)
            new_pool = _apply_galactic_latitude_cut_to_pool(new_pool, b_min)
            return new_pool, field_pool_label

        first_field_pool = field_evaluator is None
        if first_field_pool:
            field_config = dict(base_field_config)
            field_config.setdefault("nsim", selected_field_index)
            field_loader = name2field_loader(field_name)(**field_config)
            fprint("PPC: preparing field evaluator once for "
                   f"{field_name}[{selected_field_index}].")
            field_evaluator = build_field_pool_evaluator(
                field_loader,
                density_divisor=_density_divisor(field_name),
                field_smoothing_scale=field_smoothing_scale_from_config(
                    config),
                velocity_field_smoothing_scale=(
                    velocity_field_smoothing_scale_from_config(config)))
        r_sphere = r_max * (float(np.max(H0)) / 100)
        new_pool = build_field_pool(
            field_loader, r_sphere, pool_size, gen,
            density_divisor=_density_divisor(field_name),
            field_smoothing_scale=field_smoothing_scale_from_config(config),
            velocity_field_smoothing_scale=(
                velocity_field_smoothing_scale_from_config(config)),
            field_evaluator=field_evaluator,
            verbose=False)
        new_pool["base_weight"] = None
        new_pool = _apply_galactic_latitude_cut_to_pool(new_pool, b_min)
        new_pool["rho_support"] = _rho_support(new_pool["rho"])
        if first_field_pool:
            fprint("PPC: field evaluator ready; first pool has "
                   f"{len(new_pool['r_h'])} positions and refills reuse "
                   "the loaded field.")
        return new_pool, field_pool_label

    if pool is None:
        pool, pool_label = build_pool()

    n_pools_built = 1

    h_post = H0 / 100

    def reset_pool_state(pool, pool_label):
        if pool["base_weight"] is not None:
            raise ValueError(
                "PPC pool positions are sampled without replacement; "
                "weighted base pools are not supported.")
        n_pool = len(pool["r_h"])
        return {
            "pool": pool,
            "label": pool_label,
            "r_h": pool["r_h"],
            "rho": pool["rho"],
            "v_los": pool["v_los"],
            "rhat": pool["rhat_icrs"],
            "n": n_pool,
            "bias_upper": _bias_upper_bound(
                pool["rho_support"], bias, np.arange(n_post)),
            "order": gen.permutation(n_pool),
            "cursor": 0,
            "used": 0,
        }

    pool_state = reset_pool_state(pool, pool_label)
    total_pool_positions_used = 0

    # Vectorized rejection-sampling loop
    collected_mag = []
    collected_cz = []
    collected_r = []
    collected_colour = []
    n_accepted = 0
    # Adapt to low final acceptance after bias and selection rejection.
    min_batch_size = 1000
    max_batch_size = 250_000
    acceptance_rate = 1.0e-3

    fprint(f"PPC: generating {n_ppc} galaxies "
           f"(n_post={n_post}, n_hosts={n_hosts}, "
           f"pool={pool_state['n']} without replacement, "
           f"{pool_state['label']})")

    with tqdm(total=n_ppc, desc="TRGB PPC", unit="gal",
              disable=not progress) as pbar:
        while n_accepted < n_ppc:
            n_remaining_pool = pool_state["n"] - pool_state["cursor"]
            if n_remaining_pool <= 0:
                total_pool_positions_used += pool_state["used"]
                pool, pool_label = build_pool()
                n_pools_built += 1
                pool_state = reset_pool_state(pool, pool_label)
                acceptance_rate = max(acceptance_rate, 1.0e-5)
                if progress:
                    pbar.set_postfix(pool=n_pools_built, refresh=False)
                else:
                    msg = ("PPC: regenerated unique field-position pool "
                           f"#{n_pools_built} after accepting "
                           f"{n_accepted}/{n_ppc} galaxies "
                           f"({pool_state['label']}).")
                    fprint(msg)
                n_remaining_pool = pool_state["n"]
            n_need = n_ppc - n_accepted
            target_accepts = max(64, int(np.ceil(2.0 * n_need)))
            batch = int(np.ceil(
                target_accepts / max(acceptance_rate, 1.0e-5)))
            batch = min(max_batch_size, max(min_batch_size, batch))
            batch = min(batch, n_remaining_pool)

            # Draw posterior sample and pool indices
            idx_post = gen.integers(0, n_post, batch)
            start = pool_state["cursor"]
            idx_pool = pool_state["order"][start: start + batch]
            pool_state["cursor"] += batch
            pool_state["used"] += batch

            # Pool values
            r_h = pool_state["r_h"][idx_pool]
            rho = pool_state["rho"][idx_pool]
            v_los = pool_state["v_los"][idx_pool]
            rhat = pool_state["rhat"][idx_pool]

            # Posterior values
            h = h_post[idx_post]
            M = M_TRGB[idx_post]
            cs = c_star[idx_post]
            if colour_model["has_colour"]:
                ac = colour_model["alpha_c"][idx_post]
                c0 = colour_model["c_bar"][idx_post]
                wc = colour_model["w_c"][idx_post]
            sint = sigma_int[idx_post]
            if use_density_sigma_v:
                sv_low = sigma_v[0][idx_post]
                sv_high = sigma_v[1][idx_post]
                sv_log_rho_t = sigma_v[2][idx_post]
                sv_k = sigma_v[3][idx_post]
            else:
                sv = sigma_v[idx_post]
            Vext_cand = Vext[idx_post]
            bt = beta[idx_post]

            # Distance cut: r_Mpc = r_h / h
            r_Mpc = r_h / h
            in_range = (r_Mpc >= r_min) & (r_Mpc <= r_max)

            # Density rejection
            weight = _bias_values(rho, bias, idx_post)
            weight_max = pool_state["bias_upper"][idx_post]
            accept_bias = gen.random(batch) < np.minimum(
                weight / weight_max, 1.0)

            accept = in_range & accept_bias
            if not np.any(accept):
                acceptance_rate *= 0.5
                continue

            # Apply mask
            idx_post_acc = idx_post[accept]
            r_Mpc = r_Mpc[accept]
            h_acc = h[accept]
            M_acc = M[accept]
            cs_acc = cs[accept]
            if colour_model["has_colour"]:
                ac_acc = ac[accept]
                c0_acc = c0[accept]
                wc_acc = wc[accept]
            sint_acc = sint[accept]
            if use_density_sigma_v:
                sv_acc = _sigma_v_from_density(
                    rho[accept], sv_low[accept], sv_high[accept],
                    sv_log_rho_t[accept], sv_k[accept])
            else:
                sv_acc = sv[accept]
            Vext_acc = Vext_cand[accept]
            bt_acc = bt[accept]
            v_los_acc = v_los[accept]
            rhat_acc = rhat[accept]
            nu_acc = None if nu_cz is None else nu_cz[idx_post_acc]
            n_batch = len(r_Mpc)

            # Distance modulus
            mu = np.asarray(r2mu(r_Mpc, h=h_acc))

            # Measurement errors and colour population draw.
            e_mag = _sample_empirical(gen, e_mag_obs_all, n_batch)
            e_cz = _sample_empirical(gen, e_czcmb_all, n_batch)
            if colour_model["has_colour"]:
                e_colour = _sample_empirical(
                    gen, colour_model["e_colour"], n_batch)
                colour_true = gen.normal(
                    c0_acc, np.clip(wc_acc, 1e-6, None))
                colour_obs = gen.normal(colour_true, e_colour)
                colour_term = ac_acc * (colour_true - cs_acc)
            else:
                colour_obs = None
                colour_term = 0.0

            # Apparent magnitude with colour standardisation
            sigma_mag = np.sqrt(e_mag**2 + sint_acc**2)
            mag_sim = gen.normal(
                M_acc + colour_term + mu, sigma_mag)

            # Redshift with peculiar velocity from field
            z_cosmo = np.asarray(r2z(r_Mpc, h=h_acc))
            Vext_rad = np.sum(Vext_acc * rhat_acc, axis=1)
            Vpec = bt_acc * v_los_acc + Vext_rad

            cz_pred = SPEED_OF_LIGHT * (
                (1 + z_cosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)
            sigma_cz = np.sqrt(e_cz**2 + sv_acc**2)
            cz_sim = _draw_cz(gen, cz_pred, sigma_cz, nu=nu_acc)

            # Selection
            accept_sel = _apply_selection_ppc(
                mag_sim, which_sel, idx_post_acc,
                mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                mag_width_samples, mag_width_fixed, gen,
                cz_sim=cz_sim, cz_lim_samples=cz_lim_samples,
                cz_lim_fixed=cz_lim_fixed, cz_width_samples=cz_width_samples,
                cz_width_fixed=cz_width_fixed)

            n_new = int(np.sum(accept_sel))
            collected_mag.append(mag_sim[accept_sel])
            collected_cz.append(cz_sim[accept_sel])
            collected_r.append(r_Mpc[accept_sel])
            if colour_obs is not None:
                collected_colour.append(colour_obs[accept_sel])
            n_accepted += n_new
            batch_acceptance_rate = n_new / batch
            acceptance_rate = (
                0.5 * acceptance_rate + 0.5 * batch_acceptance_rate)
            pbar.update(min(n_new, n_ppc - pbar.n))

    mag_sim = np.concatenate(collected_mag)[:n_ppc]
    cz_sim = np.concatenate(collected_cz)[:n_ppc]
    r_sim = np.concatenate(collected_r)[:n_ppc]
    if collected_colour:
        colour_sim = np.concatenate(collected_colour)[:n_ppc]
    else:
        colour_sim = None
    total_pool_positions_used += pool_state["used"]
    fprint(f"PPC: generated {n_ppc} simulated galaxies "
           f"using {total_pool_positions_used} unique pool positions "
           f"across {n_pools_built} pool(s) ({pool_state['label']}).")
    return mag_sim, cz_sim, r_sim, colour_sim


###############################################################################
#                        Selection helper                                     #
###############################################################################


def _apply_selection_ppc(mag_sim, which_sel, idx_post,
                         mag_min_fixed, mag_lim_samples, mag_lim_fixed,
                         mag_width_samples, mag_width_fixed, gen,
                         cz_sim=None, cz_lim_samples=None, cz_lim_fixed=None,
                         cz_width_samples=None, cz_width_fixed=None):
    """Apply selection function, returning boolean mask."""
    n = len(mag_sim)
    if which_sel in ("TRGB_magnitude", "TRGB_magnitude_redshift"):
        ml = mag_lim_samples[idx_post] \
            if mag_lim_samples is not None else mag_lim_fixed
        mw = mag_width_samples[idx_post] \
            if mag_width_samples is not None else mag_width_fixed
        p_sel = norm.cdf((ml - mag_sim) / mw)
        if mag_min_fixed is not None:
            p_sel -= norm.cdf((mag_min_fixed - mag_sim) / mw)
        if which_sel == "TRGB_magnitude_redshift":
            cl = cz_lim_samples[idx_post] \
                if cz_lim_samples is not None else cz_lim_fixed
            cw = cz_width_samples[idx_post] \
                if cz_width_samples is not None else cz_width_fixed
            if cl is not None and cw is not None:
                p_sel = p_sel * norm.cdf((cl - cz_sim) / cw)
        p_sel = np.clip(p_sel, 0.0, 1.0)
        return gen.random(n) < p_sel
    return np.ones(n, dtype=bool)


###############################################################################
#                            PPC plotting                                     #
###############################################################################


def _hist_contour_levels(hist, enclosed=(0.68, 0.95)):
    """Return histogram levels enclosing the requested probability masses."""
    vals = np.asarray(hist, dtype=float).ravel()
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return np.array([])

    vals = np.sort(vals)[::-1]
    cdf = np.cumsum(vals)
    cdf /= cdf[-1]
    levels = []
    for p in enclosed:
        idx = min(np.searchsorted(cdf, p), len(vals) - 1)
        levels.append(vals[idx])
    return np.unique(np.sort(levels))


def _plot_2d_contours(ax, x, y, bins, color, linestyle, label):
    """Plot smoothed 2D histogram contours for one sample."""
    from scipy.ndimage import gaussian_filter

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
    hist = gaussian_filter(hist.astype(float), sigma=1.0)
    levels = _hist_contour_levels(hist)
    levels = levels[(levels > np.min(hist)) & (levels < np.max(hist))]
    if len(levels) == 0:
        return None

    xmid = 0.5 * (xedges[:-1] + xedges[1:])
    ymid = 0.5 * (yedges[:-1] + yedges[1:])
    ax.contour(xmid, ymid, hist.T, levels=levels, colors=color,
               linestyles=linestyle, linewidths=1.5)
    return ax.plot([], [], color=color, linestyle=linestyle,
                   linewidth=1.5, label=label)[0]


def _format_ppc_pvalue(pvalue):
    """Format a KS-test p-value for compact plot annotations."""
    if pvalue < 1e-3:
        return r"p<10^{-3}"
    return f"p={pvalue:.3f}"


def _plot_trgb_ppc_mnras(ppc, fname, ks_mag, ks_cz):
    """Plot a two-column MNRAS-style PPC figure."""
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401

    mag_sim = ppc["mag_sim"]
    cz_sim = ppc["cz_sim"]
    mag_obs = ppc["mag_obs"]
    cz_obs = ppc["cz_obs"]

    sim_color = "#2C7FB8"
    obs_color = "0.08"
    rc = {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
    }

    with plt.style.context("science"):
        with plt.rc_context(rc):
            fig = plt.figure(figsize=(7.09, 4.35),
                             constrained_layout=True)
            gs = fig.add_gridspec(2, 2, height_ratios=(0.9, 1.25))
            ax_mag = fig.add_subplot(gs[0, 0])
            ax_cz = fig.add_subplot(gs[0, 1])
            ax_joint = fig.add_subplot(gs[1, :])

            bins_mag = np.linspace(
                min(mag_obs.min(), mag_sim.min()) - 0.5,
                max(mag_obs.max(), mag_sim.max()) + 0.5,
                36)
            ax_mag.hist(mag_sim, bins=bins_mag, density=True,
                        histtype="stepfilled", color=sim_color, alpha=0.22,
                        edgecolor=sim_color, linewidth=0.9, label="PPC")
            ax_mag.hist(mag_obs, bins=bins_mag, density=True,
                        histtype="step", color=obs_color, linewidth=1.0,
                        label="Observed")
            ax_mag.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
            ax_mag.set_ylabel("Probability density")
            ax_mag.text(
                0.04, 0.94,
                rf"KS ${_format_ppc_pvalue(ks_mag.pvalue)}$",
                transform=ax_mag.transAxes, ha="left", va="top")
            ax_mag.legend(frameon=False, loc="upper right")

            bins_cz = np.linspace(
                min(cz_obs.min(), cz_sim.min()) - 200,
                max(cz_obs.max(), cz_sim.max()) + 200,
                36)
            ax_cz.hist(cz_sim, bins=bins_cz, density=True,
                       histtype="stepfilled", color=sim_color, alpha=0.22,
                       edgecolor=sim_color, linewidth=0.9, label="PPC")
            ax_cz.hist(cz_obs, bins=bins_cz, density=True, histtype="step",
                       color=obs_color, linewidth=1.0, label="Observed")
            ax_cz.set_xlabel(r"$cz_{\rm CMB}$ [km s$^{-1}$]")
            ax_cz.set_ylabel("Probability density")
            ax_cz.text(
                0.04, 0.94,
                rf"KS ${_format_ppc_pvalue(ks_cz.pvalue)}$",
                transform=ax_cz.transAxes, ha="left", va="top")

            bins_2d = (
                np.linspace(min(mag_obs.min(), mag_sim.min()) - 0.5,
                            max(mag_obs.max(), mag_sim.max()) + 0.5, 34),
                np.linspace(min(cz_obs.min(), cz_sim.min()) - 200,
                            max(cz_obs.max(), cz_sim.max()) + 200, 34),
            )
            handles = [
                _plot_2d_contours(
                    ax_joint, mag_sim, cz_sim, bins_2d, sim_color, "-",
                    "PPC"),
            ]
            obs_handle = ax_joint.scatter(
                mag_obs, cz_obs, s=12, marker="o", facecolor="white",
                edgecolor=obs_color, linewidth=0.6, zorder=3,
                label="Observed")
            handles = [h for h in handles if h is not None]
            ax_joint.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
            ax_joint.set_ylabel(r"$cz_{\rm CMB}$ [km s$^{-1}$]")
            handles.append(obs_handle)
            if handles:
                labels = [handle.get_label() for handle in handles]
                ax_joint.legend(handles, labels, frameon=False,
                                loc="upper left")

            for ax in (ax_mag, ax_cz, ax_joint):
                ax.tick_params(direction="in", which="both", top=True,
                               right=True)

            fig.savefig(fname, dpi=500, bbox_inches="tight")
            plt.close(fig)


def plot_trgb_ppc(ppc, fname, *, mnras=False):
    """Plot 3-panel PPC comparison: 1D histograms and 2D contours.

    Parameters
    ----------
    ppc : dict
        Output of ``generate_trgb_ppc``.
    fname : str
        Output filename for the figure.
    mnras : bool
        If true, use a two-column MNRAS-style layout with the SciencePlots
        ``science`` style.
    """
    import matplotlib.pyplot as plt

    mag_sim = ppc["mag_sim"]
    cz_sim = ppc["cz_sim"]
    mag_obs = ppc["mag_obs"]
    cz_obs = ppc["cz_obs"]

    ks_mag = ks_2samp(mag_obs, mag_sim)
    ks_cz = ks_2samp(cz_obs, cz_sim)

    if mnras:
        _plot_trgb_ppc_mnras(ppc, fname, ks_mag, ks_cz)
        fprint(f"PPC plot saved to {fname}")
        return {
            "ks_mag_statistic": float(ks_mag.statistic),
            "ks_mag_pvalue": float(ks_mag.pvalue),
            "ks_cz_statistic": float(ks_cz.statistic),
            "ks_cz_pvalue": float(ks_cz.pvalue),
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: magnitude histogram
    ax = axes[0]
    bins_mag = np.linspace(
        min(mag_obs.min(), mag_sim.min()) - 0.5,
        max(mag_obs.max(), mag_sim.max()) + 0.5,
        40)
    ax.hist(mag_sim, bins=bins_mag, density=True, alpha=0.5,
            color="C0", label="PPC")
    ax.hist(mag_obs, bins=bins_mag, density=True, histtype="step",
            color="k", linewidth=1.5, label="Observed")
    ax.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95, f"KS $p = {ks_mag.pvalue:.3f}$",
            transform=ax.transAxes, va="top", fontsize=8)

    # Panel 2: cz histogram
    ax = axes[1]
    bins_cz = np.linspace(
        min(cz_obs.min(), cz_sim.min()) - 200,
        max(cz_obs.max(), cz_sim.max()) + 200,
        40)
    ax.hist(cz_sim, bins=bins_cz, density=True, alpha=0.5,
            color="C0", label="PPC")
    ax.hist(cz_obs, bins=bins_cz, density=True, histtype="step",
            color="k", linewidth=1.5, label="Observed")
    ax.set_xlabel(r"$cz_{\rm CMB}$ [km/s]")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.text(0.05, 0.95, f"KS $p = {ks_cz.pvalue:.3f}$",
            transform=ax.transAxes, va="top", fontsize=8)

    # Panel 3: 2D distribution contours
    ax = axes[2]
    bins_2d = (
        np.linspace(min(mag_obs.min(), mag_sim.min()) - 0.5,
                    max(mag_obs.max(), mag_sim.max()) + 0.5, 35),
        np.linspace(min(cz_obs.min(), cz_sim.min()) - 200,
                    max(cz_obs.max(), cz_sim.max()) + 200, 35),
    )
    handles = [
        _plot_2d_contours(ax, mag_sim, cz_sim, bins_2d, "C0", "-", "PPC"),
        _plot_2d_contours(ax, mag_obs, cz_obs, bins_2d, "k", "--",
                          "Observed"),
    ]
    handles = [h for h in handles if h is not None]
    ax.set_xlabel(r"$m_{\rm TRGB}$ [mag]")
    ax.set_ylabel(r"$cz_{\rm CMB}$ [km/s]")
    if handles:
        ax.legend(handles=handles, fontsize=8)

    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    fprint(f"PPC plot saved to {fname}")
    return {
        "ks_mag_statistic": float(ks_mag.statistic),
        "ks_mag_pvalue": float(ks_mag.pvalue),
        "ks_cz_statistic": float(ks_cz.statistic),
        "ks_cz_pvalue": float(ks_cz.pvalue),
    }


def plot_trgb_ppc_distance(ppc, fname, *, mnras=False):
    """Plot the distance distribution of retained PPC galaxies."""
    import matplotlib.pyplot as plt
    if mnras:
        import scienceplots  # noqa: F401

    r_sim = np.asarray(ppc["r_sim"])
    q16, q50, q84 = np.percentile(r_sim, [16, 50, 84])
    bins = np.linspace(0.0, max(r_sim.max() * 1.05, q84 * 1.2), 36)

    context = plt.style.context("science") if mnras else plt.style.context([])
    rc = {
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    } if mnras else {}

    with context:
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(3.42, 2.35))
            ax.hist(r_sim, bins=bins, density=True, histtype="stepfilled",
                    color="#2C7FB8", alpha=0.24, edgecolor="#2C7FB8",
                    linewidth=0.9)
            ax.axvline(q50, color="0.08", linewidth=0.9,
                       label=rf"median ${q50:.1f}$ Mpc")
            ax.axvspan(q16, q84, color="0.08", alpha=0.10,
                       label=r"$16$--$84\%$")
            ax.set_xlabel(r"Distance [Mpc]")
            ax.set_ylabel("Probability density")
            ax.legend(frameon=False, loc="upper left")
            ax.tick_params(direction="in", which="both", top=True,
                           right=True)
            fig.tight_layout()
            fig.savefig(fname, dpi=500 if mnras else 200,
                        bbox_inches="tight")
            plt.close(fig)

    fprint(f"PPC distance plot saved to {fname}")
    return {
        "r_median": float(q50),
        "r_p16": float(q16),
        "r_p84": float(q84),
    }
