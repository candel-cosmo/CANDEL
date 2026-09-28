"""Reusable functions for mock MW Cepheid tests.

Provides a single mock data generator (`generate_one_campaign`) shared by
both the simplified photometric-parallax model and the full forward model.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.stats import norm

# ---------------------------------------------------------------------------
# True parameters (canonical source of truth)
# ---------------------------------------------------------------------------
TRUE_PARAMS = {
    "M_H_1": -5.90,
    "b_W": -3.30,
    "Z_W": -0.22,
    "delta_pi": -0.014,
    "sigma_int": 0.06,
    "f_pi": 1.0,
}

EPSILON_OH = 0.06  # fixed metallicity measurement uncertainty

# Convenience constants for simplified model diagnostics and plotting.
MWH_true = TRUE_PARAMS["M_H_1"]
bW_true = TRUE_PARAMS["b_W"]
ZW_true = TRUE_PARAMS["Z_W"]
delta_pi_true = TRUE_PARAMS["delta_pi"]
sigma_int = TRUE_PARAMS["sigma_int"]

sigma_m_obs = 0.028
sigma_varpi_obs = 0.019

# Simplified model name mapping
TRUE_VALS = {
    "MWH": TRUE_PARAMS["M_H_1"],
    "bW": TRUE_PARAMS["b_W"],
    "ZW": TRUE_PARAMS["Z_W"],
    "delta_pi": TRUE_PARAMS["delta_pi"],
}

# Default per-campaign configurations. Selection: soft parallax cut
# Phi((varpi_obs - varpi_cut) / varpi_width), hard mW_obs < mW_max and
# logP >= logP_min; a None threshold disables that cut.
DEFAULT_CONFIGS = {
    "C22": dict(
        N_parent=2000, d_min=0.3, d_max=10.0, dist_k=2,
        sigma_m_obs=sigma_m_obs, sigma_pi_obs=sigma_varpi_obs,
        mu_logP=0.8, sigma_logP=0.3, mu_OH=0.0, sigma_OH=0.15,
        varpi_cut=None, mW_max=6.5, logP_min=np.log10(8),
    ),
    "C27": dict(
        N_parent=100, d_min=0.3, d_max=2, dist_k=2,
        sigma_m_obs=sigma_m_obs, sigma_pi_obs=sigma_varpi_obs,
        mu_logP=0.75, sigma_logP=0.2, mu_OH=0.0, sigma_OH=0.15,
        varpi_cut=0.8, varpi_width=0.05, mW_max=None, logP_min=None,
    ),
}


# ---------------------------------------------------------------------------
# NumPyro model (simplified photometric-parallax)
# ---------------------------------------------------------------------------
def model(m_obs, varpi_obs, logP, OH, sigma_m, sigma_varpi, sigma_int,
          use_gaussian=True, varpi_cut=None):
    """NumPyro model for the photometric parallax method.

    use_gaussian: True = Gaussian, False = pure chi2,
                  "parallax_selection" = Gaussian + parallax selection
                  correction.
    """
    MWH = numpyro.sample("MWH", dist.Normal(-5.9, 1.0))
    bW = numpyro.sample("bW", dist.Normal(-3.3, 1.0))
    ZW = numpyro.sample("ZW", dist.Normal(0.0, 1.0))
    delta_pi = numpyro.sample("delta_pi", dist.Uniform(-0.1, 0.1))

    M_pred = MWH + bW * (logP - 1) + ZW * OH
    sigma_m_tot = jnp.sqrt(sigma_m**2 + sigma_int**2)
    varpi_phot = 10**(-0.2 * (m_obs - M_pred - 10))

    sigma_varpi_m = 0.2 * jnp.log(10) * varpi_phot * sigma_m_tot
    sigma_tilde = jnp.sqrt(sigma_varpi_m**2 + sigma_varpi**2)

    if use_gaussian is True or use_gaussian == "parallax_selection":
        with numpyro.plate("data", len(m_obs)):
            numpyro.sample(
                "obs",
                dist.Normal(varpi_phot - delta_pi, sigma_tilde),
                obs=varpi_obs,
            )

    if use_gaussian == "parallax_selection":
        sel_corr = jnp.sum(
            3 * jnp.log(varpi_cut + delta_pi) - 3 * jnp.log(varpi_phot))
        numpyro.factor("selection_correction", sel_corr)
    elif use_gaussian is not True:
        chi2 = jnp.sum(
            ((varpi_obs - varpi_phot + delta_pi) / sigma_tilde)**2)
        numpyro.factor("chi2", -0.5 * chi2)


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------
def apply_selection(varpi_obs, m_obs, logP, rng, cfg):
    """Build selection mask from active cuts in cfg."""
    prob = np.ones(len(varpi_obs))

    if cfg["varpi_cut"] is not None:
        prob *= norm.cdf((varpi_obs - cfg["varpi_cut"]) / cfg["varpi_width"])
    if cfg["mW_max"] is not None:
        prob *= m_obs < cfg["mW_max"]
    if cfg["logP_min"] is not None:
        prob *= logP >= cfg["logP_min"]

    return rng.uniform(size=len(varpi_obs)) < prob


def generate_one_campaign(rng, cfg, true_params=None):
    """Generate a parent sample + selection for one campaign config.

    Returns dict with keys: d_true, logP, OH_true, OH_obs, m_obs,
    varpi_obs, sigma_m, sigma_varpi, ell, b, sel.
    """
    if true_params is None:
        true_params = TRUE_PARAMS

    N = cfg["N_parent"]
    e = cfg["dist_k"] + 1
    u = rng.uniform(0, 1, N)
    d_true = (cfg["d_min"]**e + u * (cfg["d_max"]**e - cfg["d_min"]**e)
              )**(1 / e)

    # Sky coordinates
    ell = rng.uniform(0, 360, N)
    sin_b = rng.uniform(-1, 1, N)
    b = np.degrees(np.arcsin(sin_b))

    # Period and metallicity from population priors
    logP = rng.normal(cfg["mu_logP"], cfg["sigma_logP"], N)
    OH_true = rng.normal(cfg["mu_OH"], cfg["sigma_OH"], N)

    # Magnitude
    M = (true_params["M_H_1"]
         + true_params["b_W"] * (logP - 1)
         + true_params["Z_W"] * OH_true)
    mu = 5 * np.log10(d_true) + 10
    m = M + mu + rng.normal(0, true_params["sigma_int"], N)

    sm = cfg["sigma_m_obs"]
    sv = cfg["sigma_pi_obs"]
    sigma_m = np.full(N, sm)
    sigma_varpi = np.full(N, sv)
    m_obs = m + rng.normal(0, sm, N)

    # Parallax
    varpi_true = 1.0 / d_true
    varpi_obs = (varpi_true - true_params["delta_pi"]
                 + rng.normal(0, true_params["f_pi"] * sv, N))

    # Metallicity observation
    OH_obs = OH_true + rng.normal(0, EPSILON_OH, N)

    # Selection
    sel = apply_selection(varpi_obs, m_obs, logP, rng, cfg)

    return {
        "d_true": d_true, "logP": logP,
        "OH_true": OH_true, "OH_obs": OH_obs,
        "m_obs": m_obs, "varpi_obs": varpi_obs,
        "sigma_m": sigma_m, "sigma_varpi": sigma_varpi,
        "ell": ell, "b": b, "sel": sel,
    }


def generate_mock(seed, which, configs, true_params=None):
    """Generate mock dataset by concatenating campaigns in `which`.

    Returns (combined_dict, per_campaign_dict).
    """
    rng = np.random.default_rng(seed)
    per_campaign = {
        c: generate_one_campaign(rng, configs[c], true_params)
        for c in which}

    keys = list(per_campaign[which[0]].keys())
    combined = {
        k: np.concatenate([per_campaign[c][k] for c in which])
        for k in keys}
    return combined, per_campaign


# ---------------------------------------------------------------------------
# MCMC runner (simplified model)
# ---------------------------------------------------------------------------
def run_one_mock(seed, which, configs, true_vals=None, verbose=False,
                 num_chains=1, use_gaussian=True, sigma_int_val=None):
    """Generate one mock, run MCMC, return (mean - true) / std per param."""
    if true_vals is None:
        true_vals = TRUE_VALS
    if sigma_int_val is None:
        sigma_int_val = sigma_int

    tp = dict(TRUE_PARAMS)
    tp["sigma_int"] = sigma_int_val

    combined, _ = generate_mock(seed, which, configs, tp)
    sel = combined["sel"]

    vc = None
    if use_gaussian == "parallax_selection":
        cuts = [configs[c]["varpi_cut"] for c in which
                if configs[c].get("varpi_cut") is not None]
        if cuts:
            vc = cuts[0]

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=300, num_samples=1000,
                num_chains=num_chains, progress_bar=verbose,
                chain_method="parallel" if num_chains > 1 else "sequential")
    mcmc.run(
        jax.random.key(seed),
        jnp.array(combined["m_obs"][sel]),
        jnp.array(combined["varpi_obs"][sel]),
        jnp.array(combined["logP"][sel]),
        jnp.array(combined["OH_obs"][sel]),
        jnp.array(combined["sigma_m"][sel]),
        jnp.array(combined["sigma_varpi"][sel]),
        sigma_int_val, use_gaussian, vc,
    )

    if verbose:
        mcmc.print_summary()

    posterior = mcmc.get_samples()
    biases = {}
    for lab, tv in true_vals.items():
        samp = np.asarray(posterior[lab])
        biases[lab] = (samp.mean() - tv) / samp.std()
    return biases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# (campaign, likelihood) combinations run by run_mock_simple.py
SIMPLE_TASKS = [
    ("C22", "gaussian"),
    ("C22", "chi2"),
    ("C27", "gaussian"),
    ("C27", "parallax_selection"),
    ("C27", "chi2"),
]


def print_bias_table(all_results, params=tuple(TRUE_VALS)):
    """Print mean +/- std of the standardised biases per run."""
    header = f"{'Run':<28s}" + "".join(f"{p:>18s}" for p in params)
    print(header)
    print("-" * len(header))
    for label, biases in all_results.items():
        row = f"{label:<28s}"
        for p in params:
            if p in biases:
                b = biases[p]
                row += f"{f'{b.mean():+.2f} +/- {b.std():.2f}':>18s}"
            else:
                row += f"{'---':>18s}"
        print(row)

def likelihood_label(use_gaussian):
    if use_gaussian is True:
        return "gaussian"
    elif use_gaussian == "parallax_selection":
        return "parallax_selection"
    else:
        return "chi2"


def parse_likelihood(s):
    """Convert CLI string to use_gaussian argument."""
    s = s.lower().strip()
    if s == "gaussian":
        return True
    elif s == "parallax_selection":
        return "parallax_selection"
    elif s == "chi2":
        return False
    else:
        raise ValueError(f"Unknown likelihood type: {s!r}")
