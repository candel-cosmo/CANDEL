"""CH0 JWST-forecast mock generator and per-host redshift windows."""
import numpy as np
import pytest

from candel.mock.CH0_mock import (CZ_NO_LOWER_LIMIT, draw_CH0_hosts,
                                  sigma_mu_law)
from candel.model.utils import (log_prob_integrand_sel,
                                log_prob_integrand_window_sel)

TRUTH = {"H0": 72.5, "beta": 1.0, "Vext_mag": 262.0, "Vext_ell": 302.0,
         "Vext_b": -27.0}


def test_window_without_lower_limit_matches_one_sided():
    x = np.linspace(500.0, 6000.0, 50)
    e = np.full_like(x, 150.0)
    one = log_prob_integrand_sel(x, e, 3300.0, 300.0)
    win = log_prob_integrand_window_sel(x, e, CZ_NO_LOWER_LIMIT, 3300.0,
                                        300.0)
    np.testing.assert_allclose(np.asarray(win), np.asarray(one), atol=1e-6)


@pytest.fixture
def pool():
    gen = np.random.default_rng(0)
    n = 20000
    r_h = 70.0 * gen.random(n)**(1 / 3)
    return {"r_h": r_h, "v_los": gen.normal(0, 200, n),
            "RA": gen.uniform(0, 360, n),
            "dec": np.rad2deg(np.arcsin(gen.uniform(-1, 1, n)))}


def test_draw_hosts_selection_and_error_law(pool):
    samples = [
        {"n": 35, "sigma_factor": 1.0, "cz_low": CZ_NO_LOWER_LIMIT,
         "cz_high": 3300.0},
        {"n": 10, "sigma_factor": 0.5, "cz_low": 3300.0, "cz_high": 5000.0},
    ]
    h = draw_CH0_hosts(pool, samples, TRUTH, 300.0, 150.0, 10.0,
                       np.random.default_rng(1))
    assert len(h["cz_obs"]) == 45
    assert len(np.unique(h["pool_index"])) == 45
    prim, far = slice(0, 35), slice(35, 45)
    assert h["cz_obs"][prim].max() < 3300 + 4 * 300
    assert 3300 - 4 * 300 < h["cz_obs"][far].min()
    assert h["cz_obs"][far].max() < 5000 + 4 * 300
    # The error follows the true distance, not the observed redshift.
    d_L = 10**((h["mu_true"] - 25) / 5)
    np.testing.assert_allclose(h["sigma_mu"][prim], sigma_mu_law(d_L[prim]))
    np.testing.assert_allclose(h["sigma_mu"][far],
                               sigma_mu_law(d_L[far], 0.5))
