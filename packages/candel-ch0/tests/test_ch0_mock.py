"""CH0 JWST-forecast mock generator and per-host redshift windows."""
import numpy as np
import pytest

from candel_ch0.mock import CZ_NO_LOWER_LIMIT, draw_CH0_hosts, sigma_mu_law

TRUTH = {"H0": 72.5, "beta": 1.0, "Vext_mag": 262.0, "Vext_ell": 302.0,
         "Vext_b": -27.0}


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
