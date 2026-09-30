import numpy as np
from scipy.special import ndtr

from candel.model.utils import (log_prob_integrand_sel,
                                log_prob_integrand_window_sel)


def test_log_prob_integrand_window_sel_matches_cdf_difference():
    x = np.array([21.0, 22.5, 24.0, 27.0])
    low = 22.0
    high = 25.0
    width = 0.7

    actual = np.asarray(log_prob_integrand_window_sel(
        x, 0.0, low, high, width))
    expected = np.log(
        ndtr((high - x) / width) - ndtr((low - x) / width))

    assert np.allclose(actual, expected)


def test_log_prob_integrand_window_sel_is_stable_in_positive_tail():
    actual = np.asarray(log_prob_integrand_window_sel(
        np.array([-100.0]), 0.0, 22.0, 25.0, 1.0))

    assert np.all(np.isfinite(actual))


def test_window_without_lower_limit_matches_one_sided():
    x = np.linspace(500.0, 6000.0, 50)
    e = np.full_like(x, 150.0)
    one = log_prob_integrand_sel(x, e, 3300.0, 300.0)
    win = log_prob_integrand_window_sel(x, e, -1e5, 3300.0, 300.0)
    np.testing.assert_allclose(np.asarray(win), np.asarray(one), atol=1e-6)
