
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from numpyro import handlers

from candel_trgb.model import TRGBModel
from candel_trgb.ppc import (_apply_selection_ppc, _bias_upper_bound,
                             sky_exposure_from_posterior_theta)


def test_trgb_ppc_bias_upper_bound_includes_quadratic_peak():
    bias = {
        "which_bias": "quadratic",
        "b1": np.array([0.0]),
        "b2": np.array([-1.0]),
    }

    bound = _bias_upper_bound(
        np.array([0.0, 2.0]), bias, np.array([0]))

    assert bound[0] > 1.0


def test_trgb_model_sky_exposure_applies_theta_to_hosts_and_volume():
    model = object.__new__(TRGBModel)
    model.TRGB_sky_exposure_n_pix = 4
    model.TRGB_sky_exposure_n_support = 3
    model.TRGB_sky_exposure_kappa = 9.0
    model._TRGB_sky_exposure_support_pix = jnp.array([0, 2, 3])
    model._TRGB_sky_exposure_host_pix = jnp.array([3, 0, 2])

    log_S_pix = jnp.array([[np.log(2.0), -jnp.inf,
                            np.log(5.0), np.log(7.0)]])
    theta_support = jnp.array([0.2, 0.3, 0.5])
    theta_full = jnp.array([0.2, 0.0, 0.3, 0.5])
    fn = handlers.seed(
        handlers.substitute(
            model._sample_TRGB_sky_exposure,
            data={"TRGB_sky_exposure_theta": theta_support}),
        rng_seed=0)

    log_S, host_log_theta, log_theta = fn(log_S_pix)

    np.testing.assert_allclose(
        np.asarray(log_S),
        np.asarray(logsumexp(log_S_pix + jnp.log(theta_full)[None, :],
                             axis=-1)))
    np.testing.assert_allclose(
        np.asarray(host_log_theta),
        np.asarray(jnp.log(theta_full)[model._TRGB_sky_exposure_host_pix]))
    np.testing.assert_array_equal(np.isneginf(np.asarray(log_theta)),
                                  np.array([False, True, False, False]))


def test_trgb_model_sky_exposure_uniform_theta_cancels_normalization():
    model = object.__new__(TRGBModel)
    model.TRGB_sky_exposure_n_pix = 4
    model.TRGB_sky_exposure_n_support = 2
    model.TRGB_sky_exposure_kappa = 8.0
    model._TRGB_sky_exposure_support_pix = jnp.array([1, 3])
    model._TRGB_sky_exposure_host_pix = jnp.array([1, 3, 3])

    log_S_pix = jnp.array([[-jnp.inf, np.log(4.0), -jnp.inf, np.log(6.0)]])
    theta_support = jnp.array([0.5, 0.5])
    fn = handlers.seed(
        handlers.substitute(
            model._sample_TRGB_sky_exposure,
            data={"TRGB_sky_exposure_theta": theta_support}),
        rng_seed=0)

    log_S, host_log_theta, _ = fn(log_S_pix)

    log_S_no_sky = logsumexp(log_S_pix, axis=-1)
    np.testing.assert_allclose(
        np.asarray(log_S),
        np.asarray(log_S_no_sky - np.log(model.TRGB_sky_exposure_n_support)))
    np.testing.assert_allclose(
        np.asarray(host_log_theta),
        np.full(3, -np.log(model.TRGB_sky_exposure_n_support)))


def test_trgb_model_sky_exposure_marginalizes_over_field_realizations():
    """log_S_pix is (num_fields, n_pix): the sky-exposure baseline fraction
    must reflect all field realizations, not just the first, and the
    single-field case must reduce to the old field-0-only behaviour."""
    model = object.__new__(TRGBModel)
    model.TRGB_sky_exposure_n_pix = 4
    model.TRGB_sky_exposure_n_support = 3
    model.TRGB_sky_exposure_kappa = 9.0
    model._TRGB_sky_exposure_support_pix = jnp.array([0, 2, 3])
    model._TRGB_sky_exposure_host_pix = jnp.array([3, 0, 2])

    log_S_pix_multi = jnp.array([
        [np.log(2.0), -jnp.inf, np.log(5.0), np.log(7.0)],
        [np.log(100.0), -jnp.inf, np.log(1.0), np.log(1.0)],
    ])
    with handlers.trace() as tr, handlers.seed(rng_seed=0):
        model._sample_TRGB_sky_exposure(log_S_pix_multi)
    q = np.asarray(tr["TRGB_sky_exposure_baseline_fraction"]["value"])

    log_S_marg = logsumexp(log_S_pix_multi, axis=0)
    q_expected = np.asarray(jnp.exp(log_S_marg - logsumexp(log_S_marg)))
    q_field0_only = np.asarray(
        jnp.exp(log_S_pix_multi[0] - logsumexp(log_S_pix_multi[0])))

    np.testing.assert_allclose(q, q_expected, atol=1e-6)
    assert not np.allclose(q, q_field0_only, atol=1e-3), (
        "baseline fraction must use both field realizations, not just "
        "field 0")

    # single-field case must be unchanged (regression safety)
    with handlers.trace() as tr_single, handlers.seed(rng_seed=0):
        model._sample_TRGB_sky_exposure(log_S_pix_multi[:1])
    q_single = np.asarray(
        tr_single["TRGB_sky_exposure_baseline_fraction"]["value"])
    np.testing.assert_allclose(q_single, q_field0_only, atol=1e-6)


def test_trgb_ppc_sky_exposure_from_posterior_theta_uses_baseline_support():
    theta_samples = np.zeros((2, 12), dtype=float)
    theta_samples[:, 0] = [0.2, 0.7]
    theta_samples[:, 1] = [0.3, 0.2]
    theta_samples[:, 3] = [0.5, 0.1]
    baseline_fraction = np.zeros((2, 12), dtype=float)
    baseline_fraction[:, 0] = 0.25
    baseline_fraction[:, 3] = 0.75

    exposure = sky_exposure_from_posterior_theta(
        theta_samples, np.array([]), np.array([]), nside=1, kappa=12.0,
        baseline_fraction=baseline_fraction)

    assert exposure["n_support"] == 2
    assert exposure["baseline_fraction"][1] == 0.0
    assert np.all(exposure["theta_samples"][:, 1] == 0.0)
    np.testing.assert_allclose(
        np.sum(exposure["theta_samples"], axis=1),
        np.ones(theta_samples.shape[0]))


def test_trgb_ppc_selection_soft_lower_edge_is_not_hard_cut():
    class ZeroRandom:
        def random(self, n):
            return np.zeros(n)

    keep = _apply_selection_ppc(
        np.array([22.0]), "TRGB_magnitude", np.array([0]),
        22.1, None, 26.0, None, 0.5, ZeroRandom())

    assert keep[0]
