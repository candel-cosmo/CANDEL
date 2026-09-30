import numpy as np
from jax import numpy as jnp
from scipy.integrate import simpson

from candel.model.pv_utils import lp_galaxy_bias
from candel_pv.redshift2real import Redshift2Real
from candel.util import SPEED_OF_LIGHT


def _model(which_bias=None, e_zcmb=None, Vext=None,
           Vext_decay_start=None, Vext_decay_scale=None):
    los_r = np.linspace(0.1, 60.0, 301)
    density = 1 + 0.2 * np.sin(los_r / 5)[None, None, :]
    calibration = {"sigma_v": np.array([150.0]), "beta": np.array([1.0])}
    if Vext is not None:
        calibration["Vext"] = np.asarray(Vext)[None, :]
    if which_bias == "linear":
        calibration["b1"] = np.array([0.8])
    return Redshift2Real(
        RA=np.array([10.0]), dec=np.array([20.0]), zcmb=np.array([0.01]),
        e_zcmb=e_zcmb, los_r=los_r, los_density=density,
        los_velocity=np.zeros_like(density), which_bias=which_bias,
        calibration_samples=calibration, Rmin=los_r[0], Rmax=los_r[-1],
        num_rgrid=len(los_r), verbose=False,
        Vext_decay_start=Vext_decay_start,
        Vext_decay_scale=Vext_decay_scale)


def test_bias_normalization_includes_radial_measure():
    model = _model(which_bias="linear")
    r = model.los_grid_r
    log_norm = model._compute_bias_normalization(r, batch_size=1)
    delta = jnp.asarray(model._bias_interp.interp_many(r))
    log_bias = np.asarray(lp_galaxy_bias(
        delta[:, :, None, :], None,
        [jnp.asarray(model._bias_params[0])[None, None, :, None]], "linear"))
    prior = np.exp(2 * np.log(r) + log_bias - log_norm[..., None])
    np.testing.assert_allclose(simpson(prior, x=r, axis=-1), 1, rtol=2e-5)


def test_redshift_error_broadens_normalized_posterior():
    model = _model()
    z_grid, log_posterior = model(batch_size=1)
    broad = _model(e_zcmb=np.array([500 / SPEED_OF_LIGHT]))
    _, broad_log_posterior = broad(batch_size=1)

    np.testing.assert_allclose(
        simpson(np.exp(log_posterior), x=z_grid, axis=-1), 1, rtol=2e-5)
    assert (broad.posterior_summary(z_grid, broad_log_posterior)["std"]
            > model.posterior_summary(z_grid, log_posterior)["std"])
