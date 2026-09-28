"""Tests for the log-space Simpson integrator."""
import numpy as np
import pytest
from scipy.integrate import simpson

from candel.model.integration import ln_simpson


@pytest.mark.parametrize("axis", [-1, 0])
def test_ln_simpson_shared_1d_grid(axis):
    """A 1D grid is shared by every slice of a 2D integrand."""
    x = np.linspace(0.3, 3.0, 51) ** 1.5  # non-uniform spacing
    scale = np.array([0.5, 1.0, 2.0])[:, None]
    ln_y = 2 * np.log(x)[None, :] - x[None, :] / scale
    if axis == 0:
        ln_y = ln_y.T

    expected = np.log(simpson(np.exp(ln_y), x=x, axis=axis))
    np.testing.assert_allclose(ln_simpson(ln_y, x, axis=axis), expected,
                               rtol=1e-6)

    x_bcast = x[None, :] if axis == -1 else x[:, None]
    np.testing.assert_allclose(ln_simpson(ln_y, x_bcast, axis=axis),
                               expected, rtol=1e-6)
