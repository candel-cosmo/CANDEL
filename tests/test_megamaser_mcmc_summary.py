"""Regressions for megamaser MCMC summary performance."""
import contextlib
import io
import os
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEGAMASER_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser")
if MEGAMASER_DIR not in sys.path:
    sys.path.insert(0, MEGAMASER_DIR)

import run_maser as rm  # noqa: E402


def _direct_ess(x):
    y = x - x.mean()
    acov = np.correlate(y, y, mode="full")[x.size - 1:] / x.size
    rho = acov / acov[0]
    positive = rho[1:][rho[1:] > 0]
    first_nonpositive = np.flatnonzero(rho[1:] <= 0)
    if first_nonpositive.size:
        positive = rho[1:first_nonpositive[0] + 1]
    tau = 1.0 + 2.0 * positive.sum()
    return max(1.0, min(x.size, x.size / tau))


def _latent_summary_text(samples):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        rm._print_r_ang_summary(samples)
        rm._print_phi_summary(samples)
    return output.getvalue()


def test_fft_ess_matches_direct_autocorrelation():
    rng = np.random.default_rng(4)
    x = np.empty(1024)
    x[0] = rng.normal()
    for i in range(1, x.size):
        x[i] = 0.85 * x[i - 1] + rng.normal()

    np.testing.assert_allclose(rm._ess_1d(x), _direct_ess(x), rtol=1e-12)


def test_latent_summary_chunking_preserves_output(monkeypatch):
    rng = np.random.default_rng(5)
    samples = {
        "r_ang": np.exp(rng.normal(size=(2, 64, 11))),
        "phi": rng.uniform(-np.pi, np.pi, size=(2, 64, 11)),
    }

    monkeypatch.setattr(rm, "_LATENT_SUMMARY_SPOT_CHUNK", 11)
    expected = _latent_summary_text(samples)

    widths = []
    original_ess_rhat = rm._ess_rhat

    def tracked_ess_rhat(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            widths.append(arr.shape[-1])
        return original_ess_rhat(arr)

    monkeypatch.setattr(rm, "_LATENT_SUMMARY_SPOT_CHUNK", 3)
    monkeypatch.setattr(rm, "_ess_rhat", tracked_ess_rhat)
    actual = _latent_summary_text(samples)

    assert actual == expected
    assert widths == [3, 3, 3, 2, 3, 3, 3, 2]
