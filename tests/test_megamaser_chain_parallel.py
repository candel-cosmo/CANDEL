"""Regression check for bounded parallel chains and stable progress rows."""
import contextlib
import io
import threading

import jax
import numpy as np

from candel.model import maser_blackjax as mbj


def test_multiple_chains_use_available_workers(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.delenv("PBS_NP", raising=False)
    monkeypatch.delenv("NSLOTS", raising=False)
    monkeypatch.setattr(mbj.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        mbj.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    assert mbj._chain_worker_count(5) == 2

    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    monkeypatch.setattr(
        mbj.os, "sched_getaffinity", lambda _: set(range(32)), raising=False)
    monkeypatch.setattr(mbj.os, "cpu_count", lambda: 32)
    assert mbj._chain_worker_count(28) == 8
    assert mbj._chain_worker_count(28, 12) == 12

    barrier = threading.Barrier(2)
    positions = {}

    def run_one(*args, progress_label="", progress_positions=None, **kwargs):
        barrier.wait(timeout=5)
        chain = int(progress_label.split("/")[0])
        positions[chain] = progress_positions
        return mbj.MaserBlackJaxResult(
            samples={"x": np.array([chain])},
            log_density=np.array([-chain]),
            info={},
            warmup_info={},
            parameters={"step_size": np.array(chain)},
            theta_sites=("x",),
            runtime_seconds=99.0)

    monkeypatch.setattr(mbj, "_run_blackjax_mcmc_one", run_one)
    result = mbj.run_blackjax_mcmc(
        object(), {}, jax.random.PRNGKey(0), num_chains=2,
        num_latent_burnin=1, progress_bar=False)

    assert result.chain_method == "parallel"
    assert result.chain_workers == 2
    assert positions == {1: (0, 3, 6), 2: (1, 4, 7)}
    np.testing.assert_array_equal(result.samples["x"], [[1], [2]])
    assert result.runtime_seconds < 10.0

    class Progress:
        def set_postfix(self, values, refresh=True):
            self.refresh = refresh

    progress = Progress()
    mbj._update_mcmc_progress_postfix(
        progress,
        {"theta_acceptance_rate": 0.9,
         "latent_accept_mean": 0.3,
         "reflect_accept_mean": 0.2},
        0.01)
    assert progress.refresh is False

    stream = io.StringIO()
    with contextlib.redirect_stderr(stream):
        bar = mbj._step_bar(10, True, "MCMC warmup 1/2")
        bar.n = 5
        mbj._update_mcmc_progress_postfix(
            bar,
            {"theta_acceptance_rate": 0.9,
             "latent_accept_mean": 0.3,
             "reflect_accept_mean": 0.2},
            0.01)
        bar.refresh()
        bar.close()
    assert "theta=1.00e-02" in stream.getvalue()
