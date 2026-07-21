"""CPU checks for peak-partition validation orchestration."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import candel.model.model_H0_maser as maser_module  # noqa: E402
from candel.model.integration import trapz_log_weights  # noqa: E402
from candel.model.model_H0_maser import MaserDiskModel  # noqa: E402
from scripts.megamaser.convergence.convergence_utils import (  # noqa: E402
    dense_phi_reference_per_spot)
from scripts.megamaser.convergence.validate_phi_partition import (  # noqa: E402
    _load_reference_cache, _reflect_unit_box, _save_reference_cache,
    error_statistics, ranking_inversions, reference_cache_key,
    reference_convergence)


def test_per_spot_gate_catches_cancelled_total_error():
    stats = error_statistics([1.0, -1.0], [0.0, 0.0])
    assert stats["absolute_total_error"] == 0.0
    assert stats["max_absolute_spot_error"] == 1.0

    verdict = reference_convergence(
        [np.zeros(2), np.array([1.0, -1.0])], 2,
        {"total_atol": 0.1, "spot_atol": 0.5, "rms_atol": 1.0})
    assert not verdict["converged"]


def test_reference_tail_and_finite_masks_are_hard_gates():
    levels = [np.array([0.0, 0.0]),
              np.array([0.2, 0.0]),
              np.array([0.2001, 0.0])]
    criteria = {"total_atol": 1e-3, "spot_atol": 1e-3,
                "rms_atol": 1e-3}
    assert reference_convergence(levels, 2, criteria)["converged"]
    assert not reference_convergence(levels, 3, criteria)["converged"]

    mismatched = [np.array([0.0, np.inf]), np.array([0.0, 1.0])]
    assert not reference_convergence(mismatched, 2, criteria)["converged"]


def test_ranking_ignores_ties_but_reports_real_inversions():
    labels = ["a", "b", "c"]
    previous = [3.0, 2.0, 2.0 + 1e-6]
    current = [1.0, 2.5, 2.0]
    inversions = ranking_inversions(previous, current, labels, atol=1e-4)
    assert [(row["candidate_a"], row["candidate_b"])
            for row in inversions] == [("a", "b"), ("a", "c")]


def test_cache_key_is_order_stable_and_value_sensitive():
    first = reference_cache_key({"candidate": [1.0, 2.0], "grid": [5, 9]})
    reordered = reference_cache_key({"grid": [5, 9],
                                     "candidate": [1.0, 2.0]})
    changed = reference_cache_key({"candidate": [1.0, 2.1],
                                   "grid": [5, 9]})
    assert first == reordered
    assert first != changed


def test_reference_cache_rejects_stale_metadata(tmp_path):
    metadata = {"candidate": [1.0, 2.0], "grid": [5, 9]}
    path = tmp_path / f"{reference_cache_key(metadata)}.npz"
    values = np.arange(4.0).reshape(2, 2)
    _save_reference_cache(path, metadata, values)
    cached, loaded_path = _load_reference_cache(tmp_path, metadata)
    np.testing.assert_array_equal(cached, values)
    assert loaded_path == path

    _save_reference_cache(path, {"candidate": [9.0]}, values)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        _load_reference_cache(tmp_path, metadata)


def test_local_cloud_reflection_stays_in_unit_box():
    got = _reflect_unit_box(np.array([-0.2, 0.2, 1.2, 2.2]))
    np.testing.assert_allclose(got, [0.2, 0.2, 0.8, 0.2])


def test_float32_phi_eval_does_not_take_float64_quadform(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("float32 entered the float64 quadratic form")

    monkeypatch.setattr(maser_module, "_neg_half_chi2_quadform", forbidden)
    model = object.__new__(MaserDiskModel)
    model._predict_on_grid = lambda *args: (
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
        jnp.zeros((1, 3), dtype=jnp.float32),
        None)
    r_pre = {
        "ecc2": None,
        "r_ang": jnp.ones(1, dtype=jnp.float32),
        "all_x": jnp.zeros(1, dtype=jnp.float32),
        "all_y": jnp.zeros(1, dtype=jnp.float32),
        "all_v_rel": jnp.zeros(1, dtype=jnp.float32),
        "var_x": jnp.ones(1, dtype=jnp.float32),
        "var_y": jnp.ones(1, dtype=jnp.float32),
        "var_v": jnp.ones(1, dtype=jnp.float32),
        "has_any_accel": False,
    }
    got = model._phi_eval(
        r_pre, jnp.zeros(3, dtype=jnp.float32),
        jnp.ones(3, dtype=jnp.float32))
    assert got.dtype == jnp.float32
    assert got.shape == (1, 3)


def test_dense_acceptance_reference_uses_partition_support():
    class Model:
        n_spots = 1
        _idx_sys = jnp.array([], dtype=int)
        _idx_red = jnp.array([0])
        _idx_blue = jnp.array([], dtype=int)
        _phi_subranges = {"red": ((0.0, np.pi, 3),)}

        @staticmethod
        def _group_has_any_accel(type_key):
            del type_key
            return False

        @staticmethod
        def _r_precompute(r_ang, idx, *args, **kwargs):
            del idx, args, kwargs
            return {
                "r_ang": r_ang,
                "lnorm": jnp.zeros_like(r_ang),
                "lnorm_a": jnp.zeros_like(r_ang),
            }

        @staticmethod
        def _phi_eval(r_pre, sin_phi, cos_phi):
            del cos_phi
            return jnp.zeros_like(r_pre["r_ang"])[..., None] + 3 * sin_phi

    model = Model()
    phys_args = (None, None, jnp.asarray(1.0, dtype=jnp.float64))
    got = dense_phi_reference_per_spot(
        model, phys_args, {}, np.ones(1), 1001, 1,
        partition_support=True)
    phi = jnp.linspace(0.0, jnp.pi, 1001, dtype=jnp.float64)
    expected = jax.scipy.special.logsumexp(
        3 * jnp.sin(phi) + trapz_log_weights(phi))
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)

    full = dense_phi_reference_per_spot(
        model, phys_args, {}, np.ones(1), 1001, 1)
    assert not np.isclose(full[0], got[0])
