"""CPU checks for peak-partition validation orchestration."""
import os
import time
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import candel.model.model_H0_maser as maser_module  # noqa: E402
from candel.model.integration import trapz_log_weights  # noqa: E402
from candel.model.model_H0_maser import MaserDiskModel  # noqa: E402
from scripts.megamaser.convergence.convergence_utils import (  # noqa: E402
    dense_phi_reference_per_spot, dense_r_phi_reference_per_spot)
from scripts.megamaser.convergence.validate_phi_partition import (  # noqa: E402
    INTEGRATION_SCHEMES, METHODS, REFERENCE_CACHE_TTL_SECONDS,
    _add_pesce_deltas, _aggregate, _calibrate_gate, _candidate_table_row,
    _case_rankings, _case_worst, _clean_reference_cache, _gate,
    _irrelevant_decision, _load_reference_cache, _paired, _parser,
    _reference_grids, _reference_levels, _reference_metadata,
    _reflect_unit_box,
    _relevant_pass, _save_reference_cache, _scheme_overrides,
    _timing_comparison, _validate_args, de, error_statistics,
    ranking_inversions, reference_cache_key, reference_convergence)


def test_parser_defaults_to_all_galaxies_and_accepts_subset():
    parser = _parser()
    expected = list(de._MASTER_CFG["model"]["galaxies"])
    defaults = parser.parse_args([])
    assert defaults.galaxies == expected
    assert defaults.sobol_candidates == 4
    assert defaults.reference_r_levels == (5001, 10001, 20001)
    assert defaults.reference_phi_levels is None
    assert _reference_grids("NGC6264", defaults) == (
        (5001, 2501), (10001, 5001), (20001, 10001))
    assert _reference_grids("NGC4258", defaults) == (
        (5001, 50001), (10001, 100001), (20001, 200001))
    assert defaults.reference_spot_batch == 4
    assert not defaults.clean_cache
    assert parser.parse_args(
        ["--galaxies", "NGC6323"]).galaxies == ["NGC6323"]

    override = parser.parse_args([
        "--reference-r-levels", "5,9",
        "--reference-phi-levels", "7,11"])
    assert _reference_grids("NGC4258", override) == ((5, 7), (9, 11))


def test_integration_scheme_config_is_centralised():
    assert tuple(INTEGRATION_SCHEMES) == METHODS
    assert [spec["phi_integration"]
            for spec in INTEGRATION_SCHEMES.values()] == list(METHODS)


def test_scheme_settings_are_typed_whitelisted_and_duplicate_safe():
    parser = _parser()
    args = parser.parse_args([
        "--scheme-setting",
        "peak-partition.n_phi_partition_sys=257",
        "--scheme-setting", "peak-partition.K_sigma=6.5",
        "--scheme-setting", "fixed-grid.refine_r_center=false",
    ])
    assert _scheme_overrides(args.scheme_setting) == {
        "fixed-grid": {"refine_r_center": False},
        "peak-partition": {
            "n_phi_partition_sys": 257, "K_sigma": 6.5},
    }

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--scheme-setting", "peak-partition.n_phi_sys=257"])
    duplicate = parser.parse_args([
        "--scheme-setting", "peak-partition.n_phi_partition_hv=129",
        "--scheme-setting", "peak-partition.n_phi_partition_hv=65",
    ])
    with pytest.raises(ValueError, match="duplicate"):
        _validate_args(duplicate)


def test_reference_grid_levels_must_be_paired():
    args = _parser().parse_args([
        "--reference-r-levels", "3,5",
        "--reference-phi-levels", "3,5,7"])
    with pytest.raises(ValueError, match="same number"):
        _validate_args(args)


def test_paired_values_use_explicit_fixed_peak_separator():
    assert _paired([1.0, 2.0]) == "1 | 2"
    assert _paired([None, False], precision=None) == "n/a | False"


def test_pesce_deltas_use_tested_log_likelihood_by_method():
    case = {"candidates": [
        {"source_kind": "sobol", "methods": {
            "fixed-grid": {"test_total_log_likelihood": -8.0},
            "peak-partition": {"test_total_log_likelihood": -5.0},
        }},
        {"source_kind": "pesce-reid", "methods": {
            "fixed-grid": {"test_total_log_likelihood": -10.0},
            "peak-partition": {"test_total_log_likelihood": -4.0},
        }},
    ]}
    _add_pesce_deltas(case)
    assert case["candidates"][0]["methods"]["fixed-grid"][
        "delta_log_likelihood_vs_pesce"] == 2.0
    assert case["candidates"][0]["methods"]["peak-partition"][
        "delta_log_likelihood_vs_pesce"] == -1.0
    assert case["candidates"][1]["methods"]["fixed-grid"][
        "delta_log_likelihood_vs_pesce"] == 0.0

    case["candidates"].pop()
    _add_pesce_deltas(case)
    assert case["candidates"][0]["methods"]["fixed-grid"][
        "delta_log_likelihood_vs_pesce"] is None


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


def test_reference_metadata_excludes_production_policy_and_git():
    args = _parser().parse_args([
        "--scheme-setting", "peak-partition.n_phi_partition_sys=65"])
    case = {
        "source_hash": "source",
        "config_hash": "config",
        "galaxy": "NGC6264",
        "variant": "circular",
        "production_dtype": "float32",
        "names": ("D_A",),
    }
    candidate = {"values": np.asarray([100.0])}

    metadata = _reference_metadata(
        case, candidate, ((3, 3), (5, 5)), args)

    assert "objective_policies" not in metadata
    assert "git_revision" not in metadata
    assert "scheme_setting_overrides" not in metadata


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


def test_reference_cache_expires_after_two_days(tmp_path):
    metadata = {"candidate": [1.0], "grid": [5, 9]}
    path = tmp_path / f"{reference_cache_key(metadata)}.npz"
    _save_reference_cache(path, metadata, np.arange(2.0))
    stale = time.time() - REFERENCE_CACHE_TTL_SECONDS - 1.0
    os.utime(path, (stale, stale))

    cached, loaded_path = _load_reference_cache(tmp_path, metadata)

    assert cached is None
    assert loaded_path == path
    assert not path.exists()


def test_reference_cache_hit_reports_dense_ladder_skip(tmp_path, capsys):
    args = _parser().parse_args([
        "--reference-r-levels", "3,5",
        "--reference-phi-levels", "3,5",
        "--reference-tail-levels", "2",
    ])
    case = {
        "source_hash": "source",
        "config_hash": "config",
        "galaxy": "NGC6264",
        "variant": "circular",
        "production_dtype": "float32",
        "names": ("D_A",),
        "model": SimpleNamespace(n_spots=2),
    }
    candidate = {"values": np.asarray([100.0])}
    grids = ((3, 3), (5, 5))
    metadata = _reference_metadata(case, candidate, grids, args)
    path = tmp_path / f"{reference_cache_key(metadata)}.npz"
    expected = np.arange(4.0).reshape(2, 2)
    _save_reference_cache(path, metadata, expected)

    got, info = _reference_levels(
        case, candidate, grids, args, tmp_path)

    np.testing.assert_array_equal(got, expected)
    assert info["cache_hit"]
    assert info["level_seconds"] == []
    output = capsys.readouterr().out
    assert "Reference cache HIT" in output
    assert "dense ladder skipped" in output


def test_clean_reference_cache_removes_only_npz_files(tmp_path):
    np.savez(tmp_path / "first.npz", value=1)
    np.savez(tmp_path / "second.tmp.npz", value=2)
    keep = tmp_path / "keep.txt"
    keep.write_text("keep\n")

    assert _clean_reference_cache(tmp_path) == 2
    assert list(tmp_path.glob("*.npz")) == []
    assert keep.is_file()


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
                "lnorm": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
                "lnorm_a": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
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


def test_dense_2d_reference_integrates_full_radial_support_in_chunks():
    class Model:
        n_spots = 1
        _idx_sys = jnp.array([], dtype=int)
        _idx_red = jnp.array([0])
        _idx_blue = jnp.array([], dtype=int)
        _phi_subranges = {"red": ((0.0, np.pi, 3),)}

        @staticmethod
        def r_ang_range(distance):
            del distance
            return 1.0, 2.0

        @staticmethod
        def _group_has_any_accel(type_key):
            del type_key
            return False

        @staticmethod
        def _r_precompute(r_ang, idx, *args, **kwargs):
            del idx, args, kwargs
            return {
                "r_ang": r_ang,
                "lnorm": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
                "lnorm_a": jnp.zeros(r_ang.shape[0], dtype=r_ang.dtype),
            }

        @staticmethod
        def _phi_eval(r_pre, sin_phi, cos_phi):
            del cos_phi
            return -r_pre["r_ang"][..., None] + 3 * sin_phi

    phys_args = (None, None, jnp.asarray(1.0, dtype=jnp.float64))
    got = dense_r_phi_reference_per_spot(
        Model(), phys_args, {}, 101, 201, 16, 1)
    r = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(2.0), 101))
    phi = jnp.linspace(0.0, jnp.pi, 201)
    expected = jax.scipy.special.logsumexp(
        -r[:, None] + 3 * jnp.sin(phi)[None, :]
        + trapz_log_weights(r)[:, None]
        + trapz_log_weights(phi)[None, :])
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="requires float64"):
        dense_r_phi_reference_per_spot(
            Model(), (None, None, jnp.asarray(1.0, dtype=jnp.float32)),
            {}, 3, 3, 2, 1)


def test_gate_uses_floor_and_worst_deficit():
    # Near-anchor legit points: the -1000 nat floor dominates.
    assert _gate([-5.0, -4.0], -3.0) == -1000.0
    # A legit point 100 nats below the anchor: 100x deficit dominates.
    assert _gate([-100.0, -50.0], 0.0) == -10000.0
    # Worst legit deficit is always <= 0 (best anchor sits at the top).
    assert _gate([-10.0, -11.0], -10.0) == -1000.0


def test_irrelevant_decision_requires_both_deficits_and_railing():
    gate = -1000.0
    assert _irrelevant_decision(-2000.0, -2000.0, gate, True, False)
    assert _irrelevant_decision(-2000.0, -2000.0, gate, False, True)
    # Both deficits below the gate but nothing rails -> keep as relevant.
    assert not _irrelevant_decision(-2000.0, -2000.0, gate, False, False)
    # Only the production deficit clears the gate -> keep.
    assert not _irrelevant_decision(-2000.0, -500.0, gate, True, True)
    # Only the reference deficit clears the gate -> keep.
    assert not _irrelevant_decision(-500.0, -2000.0, gate, True, True)


def _anchor_result(kind, test, ref_total, converged):
    return {
        "source_kind": kind,
        "methods": {
            "fixed-grid": {
                "test_total_log_likelihood": test,
                "reference_total_log_likelihood": ref_total,
                "reference_convergence": {"converged": converged}},
            "peak-partition": {
                "test_total_log_likelihood": test - 1.0}},
    }


def test_calibration_needs_a_converged_anchor():
    unconverged = [_anchor_result("config", -10.0, -9.0, False)]
    assert _calibrate_gate(unconverged) is None


def test_calibration_gate_from_converged_anchor_and_local_cloud():
    legit = [
        _anchor_result("config", -10.0, -9.0, True),
        {"source_kind": "local-config", "methods": {
            "fixed-grid": {"test_total_log_likelihood": -30.0},
            "peak-partition": {"test_total_log_likelihood": -28.0}}},
    ]
    cal = _calibrate_gate(legit)
    # Anchor score is max(-10, -11) = -10; local-cloud score is -28.
    assert cal["anchor_score"] == -10.0
    assert cal["anchor_reference_total"] == -9.0
    assert cal["worst_legit"] == -18.0
    assert cal["gate"] == -1800.0


def test_relevant_pass_excuses_irrelevant_in_both_directions():
    passing = {"passed": True, "irrelevant": False}
    # A failing irrelevant needle cannot fail the case.
    assert _relevant_pass([passing, {"passed": False, "irrelevant": True}])
    # An all-irrelevant Sobol set never counts toward the verdict either way.
    assert _relevant_pass([
        passing,
        {"passed": False, "irrelevant": True},
        {"passed": False, "irrelevant": True}])
    # A failing RELEVANT candidate still fails the case.
    assert not _relevant_pass([{"passed": False, "irrelevant": False}])


def _method_record(test, converged=True, passed=True):
    return {
        "_reference_totals": [test, test, test],
        "_test_total": test,
        "test_total_log_likelihood": test,
        "reference_convergence": {"converged": converged},
        "reference_total_log_likelihood": test,
        "root_capacity_overflow_count": 0,
        "comparison": {
            "finite_mask_mismatches": 0,
            "absolute_total_error": 0.01,
            "max_absolute_spot_error": 0.02,
            "p99_absolute_spot_error": 0.0,
            "rms_spot_error": 0.0},
        "passed": passed,
        "spots": [],
    }


def _candidate(cid, kind, irrelevant, converged=True, passed=True,
               test=-10.0, seconds=None):
    result = {
        "id": cid, "source_kind": kind, "irrelevant": irrelevant,
        "passed": passed,
        "methods": {method: _method_record(test, converged, passed)
                    for method in METHODS}}
    if seconds is not None:
        result["production_seconds"] = seconds
    if irrelevant:
        result["irrelevance"] = {
            "gate": -1000.0, "anchor_score": -9.0,
            "anchor_reference_total": -9.0,
            "delta_production": -5.0e4, "delta_reference": -5.0e4,
            "radius_railed_fraction": 1.0, "phi_railed_fraction": 0.9}
    return result


def _irrelevant_case():
    return {
        "galaxy": "G", "variant": "circular", "passed": True,
        "candidates": [
            _candidate("config-0000", "config", irrelevant=False),
            _candidate("sobol-0000", "sobol", irrelevant=True,
                       converged=False, passed=False, test=-5.0e4)],
    }


def test_irrelevant_candidates_excluded_from_rankings_and_worst():
    case = _irrelevant_case()
    args = _parser().parse_args([])
    _case_rankings(case, args)
    _case_worst(case)
    for method in METHODS:
        for row in case["rankings"][method]["reference_consecutive"]:
            assert row["inversions"] == []
        assert case["worst"][method]["candidate"] == "config-0000"


def test_aggregate_separates_unconverged_excused():
    rows = _aggregate({"cases": [_irrelevant_case()]})
    by_source = {(row["point_source"], row["method"]): row for row in rows}
    sobol = by_source[("sobol", "fixed-grid")]
    # The failing, unconverged needle is excused, not a hard failure.
    assert sobol["irrelevant"] == 1
    assert sobol["reference_unconverged"] == 0
    assert sobol["unconverged_excused"] == 1
    assert sobol["passed"] is None
    config = by_source[("config", "fixed-grid")]
    assert config["irrelevant"] == 0
    assert config["unconverged_excused"] == 0
    assert config["passed"] is True


def test_markdown_row_shows_real_values_and_marker_for_irrelevant():
    case = _irrelevant_case()
    irrelevant = case["candidates"][1]
    row = _candidate_table_row("G", "circular", irrelevant, " \\| ")
    assert "[IRRELEVANT]" in row
    assert "SKIP" not in row
    # Real error columns are rendered, not placeholders.
    assert "0.01" in row and "0.02" in row
    relevant = case["candidates"][0]
    relevant_row = _candidate_table_row("G", "circular", relevant, " \\| ")
    assert "[IRRELEVANT]" not in relevant_row


def test_candidate_row_renders_paired_seconds_present_and_absent():
    present = _candidate(
        "config-0000", "config", irrelevant=False,
        seconds={"fixed-grid": 0.002, "peak-partition": 0.001})
    row = _candidate_table_row("G", "circular", present, " \\| ")
    assert "0.002 \\| 0.001" in row
    absent = _candidate("config-0001", "config", irrelevant=False)
    absent_row = _candidate_table_row("G", "circular", absent, " \\| ")
    assert "n/a \\| n/a" in absent_row


def test_timing_comparison_reports_peak_over_fixed_speedups():
    timing = {
        "fixed-grid": {
            "steady_evaluation_seconds": 1.0,
            "cold_compile_and_evaluate_seconds": 4.0,
            "throughput_candidates_per_second": 6.0,
        },
        "peak-partition": {
            "steady_evaluation_seconds": 0.5,
            "cold_compile_and_evaluate_seconds": 5.0,
            "throughput_candidates_per_second": 12.0,
        },
    }
    comparison = _timing_comparison(timing)
    assert comparison["steady_speedup_peak_over_fixed"] == 2.0
    assert comparison["cold_speedup_peak_over_fixed"] == 0.8
    assert comparison["throughput_ratio_peak_over_fixed"] == 2.0
