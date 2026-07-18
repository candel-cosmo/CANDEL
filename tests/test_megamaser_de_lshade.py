"""Regressions for the sole, Pesce-unseeded megamaser DE path."""
import os
import sqlite3
import sys

import numpy as np
import pytest
import tomli


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEGAMASER_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser")
if MEGAMASER_DIR not in sys.path:
    sys.path.insert(0, MEGAMASER_DIR)

import run_de_map as de  # noqa: E402
import benchmark_de_batching as batching  # noqa: E402


DATA_SEEDS = np.array([[140.0, 5.31], [150.0, 5.33]])


def test_initial_population_accepts_only_data_seeds():
    seeds = de._initial_de_seed_points(DATA_SEEDS)

    np.testing.assert_allclose(seeds, DATA_SEEDS)
    assert seeds is not DATA_SEEDS
    seeds[0, 0] = -1.0
    assert DATA_SEEDS[0, 0] == 140.0
    assert de._initial_de_seed_points(None) is None


def test_de_is_only_lshade_and_config_has_no_hybrid_settings():
    assert de._DE_ALGORITHM == "lshade"
    assert de._DE_SEED_POLICY == "data_sobol_only"

    path = os.path.join(MEGAMASER_DIR, "config_maser.toml")
    with open(path, "rb") as f:
        optimise = tomli.load(f)["optimise"]
    assert "algorithm" not in optimise
    assert not any(key.startswith("adam_") for key in optimise)
    assert 4 <= optimise["min_pop_size"] <= optimise["pop_size"]
    assert optimise["min_pop_size"] == 128
    assert "eval_chunk" not in optimise
    assert de._CANDIDATES_PER_GPU_WAVE == 1
    assert de._DEVICE_LOCAL_BLOCK_SIZE == 8
    assert (optimise["population_reduction_evaluations"]
            >= optimise["pop_size"])


def test_de_cli_has_no_algorithm_hybrid_or_pesce_seed_switch(capsys):
    with pytest.raises(SystemExit) as exc:
        de.main(["NGC6264", "--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--de-algorithm" not in help_text
    assert "--no-pesce-seed" not in help_text
    assert "--adam-" not in help_text
    assert "--eval-chunk" not in help_text
    assert "--population-reduction-evaluations" in help_text
    assert "{median,config}" in help_text
    assert "never uses the Pesce/Reid point" in " ".join(help_text.split())


def test_de_cli_rejects_reid_initialisation(capsys):
    with pytest.raises(SystemExit) as exc:
        de.main(["NGC6264", "--init-strategy", "reid"])

    assert exc.value.code == 2
    assert "invalid choice: 'reid'" in capsys.readouterr().err


def test_production_de_has_no_candidate_vectorisation_option(capsys):
    with pytest.raises(SystemExit) as exc:
        de.main(["NGC6264", "--eval-chunk", "1"])

    assert exc.value.code == 2
    assert "unrecognized arguments: --eval-chunk 1" in capsys.readouterr().err


def test_batching_benchmark_only_varies_spots(capsys):
    parser = batching._parser()
    help_text = parser.format_help()
    assert "--eval-chunk" not in help_text
    assert parser.parse_args(
        ["UGC3789", "--spot-batch", "all"]).spot_batch is None

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["UGC3789", "--eval-chunk", "1"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --eval-chunk 1" in capsys.readouterr().err


def test_batching_benchmark_has_deterministic_sobol_fallback(tmp_path):
    class FakeDe:
        Sobol = de.Sobol

        @staticmethod
        def _layout(target, sobol_n_sigma):
            return ["a", "b", "c"], [1, 1, 1], np.zeros(3), np.ones(3)

        @staticmethod
        def results_path(*parts):
            return os.path.join(*map(str, parts))

        @staticmethod
        def _variant_suffix(model):
            return ""

    master = {
        "optimise": {"sobol_n_sigma": 5},
        "io": {"root_output": str(tmp_path)},
    }
    first = batching._checkpoint_points(
        FakeDe, object(), object(), master, "TEST", 17, 123)
    second = batching._checkpoint_points(
        FakeDe, object(), object(), master, "TEST", 17, 123)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[0].shape == (17, 3)
    assert first[0].dtype == np.float32
    assert np.all((first[0] >= 0.0) & (first[0] <= 1.0))
    assert "scrambled Sobol seed=123" in first[-1]


def test_memory_planner_only_sizes_spot_batch(monkeypatch):
    class Model:
        _n_r_local = 4
        _n_r_global = 2
        _n_sys = 8
        _n_red = 4
        _n_blue = 3
        _phi_concat = {
            key: {"sin_phi": np.empty(10)}
            for key in ("sys", "red", "blue")}

    monkeypatch.setattr(
        de, "_device_budget_bytes", lambda gpu_mem_gb=None: (1e12, "test"))
    spot_batch, budget_known, info = de._plan_de_batch(
        Model(), pop_size=2000)

    assert spot_batch is None
    assert budget_known is True
    assert "one candidate per GPU wave" in info


def test_resume_rejects_checkpoint_that_may_have_seeded_pesce(tmp_path):
    seeded = tmp_path / "de_ckpt_rmap_lshade.npz"
    np.savez(seeded, algorithm=np.asarray("lshade"))
    with np.load(seeded) as checkpoint:
        with pytest.raises(ValueError, match="no seed-policy marker"):
            de._validate_de_checkpoint_policy(checkpoint, str(seeded))

    explicit = tmp_path / "de_ckpt_rmap_lshade_nopesce.npz"
    np.savez(explicit, algorithm=np.asarray("lshade"),
             seed_policy=np.asarray("pesce_data_sobol"),
             population_schedule=np.asarray("nfe_linear"))
    with np.load(explicit) as checkpoint:
        with pytest.raises(ValueError, match="Checkpoint seed policy"):
            de._validate_de_checkpoint_policy(checkpoint, str(explicit))


def test_resume_accepts_explicit_nfe_checkpoint(tmp_path):
    explicit = tmp_path / "new.npz"
    np.savez(explicit, algorithm=np.asarray("lshade"),
             seed_policy=np.asarray("data_sobol_only"),
             population_schedule=np.asarray("nfe_linear"),
             objective_policy=np.asarray(de._DE_OBJECTIVE_POLICY))
    with np.load(explicit) as checkpoint:
        de._validate_de_checkpoint_policy(checkpoint, str(explicit))


def test_resume_rejects_legacy_generation_schedule(tmp_path):
    legacy = tmp_path / "de_ckpt_rmap_lshade_nopesce.npz"
    np.savez(legacy, algorithm=np.asarray("lshade"),
             seed_policy=np.asarray("data_sobol_only"))
    with np.load(legacy) as checkpoint:
        with pytest.raises(ValueError, match="legacy generation-linear"):
            de._validate_de_checkpoint_policy(checkpoint, str(legacy))

    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, algorithm=np.asarray("lshade"),
             seed_policy=np.asarray("data_sobol_only"),
             population_schedule=np.asarray("generation_linear"))
    with np.load(wrong) as checkpoint:
        with pytest.raises(ValueError, match="population schedule"):
            de._validate_de_checkpoint_policy(checkpoint, str(wrong))


def test_resume_rejects_legacy_objective(tmp_path):
    legacy = tmp_path / "legacy_objective.npz"
    np.savez(legacy, algorithm=np.asarray("lshade"),
             seed_policy=np.asarray("data_sobol_only"),
             population_schedule=np.asarray("nfe_linear"))
    with np.load(legacy) as checkpoint:
        with pytest.raises(ValueError, match="objective policy"):
            de._validate_de_checkpoint_policy(checkpoint, str(legacy))


def test_lshade_trials_are_reproducible_and_bounded():
    population = np.linspace(0.05, 0.95, 24).reshape(8, 3)
    fitness = np.arange(8.0)
    archive = np.empty((0, 3))
    m_f = np.full(6, 0.5)
    m_cr = np.full(6, 0.5)

    out1 = de._lshade_trials(
        population, fitness, archive, m_f, m_cr,
        np.random.default_rng(123))
    out2 = de._lshade_trials(
        population, fitness, archive, m_f, m_cr,
        np.random.default_rng(123))

    for first, second in zip(out1, out2):
        np.testing.assert_allclose(first, second)
    trials, mutation, crossover = out1
    assert trials.shape == population.shape
    assert np.all((trials >= 0.0) & (trials <= 1.0))
    assert np.all((mutation > 0.0) & (mutation <= 1.0))
    assert np.all((crossover >= 0.0) & (crossover <= 1.0))


def test_weighted_round_robin_balances_counts_and_restores_order():
    np.testing.assert_array_equal(
        de._weighted_round_robin_assignment(8, [0.5, 0.5]),
        [0, 1, 0, 1, 0, 1, 0, 1])
    assignment = de._weighted_round_robin_assignment(100, [0.4, 0.6])
    np.testing.assert_array_equal(
        np.bincount(assignment, minlength=2), [40, 60])
    np.testing.assert_allclose(
        de._project_device_weights([0.0, 1.0]), [0.25, 0.75])

    indices = [np.flatnonzero(assignment == i) for i in range(2)]
    values = [idx.astype(float) + 0.5 for idx in indices]
    restored = de._restore_device_outputs(values, indices, 100)
    np.testing.assert_allclose(restored, np.arange(100) + 0.5)


def test_fixed_device_block_padding_and_rebalance_gate():
    assert de._device_block_capacity(0) == 8
    assert de._device_block_capacity(1) == 8
    assert de._device_block_capacity(8) == 8
    assert de._device_block_capacity(9) == 16

    # A tiny measured imbalance would turn a 64/64 split into padded 72/64,
    # so it must not trigger an assignment change.
    assert de._predicted_rebalance_gain(
        128, [0.5, 0.5], [0.51, 0.49], [100.0, 100.0]) == 0.0
    # A real 2x throughput asymmetry survives block padding and clears 2%.
    assert de._predicted_rebalance_gain(
        128, [0.5, 0.5], [1.0, 2.0], [50.0, 100.0]) > 0.02


def test_fixed_device_block_evaluator_accepts_arbitrary_population_sizes():
    evaluate = de._make_batched_fitness(
        lambda row: de.jnp.sum(row ** 2), 1, ())
    first = np.arange(9.0).reshape(3, 3)
    second = np.arange(51.0).reshape(17, 3)

    np.testing.assert_allclose(
        evaluate(first), np.sum(first ** 2, axis=1))
    np.testing.assert_allclose(
        evaluate(second), np.sum(second ** 2, axis=1))
    profile = evaluate.device_profile()
    np.testing.assert_array_equal(profile["last_real_candidates"], [17])
    np.testing.assert_array_equal(profile["last_candidates"], [24])
    assert profile["block_size"] == 8
    assert profile["rebalances"] == 0


def test_device_profile_uses_first_measurement_then_smooths():
    first = de._updated_device_profile_weights(
        [0.5, 0.5], [0.4, 0.6], profile_samples=0)
    np.testing.assert_allclose(first, [0.4, 0.6])

    second = de._updated_device_profile_weights(
        first, [0.5, 0.5], profile_samples=1)
    np.testing.assert_allclose(second, [0.43, 0.57])


def test_f32_five_galaxy_policy_defaults_to_all_spots():
    for galaxy in de._F32_ALL_SPOT_GALAXIES:
        assert de._de_spot_batch_policy(
            galaxy, False, None, None, 17) == (
                None, "measured float32 all-spots default")
        assert de._de_spot_batch_policy(
            galaxy, True, None, None, 17) == (
                17, "automatic VRAM plan")

    assert de._de_spot_batch_policy(
        "NGC6264", False, 3, None, 17) == (
            3, "explicit --spot-batch")
    assert de._de_spot_batch_policy(
        "NGC4258", True, None, 16, 2) == (
            16, "per-galaxy config")


def test_reference_point_uses_de_coordinates_without_becoming_a_seed():
    point = {"a": 3.0, "b": -1.0}
    normalised = de._normalise_theta_point(
        point, ("a", "b"), np.array([1.0, -5.0]),
        np.array([5.0, 3.0]))
    np.testing.assert_allclose(normalised, [0.5, 0.5])


def test_exact_archive_filters_new_keys_and_preserves_cache(tmp_path):
    path = tmp_path / "exact.sqlite"
    calls = []

    def batch_eval(points, desc=None):
        points = np.asarray(points)
        calls.append(points.copy())
        return np.sum(points, axis=1)

    archive = de._ExactArchive(
        str(path), dimension=2, objective_policy=de._DE_OBJECTIVE_POLICY)
    first = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]])
    np.testing.assert_allclose(
        archive(batch_eval, first), [3.0, 7.0, 3.0])
    assert len(calls) == 1
    assert calls[0].shape == (2, 2)
    assert archive.evaluations == 2
    assert archive.hits == 1
    assert archive.lookup_queries == 0

    second = np.array([[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(
        archive(batch_eval, second), [7.0, 3.0, 11.0])
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1], [[5.0, 6.0]])
    assert archive.evaluations == 3
    assert archive.hits == 3
    assert archive.lookup_queries == 1
    archive.close()

    resumed = de._ExactArchive(
        str(path), dimension=2, resume=True,
        objective_policy=de._DE_OBJECTIVE_POLICY)

    def should_not_evaluate(points, desc=None):
        raise AssertionError("persisted cache entry was evaluated again")

    np.testing.assert_allclose(
        resumed(should_not_evaluate, second[:2]), [7.0, 3.0])
    assert resumed.hits == 2
    assert resumed.lookup_queries == 1
    resumed.close()

    with pytest.raises(ValueError, match="archive objective policy"):
        de._ExactArchive(
            str(path), dimension=2, resume=True,
            objective_policy="different_objective")


def test_exact_archive_backfills_compact_fingerprints(tmp_path):
    path = tmp_path / "legacy.sqlite"
    point = np.array([1.25, 2.5])
    key = de._ExactArchive._key(point)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE evaluations "
        "(point BLOB PRIMARY KEY, fitness REAL NOT NULL) WITHOUT ROWID")
    connection.execute(
        "CREATE TABLE metadata "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
    connection.execute(
        "INSERT INTO evaluations VALUES (?, ?)", (key, 3.75))
    connection.execute(
        "INSERT INTO metadata VALUES ('dimension', '2')")
    connection.commit()
    connection.close()

    archive = de._ExactArchive(str(path), dimension=2, resume=True)

    def should_not_evaluate(points, desc=None):
        raise AssertionError("legacy exact value was not recovered")

    np.testing.assert_allclose(
        archive(should_not_evaluate, point[None]), [3.75])
    fingerprint = de._ExactArchive._fingerprint(key)
    assert fingerprint in archive._known_fingerprints
    row = archive.connection.execute(
        "SELECT value FROM metadata "
        "WHERE key='fingerprint_version'").fetchone()
    assert row == (de._ARCHIVE_FINGERPRINT_VERSION,)
    assert archive.connection.execute(
        "SELECT COUNT(*) FROM evaluation_fingerprints").fetchone() == (1,)
    archive.close()

    with pytest.raises(ValueError, match="'legacy'"):
        de._ExactArchive(
            str(path), dimension=2, resume=True,
            objective_policy=de._DE_OBJECTIVE_POLICY)


def test_exact_archive_fingerprint_collision_is_only_a_sql_probe(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        de._ExactArchive, "_fingerprint", staticmethod(lambda key: 7))
    archive = de._ExactArchive(
        str(tmp_path / "collision.sqlite"), dimension=2)
    points = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.array([3.0, 7.0])

    archive(lambda x, desc=None: np.sum(x, axis=1), points)

    def should_not_evaluate(points, desc=None):
        raise AssertionError("full BLOB keys were not used after collision")

    np.testing.assert_allclose(
        archive(should_not_evaluate, points), expected)
    assert archive.connection.execute(
        "SELECT COUNT(*) FROM evaluation_fingerprints").fetchone() == (1,)
    assert archive.lookup_queries == 1
    archive.close()


def test_lshade_population_reduction_uses_evaluations_not_generations():
    assert de._linear_population_size(2000, 128, 0, 3_400_000) == 2000
    assert de._linear_population_size(
        2000, 128, 1_700_000, 3_400_000) == 1064
    assert de._linear_population_size(
        2000, 128, 3_400_000, 3_400_000) == 128
    assert de._linear_population_size(
        2000, 128, 6_800_000, 3_400_000) == 128

    evaluations = 2000
    population = 2000
    for _ in range(5000):
        evaluations += population
        population = de._linear_population_size(
            2000, 128, evaluations, 3_400_000)
    assert population == 128
    assert 3_400_000 <= evaluations < 3_402_000
