"""Small deterministic checks for the megamaser hybrid DE bookkeeping."""
import numpy as np

from scripts.megamaser.run_de_map import (
    _ExactArchive, _draw_reshuffled_batch, _linear_population_size,
    _lshade_trials, _polish_elites, _update_lshade_memory)


def test_reshuffled_batches_cover_epoch_before_reuse():
    rng = np.random.default_rng(4)
    order = rng.permutation(11)
    cursor = 0
    batches = []
    for _ in range(4):
        batch, order, cursor = _draw_reshuffled_batch(
            order, cursor, 6, rng)
        assert len(np.unique(batch)) == len(batch)
        batches.append(batch)
    stream = np.concatenate(batches)
    assert len(np.unique(stream[:11])) == 11
    assert len(np.unique(stream[11:22])) == 11


def test_lshade_trials_and_memory_are_bounded():
    rng = np.random.default_rng(8)
    population = rng.random((8, 3))
    fitness = np.sum(population**2, axis=1)
    m_f = np.full(6, 0.5)
    m_cr = np.full(6, 0.5)
    trials, f, cr = _lshade_trials(
        population, fitness, np.empty((0, 3)), m_f, m_cr, rng)
    assert trials.shape == population.shape
    assert np.all((0 <= trials) & (trials <= 1))
    assert np.all((0 < f) & (f <= 1))
    assert np.all((0 <= cr) & (cr <= 1))

    index = _update_lshade_memory(
        m_f, m_cr, 0, np.array([0.4, 0.8]), np.array([0.2, 0.6]),
        np.array([1.0, 3.0]))
    assert index == 1
    assert 0.4 < m_f[0] < 0.8
    assert 0.2 < m_cr[0] < 0.6
    assert [_linear_population_size(20, 4, g, 4)
            for g in range(5)] == [20, 16, 12, 8, 4]


def test_exact_archive_deduplicates_and_resumes(tmp_path):
    path = tmp_path / "exact.sqlite"
    calls = []

    def evaluate(points, desc=None):
        calls.append(len(points))
        return np.sum(np.asarray(points)**2, axis=1)

    points = np.array([[0.1, 0.2], [0.3, 0.4], [0.1, 0.2]])
    archive = _ExactArchive(str(path), 2)
    expected = np.array([0.05, 0.25, 0.05])
    np.testing.assert_allclose(archive(evaluate, points), expected)
    np.testing.assert_allclose(archive(evaluate, points), expected)
    assert calls == [2]
    assert archive.count() == 2
    archive.close()

    resumed = _ExactArchive(str(path), 2, resume=True)
    np.testing.assert_allclose(resumed(evaluate, points), expected)
    assert calls == [2]
    resumed.close()


def test_adam_endpoint_needs_exact_improvement():
    population = np.array([[0.2], [0.6]])
    fitness = np.sum(population**2, axis=1)
    rng = np.random.default_rng(9)
    archive = _ExactArchive(":memory:", 1)

    def exact(points, desc=None):
        return np.sum(np.asarray(points)**2, axis=1)

    def noisy_value_grad(point, batches):
        return 0.0, -np.ones_like(point)  # pushes away from the exact optimum

    updated, updated_fitness, replaced = _polish_elites(
        population, fitness, exact, archive, noisy_value_grad, [4], rng,
        n_elites=1, n_steps=5, learning_rate=0.05, batch_fraction=0.5,
        min_dist_frac=0.0)
    np.testing.assert_allclose(updated, population)
    np.testing.assert_allclose(updated_fitness, fitness)
    assert replaced.shape == (0, 1)
    archive.close()
