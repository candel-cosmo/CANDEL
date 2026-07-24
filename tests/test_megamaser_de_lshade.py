"""Regressions for the sole, Pesce-unseeded megamaser DE path."""
import os
import sqlite3
import sys

import numpy as np
import pytest
import tomli
from scipy.stats import chisquare, ks_2samp


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEGAMASER_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser")
if MEGAMASER_DIR not in sys.path:
    sys.path.insert(0, MEGAMASER_DIR)

import run_de_map as de  # noqa: E402
import benchmark_de_batching as batching  # noqa: E402


DATA_SEEDS = np.array([[140.0, 5.31], [150.0, 5.33]])


def _reference_lshade_trials(population, fitness, mutation_archive, m_f, m_cr,
                             rng, pbest_fraction=0.11):
    """Pre-vectorisation reference implementation of ``de._lshade_trials``.

    Verbatim copy of the original per-member Python loop, kept private to this
    test file so the vectorised version can be checked distributionally.
    """
    pop = np.asarray(population)
    n, dimension = pop.shape
    if n < 4:
        raise ValueError("L-SHADE requires at least four population members.")
    archive = np.asarray(mutation_archive).reshape(-1, dimension)
    union = np.vstack([pop, archive]) if archive.size else pop
    order = np.argsort(fitness)
    n_pbest = max(2, min(n, int(np.ceil(pbest_fraction * n))))
    memory_slots = rng.integers(len(m_f), size=n)
    f = np.empty(n)
    cr = np.empty(n)
    mutants = np.empty_like(pop)

    def draw_index(limit, forbidden):
        while True:
            value = int(rng.integers(limit))
            if value not in forbidden:
                return value

    for i, slot in enumerate(memory_slots):
        value = -1.0
        while value <= 0.0:
            value = m_f[slot] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
        f[i] = min(value, 1.0)
        cr[i] = (0.0 if m_cr[slot] < 0.0 else
                 np.clip(rng.normal(m_cr[slot], 0.1), 0.0, 1.0))
        pbest_pool = order[:n_pbest]
        pbest_pool = pbest_pool[pbest_pool != i]
        pbest = int(rng.choice(pbest_pool))
        r1 = draw_index(n, {i, pbest})
        r2 = draw_index(len(union), {i, pbest, r1})
        mutants[i] = (pop[i] + f[i] * (pop[pbest] - pop[i])
                      + f[i] * (pop[r1] - union[r2]))

    mutants = np.abs(mutants)
    cycle = np.floor(mutants).astype(np.int32)
    frac = mutants - np.floor(mutants)
    mutants = np.where(cycle % 2 == 0, frac, 1.0 - frac)
    cross = rng.random((n, dimension)) < cr[:, None]
    cross[np.arange(n), rng.integers(dimension, size=n)] = True
    return np.where(cross, mutants, pop), f, cr


def test_initial_population_puts_required_base_model_seed_first():
    base_model_seed = np.array([145.0, 5.32])
    seeds = de._initial_de_seed_points(DATA_SEEDS, base_model_seed)

    np.testing.assert_allclose(seeds[0], base_model_seed)
    np.testing.assert_allclose(seeds[1:], DATA_SEEDS)
    seeds[1, 0] = -1.0
    assert DATA_SEEDS[0, 0] == 140.0
    assert de._initial_de_seed_points(None) is None


def test_expanded_model_seed_cloud_varies_around_vanilla_map(monkeypatch):
    names = (
        "D_A", "eta", "x0", "d2i_dr2", "d2Omega_dr2",
        "e_x", "e_y", "dperiapsis_dr")
    bounds = {
        "D_A": (6.0, 9.0),
        "eta": (3.0, 9.0),
        "x0": (-750.0, 750.0),
        "d2i_dr2": (-450.0, 450.0),
        "d2Omega_dr2": (-450.0, 450.0),
        "e_x": (-0.125, 0.125),
        "e_y": (-0.125, 0.125),
        "dperiapsis_dr": (-360.0, 360.0),
    }
    monkeypatch.setattr(
        de, "_prior_bounds",
        lambda prior, sobol_n_sigma=5: bounds[prior])
    model = type("Model", (), {
        "_r_ang_ref_i": 5.1,
        "_r_ang_ref_Omega": 5.1,
        "_r_ang_ref_periapsis": 5.1,
    })()
    target = type("Target", (), {
        "names": names,
        "sites": [(name, None, name) for name in names],
    })()
    base = {
        "D_A": 7.416,
        "eta": 6.7223,
        "x0": -168.3482,
        "d2i_dr2": 0.0,
        "d2Omega_dr2": 0.0,
        "e_x": 0.0,
        "e_y": 0.0,
        "dperiapsis_dr": 0.0,
        "r_ang": np.array([3.2, 5.1, 8.1]),
    }

    seeds, info = de._base_model_variation_seeds(
        model, target, base, 2048, seed=45)

    assert seeds.shape == (2048, len(names))
    np.testing.assert_allclose(
        seeds[:, :3],
        np.tile([7.416, 6.7223, -168.3482], (len(seeds), 1)))
    for i in range(3, len(names)):
        assert np.std(seeds[:, i]) > 0.0
    for i, name in enumerate(names):
        assert np.all(seeds[:, i] >= bounds[name][0])
        assert np.all(seeds[:, i] <= bounds[name][1])
    assert "all fitted linear-model coordinates are copied exactly" in info
    assert "e_x, e_y, dperiapsis_dr, d2i_dr2, d2Omega_dr2" in info
    assert "0.01-10 deg sigma" in info


def test_required_base_model_seed_cannot_be_dropped():
    def evaluate(points, desc=None):
        del desc
        return de.jnp.sum(points, axis=1)

    anchor = np.array([[0.25, 0.75]])
    population, _ = de._make_de_initial_population(
        evaluate, np.zeros(2), np.ones(2), pop_size=4, seed=3,
        N_sobol=8, min_dist_frac=0.0, seed_points=anchor,
        required_seed_points=1)
    np.testing.assert_allclose(population[0], anchor[0])

    with pytest.raises(ValueError, match="Required DE seed point"):
        de._make_de_initial_population(
            evaluate, np.zeros(2), np.ones(2), pop_size=4, seed=3,
            N_sobol=8, min_dist_frac=0.0,
            seed_points=np.array([[1.25, 0.75]]),
            required_seed_points=1)


def test_required_base_model_seed_audit_prints_values_and_score(capsys):
    de._print_required_de_seeds(
        ("D_A", "eta", "d2i_dr2", "d2Omega_dr2"),
        np.array([[7.416, 6.7223, 0.0, 0.0]]),
        np.array([5694.6675, 20000.0]), 1)

    output = capsys.readouterr().out
    assert "Required NGC4258 base-model seed (injected)" in output
    assert "[model.galaxies.NGC4258.init]" in output
    assert "normalised priors can shift absolute logP" in output
    assert "d2i_dr2" in output and "= 0" in output
    assert "d2Omega_dr2" in output
    assert "log_MBH (derived)" in output
    assert "logP = -5694.667500" in output
    assert "initial-population rank = 1/2" in output


def test_de_is_only_lshade_and_config_has_no_hybrid_settings():
    assert de._DE_ALGORITHM == "lshade"
    assert de._DE_SEED_POLICY == "data_sobol_only"
    assert de._DE_BASE_MODEL_SEED_POLICY == (
        "vanilla_expansion_variations_sobol_ngc4258_base_config_v3")

    path = os.path.join(MEGAMASER_DIR, "config_maser.toml")
    with open(path, "rb") as f:
        config = tomli.load(f)
    optimise = config["optimise"]
    assert config["model"]["phi_integration"] == "peak-partition"
    assert config["model"]["n_phi_partition_sys"] == 129
    assert config["model"]["n_phi_partition_hv"] == 65
    assert config["model"]["n_r_global"] == 176
    assert config["model"]["global_r_full_support"] is True
    assert config["model"]["asymmetric_r_local"] is True
    assert config["model"]["peak_r_refine_steps"] == 0
    assert config["model"]["peak_r_refine_order"] == 7
    assert config["model"]["peak_r_refine_hv_only"] is False
    assert config["model"]["peak_r_width_steps"] == 0
    assert config["model"]["priors"]["sigma_v_sys"]["low"] == 0.1
    assert config["model"]["priors"]["sigma_v_hv"]["low"] == 0.1
    cgcg = config["model"]["galaxies"]["CGCG074-064"]
    assert cgcg["n_phi_partition_sys"] == 97
    assert cgcg["n_phi_partition_hv"] == 49
    ngc4258 = config["model"]["galaxies"]["NGC4258"]
    assert ngc4258["phi_integration"] == "peak-partition"
    assert ngc4258["n_phi_partition_sys"] == 513
    assert ngc4258["n_phi_partition_hv"] == 65
    assert ngc4258["conditional_spot_batch"] == 32
    assert ngc4258["peak_r_refine_steps"] == 3
    assert ngc4258["peak_r_refine_hv_only"] is False
    assert ngc4258["peak_r_width_steps"] == 8
    linear_map = ngc4258["init"]
    assert {
        key: linear_map[key]
        for key in (
            "D_A", "Omega0", "dOmega_dr", "di_dr", "dv_sys", "eta", "i0",
            "log_MBH", "sigma_a_floor", "sigma_v_hv", "sigma_v_sys",
            "sigma_x_floor", "sigma_y_floor", "x0", "y0")
    } == {
        "D_A": 7.416,
        "Omega0": 86.205,
        "dOmega_dr": 2.0856,
        "di_dr": -2.2856,
        "dv_sys": -192.6809,
        "eta": 6.7223,
        "i0": 95.7056,
        "log_MBH": 7.5925,
        "sigma_a_floor": 0.4666,
        "sigma_v_hv": 4.0889,
        "sigma_v_sys": 0.1707,
        "sigma_x_floor": 2.1823,
        "sigma_y_floor": 5.4207,
        "x0": -168.3482,
        "y0": 557.1408,
    }
    assert len(linear_map["r_ang"]) == 358
    expanded = type("Model", (), {
        "_D_A_uniform": True,
        "mass_parameterization": "eta",
        "use_ecc": True,
        "ecc_cartesian": True,
        "use_quadratic_warp": True,
    })()
    lifted = de._clean_init(expanded, linear_map)
    assert float(lifted["e_x"]) == 0.0
    assert float(lifted["e_y"]) == 0.0
    assert float(lifted["dperiapsis_dr"]) == 0.0
    assert float(lifted["d2i_dr2"]) == 0.0
    assert float(lifted["d2Omega_dr2"]) == 0.0
    ngc5765b = config["model"]["galaxies"]["NGC5765b"]
    assert ngc5765b["n_phi_partition_sys"] == 97
    assert ngc5765b["n_phi_partition_hv"] == 49
    assert ngc5765b["n_r_local"] == 321
    assert config["model"]["galaxies"]["NGC6264"][
        "conditional_spot_batch"] == 32
    ugc3789 = config["model"]["galaxies"]["UGC3789"]
    assert ugc3789["n_phi_partition_sys"] == 97
    assert ugc3789["n_phi_partition_hv"] == 49
    assert ugc3789["n_r_local"] == 384
    assert ugc3789["scan_width_drop"] == 50.0
    assert all(
        "n_r_global" not in galaxy
        for galaxy in config["model"]["galaxies"].values())
    assert "algorithm" not in optimise
    assert not any(key.startswith("adam_") for key in optimise)
    assert optimise["pop_size"] == 2000
    assert 4 <= optimise["min_pop_size"] <= optimise["pop_size"]
    assert optimise["min_pop_size"] == 1024
    assert "eval_chunk" not in optimise
    assert de._CANDIDATES_PER_GPU_WAVE == 1
    assert de._PEAK_PARTITION_CANDIDATES_PER_GPU_WAVE == 8
    assert de._DEVICE_LOCAL_BLOCK_SIZE == 8
    fixed = type("Model", (), {"phi_integration": "fixed-grid"})()
    peak = type("Model", (), {"phi_integration": "peak-partition"})()
    assert de._de_candidates_per_wave(fixed) == 1
    assert de._de_candidates_per_wave(peak) == 8
    assert de._de_candidates_per_wave(peak, 2) == 2
    with pytest.raises(ValueError, match="requires peak-partition"):
        de._de_candidates_per_wave(fixed, 2)
    assert optimise["population_reduction_evaluations"] == 5_000_000


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
    assert "--phi-integration {fixed-grid,peak-partition}" in help_text
    assert "--peak-candidates-per-wave {1,2,4,8}" in help_text
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


def test_batching_benchmark_only_varies_exact_gpu_tiling(capsys):
    parser = batching._parser()
    help_text = parser.format_help()
    assert "--eval-chunk" not in help_text
    assert parser.parse_args(
        ["UGC3789", "--spot-batch", "all"]).spot_batch is None
    peak = parser.parse_args([
        "UGC3789", "--phi-integration", "peak-partition",
        "--candidate-wave", "4"])
    assert peak.phi_integration == "peak-partition"
    assert peak.candidate_wave == 4

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["UGC3789", "--eval-chunk", "1"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --eval-chunk 1" in capsys.readouterr().err


def test_batching_memory_geometry_uses_partition_scans():
    class PeakModel:
        phi_integration = "peak-partition"
        _n_r_local = 256
        _n_r_global = 128
        _n_sys = 3
        _n_red = 2
        _n_blue = 1

        @staticmethod
        def _phi_partition_scan_size(name):
            return 129 if name == "sys" else 65

    geometry = batching._memory_geometry(PeakModel(), dtype_bytes=4)
    assert geometry["n_r"] == 256
    assert geometry["groups"]["sys"]["n_phi_scan"] == 129
    assert geometry["groups"]["red"]["n_phi_scan"] == 65
    assert geometry["groups"]["sys"]["n_half_planes"] == 2
    assert geometry["groups"]["red"]["n_half_planes"] == 1
    assert geometry["groups"]["sys"][
        "all_spots_one_candidate_bytes"] == 2 * 3 * 256 * 129 * 4


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

        @staticmethod
        def _phi_integration_suffix(model):
            return "_peakpartition"

        @staticmethod
        def _objective_policy(model):
            return "peak-policy"

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
        with pytest.raises(ValueError, match="Checkpoint seed policy"):
            de._validate_de_checkpoint_policy(
                checkpoint, str(explicit),
                seed_policy=de._DE_BASE_MODEL_SEED_POLICY)


@pytest.mark.parametrize(
    ("enable_x64", "saved_dtype", "should_raise"),
    ((False, np.float32, False), (True, np.float64, False),
     (False, np.float64, True), (True, np.float32, True)))
def test_resume_requires_matching_checkpoint_precision(
        tmp_path, enable_x64, saved_dtype, should_raise):
    path = tmp_path / f"checkpoint_{enable_x64}_{saved_dtype.__name__}.npz"
    np.savez(
        path, lo=np.array([0.0]), hi=np.array([1.0]),
        names=np.array(["x"]), sizes=np.array([1]),
        population=np.zeros((4, 1), dtype=saved_dtype),
        fitness=np.zeros(4, dtype=saved_dtype))
    original_x64 = de.jax.config.jax_enable_x64
    try:
        de.jax.config.update("jax_enable_x64", enable_x64)
        if should_raise:
            with pytest.raises(ValueError, match="DE-state precision"):
                de._load_de_checkpoint(
                    str(path), np.array([0.0]), np.array([1.0]), ["x"], [1])
        else:
            checkpoint = de._load_de_checkpoint(
                str(path), np.array([0.0]), np.array([1.0]), ["x"], [1])
            checkpoint.close()
    finally:
        de.jax.config.update("jax_enable_x64", original_x64)


def test_checkpoint_updates_progress_plot_and_restores_history(tmp_path):
    checkpoint = tmp_path / "de.npz"
    history = {
        "history_generation": np.arange(600),
        "history_logp": np.linspace(-12.0, -7.0, 600),
        "history_D_A": np.linspace(7.0, 7.2, 600),
    }
    args = [
        str(checkpoint), np.zeros((4, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float32), np.array([0.5, 0.5]),
        np.array(7.0), 599, np.array([0, 1], dtype=np.uint32), 0, -7.0,
        np.zeros(2), np.ones(2), ["D_A", "eta"], [1, 1],
    ]

    de._save_de_checkpoint(*args, extra=history)

    plot = tmp_path / "de_progress.png"
    png = plot.read_bytes()
    assert png.startswith(b"\x89PNG")
    assert tuple(
        int.from_bytes(png[i:i + 4], "big") for i in (16, 20)
    ) == (2700, 2100)
    with np.load(checkpoint) as saved:
        restored = de._load_de_history(saved, 599, -7.0, 7.2)
    for values, expected in zip(restored, history.values()):
        np.testing.assert_allclose(values, expected)

    first_plot = plot.read_bytes()
    extended = {
        key: np.append(values, value)
        for (key, values), value in zip(
            history.items(), (600, -6.5, 7.25))
    }
    args[4:6] = [np.array(6.5), 600]
    args[8] = -6.5
    de._save_de_checkpoint(*args, extra=extended)
    assert plot.read_bytes() != first_plot


def test_peak_partition_uses_distinct_checkpoint_policy(tmp_path):
    class Model:
        phi_integration = "peak-partition"
        _n_phi_partition_sys = 129
        _n_phi_partition_hv = 65
        _phi_partition_root_capacity = 4
        _n_r_local = 256
        _n_r_global = 128
        _K_sigma = 10.0
        _global_r_full_support = True
        _asymmetric_r_local = True
        _scan_width_drop = 0.0
        _peak_r_refine_steps = 4
        _peak_r_refine_order = 7
        _peak_r_refine_hv_only = True
        _peak_r_width_steps = 12

    model = Model()
    model.use_ecc = False
    policy = de._objective_policy(model)
    assert policy.startswith(de._DE_PEAK_PARTITION_POLICY)
    assert ":rrhv:" in policy
    assert ":rw12:" in policy
    assert policy != de._DE_OBJECTIVE_POLICY
    model.use_ecc = True
    eccentric_policy = de._objective_policy(model)
    assert ":ecc_hybrid_qf1:" in eccentric_policy
    assert eccentric_policy != policy

    explicit = tmp_path / "peak.npz"
    np.savez(explicit, algorithm=np.asarray("lshade"),
             seed_policy=np.asarray("data_sobol_only"),
             population_schedule=np.asarray("nfe_linear"),
             objective_policy=np.asarray(policy))
    with np.load(explicit) as checkpoint:
        de._validate_de_checkpoint_policy(
            checkpoint, str(explicit), objective_policy=policy)
        with pytest.raises(ValueError, match="objective policy"):
            de._validate_de_checkpoint_policy(checkpoint, str(explicit))


@pytest.mark.parametrize(
    ("phi_integration", "use_ecc", "expected_reuse"),
    (("fixed-grid", False, True),
     ("fixed-grid", True, False),
     ("peak-partition", False, True),
     ("peak-partition", True, True)))
def test_compatible_objectives_reuse_scan_cache(
        phi_integration, use_ecc, expected_reuse):
    class Model:
        def __init__(self):
            self.phi_integration = phi_integration
            self.use_ecc = use_ecc
            self.return_scan_cache = None
            self.received_scan_cache = None

        def phys_from_params_jax(self, theta, h):
            del theta, h
            return tuple(de.jnp.ones(()) for _ in range(17)), {}

        def _build_conditional_r_grids(self, *args, return_scan_cache):
            del args
            self.return_scan_cache = return_scan_cache
            return (([], ["cache"]) if return_scan_cache else [])

        def _sum_phi_marginal(self, groups, phys_args, phys_kw,
                              spot_batch, remat, scan_cache):
            del groups, phys_args, phys_kw, spot_batch, remat
            self.received_scan_cache = scan_cache
            return de.jnp.asarray(0.0)

    model = Model()

    class Target:
        h = 0.73
        sites = ()
        spot_batch = None
        mass_parameterization = "log_mbh"

    target = Target()
    target.model = model
    de._logp_2d_terms(target, {})

    assert model.return_scan_cache is expected_reuse
    assert (model.received_scan_cache is not None) is expected_reuse


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


@pytest.mark.parametrize("n,dimension", [(4, 2), (4, 6), (17, 3), (64, 6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("with_archive", [False, True])
def test_lshade_trials_invariants(n, dimension, dtype, with_archive):
    rng0 = np.random.default_rng(20240721)
    pop = rng0.random((n, dimension)).astype(dtype)
    fitness = rng0.random(n)
    archive = (rng0.random((max(1, n // 2), dimension)).astype(dtype)
               if with_archive else np.empty((0, dimension), dtype=dtype))
    m_f = np.full(6, 0.5)
    m_cr = np.array([0.5, 0.3, -1.0, 0.8, 0.6, 0.2])

    trials, f, cr = de._lshade_trials(
        pop, fitness, archive, m_f, m_cr, np.random.default_rng(1))

    assert trials.shape == (n, dimension)
    assert trials.dtype == pop.dtype
    assert f.shape == (n,) and cr.shape == (n,)
    assert np.all((trials >= 0.0) & (trials <= 1.0))
    assert np.all((f > 0.0) & (f <= 1.0))
    assert np.all((cr >= 0.0) & (cr <= 1.0))
    # Forced-crossover column: every member differs from its parent somewhere.
    assert np.all(np.any(trials != pop, axis=1))


def test_lshade_trials_match_reference_distribution():
    rng0 = np.random.default_rng(7)
    n, dimension = 64, 6
    pop = rng0.random((n, dimension))
    fitness = rng0.random(n)
    archive = rng0.random((32, dimension))
    m_f = np.array([0.3, 0.5, 0.7, 0.5, 0.9, 0.4])
    m_cr = np.array([0.5, 0.2, -1.0, 0.8, 0.6, 0.3])
    n_batches = 400

    def collect(fn):
        fs, crs, disp, nocross = [], [], [], []
        for b in range(n_batches):
            rng = np.random.default_rng(1000 + b)
            trials, f, cr = fn(pop, fitness, archive, m_f, m_cr, rng)
            fs.append(f)
            crs.append(cr)
            disp.append(trials - pop)
            nocross.append(trials == pop)
        return (np.concatenate(fs), np.concatenate(crs),
                np.concatenate(disp, axis=0), np.concatenate(nocross, axis=0))

    f_new, cr_new, disp_new, nc_new = collect(de._lshade_trials)
    f_ref, cr_ref, disp_ref, nc_ref = collect(_reference_lshade_trials)

    ks_f = ks_2samp(f_new, f_ref)
    assert ks_f.pvalue > 1e-3, ks_f
    ks_cr = ks_2samp(cr_new[cr_new > 0], cr_ref[cr_ref > 0])
    assert ks_cr.pvalue > 1e-3, ks_cr

    np.testing.assert_allclose(
        disp_new.mean(axis=0), disp_ref.mean(axis=0), atol=0.02)
    np.testing.assert_allclose(
        np.cov(disp_new, rowvar=False), np.cov(disp_ref, rowvar=False),
        atol=0.02)
    np.testing.assert_allclose(nc_new.mean(), nc_ref.mean(), rtol=0.05)


def test_lshade_draw_indices_respect_exclusions():
    rng = np.random.default_rng(2024)
    for n, n_extra in [(4, 0), (8, 8), (17, 5), (32, 32)]:
        n_union = n + n_extra
        order = rng.permutation(n)
        n_pbest = max(2, int(np.ceil(0.11 * n)))
        pool = order[:n_pbest]
        idx = np.arange(n)
        for _ in range(400):
            pbest, r1, r2 = de._lshade_draw_indices(
                order, n_pbest, n, n_union, rng)
            assert np.all(np.isin(pbest, pool))
            assert np.all(pbest != idx)
            assert np.all(r1 != idx) and np.all(r1 != pbest)
            assert np.all((r2 != idx) & (r2 != pbest) & (r2 != r1))
            assert np.all(r1 < n) and np.all(r2 < n_union)


def test_lshade_draw_indices_marginals_are_uniform():
    n, n_union = 8, 16
    order = np.arange(n)          # pool = {0, 1}
    n_pbest = 2
    i0, p0, q0 = 5, 0, 1          # i0 outside the pbest pool
    rng = np.random.default_rng(99)
    pbest_s, r1_s, r2_s = [], [], []
    for _ in range(20000):
        pbest, r1, r2 = de._lshade_draw_indices(order, n_pbest, n, n_union, rng)
        pbest_s.append(pbest[i0])
        r1_s.append(r1[i0])
        r2_s.append(r2[i0])
    pbest_s = np.array(pbest_s)
    r1_s = np.array(r1_s)
    r2_s = np.array(r2_s)

    # pbest uniform over the pool {0, 1}.
    counts = np.bincount(pbest_s, minlength=2)[[0, 1]]
    assert chisquare(counts).pvalue > 1e-4

    # r1 | pbest==p0 uniform over [0, n) \ {i0, p0}.
    sel = r1_s[pbest_s == p0]
    support = [v for v in range(n) if v not in (i0, p0)]
    counts = np.array([(sel == v).sum() for v in support])
    assert chisquare(counts).pvalue > 1e-4

    # r2 | pbest==p0, r1==q0 uniform over [0, n_union) \ {i0, p0, q0}.
    sel = r2_s[(pbest_s == p0) & (r1_s == q0)]
    support = [v for v in range(n_union) if v not in (i0, p0, q0)]
    counts = np.array([(sel == v).sum() for v in support])
    assert chisquare(counts).pvalue > 1e-4


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
    assert profile["candidates_per_wave"] == 1
    assert profile["rebalances"] == 0


def test_peak_partition_device_block_evaluates_candidates_concurrently():
    evaluate = de._make_batched_fitness(
        lambda row: de.jnp.sum(row ** 2), 1, (),
        candidates_per_wave=8)
    points = np.arange(51.0).reshape(17, 3)

    np.testing.assert_allclose(
        evaluate(points), np.sum(points ** 2, axis=1))
    profile = evaluate.device_profile()
    assert profile["candidates_per_wave"] == 8
    np.testing.assert_array_equal(profile["last_candidates"], [24])


def test_candidate_wave_size_must_divide_fixed_device_block():
    for size in (0, 3):
        with pytest.raises(ValueError, match="positive divisor"):
            de._make_batched_fitness(
                lambda row: de.jnp.sum(row), 1, (),
                candidates_per_wave=size)


def test_homogeneous_devices_use_one_shared_pmap(monkeypatch):
    class Device:
        device_kind = "A100"

    calls = []

    def fake_pmap(_, devices):
        calls.append(tuple(devices))
        return lambda blocks: np.sum(blocks ** 2, axis=-1)

    devices = (Device(), Device())
    monkeypatch.setattr(de.jax, "pmap", fake_pmap)
    evaluate = de._make_batched_fitness(
        lambda row: de.jnp.sum(row ** 2), 2, devices)
    points = np.arange(51.0).reshape(17, 3)

    np.testing.assert_allclose(
        evaluate(points), np.sum(points ** 2, axis=1))
    profile = evaluate.device_profile()
    assert calls == [devices]
    assert profile["execution_mode"] == "shared pmap"
    np.testing.assert_array_equal(profile["last_real_candidates"], [9, 8])
    np.testing.assert_array_equal(profile["last_candidates"], [16, 16])
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
    # Both cache hits resolve from the RAM write buffer, so no SQL is issued.
    assert archive.lookup_queries == 0
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
    # The colliding hit resolves from the RAM write buffer (no SQL probe); the
    # persisted table still dedups the single fingerprint once flushed.
    assert archive.lookup_queries == 0
    archive.flush()
    assert archive.connection.execute(
        "SELECT COUNT(*) FROM evaluation_fingerprints").fetchone() == (1,)
    archive.close()


def test_lshade_population_reduction_uses_evaluations_not_generations():
    assert de._linear_population_size(2000, 1024, 0, 5_000_000) == 2000
    assert de._linear_population_size(
        2000, 1024, 2_500_000, 5_000_000) == 1512
    assert de._linear_population_size(
        2000, 1024, 5_000_000, 5_000_000) == 1024
    assert de._linear_population_size(
        2000, 1024, 10_000_000, 5_000_000) == 1024

    evaluations = 2000
    population = 2000
    for _ in range(5000):
        evaluations += population
        population = de._linear_population_size(
            2000, 1024, evaluations, 5_000_000)
    assert population == 1024
    assert evaluations > 5_000_000
