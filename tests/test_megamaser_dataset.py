"""Regressions for the megamaser spot-table dataset switch.

The failure this guards against is silent: a wrong-dataset init block runs to
completion because ``maser_blackjax`` falls back to ``zeros_like(r_hat)`` when
``r_ang``'s length does not match ``n_spots``, and ``r_ang`` is not among the
validated scalar sites.
"""
import csv
import os
import sys

import numpy as np
import pytest
import tomli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEGAMASER_DIR = os.path.join(REPO_ROOT, "scripts", "megamaser")
if MEGAMASER_DIR not in sys.path:
    sys.path.insert(0, MEGAMASER_DIR)

from candel.pvdata import megamaser_data  # noqa: E402
from candel.pvdata.megamaser_data import (  # noqa: E402
    DEFAULT_MASER_DATASET, MASER_DATASETS, _apply_clipped_mask,
    clipped_mask_path, load_megamaser_spots,
    maser_data_root, megamaser_velocity_frame)
from maser_config import (apply_dataset, check_chain_dataset,  # noqa: E402
                          check_init_block, dataset_init_path,
                          resolve_dataset)

CONFIG_PATH = os.path.join(MEGAMASER_DIR, "config_maser.toml")

# Spot counts of each dataset, from the tables themselves. The fiducial counts
# match the per-galaxy provenance the MCP gave for their vetting (see the
# checked-in P20 clipping audit).
EXPECTED_N_SPOTS = {
    "original_published": {"CGCG074-064": 165, "NGC4258": 358,
                           "NGC5765b": 212, "NGC6264": 66,
                           "NGC6323": 68, "UGC3789": 156},
    "fiducial": {"CGCG074-064": 165, "NGC4258": 358,
                 "NGC5765b": 169, "NGC6264": 61,
                 "NGC6323": 87, "UGC3789": 153},
    "unpruned": {"CGCG074-064": 165, "NGC4258": 358,
                 "NGC5765b": 212, "NGC6264": 66,
                 "NGC6323": 87, "UGC3789": 156},
}
SOURCE_DATASETS = tuple(EXPECTED_N_SPOTS)

# Byte-identical published/fiducial tables, so their spots must agree exactly.
SHARED_GALAXIES = ("CGCG074-064", "NGC4258")


def _config():
    with open(CONFIG_PATH, "rb") as f:
        return tomli.load(f)


def _load(dataset, galaxy):
    gcfg = _config()["model"]["galaxies"][galaxy]
    root = maser_data_root(dataset)
    if not os.path.isdir(root):
        pytest.skip(f"external megamaser dataset is not provisioned: {root}")
    return load_megamaser_spots(root, galaxy,
                                v_sys_obs=gcfg.get("v_sys_obs"))


def test_default_dataset_has_complete_config_init():
    assert DEFAULT_MASER_DATASET == "fiducial"
    assert MASER_DATASETS == (
        "original_published", "fiducial", "unpruned", "clipped")
    assert _config()["io"]["dataset"] == DEFAULT_MASER_DATASET


def test_maser_data_root_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="Unknown megamaser dataset"):
        maser_data_root("no_such_dataset")


def test_load_rejects_unqualified_root():
    """A bare data/Megamaser must not silently resolve to a dataset."""
    with pytest.raises(ValueError, match="not dataset-qualified"):
        load_megamaser_spots(os.path.join(REPO_ROOT, "data", "Megamaser"),
                             "NGC4258")


def test_clipped_mask_filters_unpruned_rows(tmp_path):
    data = {
        "n_spots": 3,
        "velocity": np.array([10.0, 20.0, 30.0]),
        "x": np.array([1.0, 2.0, 3.0]),
        "dataset": "unpruned",
    }
    path = tmp_path / "mask.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow((
            "unpruned_spot_index", "velocity_km_s", "clip",
            "pending_clip", "stabilised"))
        writer.writerows((
            (1, 10.0, False, False, True),
            (2, 20.0, True, False, True),
            (3, 30.0, False, False, True),
        ))

    clipped = _apply_clipped_mask(data, path)
    assert clipped["dataset"] == "clipped"
    assert clipped["n_spots"] == 2
    assert clipped["velocity"].tolist() == [10.0, 30.0]
    assert clipped["unpruned_spot_index"].tolist() == [0, 2]

    rows = list(csv.DictReader(path.open(newline="")))
    rows[1]["pending_clip"] = "True"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="not a stabilised"):
        _apply_clipped_mask(data, path)

    rows[1]["pending_clip"] = "False"
    rows[1]["clip"] = "invalid"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="invalid clip"):
        _apply_clipped_mask(data, path)


def test_clipped_dataset_prefers_model_variant_then_linear(
        tmp_path, monkeypatch):
    def fake_spots(*args, **kwargs):
        del args, kwargs
        return {
            "n_spots": 3,
            "velocity": np.array([10.0, 20.0, 30.0]),
            "x": np.array([1.0, 2.0, 3.0]),
        }

    def write_mask(path, clipped):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow((
                "unpruned_spot_index", "velocity_km_s", "clip",
                "pending_clip", "stabilised"))
            for i, velocity in enumerate((10.0, 20.0, 30.0)):
                writer.writerow((i + 1, velocity, i == clipped, False, True))

    monkeypatch.setattr(megamaser_data, "load_NGC6264_spots", fake_spots)
    root = str(tmp_path / "clipped")
    linear = clipped_mask_path(root, "NGC6264")
    quadratic = clipped_mask_path(
        root, "NGC6264", use_quadratic_warp=True)
    assert clipped_mask_path(
        root, "NGC6264", use_ecc=True,
        use_quadratic_warp=True).endswith("_ecc_qw.csv")
    write_mask(linear, 1)
    write_mask(quadratic, 2)

    data = load_megamaser_spots(
        root, "NGC6264", v_sys_obs=0.0, use_quadratic_warp=True)
    assert data["velocity"].tolist() == [10.0, 20.0]

    os.remove(quadratic)
    data = load_megamaser_spots(
        root, "NGC6264", v_sys_obs=0.0, use_quadratic_warp=True)
    assert data["velocity"].tolist() == [10.0, 30.0]


@pytest.mark.parametrize("dataset", SOURCE_DATASETS)
def test_spot_counts_and_provenance(dataset):
    for galaxy, n in EXPECTED_N_SPOTS[dataset].items():
        data = _load(dataset, galaxy)
        assert data["n_spots"] == n, (dataset, galaxy)
        assert data["dataset"] == dataset
        # Every spot is classified into exactly one of the three supports.
        n_sys = int((~data["is_highvel"]).sum())
        n_blue = int(data["is_blue"].sum())
        assert 0 < n_sys < n and 0 < n_blue < n


@pytest.mark.parametrize("galaxy", SHARED_GALAXIES)
def test_shared_galaxies_identical_across_datasets(galaxy):
    a = _load("original_published", galaxy)
    b = _load("fiducial", galaxy)
    for key in ("velocity", "x", "y", "sigma_x", "sigma_y", "a", "sigma_a",
                "accel_measured", "phi_lo", "phi_hi", "is_blue",
                "is_highvel"):
        assert np.array_equal(a[key], b[key]), (galaxy, key)


def test_systemics_without_acceleration_are_retained():
    expected = {"CGCG074-064": 0, "NGC4258": 92, "NGC5765b": 20,
                "NGC6264": 0, "NGC6323": 0, "UGC3789": 0}
    for galaxy, n_missing in expected.items():
        published = _load("original_published", galaxy)
        missing = (~published["is_highvel"]) & (~published["accel_measured"])
        assert int(missing.sum()) == n_missing, galaxy

    published = _load("original_published", "NGC5765b")
    missing = (~published["is_highvel"]) & (~published["accel_measured"])
    assert np.all(published["a"][missing] == 1.0)
    assert np.all(published["sigma_a"][missing] == 1.0)

    printed = np.loadtxt(os.path.join(
        REPO_ROOT, "data", "Megamaser", "NGC5765b_Gao2016_table6_tex.dat"))
    printed = printed[(printed[:, 5] == 1.0) & (printed[:, 6] == 1.0)]
    by_velocity = {velocity: i for i, velocity in
                   enumerate(published["velocity"])}
    idx = np.array([by_velocity[velocity] for velocity in printed[:, 0]])
    np.testing.assert_allclose(published["x"][idx],
                               1000.0 * (printed[:, 1] + 0.002), atol=1e-12)
    np.testing.assert_allclose(published["y"][idx],
                               1000.0 * (printed[:, 3] - 0.013), atol=1e-12)
    np.testing.assert_allclose(published["sigma_x"][idx],
                               1000.0 * printed[:, 2], atol=1e-12)
    np.testing.assert_allclose(published["sigma_y"][idx],
                               1000.0 * printed[:, 4], atol=1e-12)


def test_unpruned_matches_documented_dataset_policy():
    if not os.path.isdir(maser_data_root("unpruned")):
        pytest.skip("external unpruned megamaser dataset is not provisioned")

    for galaxy in EXPECTED_N_SPOTS["unpruned"]:
        original = _load("original_published", galaxy)
        fiducial = _load("fiducial", galaxy)
        unpruned = _load("unpruned", galaxy)
        oi = {v: i for i, v in enumerate(original["velocity"])}
        fi = {v: i for i, v in enumerate(fiducial["velocity"])}
        ui = {v: i for i, v in enumerate(unpruned["velocity"])}

        if galaxy == "NGC6323":
            for key in ("velocity", "x", "sigma_x", "y", "sigma_y", "a",
                        "sigma_a", "accel_measured", "phi_lo", "phi_hi",
                        "is_blue", "is_highvel"):
                assert np.array_equal(unpruned[key], fiducial[key]), key
            continue

        assert set(ui) == set(oi) | set(fi), galaxy
        astrometry = fiducial if galaxy == "UGC3789" else original
        ai = fi if galaxy == "UGC3789" else oi
        for velocity in set(oi) & set(fi):
            for key in ("x", "sigma_x", "y", "sigma_y"):
                assert (unpruned[key][ui[velocity]] ==
                        astrometry[key][ai[velocity]])
            assert (unpruned["accel_measured"][ui[velocity]] ==
                    fiducial["accel_measured"][fi[velocity]])
            if fiducial["accel_measured"][fi[velocity]]:
                for key in ("a", "sigma_a"):
                    assert (unpruned[key][ui[velocity]] ==
                            fiducial[key][fi[velocity]])
        for velocity in set(oi) - set(fi):
            for key in ("x", "sigma_x", "y", "sigma_y", "a", "sigma_a",
                        "accel_measured"):
                assert (unpruned[key][ui[velocity]] ==
                        original[key][oi[velocity]])

    with open(os.path.join(maser_data_root("unpruned"),
                           "provenance.csv"), newline="") as f:
        provenance = list(csv.DictReader(f))
    clipped = {galaxy: 0 for galaxy in EXPECTED_N_SPOTS["unpruned"]}
    for row in provenance:
        clipped[row["galaxy"]] += row["clipped_by_pesce"] == "True"
    assert clipped == {"CGCG074-064": 0, "NGC4258": 0,
                       "NGC5765b": 43, "NGC6264": 5,
                       "NGC6323": 0, "UGC3789": 3}
    assert all(row["astrometry_source"] == "fiducial" and
               row["acceleration_source"] == "fiducial"
               for row in provenance if row["galaxy"] == "NGC6323")
    assert all(row["astrometry_source"] == row["acceleration_source"] ==
               ("original_published" if row["clipped_by_pesce"] == "True"
                else "fiducial")
               for row in provenance if row["galaxy"] == "UGC3789")


def test_velocity_frame_raises_for_unknown_galaxy():
    with pytest.raises(ValueError, match="No velocity frame recorded"):
        megamaser_velocity_frame("NGC9999")


@pytest.mark.parametrize("dataset", SOURCE_DATASETS)
def test_p20_thresholds_agree_with_kmeans(dataset):
    """The fiducial tables state their own blue/red split; it must not
    repartition the spots relative to the k-means classifier the published
    tables use, or the phi supports change silently."""
    from scipy.cluster.vq import kmeans2
    for galaxy in EXPECTED_N_SPOTS[dataset]:
        data = _load(dataset, galaxy)
        if "spot_type" not in data:
            continue
        v = data["velocity"].astype(np.float64)
        centroids, lab = kmeans2(v, 3, minit="++", seed=42)
        order = np.argsort(centroids)
        remap = np.empty(3, dtype=int)
        remap[order] = np.arange(3)
        km = remap[lab]
        stated = np.array([{"b": 0, "s": 1, "r": 2}[t]
                           for t in data["spot_type"]])
        if galaxy in ("NGC4258", "CGCG074-064"):
            continue          # explicit spot_type from the source table
        assert np.array_equal(stated, km), (dataset, galaxy)


# ---- config split ----

def test_config_carries_no_dataset_specific_keys():
    """init*/r_ang_ref_* must live only in the per-dataset files: a base copy
    would let a galaxy silently inherit the other dataset's best point."""
    for galaxy, blk in _config()["model"]["galaxies"].items():
        stray = [k for k in blk
                 if k.startswith("init") or k.startswith("r_ang_ref_")]
        assert not stray, (galaxy, stray)


def test_unpruned_warp_pivots_match_other_datasets():
    keys = ("r_ang_ref_i", "r_ang_ref_Omega", "r_ang_ref_periapsis")
    configs = {}
    for dataset in SOURCE_DATASETS:
        with open(dataset_init_path(dataset), "rb") as f:
            configs[dataset] = tomli.load(f)["model"]["galaxies"]

    for galaxy in EXPECTED_N_SPOTS["unpruned"]:
        expected = {key: configs["fiducial"][galaxy][key] for key in keys}
        assert {key: configs["original_published"][galaxy][key]
                for key in keys} == expected
        assert {key: configs["unpruned"][galaxy][key]
                for key in keys} == expected


def test_original_published_init_set_is_complete():
    with open(dataset_init_path("original_published"), "rb") as f:
        galaxies = tomli.load(f)["model"]["galaxies"]
    expected = {
        "CGCG074-064": {"init", "init_qw"},
        "NGC4258": {"init", "init_qw", "init_ecc_qw"},
        "NGC5765b": {"init", "init_qw"},
        "NGC6264": {"init", "init_qw"},
        "NGC6323": {"init", "init_qw"},
        "UGC3789": {"init", "init_qw"},
    }
    for galaxy, variants in expected.items():
        assert {k for k in galaxies[galaxy] if k.startswith("init")} == variants


@pytest.mark.parametrize("dataset", SOURCE_DATASETS)
def test_init_r_ang_lengths_match_spot_counts(dataset):
    """Every init block present must belong to its own dataset."""
    with open(dataset_init_path(dataset), "rb") as f:
        init_cfg = tomli.load(f)
    for galaxy, blk in init_cfg["model"]["galaxies"].items():
        for key, sub in blk.items():
            if not key.startswith("init"):
                continue
            assert len(sub["r_ang"]) == EXPECTED_N_SPOTS[dataset][galaxy], (
                dataset, galaxy, key)


def test_fiducial_init_set_is_complete():
    with open(dataset_init_path("fiducial"), "rb") as f:
        galaxies = tomli.load(f)["model"]["galaxies"]
    expected = {
        "CGCG074-064": {"init", "init_qw"},
        "NGC4258": {"init", "init_qw", "init_ecc_qw"},
        "NGC5765b": {"init", "init_qw"},
        "NGC6264": {"init", "init_qw"},
        "NGC6323": {"init", "init_qw"},
        "UGC3789": {"init", "init_qw"},
    }
    for galaxy, variants in expected.items():
        assert {k for k in galaxies[galaxy]
                if k.startswith("init")} == variants


# ---- apply_dataset ----

def test_apply_dataset_namespaces_root_output_idempotently():
    cfg = _config()
    apply_dataset(cfg, "fiducial")
    root = cfg["io"]["root_output"]
    assert root.endswith(os.path.join("Megamaser", "fiducial"))
    apply_dataset(cfg, "fiducial")
    assert cfg["io"]["root_output"] == root


def test_apply_dataset_rejects_cross_dataset_reapply():
    cfg = _config()
    apply_dataset(cfg, "fiducial")
    with pytest.raises(ValueError, match="already namespaced"):
        apply_dataset(cfg, "original_published")


def test_resolve_dataset_falls_back_to_config():
    assert resolve_dataset(_config()) == DEFAULT_MASER_DATASET
    assert resolve_dataset(_config(), "original_published") == \
        "original_published"
    with pytest.raises(ValueError, match="Unknown megamaser dataset"):
        resolve_dataset(_config(), "nope")


# ---- chain provenance ----

def test_chain_dataset_treats_legacy_chain_as_original_published():
    assert check_chain_dataset({}, "original_published", "old.hdf5") == \
        "original_published"


def test_chain_dataset_rejects_mismatch():
    with pytest.raises(ValueError, match="sampled on dataset 'fiducial'"):
        check_chain_dataset(
            {"dataset": b"fiducial"}, "original_published", "chain.hdf5")


# ---- init guardrails ----

class _FakeModel:
    def __init__(self, n_spots=358, galaxy="NGC4258", dataset="fiducial"):
        self.n_spots = n_spots
        self.galaxy_name = galaxy
        self.dataset = dataset


def test_check_init_block_rejects_missing_block():
    with pytest.raises(SystemExit, match="init_fiducial.toml"):
        check_init_block({}, _FakeModel())


def test_check_init_block_rejects_wrong_spot_count():
    """The direct guard on the silent zeros fallback."""
    init = {"r_ang": [1.0] * 357}
    with pytest.raises(SystemExit, match="357 r_ang values but dataset"):
        check_init_block(init, _FakeModel(n_spots=358))


def test_check_init_block_accepts_matching_block():
    init = {"r_ang": [1.0] * 358, "D_A": 7.4}
    assert check_init_block(init, _FakeModel()) is init
