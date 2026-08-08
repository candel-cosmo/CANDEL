import shutil
import subprocess
from pathlib import Path

from scripts.mocks.mock_TRGB import _expected_mpi_tasks_from_env


REPO_ROOT = Path(__file__).resolve().parents[1]


def _mock_submit_checkout(tmp_path):
    root = tmp_path / "CANDEL"
    (root / "scripts/mocks").mkdir(parents=True)
    for relative in (
            "scripts/_submit_lib.sh",
            "scripts/_cluster_glamdring.sh",
            "scripts/mocks/mock_TRGB.sh",
            "scripts/mocks/submit_TRGBH0_mock_bias.sh"):
        source = REPO_ROOT / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (root / "local_config.toml").write_text(
        'machine = "glamdring"\npython_exec = "/usr/bin/python3"\n',
        encoding="utf-8")
    return root


def test_gpulong_queue_uses_exact_two_cpu_gpu_shards(tmp_path):
    root = _mock_submit_checkout(tmp_path)
    result = subprocess.run(
        [root / "scripts/mocks/submit_TRGBH0_mock_bias.sh",
         "-q", "gpulong", "--n-mocks", "100", "--master-seed", "42",
         "--gpu-shards", "10", "--dry"],
        input="y\n", text=True, capture_output=True, check=True)

    assert "Mode:        GPU shards" in result.stdout
    assert "GPU shard:   1 GPU, 2 CPU cores" in result.stdout
    assert "GPU queues:  gpulong" in result.stdout
    assert "GPU jobs:    10" in result.stdout
    assert "Mocks/job:   10" in result.stdout
    assert "Field index: 42" in result.stdout
    assert result.stdout.count("--gpus 1 -n 2") == 10
    assert result.stdout.count("CANDEL_MOCK_SEQUENTIAL=1") == 10
    assert result.stdout.count("mocks=10") == 10


def test_gpu_shards_require_exact_division(tmp_path):
    root = _mock_submit_checkout(tmp_path)
    result = subprocess.run(
        [root / "scripts/mocks/submit_TRGBH0_mock_bias.sh",
         "-q", "gpulong", "--n-mocks", "100", "--gpu-shards", "6",
         "--dry"],
        input="y\n", text=True, capture_output=True)

    assert result.returncode == 1
    assert "must be divisible by --gpu-shards (6)" in result.stderr


def test_gpu_cpu_slots_do_not_imply_mpi_ranks(monkeypatch):
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("CANDEL_MOCK_SEQUENTIAL", "1")
    assert _expected_mpi_tasks_from_env() == 1

    monkeypatch.delenv("CANDEL_MOCK_SEQUENTIAL")
    assert _expected_mpi_tasks_from_env() == 2
