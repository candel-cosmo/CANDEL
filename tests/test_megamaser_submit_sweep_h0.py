import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_standard_sweep_forwards_quadratic_warp():
    result = subprocess.run(
        ["bash", "scripts/megamaser/submit_sweep_H0.sh", "--local",
         "--dataset", "original_published", "--add-quadratic-warp", "--dry"],
        cwd=ROOT, text=True, capture_output=True, check=True)
    commands = [line for line in result.stdout.splitlines()
                if line.startswith("[dry]")]
    assert len(commands) == 11
    assert all("--add-quadratic-warp" in line for line in commands)
