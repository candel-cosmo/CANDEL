"""The core stays independent of the probe packages that register with it."""
import subprocess
import sys
from pathlib import Path

import pytest

from candel import load_config
from candel.probe import get_probe, probes, task_specs


def test_core_does_not_import_probe_packages():
    code = ("import sys, candel, candel.field.los_prep; "
            "print([m for m in sys.modules if m.startswith('candel_')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_installed_probes_have_runnable_task_specs():
    if not probes():
        pytest.skip("no probe packages installed")
    for name, spec in task_specs().items():
        assert Path(spec["config_path"]).is_file(), name


def test_explicit_pv_which_run_matches_unset(tmp_path):
    if None not in probes():
        pytest.skip("candel-pv not installed")
    assert get_probe("PV") is get_probe(None)
    for which_run in ('', 'which_run = "PV"'):
        path = tmp_path / "run.toml"
        path.write_text(f"[model]\n{which_run}\n")
        config = load_config(path, fill_paths=False)
        assert config["model"].get("which_run") is None
