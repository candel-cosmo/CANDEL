"""The core stays independent of the probe packages that register with it."""
import subprocess
import sys
from pathlib import Path

from candel.probe import probes, task_specs


def test_core_does_not_import_probe_packages():
    code = ("import sys, candel, candel.field.los_prep; "
            "print([m for m in sys.modules if m.startswith('candel_')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_installed_probes_have_runnable_task_specs():
    assert probes(), "no probe packages installed"
    for name, spec in task_specs().items():
        assert Path(spec["config_path"]).is_file(), name
