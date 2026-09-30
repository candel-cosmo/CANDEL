# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""
Run inference for one task config. The config's `model/which_run` selects the
probe (from the installed CANDEL packages) that loads the data, builds the
model and runs it.

This script is expected to be run either from the command line or from a shell
submission script.
"""
import os
import subprocess
import threading
import time
from argparse import ArgumentParser
from os.path import exists

# ---- Pre-parse device args BEFORE importing anything that pulls JAX/NumPyro
_pre = ArgumentParser(add_help=False)
_pre.add_argument(
    "--host-devices", type=int,
    help="Set NumPyro host device count before importing candel."
)
_pre_args, _ = _pre.parse_known_args()

if _pre_args.host_devices:
    import numpyro  # safe to import here; must be before candel/JAX use
    numpyro.set_host_device_count(_pre_args.host_devices)

# Only now import candel (which may import jax/numpyro internally)
import candel  # noqa
from candel import fprint, get_nested  # noqa
from candel.field.field_products import cleanup_temporary_los_files  # noqa


class GPUMonitor:
    """Poll nvidia-smi in a background thread and report a summary on stop."""

    def __init__(self, interval=10):
        self.interval = interval
        self._util, self._mem_used, self._mem_total = [], [], []
        self._times = []
        self._t0 = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _query(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode()
            u, mu, mt = out.strip().split(",")
            self._util.append(float(u))
            self._mem_used.append(float(mu))
            self._mem_total.append(float(mt))
            self._times.append(time.time() - self._t0)
        except Exception:
            pass

    def _run(self):
        while not self._stop.wait(self.interval):
            self._query()

    def start(self):
        self._t0 = time.time()
        self._query()
        self._thread.start()

    def stop(self):
        self._query()
        self._stop.set()
        self._thread.join()
        if not self._util:
            return
        n = len(self._util)
        mem_total = self._mem_total[0]
        mem_pct = [100 * m / mem_total for m in self._mem_used]
        mean_util = sum(self._util) / n
        mean_mem = sum(mem_pct) / n
        peak_util = max(self._util)
        peak_mem = max(mem_pct)
        width = min(60, max(n, 2))

        def _chart(values, vmin, vmax, label):
            if vmax == vmin:
                vmax = vmin + 1
            height = 5
            lines = [f"   {label}"]
            for row in range(height):
                threshold = vmax - (row / (height - 1)) * (vmax - vmin)
                if n >= width:
                    step = n / width
                    idxs = [int(i * step) for i in range(width)]
                else:
                    idxs = [int(i * (n - 1) / max(width - 1, 1))
                            for i in range(width)]
                bar = "".join(
                    "█" if values[min(i, n - 1)] >= threshold else " "
                    for i in idxs)
                lines.append(f"   {threshold:4.0f}% |{bar}|")
            t_total = int(self._times[-1])
            t_min, t_sec = divmod(t_total, 60)
            t_label = f"{t_min}m{t_sec:02d}s" if t_min else f"{t_sec}s"
            lines.append(f"         +{'-' * width}+")
            lines.append(f"         0{t_label:>{width}}")
            return "\n".join(lines)

        fprint("── GPU usage summary ────────────────────────────────────────")
        fprint(
            f"   utilization : mean {mean_util:.0f}%,  "
            f"peak {peak_util:.0f}%")
        fprint(f"   memory      : mean {mean_mem:.1f}%,  peak {peak_mem:.1f}%"
               f"  ({max(self._mem_used):.0f} / {mem_total:.0f} MiB)")
        fprint(_chart(self._util, 0, 100, "GPU utilization (%)"))
        fprint(_chart(mem_pct, 0, 100, "GPU memory (%)"))


def insert_comment_at_top(path: str, label: str):
    if not exists(path):
        fprint(f"[WARN] cannot mark config as {label}: `{path}` is missing.")
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    comment = f"# Job {label} at: {timestamp}\n"
    with open(path, "r") as f:
        original = f.readlines()
    with open(path, "w") as f:
        f.write(comment)
        f.writelines(original)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run CANDEL inference for one task config.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    # Re-expose the pre-parsed options so they show up in --help
    parser.add_argument("--host-devices", type=int,
                        help="NumPyro host device count (handled pre-import).")
    args = parser.parse_args()
    if args.host_devices is not None:
        os.environ.setdefault("CANDEL_PPC_N_WORKERS",
                              str(args.host_devices))

    insert_comment_at_top(args.config, "started")

    gpu_monitor = GPUMonitor(interval=10)
    gpu_monitor.start()

    config = candel.load_config(args.config, replace_los_prior=False)
    which_run = get_nested(config, "model/which_run", None)
    try:
        fprint(f"selected `{which_run or 'PV'}` run.")
        candel.get_probe(which_run).run(args.config)
        insert_comment_at_top(args.config, "finished")
    finally:
        try:
            cleanup_temporary_los_files()
        finally:
            gpu_monitor.stop()
