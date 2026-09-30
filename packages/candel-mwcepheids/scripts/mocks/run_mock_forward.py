#!/usr/bin/env python
"""Batch mock runner for full forward model closure tests (MPI version).

Generates synthetic Cepheid data and runs MWCepheidModel inference,
checking that true parameters are recovered without bias.

Rank 0 is the master that distributes seeds; ranks 1..N are workers
that each JIT-compile once and reuse the cache for all subsequent mocks.

Usage:
    mpirun -np 28 python packages/candel-mwcepheids/scripts/mocks/run_mock_forward.py \
        --n-mocks 1000 --campaigns C22 C27
"""
import argparse
import logging
import os
import signal
import sys
import time
import tomllib
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from mock_forward_utils import MOCK_CFG_MW, MOCK_CFG_PI, generate_mock_forward
from mock_utils import TRUE_PARAMS
from mpi4py import MPI

from candel.inference import run_inference
from candel_mwcepheids import MWCepheidModel

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTDIR = REPO_ROOT / "results" / "MWCepheids" / "mocks"

ALL_TASKS = {
    "C22": ("C22", SCRIPT_DIR / "config_mock_mW.toml", MOCK_CFG_MW),
    "C27": ("C27", SCRIPT_DIR / "config_mock_pi.toml", MOCK_CFG_PI),
}

# Global parameters (same name in TRUE_PARAMS and the NumPyro model) and
# per-campaign population hyperparameters (suffixed with the campaign name)
PRIMARY_PARAMS = ["M_H_1", "b_W", "Z_W", "delta_pi", "sigma_int"]
POPULATION_PARAMS = ["mu_logP", "sigma_logP", "mu_OH", "sigma_OH"]

# Remap forward model names to match simplified model keys
PARAM_REMAP = {"M_H_1": "MWH", "b_W": "bW", "Z_W": "ZW"}

TAG_WORK = 1
TAG_RESULT = 2
TAG_DONE = 3


def load_mock_config(toml_path):
    """Load a mock config TOML."""
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


def run_one_mock_forward(seed, campaign, toml_path, mock_cfg,
                         true_params, quiet=True):
    """Run one mock: generate data, run inference, compute biases."""
    data, n_parent, n_sel = generate_mock_forward(
        seed, true_params, mock_cfg, campaign)

    mw_log = logging.getLogger("candel_mwcepheids")
    prev_level = mw_log.level
    if quiet:
        mw_log.setLevel(logging.ERROR)

    config = load_mock_config(toml_path)
    # Override inference seed per mock for independent chains
    config["inference"]["seed"] = seed

    # Model period cut must equal the one used to generate the mock
    if mock_cfg.get("logP_min") is not None:
        config["model"][campaign]["selection"]["logP_min"] = float(
            mock_cfg["logP_min"])

    model = MWCepheidModel(config, data)
    mcmc, samples = run_inference(
        model, print_summary=not quiet, save_samples=False,
        progress_bar=not quiet, return_mcmc=True)
    mw_log.setLevel(prev_level)

    biases = {}
    # Primary parameters (same name in true_params and NumPyro)
    for param in PRIMARY_PARAMS:
        samp = np.asarray(samples[param])
        biases[param] = (samp.mean() - true_params[param]) / samp.std()

    # Population hyperparameters (suffixed with campaign in NumPyro); their
    # true values are the campaign-specific generator settings.
    for param in POPULATION_PARAMS:
        key = f"{param}_{campaign}"
        samp = np.asarray(samples[key])
        biases[key] = (samp.mean() - mock_cfg[param]) / samp.std()

    # Remap parameter names
    for old, new in PARAM_REMAP.items():
        if old in biases:
            biases[new] = biases.pop(old)

    return biases, n_parent, n_sel


def _make_label(campaign, mock_cfg, sigma_int, logP_min=None):
    """Build a unique label encoding campaign, selection, and sigma_int."""
    sel = "mW" if mock_cfg.get("mW_max") is not None else "pi"
    label = f"{campaign}_{sel}_si{sigma_int:.4f}"
    if logP_min is not None:
        label += f"_logPmin{logP_min:.3f}"
    return label


def master(comm, n_workers, n_mocks, campaign, toml_path, mock_cfg,
           true_params, outdir, logP_min=None):
    """Rank 0: distribute seeds to workers and collect results."""
    label = _make_label(campaign, mock_cfg, true_params["sigma_int"],
                        logP_min)

    print(f"\n{'=' * 60}")
    print(f"[INFO] {label}: {n_mocks} mocks, {n_workers} workers")

    t0 = time.time()
    seeds = list(range(n_mocks))
    results = []
    n_sent = 0
    n_done = 0
    n_skipped = 0
    status = MPI.Status()

    # Send initial work to each worker
    for rank in range(1, n_workers + 1):
        if n_sent < n_mocks:
            comm.send(seeds[n_sent], dest=rank, tag=TAG_WORK)
            n_sent += 1
        else:
            comm.send(None, dest=rank, tag=TAG_DONE)

    # Collect results and send more work
    while n_done < n_mocks:
        result = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT,
                           status=status)
        source = status.Get_source()
        n_done += 1

        elapsed = time.time() - t0
        now = datetime.now().strftime("%H:%M:%S")
        if result is None:
            n_skipped += 1
            print(f"[WARN {now}] {label}: {n_done}/{n_mocks} — seed failed "
                  f"or timed out on rank {source} ({elapsed:.0f}s)",
                  flush=True)
        else:
            results.append(result)
            print(f"[INFO {now}] {label}: {n_done}/{n_mocks} done "
                  f"({elapsed:.0f}s)", flush=True)

        if n_sent < n_mocks:
            comm.send(seeds[n_sent], dest=source, tag=TAG_WORK)
            n_sent += 1
        else:
            comm.send(None, dest=source, tag=TAG_DONE)

    elapsed = time.time() - t0
    print(f"[INFO] {label}: done in {elapsed:.0f}s "
          f"({elapsed / n_mocks:.1f}s/mock)")
    if n_skipped:
        print(f"[WARN] {label}: {n_skipped}/{n_mocks} mocks skipped "
              f"(failed or timed out)")

    # Collect biases (use remapped names)
    all_params = [PARAM_REMAP.get(p, p) for p in PRIMARY_PARAMS] + [
        f"{p}_{campaign}" for p in POPULATION_PARAMS]
    biases = {p: [] for p in all_params}
    n_selected_list = []
    for b, _, n_sel in results:
        n_selected_list.append(n_sel)
        for p in all_params:
            if p in b:
                biases[p].append(b[p])

    if n_selected_list:
        print(f"[INFO] {label}: median N_selected = "
              f"{np.median(n_selected_list):.0f}")

    save_dict = {}
    param_names = []
    for p in list(biases):
        if biases[p]:
            biases[p] = np.array(biases[p])
            save_dict[p] = biases[p]
            param_names.append(p)
        else:
            del biases[p]

    save_dict["params"] = np.array(param_names)
    save_dict["campaign"] = np.array(campaign)
    save_dict["n_mocks"] = np.array(n_mocks)
    save_dict["n_skipped"] = np.array(n_skipped)
    save_dict["n_selected"] = np.array(n_selected_list)

    outfile = os.path.join(outdir, f"mock_forward_{label}.npz")
    np.savez(outfile, **save_dict)
    print(f"[INFO] Saved to {outfile}")

    return label, biases, param_names, n_skipped


def worker(comm, campaign, toml_path, mock_cfg, true_params, timeout):
    """Ranks 1..N: receive seeds, run mocks, send back results."""
    def _alarm_handler(signum, frame):
        raise TimeoutError

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)

    while True:
        status = MPI.Status()
        seed = comm.recv(source=0, status=status)

        if status.Get_tag() == TAG_DONE:
            break

        if timeout > 0:
            signal.alarm(timeout)
        try:
            result = run_one_mock_forward(
                seed, campaign, toml_path, mock_cfg, true_params, quiet=True)
        except TimeoutError:
            result = None
        except Exception:
            # Report the failure instead of leaving the master waiting.
            print(f"[ERROR] seed {seed} failed:\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
            result = None
        finally:
            signal.alarm(0)

        comm.send(result, dest=0, tag=TAG_RESULT)

    signal.signal(signal.SIGALRM, old_handler)


def print_table(all_results):
    """Print summary table of mean standardised biases."""
    # Collect all param names
    all_params = []
    for _, biases, params in all_results:
        for p in params:
            if p not in all_params:
                all_params.append(p)

    w = max(len(label) for label, _, _ in all_results) + 2
    header = f"{'Run':<{w}s}" + "".join(f"{p:>20s}" for p in all_params)
    print(f"\n{'=' * 60}")
    print("Summary: mean standardised bias +/- std")
    print("=" * 60)
    print(header)
    print("-" * len(header))
    for label, biases, _ in all_results:
        row = f"{label:<{w}s}"
        for p in all_params:
            if p in biases:
                b = biases[p]
                row += f"{f'{b.mean():+.2f} +/- {b.std():.2f}':>20s}"
            else:
                row += f"{'---':>20s}"
        print(row)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_workers = size - 1

    # Rank 0 parses arguments and broadcasts config
    if rank == 0:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--n-mocks", type=int, default=50)
        parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
        parser.add_argument("--campaigns", nargs="+",
                            default=list(ALL_TASKS),
                            choices=list(ALL_TASKS),
                            help="Campaigns to run (default: all)")
        parser.add_argument("--sigma-int", type=float, default=None,
                            help="Override intrinsic scatter (default: use "
                            "TRUE_PARAMS value)")
        parser.add_argument("--timeout", type=int, default=900,
                            help="Per-mock wall-time limit in seconds "
                            "(default: 900 = 15 min, 0 = no limit)")
        parser.add_argument("--logP-min", type=float, default=None,
                            help="Override logP_min selection threshold "
                            "(default: use config value)")
        args = parser.parse_args()

        if n_workers < 1:
            print("[ERROR] Need at least 2 MPI ranks (1 master + 1 worker).")
            sys.exit(1)

        tp = dict(TRUE_PARAMS)
        if args.sigma_int is not None:
            tp["sigma_int"] = args.sigma_int
        print(f"[INFO] sigma_int = {tp['sigma_int']}")
        if args.logP_min is not None:
            print(f"[INFO] logP_min = {args.logP_min}")
        print(f"[INFO] timeout = {args.timeout}s"
              if args.timeout > 0 else "[INFO] timeout = none")
        print(f"[INFO] MPI size = {size} ({n_workers} workers)")

        os.makedirs(args.outdir, exist_ok=True)

        tasks = []
        for c in args.campaigns:
            campaign, toml_path, mock_cfg = ALL_TASKS[c]
            # The override applies only to campaigns with a period cut; it
            # is used for both data generation and the model.
            logP_min = None
            if (args.logP_min is not None
                    and mock_cfg.get("logP_min") is not None):
                logP_min = args.logP_min
                mock_cfg = dict(mock_cfg, logP_min=logP_min)
            tasks.append((campaign, toml_path, mock_cfg, logP_min))

        config = {
            "n_mocks": args.n_mocks,
            "outdir": args.outdir,
            "true_params": tp,
            "tasks": tasks,
            "timeout": args.timeout,
        }
    else:
        config = None

    config = comm.bcast(config, root=0)

    n_mocks = config["n_mocks"]
    outdir = config["outdir"]
    tp = config["true_params"]
    tasks = config["tasks"]
    timeout = config["timeout"]

    all_results = []
    labels = []
    total_skipped = 0
    for campaign, toml_path, mock_cfg, logP_min in tasks:
        if rank == 0:
            label, biases, params, n_skipped = master(
                comm, n_workers, n_mocks, campaign, toml_path, mock_cfg,
                tp, outdir, logP_min)
            all_results.append((label, biases, params))
            labels.append(label)
            total_skipped += n_skipped
        else:
            worker(comm, campaign, toml_path, mock_cfg, tp, timeout)

    single_task = len(tasks) == 1

    if rank == 0:
        print_table(all_results)
        if total_skipped:
            print(f"\n[WARN] Total skipped (failed or timed out): "
                  f"{total_skipped}")

        # In single-task mode, keep the per-task .npz (a separate
        # collector or multi-campaign run will combine them later).
        if not single_task:
            combined = {}
            for label, biases, _ in all_results:
                for p, vals in biases.items():
                    combined[f"{label}/{p}"] = vals
            combined["labels"] = np.array(labels)
            combined["n_mocks"] = np.array(n_mocks)

            si_tag = f"_si{tp['sigma_int']:.4f}"
            outfile = os.path.join(
                outdir, f"mock_forward_batch{si_tag}.npz")
            np.savez(outfile, **combined)
            print(f"\n[INFO] Combined results saved to {outfile}")

            for label in labels:
                f = os.path.join(outdir, f"mock_forward_{label}.npz")
                if os.path.exists(f):
                    os.remove(f)
                    print(f"[INFO] Removed {f}")


if __name__ == "__main__":
    main()
