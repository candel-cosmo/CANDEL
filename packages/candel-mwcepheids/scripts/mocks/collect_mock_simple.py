#!/usr/bin/env python
"""Collect individual mock batch results into a single combined file.

Run this after the individual mock runs have finished.

Usage:
    python scripts/mocks/collect_mock_simple.py
"""
import argparse
import os

import mock_utils
import numpy as np
from mock_utils import (SIMPLE_TASKS, TRUE_VALS, likelihood_label,
                        parse_likelihood, print_bias_table)

from candel.util import CANDEL_ROOT as REPO_ROOT  # noqa: E402
DEFAULT_OUTDIR = REPO_ROOT / "results" / "MWCepheids" / "mocks"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--sigma-int", type=float, default=None,
                        help="sigma_int tag used in the run labels")
    parser.add_argument("--keep", action="store_true",
                        help="Keep intermediate files after combining")
    args = parser.parse_args()

    sigma_int_tag = (f"{args.sigma_int:.3f}"
                     if args.sigma_int is not None else None)

    # Build expected labels
    labels = []
    for campaign, likelihood in SIMPLE_TASKS:
        use_gaussian = parse_likelihood(likelihood)
        label = f"{campaign}_{likelihood_label(use_gaussian)}"
        if sigma_int_tag is not None:
            label += f"_sint{sigma_int_tag}"
        labels.append(label)

    # Load results
    all_results = {}
    files = []
    missing = []
    for label in labels:
        fpath = f"{args.outdir}/mock_{label}.npz"
        if not os.path.exists(fpath):
            missing.append(label)
            continue
        data = np.load(fpath)
        biases = {p: data[p] for p in TRUE_VALS if p in data}
        all_results[label] = biases
        files.append(fpath)

    if missing:
        print(f"[WARN] Missing results for: {', '.join(missing)}")
        print("       Are all jobs finished?")

    if not all_results:
        print("[ERROR] No results found. Exiting.")
        return

    # Print summary table
    first = np.load(files[0])
    n_mocks = int(first["n_mocks"]) if "n_mocks" in first else None

    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    if n_mocks is not None:
        print(f"  n_mocks   = {n_mocks}")
    if sigma_int_tag is not None:
        print(f"  sigma_int = {args.sigma_int}")
    else:
        print(f"  sigma_int = {mock_utils.sigma_int} (default)")
    print()

    # Strip sigma_int suffix from labels for display
    sint_suffix = f"_sint{sigma_int_tag}" if sigma_int_tag is not None else ""
    print_bias_table({label.removesuffix(sint_suffix): biases
                      for label, biases in all_results.items()})

    # Save combined file
    combined = {}
    for label, biases in all_results.items():
        for p, vals in biases.items():
            combined[f"{label}/{p}"] = vals
    combined["labels"] = np.array(list(all_results.keys()))
    combined["params"] = np.array(list(TRUE_VALS))

    # Get n_mocks and sigma_int from first file
    if "n_mocks" in first:
        combined["n_mocks"] = first["n_mocks"]
    if "sigma_int" in first:
        combined["sigma_int"] = first["sigma_int"]

    tag = f"_sint{sigma_int_tag}" if sigma_int_tag is not None else ""
    outfile = f"{args.outdir}/mock_batch{tag}.npz"
    np.savez(outfile, **combined)
    print(f"\n[INFO] Combined results saved to {outfile}")

    if not args.keep:
        confirm = input("\nRemove intermediate files? [y/N]: ")
        if confirm.strip().lower() == "y":
            for f in files:
                os.remove(f)
                print(f"[INFO] Removed {f}")
        else:
            print("[INFO] Keeping intermediate files.")


if __name__ == "__main__":
    main()
