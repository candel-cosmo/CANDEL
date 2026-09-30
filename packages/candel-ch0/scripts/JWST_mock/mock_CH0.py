"""CH0 JWST forecast with mocks in a single Manticore realisation.

Mock hosts have Cepheid distance moduli known up to the common offset M_W
(see `candel_ch0.mock`), redshifts from one field realisation plus a
fixed 150 km/s scatter, and known redshift-window selections. The same
realisation is used to analyse every mock.

Steps (run from the repository root):
    python packages/candel-ch0/scripts/JWST_mock/mock_CH0.py pool
    python packages/candel-ch0/scripts/JWST_mock/mock_CH0.py run --scenario M0 --seeds 0-9
    python packages/candel-ch0/scripts/JWST_mock/mock_CH0.py collect
"""
import argparse
import inspect
import os
import tempfile
from os.path import exists, join
from pathlib import Path

import numpy as np
import tomli_w

import candel
import candel_ch0
from candel.field import name2field_loader
from candel_ch0.mock import (CZ_NO_LOWER_LIMIT, build_CH0_host_pool,
                             draw_CH0_hosts, make_CH0_mock_data)

BASE_CONFIG = ("scripts/runs/generated_configs/CH0_JWST_forecast_sigv150/"
               "CH0_MAS-PCS_sel-redshift_ManticoreLocalCOLA_jwst.toml")
OUT = "results/CH0_paper/jwst_forecast/mocks"
FIELD = "ManticoreLocalCOLA"
FIELD_INDEX = 0
SEL_RADIUS = 70.0      # Mpc/h, covers cz ~ 5000 km/s plus the edge width
POOL_SIZE = 50_000
CZ_WIDTH = 300.0
SIGMA_V = 150.0
E_CZ = 10.0

# Posterior means of the fiducial redshift-selection Manticore fit.
TRUTH = {
    "H0": 72.5, "M_W": -5.90, "mu_N4258": 29.39, "mu_LMC": 18.48,
    "beta": 1.0, "Vext_mag": 262.0, "Vext_ell": 302.0, "Vext_b": -27.0,
    "bias": [2.26, 0.66, 0.68, 0.57],   # double power law
}


def _sample(n, f, low, high):
    return {"n": n, "sigma_factor": f, "cz_low": low, "cz_high": high}


def _scenarios():
    primary = (CZ_NO_LOWER_LIMIT, 3300.0)
    far = (3300.0, 5000.0)
    sc = {"M0": [_sample(35, 1.0, *primary)],
          "M1": [_sample(35, 0.5, *primary)]}
    for k in (5, 10, 20):
        for tag, f in (("", 0.5), ("h", 1.0)):
            sc[f"M2{tag}_k{k}"] = [_sample(35, 1.0, *primary),
                                   _sample(k, f, *primary)]
            sc[f"M3{tag}_k{k}"] = [_sample(35, 1.0, *primary),
                                   _sample(k, f, *far)]
    return sc


SCENARIOS = _scenarios()


def make_config(num_warmup, num_samples):
    config = candel.load_config(BASE_CONFIG, replace_none=False,
                                replace_los_prior=False)
    config["io"]["field_indices"] = [FIELD_INDEX]
    config["model"]["selection_integral_grid_radius"] = SEL_RADIUS
    config["model"]["cz_lim_selection_width"] = CZ_WIDTH
    # Mock rows have log P = [O/H] = 0, so b_W and Z_W do not enter; delta
    # priors remove them from sampling.
    for name in ("b_W", "Z_W"):
        config["model"]["priors"][name] = {"dist": "delta", "value": 0.0}
    config["model"]["priors"]["sigma_v"] = {"dist": "delta",
                                           "value": SIGMA_V}
    config["inference"].update({
        "num_chains": 1, "num_warmup": num_warmup,
        "num_samples": num_samples, "compute_log_density": False,
        "compute_evidence": False})
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, f)
    f.close()
    return config, f.name


def load_template(config_path):
    """Real SH0ES data dict with the field-k selection volume at 70 Mpc/h."""
    return candel_ch0.load_SH0ES_from_config(config_path)


def cmd_pool(args):
    config, path = make_config(10, 10)
    try:
        template = load_template(path)     # also warms the volume cache
    finally:
        os.unlink(path)
    field_config = dict(config["io"]["reconstruction_main"][FIELD])
    loader_cls = name2field_loader(FIELD)
    if "nsim" in inspect.signature(loader_cls.__init__).parameters:
        field_config.setdefault("nsim", FIELD_INDEX)
    loader = loader_cls(**field_config)
    pool = build_CH0_host_pool(
        loader, SEL_RADIUS, POOL_SIZE, TRUTH["bias"], "double_powerlaw",
        np.asarray(template["host_los_r"]), np.random.default_rng(1))
    os.makedirs(OUT, exist_ok=True)
    np.savez(join(OUT, f"pool_field{FIELD_INDEX}.npz"), **pool)
    print(f"saved pool of {len(pool['r_h'])} hosts.")


def _parse_seeds(text):
    lo, _, hi = text.partition("-")
    return range(int(lo), int(hi or lo) + 1)


def cmd_run(args):
    samples = SCENARIOS[args.scenario]
    config, path = make_config(args.num_warmup, args.num_samples)
    pool = dict(np.load(join(OUT, f"pool_field{FIELD_INDEX}.npz")))
    try:
        template = load_template(path)
        for seed in _parse_seeds(args.seeds):
            fout = join(OUT, f"{args.scenario}_seed{seed:03d}.npz")
            if exists(fout):
                continue
            gen = np.random.default_rng(seed)
            hosts = draw_CH0_hosts(pool, samples, TRUTH, CZ_WIDTH, SIGMA_V,
                                   E_CZ, gen)
            data = make_CH0_mock_data(template, pool, hosts, TRUTH, E_CZ,
                                      FIELD_INDEX, gen)
            model = candel_ch0.CH0Model(path, data)
            post = candel.run_inference(model, save_samples=False,
                                        print_summary=False,
                                        progress_bar=False)
            np.savez(fout, H0=np.asarray(post["H0"]),
                     M_W=np.asarray(post["M_W"]),
                     n_hosts=len(hosts["cz_obs"]), H0_true=TRUTH["H0"])
            H0 = np.asarray(post["H0"])
            print(f"{args.scenario} seed {seed}: H0 = {H0.mean():.2f} "
                  f"+- {H0.std():.2f}", flush=True)
    finally:
        os.unlink(path)


def cmd_collect(args):
    print("| Scenario | runs | median sigma(H0) | mean (H0 - true)/sigma |")
    print("|---|---|---|---|")
    for name in SCENARIOS:
        files = sorted(f for f in os.listdir(OUT)
                       if f.startswith(f"{name}_seed"))
        if not files:
            continue
        sig, pull = [], []
        for f in files:
            d = np.load(join(OUT, f))
            sig.append(d["H0"].std())
            pull.append((d["H0"].mean() - d["H0_true"]) / d["H0"].std())
        pull = np.asarray(pull)
        print(f"| {name} | {len(files)} | {np.median(sig):.2f} | "
              f"{pull.mean():+.2f} +- {pull.std() / np.sqrt(len(pull)):.2f}"
              " |")


def main():
    # Paths are relative to the repository root.
    os.chdir(Path(__file__).resolve().parents[4])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool")
    run = sub.add_parser("run")
    run.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    run.add_argument("--seeds", default="0-9")
    run.add_argument("--num-warmup", type=int, default=1000)
    run.add_argument("--num-samples", type=int, default=2000)
    sub.add_parser("collect")
    args = parser.parse_args()
    {"pool": cmd_pool, "run": cmd_run, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
