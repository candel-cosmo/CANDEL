"""CH0 JWST forecast: the real 35 SH0ES hosts plus five synthetic far hosts.

Each added host has one effective Cepheid row, `mag = mu_true + M_W`, with
log P = [O/H] = 0 (so b_W and Z_W do not enter), a 0.05 mag error (by
default) and no covariance with the real Cepheids. Its true distance is the
cosmological distance of its observed CMB-frame redshift at the fiducial H0,
and its mock redshift adds the Manticore-Local peculiar velocity at that
distance, plus the fiducial Vext. With `--field K` a single realisation sets
the velocities and is used in the inference; otherwise all realisations are
marginalised over and the velocities are their mean. The rows are noise-free, so only
the posterior widths are forecasts. The added hosts are selected in a
separate redshift window at 3300--5500 km/s.

Steps (run from the repository root):
    python scripts/CH0_JWST_mock/ch0_new_hosts.py prep
    python scripts/CH0_JWST_mock/ch0_new_hosts.py run --scenario HST_plus5
    python scripts/CH0_JWST_mock/ch0_new_hosts.py collect
"""
import argparse
import inspect
import os
import tempfile
from os.path import exists, join

import numpy as np
import tomli_w
from scipy.linalg import block_diag, cholesky

import candel
from candel.cosmo.cosmography import Distance2Distmod, Distance2Redshift
from candel.field import interpolate_los_density_velocity, name2field_loader
from candel.mock.CH0_mock import CZ_NO_LOWER_LIMIT
from candel.util import (SPEED_OF_LIGHT, galactic_to_radec_cartesian,
                         heliocentric_to_cmb)

CONFIG_DIR = "scripts/runs/generated_configs/CH0_JWST_forecast_sigv150"
OUT = candel.results_path("results/CH0_paper/jwst_forecast/new_hosts")
FIELD = "ManticoreLocalCOLA"
E_CZ = 10.0
CZ_LOW, CZ_HIGH = 3300.0, 5500.0
# Covers the far window plus its smoothing; the host LOS grid ends at 75.
SEL_RADIUS = 75.0

# Posterior means of the fiducial redshift-selection Manticore fit
# (sigma_v = 150 km/s, beta = 1).
TRUTH = {"H0": 72.33, "M_W": -5.90, "b_W": -3.30, "Vext_mag": 261.7,
         "Vext_ell": 303.5, "Vext_b": -27.6}

# Name, RA, dec (deg), heliocentric velocity (km/s), from CDS Sesame.
NEW_HOSTS = [
    ("N0105", 6.31980, 12.88386, 5174.0),
    ("N0976", 38.50016, 20.97675, 4284.0),
    ("N7038", 318.78124, -47.22047, 4929.6),
    ("N6956", 310.97385, 12.51190, 0.015561 * SPEED_OF_LIGHT),
    ("E478-6", 32.32531, -23.41507, 5171.4),
]

SCENARIOS = {
    # Real 35 hosts only, but with the per-host window code path and the
    # larger selection radius: must reproduce the Table 1 baseline.
    "HST_windows": ("", False),
    "HST_plus5": ("", True),
    "JWST35_plus5": ("_ceperr0p5-35", True),
    "JWST36_plus5": ("_ceperr0p5-36", True),
}


def base_config_path(variant):
    return join(CONFIG_DIR,
                f"CH0_MAS-PCS_sel-redshift{variant}_ManticoreLocalCOLA_jwst"
                ".toml")


def make_config(variant, num_warmup, num_samples, field=None):
    config = candel.load_config(base_config_path(variant), replace_none=False,
                                replace_los_prior=False)
    if field is not None:
        config["io"]["field_indices"] = [int(field)]
    config["model"]["selection_integral_grid_radius"] = SEL_RADIUS
    config["inference"].update({
        "num_chains": 1, "num_warmup": num_warmup,
        "num_samples": num_samples, "compute_log_density": False,
        "compute_evidence": False})
    f = tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False)
    tomli_w.dump(config, f)
    f.close()
    return config, f.name


def cmd_prep(args):
    config, path = make_config("", 10, 10)
    try:
        data = candel.pvdata.load_SH0ES_from_config(path)
    finally:
        os.unlink(path)
    r_h = np.asarray(data["host_los_r"])
    fields = np.asarray(data["host_los_field_indices"]).ravel()
    names = [h[0] for h in NEW_HOSTS]
    RA = np.array([h[1] for h in NEW_HOSTS])
    dec = np.array([h[2] for h in NEW_HOSTS])
    z_cmb = heliocentric_to_cmb(np.array([h[3] for h in NEW_HOSTS])
                                / SPEED_OF_LIGHT, RA, dec)

    field_config = dict(config["io"]["reconstruction_main"][FIELD])
    loader_cls = name2field_loader(FIELD)
    assert "nsim" in inspect.signature(loader_cls.__init__).parameters
    dens, vel = [], []
    for k in fields:
        loader = loader_cls(nsim=int(k), **field_config)
        d, v = interpolate_los_density_velocity(loader, r_h, RA, dec,
                                                verbose=False)
        dens.append(d)
        vel.append(v)
        print(f"interpolated field {k}.", flush=True)
    dens = np.asarray(dens, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)

    h = TRUTH["H0"] / 100
    r_Mpc_grid = np.linspace(1.0, 150.0, 20000)
    z_grid = np.asarray(Distance2Redshift()(r_Mpc_grid, h=h))
    r_true = np.interp(z_cmb, z_grid, r_Mpc_grid)
    new = {"names": np.array(names), "RA": RA, "dec": dec,
           "cz_cmb": z_cmb * SPEED_OF_LIGHT, "r_true": r_true,
           "mu_true": np.asarray(Distance2Distmod()(
               r_true, h=h)),
           "los_density": dens, "los_velocity": vel, "los_r": r_h,
           "field_indices": fields}
    os.makedirs(OUT, exist_ok=True)
    np.savez(join(OUT, "new_hosts.npz"), **new)

    # N0105 and N0976 are also in the SH0ES LOS cache (beyond the 3300 km/s
    # cut): the on-the-fly interpolation must match it.
    all_names = [h.removeprefix("mu_") for h in data["q_names"][:37]]
    for i, name in enumerate(names[:2]):
        j = all_names.index(name)
        for key, arr, atol in (("density", dens, 1e-2),
                               ("velocity", vel, 1.0)):
            cache = np.asarray(data[f"host_los_{key}"])[:, j]
            err = np.max(np.abs(arr[:, i] - cache))
            print(f"{name} LOS {key}: max |on-the-fly - cache| = {err:.3g}")
            assert err < atol, (name, key, err)

    print_truth(new, fields)


def select_fields(new, fields):
    """Restrict the new-host LOS to `fields` and set their mock redshifts.

    The true distance is the cosmological distance of the observed CMB-frame
    redshift. The peculiar velocity is the field radial velocity there,
    averaged over the selected realisations, plus the fiducial Vext.
    """
    new = dict(new)
    idx = np.searchsorted(new["field_indices"], fields)
    if not np.array_equal(new["field_indices"][idx], fields):
        raise ValueError(f"Fields {fields} not in the new-host LOS.")
    for key in ("los_density", "los_velocity"):
        new[key] = new[key][idx]
    new["field_indices"] = np.asarray(fields)
    h = TRUTH["H0"] / 100
    RA, dec = np.deg2rad(new["RA"]), np.deg2rad(new["dec"])
    rhat = np.stack([np.cos(dec) * np.cos(RA), np.cos(dec) * np.sin(RA),
                     np.sin(dec)], axis=1)
    Vext = TRUTH["Vext_mag"] * np.asarray(galactic_to_radec_cartesian(
        TRUTH["Vext_ell"], TRUTH["Vext_b"])).reshape(3)
    new["v_field"] = np.array([
        np.interp(new["r_true"][i] * h, new["los_r"],
                  new["los_velocity"][:, i].mean(0))
        for i in range(len(new["names"]))])
    new["Vpec"] = new["v_field"] + rhat @ Vext
    z_cos = new["cz_cmb"] / SPEED_OF_LIGHT
    new["cz_obs"] = SPEED_OF_LIGHT * (
        (1 + z_cos) * (1 + new["Vpec"] / SPEED_OF_LIGHT) - 1)
    return new


def print_truth(new, fields):
    new = select_fields(new, fields)
    print("| host | cz_cmb | r [Mpc] | V_field | V_pec | cz_mock | mu |")
    for i, name in enumerate(new["names"]):
        print(f"| {name} | {new['cz_cmb'][i]:.0f} | {new['r_true'][i]:.1f} "
              f"| {new['v_field'][i]:.0f} | {new['Vpec'][i]:.0f} | "
              f"{new['cz_obs'][i]:.0f} | {new['mu_true'][i]:.3f} |")


def add_hosts(data, new, M_W, sigma_mu, logP=0.0):
    """Append synthetic hosts with one effective Cepheid row each.

    With `logP = 0` the rows sit at the P-L pivot, so b_W does not enter;
    otherwise they are placed at `mag = mu + M_W + b_W logP` with the
    fiducial b_W, which makes them depend on b_W like real Cepheids.
    """
    data = dict(data)
    n0, k = int(data["num_hosts"]), len(new["names"])
    if not np.array_equal(np.asarray(data["host_los_field_indices"]).ravel(),
                          new["field_indices"]):
        raise ValueError("New-host LOS fields do not match the data.")

    def insert_host_cols(L):
        return np.hstack([L[:, :n0], np.zeros((len(L), k)), L[:, n0:]])

    L_new = np.zeros((k, n0 + k + 3))
    L_new[np.arange(k), n0 + np.arange(k)] = 1.0
    C = block_diag(data["C_Cepheid"], sigma_mu**2 * np.eye(k))
    PV = data["PV_covmat_cepheid_host"]
    m = np.asarray(data["mask_host"])
    data.update({
        "mag_cepheid": np.concatenate([data["mag_cepheid"],
                                       new["mu_true"] + M_W
                                       + TRUTH["b_W"] * logP]),
        "logP": np.concatenate([data["logP"], np.full(k, logP)]),
        "OH": np.concatenate([data["OH"], np.zeros(k)]),
        "C_Cepheid": C,
        "L_Cepheid": cholesky(C, lower=True),
        "L_Cepheid_host_dist": np.vstack([
            insert_host_cols(data["L_Cepheid_host_dist"]), L_new]),
        "L_SN_unique_Cepheid_host_dist": insert_host_cols(
            data["L_SN_unique_Cepheid_host_dist"]),
        "num_cepheids": int(data["num_cepheids"]) + k,
        "num_hosts": n0 + k,
        "host_names": np.concatenate([data["host_names"],
                                      [f"mu_{n}" for n in new["names"]]]),
        "czcmb_cepheid_host": np.concatenate([data["czcmb_cepheid_host"],
                                              new["cz_obs"]]),
        "e_czcmb_cepheid_host": np.concatenate([
            data["e_czcmb_cepheid_host"], np.full(k, E_CZ)]),
        "RA_host": np.concatenate([data["RA_host"], new["RA"]]),
        "dec_host": np.concatenate([data["dec_host"], new["dec"]]),
        # Unused by the reconstruction models; padded to keep shapes.
        "PV_covmat_cepheid_host": block_diag(
            PV, np.mean(np.diag(PV)) * np.eye(k)),
        "host_los_density": np.concatenate(
            [np.asarray(data["host_los_density"])[:, m],
             new["los_density"]], axis=1),
        "host_los_velocity": np.concatenate(
            [np.asarray(data["host_los_velocity"])[:, m],
             new["los_velocity"]], axis=1),
        "mask_host": np.ones(n0 + k, dtype=bool),
    })
    return data


def add_windows(data, n_far):
    """The real hosts keep cz < 3300 km/s; the added ones get 3300--5500."""
    n0 = int(data["num_hosts"]) - n_far
    data["cz_sel_low_host"] = np.concatenate(
        [np.full(n0, CZ_NO_LOWER_LIMIT), np.full(n_far, CZ_LOW)])
    data["cz_sel_high_host"] = np.concatenate(
        [np.full(n0, CZ_LOW), np.full(n_far, CZ_HIGH)])
    return data


def cmd_run(args):
    variant, with_new = SCENARIOS[args.scenario]
    tag = "" if args.sigma_mu == 0.05 else f"_smu{args.sigma_mu:g}"
    tag += "" if args.field is None else f"_field{args.field:02d}"
    tag += "" if args.logP == 0 else f"_logP{args.logP:g}"
    fout = join(OUT, f"{args.scenario}{tag}.npz")
    if exists(fout):
        print(f"{fout} exists, skipping.")
        return
    _, path = make_config(variant, args.num_warmup, args.num_samples,
                          args.field)
    try:
        data = candel.pvdata.load_SH0ES_from_config(path)
        n_far = 0
        if with_new:
            new = select_fields(
                dict(np.load(join(OUT, "new_hosts.npz"))),
                np.asarray(data["host_los_field_indices"]).ravel())
            print_truth(new, new["field_indices"])
            data = add_hosts(data, new, TRUTH["M_W"], args.sigma_mu,
                             args.logP)
            n_far = len(new["names"])
        data = add_windows(data, n_far)
        model = candel.model.CH0Model(path, data)
        post = candel.run_H0_inference(model, save_samples=False,
                                       print_summary=True,
                                       progress_bar=False)
    finally:
        os.unlink(path)
    H0 = np.asarray(post["H0"])
    np.savez(fout, H0=H0, M_W=np.asarray(post["M_W"]),
             n_hosts=int(data["num_hosts"]))
    print(f"{args.scenario}: H0 = {H0.mean():.2f} +- {H0.std():.2f}",
          flush=True)


def cmd_collect(args):
    print("| Scenario | hosts | H0 mean | sigma(H0) |")
    print("|---|---|---|---|")
    for f in sorted(f for f in os.listdir(OUT)
                    if f.endswith(".npz") and f != "new_hosts.npz"):
        d = np.load(join(OUT, f))
        print(f"| {f[:-4]} | {int(d['n_hosts'])} | {d['H0'].mean():.2f} | "
              f"{d['H0'].std():.2f} |")


def main():
    # Paths are relative to the repository root.
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    run = sub.add_parser("run")
    run.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    run.add_argument("--sigma-mu", type=float, default=0.05,
                     help="Distance-modulus error of the added hosts (mag).")
    run.add_argument("--logP", type=float, default=0.0,
                     help="log P - 1 of the added rows; nonzero makes them "
                     "depend on b_W (0.6 matches the far SH0ES hosts).")
    run.add_argument("--field", type=int, default=None,
                     help="Use a single Manticore realisation for both the "
                     "mock velocities and the inference (default: all).")
    run.add_argument("--num-warmup", type=int, default=1000)
    run.add_argument("--num-samples", type=int, default=5000)
    sub.add_parser("collect")
    args = parser.parse_args()
    {"prep": cmd_prep, "run": cmd_run, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
