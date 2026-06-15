#!/usr/bin/env python3
"""Generate Reid ``fit_disk`` inputs for the CANDEL NGC6264 setup."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "scripts/megamaser/config_maser.toml"
DEFAULT_TABLE2 = ROOT / "data/Megamaser/NGC6264_Kuo2013_table2.txt"
DEFAULT_DATA_OUT = ROOT / "data/Megamaser/NGC6264_Kuo2013_table2_reid.inp"
DEFAULT_INIT_OUT = SCRIPT_DIR / "ngc6264_init.toml"


@dataclass
class Spot:
    velocity: float
    x_mas: float
    sigma_x_mas: float
    y_mas: float
    sigma_y_mas: float
    accel: float | None
    sigma_accel: float | None


def load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def is_missing(value: str) -> bool:
    return value.strip() in {"", "...", "sdotsdotsdot", r"\ldots"}


def optional_float(value: str) -> float | None:
    if is_missing(value):
        return None
    return float(value.strip())


def parse_kuo_table2(path: Path) -> list[Spot]:
    spots: list[Spot] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            velocity = float(fields[0].strip())
            x = float(fields[1].strip())
            sigma_x = float(fields[2].strip())
            y = float(fields[3].strip())
            sigma_y = float(fields[4].strip())
        except ValueError:
            continue

        accel = optional_float(fields[5]) if len(fields) > 5 else None
        sigma_accel = optional_float(fields[6]) if len(fields) > 6 else None
        if accel is None or sigma_accel is None:
            accel = None
            sigma_accel = None

        spots.append(
            Spot(
                velocity=velocity,
                x_mas=x,
                sigma_x_mas=sigma_x,
                y_mas=y,
                sigma_y_mas=sigma_y,
                accel=accel,
                sigma_accel=sigma_accel,
            )
        )
    if not spots:
        raise ValueError(f"No Kuo Table 2 spots parsed from {path}")
    return spots


def ngc6264_config_values(config_path: Path, vcor: float) -> tuple[dict[str, float], float]:
    cfg = load_toml(config_path)
    model_cfg = cfg["model"]
    galaxy_cfg = model_cfg["galaxies"]["NGC6264"]
    init = galaxy_cfg["init"]

    v_sys = float(galaxy_cfg["v_sys_obs"]) + float(init.get("dv_sys", 0.0))
    distance = float(init["D_c"])
    h0 = (v_sys + vcor) / distance

    ecc = float(init.get("ecc", 0.0))
    peri = float(init.get("periapsis", 0.0))
    if "e_x" in init and "e_y" in init:
        ex = float(init["e_x"])
        ey = float(init["e_y"])
        ecc = math.hypot(ex, ey)
        peri = math.degrees(math.atan2(ey, ex)) % 360.0

    values = {
        "H0": h0,
        "Mbh_1e7Msun": 10.0 ** (float(init["log_MBH"]) - 7.0),
        "Vsys_km_s": v_sys,
        "x0_mas": float(init.get("x0", 0.0)) / 1000.0,
        "y0_mas": float(init.get("y0", 0.0)) / 1000.0,
        "i0_deg": 180.0 - float(init.get("i0", 90.0)),
        "di_dr_deg_mas": -float(init.get("di_dr", 0.0)),
        "d2i_dr2_deg_mas2": -float(init.get("d2i_dr2", 0.0)),
        "PA_deg": float(init.get("Omega0", 0.0)),
        "dPA_dr_deg_mas": float(init.get("dOmega_dr", 0.0)),
        "d2PA_dr2_deg_mas2": float(init.get("d2Omega_dr2", 0.0)),
        "ecc": ecc,
        "peri_az_deg": peri,
        "dperi_dr_deg_mas": float(init.get("dperiapsis_dr", 0.0)),
        "Vcor_km_s": vcor,
        "sigma_x_mas": float(init.get("sigma_x_floor", 0.0)) / 1000.0,
        "sigma_y_mas": float(init.get("sigma_y_floor", 0.0)) / 1000.0,
        "sigma_vsys_km_s": float(init.get("sigma_v_sys", 0.0)),
        "sigma_vhv_km_s": float(init.get("sigma_v_hv", 0.0)),
        "sigma_acc_km_s_yr": float(init.get("sigma_a_floor", 0.0)),
    }
    sigma_v_default = float(model_cfg.get("sigma_v_default", 0.25))
    return values, sigma_v_default


def pivot_radii(
    spots: list[Spot],
    init: dict[str, float],
    sys_vmin: float,
    sys_vmax: float,
) -> tuple[float, float]:
    hv = [spot for spot in spots if spot.velocity < sys_vmin or spot.velocity > sys_vmax]
    if not hv:
        raise ValueError("Cannot infer warp pivots: no high-velocity spots found")

    # Match CANDEL's default pivot: median projected HV radius, before applying
    # the fitted BH-centre offset.
    candel_r = sorted(math.hypot(spot.x_mas, spot.y_mas) for spot in hv)[len(hv) // 2]

    # Match Reid's internal r_ref: mean centred HV radius.
    reid_r = sum(
        math.hypot(spot.x_mas - init["x0_mas"], spot.y_mas - init["y0_mas"])
        for spot in hv
    ) / len(hv)
    return candel_r, reid_r


def shift_to_reid_pivot(
    init: dict[str, float],
    candel_r_ref: float,
    reid_r_ref: float,
) -> dict[str, float]:
    out = dict(init)
    dr = reid_r_ref - candel_r_ref
    out["i0_deg"] = (
        init["i0_deg"]
        + init["di_dr_deg_mas"] * dr
        + init["d2i_dr2_deg_mas2"] * dr * dr
    )
    out["PA_deg"] = (
        init["PA_deg"]
        + init["dPA_dr_deg_mas"] * dr
        + init["d2PA_dr2_deg_mas2"] * dr * dr
    )
    out["peri_az_deg"] = init["peri_az_deg"] - init["dperi_dr_deg_mas"] * (
        0.5 * candel_r_ref
    )
    return out


def write_reid_data(
    path: Path,
    spots: list[Spot],
    init: dict[str, float],
    sigma_v: float,
    sys_vmin: float,
    sys_vmax: float,
) -> None:
    n_sys = sum(sys_vmin <= spot.velocity <= sys_vmax for spot in spots)
    n_accel = sum(spot.sigma_accel is not None for spot in spots)
    lines = [
        (
            f"{sys_vmin:.8g} {sys_vmax:.8g} "
            f"{init['sigma_x_mas']:.8g} {init['sigma_y_mas']:.8g} "
            f"{init['sigma_vsys_km_s']:.8g} {init['sigma_vhv_km_s']:.8g} "
            f"{init['sigma_acc_km_s_yr']:.8g} Optical"
        ),
        "! Reid fit_disk input for NGC6264 generated from Kuo+2013 Table 2.",
        "! Velocities are optical LSR as stated in the Kuo+2013 table note.",
        f"! Raw velocity errors set to CANDEL model/sigma_v_default = {sigma_v:.8g} km/s.",
        "! Missing accelerations are encoded as sigma_A = -2 to exclude them.",
        f"! Spots: {len(spots)} total, {n_sys} systemic, {len(spots) - n_sys} high-velocity, {n_accel} with acceleration.",
        "!",
        "! ID  Vlsr_opt  sigma_V   x  sigma_x   y  sigma_y   Acc  sigma_Acc",
    ]
    for idx, spot in enumerate(spots, start=1):
        accel = 0.0 if spot.accel is None else spot.accel
        sigma_accel = -2.0 if spot.sigma_accel is None else spot.sigma_accel
        lines.append(
            f"{idx:5d}"
            f" {spot.velocity:12.5f} {sigma_v:10.5f}"
            f" {spot.x_mas:12.6f} {spot.sigma_x_mas:10.6f}"
            f" {spot.y_mas:12.6f} {spot.sigma_y_mas:10.6f}"
            f" {accel:12.6f} {sigma_accel:10.6f}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_reid_init(
    path: Path,
    values: dict[str, float],
    config_path: Path,
    candel_r_ref: float,
    reid_r_ref: float,
) -> None:
    try:
        config_label = config_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        config_label = str(config_path)
    ordered_names = [
        "H0",
        "Mbh_1e7Msun",
        "Vsys_km_s",
        "x0_mas",
        "y0_mas",
        "i0_deg",
        "di_dr_deg_mas",
        "d2i_dr2_deg_mas2",
        "PA_deg",
        "dPA_dr_deg_mas",
        "d2PA_dr2_deg_mas2",
        "ecc",
        "peri_az_deg",
        "dperi_dr_deg_mas",
        "Vcor_km_s",
        "sigma_x_mas",
        "sigma_y_mas",
        "sigma_vsys_km_s",
        "sigma_vhv_km_s",
        "sigma_acc_km_s_yr",
    ]
    lines = [
        "# Reid-convention NGC6264 globals generated from CANDEL config.",
        f"# Source config: {config_label}",
        "# Positions and position floors are converted from microarcsec to mas.",
        "# Inclination convention: i_Reid = 180 - i_CANDEL; di/dr and d2i/dr2 are sign-flipped.",
        f"# CANDEL default HV pivot: {candel_r_ref:.12g} mas.",
        f"# Reid internal HV r_ref: {reid_r_ref:.12g} mas.",
        "# i0 and PA are shifted from the CANDEL pivot to Reid's r_ref.",
        "",
        "[globals]",
    ]
    lines.extend(f"{name} = {values[name]:.12g}" for name in ordered_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare NGC6264 data/init files for Reid fit_disk."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--table2", type=Path, default=DEFAULT_TABLE2)
    parser.add_argument("--data-out", type=Path, default=DEFAULT_DATA_OUT)
    parser.add_argument("--init-out", type=Path, default=DEFAULT_INIT_OUT)
    parser.add_argument("--sys-vmin", type=float, default=10000.0)
    parser.add_argument("--sys-vmax", type=float, default=10350.0)
    parser.add_argument("--vcor", type=float, default=0.0)
    args = parser.parse_args(argv)

    spots = parse_kuo_table2(args.table2)
    values, sigma_v = ngc6264_config_values(args.config, args.vcor)
    candel_r_ref, reid_r_ref = pivot_radii(
        spots, values, args.sys_vmin, args.sys_vmax
    )
    values = shift_to_reid_pivot(values, candel_r_ref, reid_r_ref)
    write_reid_data(
        args.data_out,
        spots,
        values,
        sigma_v,
        args.sys_vmin,
        args.sys_vmax,
    )
    write_reid_init(args.init_out, values, args.config, candel_r_ref, reid_r_ref)

    n_sys = sum(args.sys_vmin <= spot.velocity <= args.sys_vmax for spot in spots)
    n_accel = sum(spot.sigma_accel is not None for spot in spots)
    print(f"Wrote Reid data: {args.data_out}")
    print(f"Wrote Reid init: {args.init_out}")
    print(
        f"NGC6264 spots: {len(spots)} total, {n_sys} systemic, "
        f"{len(spots) - n_sys} high-velocity, {n_accel} with acceleration"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
