# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Loaders for the peculiar-velocity catalogues (TFR, SN, FP samples)."""
from os.path import join

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from h5py import File

from candel.field.dust import read_dustmap
from candel.field.los import _filter_data, _zcmb_blat_mask, load_los
from candel.util import SPEED_OF_LIGHT, fprint


def load_SH0ES_calibration(calibration_path, pgc_CF4):
    """
    Load SH0ES distance modulus samples and match to CF4 galaxies by PGC ID.
    """
    with File(calibration_path, 'r') as f:
        mu_samples = f["distmod_samples"][...]
        pgc_SH0ES = f["pgc"][...]

    i_CF4 = []
    i_SH0ES = []

    for i, pgc_i in enumerate(pgc_CF4):
        if pgc_i in pgc_SH0ES:
            match = np.where(pgc_SH0ES == pgc_i)[0]
            assert len(match) == 1
            i_CF4.append(i)
            i_SH0ES.append(match[0])

    i_CF4 = np.array(i_CF4)
    i_SH0ES = np.array(i_SH0ES)

    is_calibrator = np.zeros(len(pgc_CF4), dtype=bool)
    is_calibrator[i_CF4] = True

    mu_cal = np.mean(mu_samples[:, i_SH0ES], axis=0)
    C_mu_cal = np.cov(mu_samples[:, i_SH0ES], rowvar=False)

    return is_calibrator, mu_cal, C_mu_cal


def load_CF4_data(root, which_band, best_mag_quality=True, eta_min=-0.3,
                  zcmb_min=None, zcmb_max=None, b_min=7.5,
                  remove_outliers=True, calibration=None, los_data_path=None,
                  field_indices=None, return_all=False, dust_model=None,
                  exclude_W1=False, **kwargs):
    """
    Load CF4 TFR data and apply optional filters and dust correction removal.
    """
    with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
        grp = f["cf4"]
        zcmb = grp["Vcmb"][...] / SPEED_OF_LIGHT
        RA = grp["RA"][...] * 15  # deg
        DEC = grp["DE"][...]
        mag = grp[which_band][...]
        mag_quality = grp["Qw"][...] if which_band == "w1" else grp["Qs"][...]
        eta = grp["lgWmxi"][...] - 2.5
        e_eta = grp["elgWi"][...]
        pgc = grp["pgc"][...]

        if dust_model is not None:
            if which_band not in ["w1", "w2"]:
                raise ValueError(
                    f"Band `{which_band}` is not supported for dust "
                    f"correction removal. Only `w1` and `w2` are supported.")

            Ab_default = grp[f"A_{which_band}"][...]
            fprint(f"switching the dust model to `{dust_model}`.")

            mag += Ab_default
            if dust_model == "default":
                ebv = Ab_default / (0.186 if which_band == "w1" else 0.123)
            else:
                ebv = read_dustmap(RA, DEC, dust_model)

            if not np.all(np.isfinite(ebv)):
                raise ValueError(
                    f"Non-finite E(B-V) values for dust map `{dust_model}`.")
        else:
            ebv = np.full_like(mag, np.nan)

    fprint(f"initially loaded {len(pgc)} galaxies from CF4 TFR data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=np.full_like(mag, 0.05),
        eta=eta,
        e_eta=e_eta,
        ebv=ebv,
    )

    if return_all:
        return data

    mask = eta > eta_min
    if best_mag_quality:
        mask &= mag_quality == 5
    else:
        mask &= mag > 5

    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)

    if remove_outliers:
        outliers = np.concatenate([
            np.genfromtxt(join(root, f"CF4_{b}_outliers.csv"),
                          delimiter=",", names=True)
            for b in ("W1", "i")
        ])
        mask &= ~np.isin(pgc, outliers["PGC"])

    if which_band == "i" and exclude_W1:
        with File(join(root, "CF4_TFR.hdf5"), 'r') as f:
            w1_quality = f["cf4"]["Qw"][...]
            w1_mag = f["cf4"]["w1"][...]
        fprint("excluding galaxies with W1 quality 5 or W1 mag < 5.")
        exclude = (w1_quality == 5) | (w1_mag > 5)
        mask &= ~exclude

    _filter_data(data, mask, los_data_path, field_indices=field_indices)
    pgc = pgc[mask]

    if calibration == "SH0ES":
        is_cal, mu, C_mu = load_SH0ES_calibration(
            join(root, "CF4_SH0ES_calibration.hdf5"), pgc)
        fprint(f"out of {len(pgc)} galaxies, {np.sum(is_cal)} are SH0ES "
               "calibrators.")
        data.update({
            "is_calibrator": is_cal,
            "mu_cal": mu,
            "C_mu_cal": C_mu,
            "std_mu_cal": np.sqrt(np.diag(C_mu)),
        })
    elif calibration:
        raise ValueError("Unknown calibration type.")

    return data


def load_CF4_mock(root, index):
    fname = join(root, f"mock_{index}.hdf5")
    with File(fname, 'r') as f:
        grp = f["mock"]
        data = {key: grp[key][...] for key in grp.keys()}
    return data


def load_2MTF(root, eta_min=-0.1, eta_max=0.2, zcmb_min=None, zcmb_max=None,
              b_min=7.5, los_data_path=None, return_all=False,
              field_indices=None, **kwargs):
    """
    Load the 2MTF data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f["2MTF"]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mag"][...]
        eta = grp["eta"][...]

        e_eta = grp["e_eta"][...]
        e_mag = grp["e_mag"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from 2MTF data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=e_mag,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = (eta > eta_min) & (eta < eta_max)
    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path, field_indices=field_indices)


def load_SFI(root, eta_min=-0.1, zcmb_min=None, zcmb_max=None,
             b_min=7.5, los_data_path=None, return_all=False,
             field_indices=None, **kwargs):
    """
    Load the SFI++ data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f["SFI_gals"]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mag"][...]
        eta = grp["eta"][...]

        e_eta = grp["e_eta"][...]
        e_mag = grp["e_mag"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from SFI++ data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        e_mag=e_mag,
        eta=eta,
        e_eta=e_eta,
    )

    if return_all:
        return data

    mask = eta > eta_min
    mask &= _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path, field_indices=field_indices)


def _load_LOSS_Foundation(which, root, zcmb_min=None, zcmb_max=None,
                          b_min=7.5, los_data_path=None, return_all=False,
                          field_indices=None, **kwargs):
    """
    Load the LOSS or Foundation SNe data from the given root directory.
    """
    with File(join(root, "PV_compilation.hdf5"), 'r') as f:
        grp = f[which]

        zcmb = grp["z_CMB"][...]
        RA = grp["RA"][...]
        DEC = grp["DEC"][...]
        mag = grp["mB"][...]
        c = grp["c"][...]
        x1 = grp["x1"][...]

        e_mag = grp["e_mB"][...]
        e_c = grp["e_c"][...]
        e_x1 = grp["e_x1"][...]

    fprint(f"initially loaded {len(zcmb)} galaxies from {which} data.")

    data = dict(
        zcmb=zcmb,
        RA=RA,
        dec=DEC,
        mag=mag,
        c=c,
        x1=x1,
        e_mag=e_mag,
        e_c=e_c,
        e_x1=e_x1
    )

    if return_all:
        return data

    mask = _zcmb_blat_mask(zcmb, RA, DEC, zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path, field_indices=field_indices)


def load_LOSS(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
              los_data_path=None, return_all=False, field_indices=None,
              **kwargs):
    return _load_LOSS_Foundation(
        "LOSS", root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
        b_min=b_min, los_data_path=los_data_path, return_all=return_all,
        field_indices=field_indices, **kwargs)


def load_Foundation(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                    los_data_path=None, return_all=False, field_indices=None,
                    **kwargs):
    return _load_LOSS_Foundation(
        "Foundation", root, zcmb_min=zcmb_min, zcmb_max=zcmb_max,
        b_min=b_min, los_data_path=los_data_path, return_all=return_all,
        field_indices=field_indices, **kwargs)


def load_PantheonPlus_Lane(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                           los_data_path=None, return_all=False,
                           field_indices=None, **kwargs):
    if zcmb_max is not None and zcmb_max > 0.075:
        raise ValueError(f"`zcmb_max` of {zcmb_max} is too high for the "
                         "LOWZ sample which goes only up to 0.075.")
    fname = join(root, "full_ps1_input_LOWZ.csv")
    x = np.genfromtxt(fname, delimiter=",", names=True, dtype=None,
                      encoding=None)

    fprint(f"initially loaded {len(x)} galaxies from Pantheon+Lane data.")

    data = dict(
        zcmb=x["zCMB"],
        RA=x["RA"],
        dec=x["DEC"],
        mag=x["mB"],
        x1=x["x1"],
        c=x["c"],
    )

    if return_all:
        return data

    C = np.loadtxt(join(root, "PP_cov_new_LOWZ.txt"))

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    _filter_data(data, mask, los_data_path, field_indices=field_indices)

    C_idx = (3 * np.where(mask)[0][:, None] + np.arange(3)).ravel()
    data["mag_covmat"] = C[C_idx][:, C_idx]

    return data


def load_PantheonPlus(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                      los_data_path=None, return_all=False,
                      removed_PV_from_covmat=True, field_indices=None,
                      **kwargs):
    """
    Load the Pantheon+ data from the given root directory, the covariance
    is expected to have peculiar velocity contribution removed.
    """
    if removed_PV_from_covmat:
        arr_fname = "Pantheon+SH0ES_zsel.dat"
        covmat_fname = "Pantheon+SH0ES_zsel_STAT+SYS_noPV.cov"
    else:
        arr_fname = "Pantheon+SH0ES.dat"
        covmat_fname = "Pantheon+SH0ES_STAT+SYS.cov"

    arr = np.genfromtxt(
        join(root, arr_fname), names=True, dtype=None, encoding=None)

    fprint(f"initially loaded {len(arr)} galaxies from Pantheon+ data.")

    data = {
        "zcmb": arr["zCMB"],
        "e_zcmb": arr["zCMBERR"],
        "RA": arr["RA"],
        "dec": arr["DEC"],
        "mag": arr["m_b_corr"],
    }

    if return_all:
        return data

    covmat = np.loadtxt(join(root, covmat_fname), delimiter=",")
    size = int(covmat[0])
    C = np.reshape(covmat[1:], (size, size))

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    _filter_data(data, mask, los_data_path, field_indices=field_indices)

    C = C[mask][:, mask]
    data["mag_covmat"] = C
    data["e_mag"] = np.sqrt(np.diag(C))  # Do not use in the inference!

    return data


def arcsec_to_radian(arcsec):
    return (arcsec * u.arcsec).to(u.radian).value


def load_SDSS_FP(root, zcmb_min=None, zcmb_max=None, b_min=7.5,
                 los_data_path=None, return_all=False, field_indices=None,
                 **kwargs):
    """Load the SDSS FP data from the given root directory."""
    fname = join(root, "SDSS_PV_public.dat")
    d_input = np.genfromtxt(fname, names=True, )

    rdev = d_input["deVRad_r"]
    e_rdev = d_input["deVRadErr_r"]
    boa = d_input["deVAB_r"]
    e_boa = d_input["deVABErr_r"]

    fprint(f"initially loaded {len(d_input)} galaxies from SDSS FP data.")

    theta_eff = arcsec_to_radian(rdev * np.sqrt(boa))
    e_theta_eff = theta_eff * np.sqrt(
        (e_rdev / rdev)**2 + (0.5 * e_boa / boa)**2)

    data = {
        "RA": d_input["RA"],
        "dec": d_input["Dec"],
        "zcmb": d_input["zcmb_group"],
        "theta_eff": theta_eff,
        "e_theta_eff": e_theta_eff,
        "log_theta_eff": np.log10(theta_eff),
        "e_log_theta_eff": e_theta_eff / (theta_eff * np.log(10)),
        "logI": d_input["i"],
        "e_logI": d_input["ei"],
        "logs": d_input["s"],
        "e_logs": d_input["es"],
        }

    if return_all:
        return data

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path, field_indices=field_indices)


def load_6dF_FP(root, which_band=None, zcmb_min=None, zcmb_max=None, b_min=7.5,
                los_data_path=None, return_all=False, field_indices=None,
                **kwargs):
    """Load the 6dF FP data from the given root directory."""
    d = np.genfromtxt(join(root, "6dF_FP.dat"))

    RA = d[:, 2] * 360 / 24
    dec = d[:, 3]
    czcmb = d[:, 4]

    data = {
        "RA": RA,
        "dec": dec,
        "zcmb": czcmb / SPEED_OF_LIGHT,
    }

    fprint(f"initially loaded {len(d)} galaxies from 6dF FP data.")

    if return_all:
        return data
    elif which_band is None:
        raise ValueError("which_band must be one of 'J', 'H', 'K'.")

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    dA_zcmb = cosmo.angular_diameter_distance(czcmb / SPEED_OF_LIGHT).value  # noqa

    if which_band == "J":
        logRe = d[:, 5]
        e_logRe = d[:, 6]

        logIe = d[:, 11]
        e_logIe = d[:, 12]
    elif which_band == "H":
        logRe = d[:, 7]
        e_logRe = d[:, 8]

        logIe = d[:, 13]
        e_logIe = d[:, 14]
    elif which_band == "K":
        logRe = d[:, 9]
        e_logRe = d[:, 10]

        logIe = d[:, 15]
        e_logIe = d[:, 16]
    else:
        raise ValueError(f"which_band must be one of 'J', 'H', 'K', got "
                         f"{which_band}.")
    logVd = d[:, 17]
    e_logVd = d[:, 18]

    log_theta_eff = logRe - np.log10(dA_zcmb * 1e3)
    e_log_theta_eff = e_logRe

    data.update({
        "logI": logIe,
        "e_logI": e_logIe,
        "logs": logVd,
        "e_logs": e_logVd,
        "log_theta_eff": log_theta_eff,
        "e_log_theta_eff": e_log_theta_eff,
    })

    mask = _zcmb_blat_mask(
        data["zcmb"], data["RA"], data["dec"], zcmb_min, zcmb_max, b_min)
    return _filter_data(data, mask, los_data_path, field_indices=field_indices)


def load_generic(filepath, los_data_path=None, field_indices=None, **kwargs):
    """
    Load generic catalog data from a .txt file with column names.

    Expected columns: RA, dec, Vcmb (in CMB frame).
    """
    d = np.genfromtxt(filepath, names=True)

    data = {
        "RA": d["RA"],
        "dec": d["dec"],
        "zcmb": d["Vcmb"] / SPEED_OF_LIGHT,
    }

    fprint(f"loaded {len(data['RA'])} galaxies from {filepath}.")

    if los_data_path is not None:
        data = load_los(
            los_data_path, data, field_indices=field_indices)

    return data


_CATALOGUE_LOADERS = {
    "2MTF": load_2MTF,
    "SFI": load_SFI,
    "SDSS_FP": load_SDSS_FP,
    "6dF_FP": load_6dF_FP,
    "LOSS": load_LOSS,
    "Foundation": load_Foundation,
    "PantheonPlus": load_PantheonPlus,
    "PantheonPlusLane": load_PantheonPlus_Lane,
}
