# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Corner plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np
from corner import corner, overplot_lines, overplot_points

from candel.util import fprint


def name2label(name):
    """
    Map internal parameter names to LaTeX labels, optionally including
    catalogue prefix.
    """
    latex_labels = {
        "a_TFR": r"$a_\mathrm{TFR}$",
        "b_TFR": r"$b_\mathrm{TFR}$",
        "c_TFR": r"$c_\mathrm{TFR}$",
        "sigma_int": r"$\sigma_{\rm int}$",
        "sigma_v": r"$\sigma_v$",
        "sigma_v_low": r"$\sigma_{v,{\rm low}}$",
        "sigma_v_high": r"$\sigma_{v,{\rm high}}$",
        "log_sigma_v_rho_t": r"$\ln\rho_{v,t}$",
        "sigma_v_k": r"$k_v$",
        "alpha": r"$\alpha$",
        "alpha_low": r"$\alpha_\mathrm{low}$",
        "alpha_high": r"$\alpha_\mathrm{high}$",
        "log_rho_t": r"$\ln \rho_t$",
        "log_rho_width": r"$\Delta \ln \rho$",
        "b1": r"$b_1$",
        "b2": r"$b_2$",
        "b3": r"$b_3$",
        "beta": r"$\beta$",
        "velocity_beta": r"$\beta$",
        "Vext_mag": r"$V_\mathrm{ext}$",
        "Vext_ell": r"$\ell_\mathrm{ext}$",
        "Vext_b": r"$b_\mathrm{ext}$",
        "logM_miss": r"$\log_{10} M_{\rm miss}$",
        "Mmiss_distance": r"$r_{\rm miss}$",
        "Mmiss_ell": r"$\ell_{\rm miss}$",
        "Mmiss_b": r"$b_{\rm miss}$",
        "h": r"$h$",
        "a": r"$a$",
        "m1": r"$m_1$",
        "m2": r"$m_2$",
        "zeropoint_dipole_mag": r"$\Delta \mathrm{ZP}$",
        "zeropoint_dipole_ell": r"$\ell_{\Delta \mathrm{ZP}}$",
        "zeropoint_dipole_b": r"$b_{\Delta \mathrm{ZP}}$",
        "SN_absmag": r"$M_{\rm SN}$",
        "SN_alpha": r"$\mathcal{A}$",
        "SN_beta": r"$\mathcal{B}$",
        "eta": r"$\eta$",
        "eta_prior_mean": r"$\hat{\eta}$",
        "eta_prior_std": r"$w_\eta$",
        "A_CL": r"$A_{\rm CL}$",
        "B_CL": r"$B_{\rm CL}$",
        "C_CL": r"$C_{\rm CL}$",
        "a_FP": r"$a_{\rm FP}$",
        "b_FP": r"$b_{\rm FP}$",
        "c_FP": r"$c_{\rm FP}$",
        "sigma_log_theta": r"$\sigma_{\log \theta}$",
        "R_dust": r"$R_{\rm W1}$",
        "R_dist_emp": r"$R_{\rm dist}$",
        "q_dist_emp": r"$q_{\rm dist}$",
        "Rmax_dist_emp": r"$R_{\rm max, dist}$",
        "rho_corr": r"$\rho_{\rm corr}$",
        "Vext_radmag_ell": r"$\ell_{\mathrm{Vext}}$",
        "Vext_radmag_b": r"$b_{\mathrm{Vext}}$",
        "H0": r"$H_0$",
        "M_SN": r"$M_{\rm SN}$",
        "M_TRGB": r"$M_{\rm TRGB}$",
        "mag_lim_SN": r"$m_{\rm lim}^{\rm SN}$",
        "mag_lim_SN_width": r"$\sigma_{m,{\rm lim}}^{\rm SN}$",
        "mu_LMC": r"$\mu_{\rm LMC}$",
        "c_star": r"$c_\star$",
        "c_bar": r"$\bar{c}$",
        "w_c": r"$w_c$",
        "mag_lim_TRGB": r"$m_{\rm lim}$",
        "mag_lim_TRGB_width": r"$\sigma_{\rm sel}$",
        "nu_cz": r"$\nu$",
        "mu_N4258": r"$\mu_{\rm N4258}$",
        "log10_D_c": r"$\log_{10} D_c$",
        "D_c": r"$D_c$",
        "D_A": r"$D_A$",
        "log_MBH": r"$\log M_{\rm BH}$",
        "M_BH": r"$M_{\rm BH}$",
        "i0": r"$i_0$",
        "di_dr": r"$\mathrm{d}i/\mathrm{d}r$",
        "d2i_dr2": r"$\mathrm{d}^2i/\mathrm{d}r^2$",
        "Omega0": r"$\Omega_0$",
        "dOmega_dr": r"$\mathrm{d}\Omega/\mathrm{d}r$",
        "d2Omega_dr2": r"$\mathrm{d}^2\Omega/\mathrm{d}r^2$",
        "x0": r"$x_0$",
        "y0": r"$y_0$",
        "dv_sys": r"$\Delta v_{\rm sys}$",
        "sigma_x_floor": r"$\sigma_{x,\mathrm{fl}}$",
        "sigma_y_floor": r"$\sigma_{y,\mathrm{fl}}$",
        "sigma_v_sys": r"$\sigma_{v,\mathrm{sys}}$",
        "sigma_v_hv": r"$\sigma_{v,\mathrm{hv}}$",
        "sigma_a_floor": r"$\sigma_{a,\mathrm{fl}}$",
        "sigma_x_floor_clump2": r"$\sigma^{(2)}_{x,\mathrm{fl}}$",
        "sigma_y_floor_clump2": r"$\sigma^{(2)}_{y,\mathrm{fl}}$",
        "sigma_v_floor_clump2": r"$\sigma^{(2)}_{v,\mathrm{sys}}$",
        "sigma_a_floor_clump2": r"$\sigma^{(2)}_{a,\mathrm{fl}}$",
        "ecc": r"$e$",
        "e_x": r"$e_x$",
        "e_y": r"$e_y$",
        "periapsis": r"$\omega$",
        "periapsis_rad": r"$\omega$",
        "dperiapsis_dr": r"$\mathrm{d}\omega/\mathrm{d}r$",
        "sigma_pec": r"$\sigma_{\rm pec}$",
        "D_lim": r"$D_{\rm lim}$",
        "D_width": r"$\sigma_{D,\rm lim}$",
        "cz_lim_selection": r"$cz_{\rm lim}$",
        "cz_lim_selection_width": r"$\sigma_{cz,\rm lim}$",
    }

    if "/" in name:
        prefix, base = name.split("/", 1)
        base_label = latex_labels.get(base, base)
        prefix_latex = prefix.replace("_", r"\,").replace(" ", "~")
        return rf"$\mathrm{{{prefix_latex}}},\,{base_label.strip('$')}$"

    return latex_labels.get(name, name)


def sort_params(keys):
    order = [
        "H0", "log10_D_c", "D_c", "log_MBH", "dv_sys",
        "sigma_pec", "D_lim", "D_width", "cz_lim_selection",
        "cz_lim_selection_width", "x0", "y0", "i0", "Omega0",
        "di_dr", "dOmega_dr", "d2i_dr2", "d2Omega_dr2",
        "sigma_x_floor", "sigma_y_floor", "sigma_v_sys", "sigma_v_hv",
        "sigma_a_floor", "sigma_x_floor_clump2", "sigma_y_floor_clump2",
        "sigma_v_floor_clump2", "sigma_a_floor_clump2",
        "ecc", "e_x", "e_y", "periapsis",
        "periapsis_rad", "dperiapsis_dr", "a_TFR", "b_TFR", "c_TFR",
        "alpha", "beta", "sigma_int", "sigma_v", "logM_miss",
        "Mmiss_distance", "Mmiss_ell", "Mmiss_b", "Vext", "Vext_mag",
        "Vext_ell", "Vext_b"
    ]

    def sort_key(k):
        prefix, base = k.split("/", 1) if "/" in k else ("", k)
        try:
            return (order.index(base), prefix, k)
        except ValueError:
            return (len(order), prefix, k)

    return sorted(keys, key=sort_key)


def plot_corner(samples, show_fig=True, filename=None, smooth=1, keys=None,
                truths=None, points=None, point_color="tab:blue",
                point_label=None, truth_label=None, log_save=True,
                map_point=None, map_color="tab:green", map_label="MAP"):
    """Plot a corner plot from posterior samples."""
    flat_samples = []
    labels = []
    truth_keys = []

    if keys is None:
        keys = sort_params(list(samples.keys()))

    for k in keys:
        if k not in samples:
            continue
        v = np.asarray(samples[k])

        if k == "Vext_radmag_mag":
            nbin = v.shape[1]
            for i in range(nbin):
                flat_samples.append(v[:, i])
                labels.append(fr"$V_{{\mathrm{{ext}}, {{{i}}}}}$")
                truth_keys.append(None)

        if v.ndim > 1:
            continue
        v = np.asarray(v).reshape(-1)
        if np.ptp(v) == 0:
            continue
        flat_samples.append(v)
        labels.append(name2label(k))
        truth_keys.append(k)

    if not flat_samples:
        raise ValueError("No valid samples to plot.")

    data = np.vstack(flat_samples).T
    truth_values = None
    if truths is not None:
        truth_values = [
            truths.get(k) if k is not None and k in truths else None
            for k in truth_keys
        ]
        if not any(v is not None for v in truth_values):
            truth_values = None
    fig = corner(
        data, labels=labels, show_titles=True, smooth=smooth,
        truths=truth_values, truth_color="red")
    legend_fontsize = max(12, min(26, 0.9 * max(fig.get_size_inches())))
    legend_handles = []
    if truth_values is not None and truth_label:
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D(
            [0], [0], color="red", lw=2.0, label=truth_label))
    if points is not None:
        point_values = [
            points.get(k) if k is not None and k in points else None
            for k in truth_keys
        ]
        if any(v is not None for v in point_values):
            overplot_lines(fig, point_values, color=point_color)
            overplot_points(
                fig,
                [[np.nan if v is None else v for v in point_values]],
                color=point_color, marker="s")
            if point_label:
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D(
                    [0], [0], color=point_color, lw=2.0, marker="s",
                    markersize=0.45 * legend_fontsize, label=point_label))
    if map_point is not None:
        map_values = [
            map_point.get(k) if k is not None and k in map_point else None
            for k in truth_keys
        ]
        if any(v is not None for v in map_values):
            overplot_lines(fig, map_values, color=map_color, linestyle=":")
            overplot_points(
                fig,
                [[np.nan if v is None else v for v in map_values]],
                color=map_color, marker="*")
            if map_label:
                from matplotlib.lines import Line2D
                legend_handles.append(Line2D(
                    [0], [0], color=map_color, lw=2.0, marker="*",
                    markersize=0.6 * legend_fontsize, label=map_label))
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper right",
                   bbox_to_anchor=(0.98, 0.98), frameon=True,
                   framealpha=0.9, fontsize=legend_fontsize,
                   borderpad=0.6, labelspacing=0.45, handlelength=2.0)

    if filename is not None:
        if log_save:
            fprint(f"saving a corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        fig.show()
    else:
        plt.close(fig)


def plot_Vext_rad_corner(samples, show_fig=True, filename=None, smooth=1):
    """Plot a corner plot of Vext_rad_{mag, ell, b} samples."""
    keys = ["Vext_rad_mag", "Vext_rad_ell", "Vext_rad_b"]
    base_labels = [r"V", r"\ell", r"b"]

    arrays = []
    labels = []
    for key, base_label in zip(keys, base_labels):
        if key not in samples:
            raise ValueError(f"Missing key: {key}")

        arr = samples[key]
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            raise ValueError(f"{key} must be 2D or 3D")

        ndim = arr.shape[1]
        arrays.append(arr)
        for i in range(ndim):
            labels.append(fr"${base_label}_{{{i}}}$")

    data = np.hstack(arrays)
    fig = corner(data, labels=labels, show_titles=True, smooth=smooth)

    if filename is not None:
        fprint(f"saving knots corner plot to {filename}")
        fig.savefig(filename, bbox_inches="tight")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)
