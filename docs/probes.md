[Documentation](README.md) / Probes

# Probes

Each probe package is described briefly here; its README and `papers/` directory hold the details.

## Peculiar velocities — candel-pv

[candel-pv](https://github.com/candel-cosmo/candel-pv) jointly calibrates distance-indicator relations and a reconstructed density and velocity field: amplitude $\beta$, external bulk flow $\mathbf{V}_\mathrm{ext}$, galaxy bias, and a density-dependent velocity dispersion.
These models work in units of $h^{-1}\,\mathrm{Mpc}$, and several catalogues can be fitted jointly with shared parameters.
Runs leave `which_run` unset and choose the model with `inference.model`.

- Tully--Fisher: 2MTF, SFI++, CF4-TFR
- Type Ia supernovae (SALT2): LOSS, Foundation, Pantheon+
- Fundamental Plane: 6dFGS-FP, SDSS-FP
- Growth rate and $S_8$, mocks, and the redshift-to-real-space mapping

Papers: [arXiv:2502.00121](https://arxiv.org/abs/2502.00121) (Velocity Field Olympics), [arXiv:2509.14997](https://arxiv.org/abs/2509.14997) ($H_0$ anisotropy), [arXiv:2509.20235](https://arxiv.org/abs/2509.20235) ($S_8$).

## Cepheid $H_0$ — candel-ch0

[candel-ch0](https://github.com/candel-cosmo/candel-ch0) forward-models the SH0ES Cepheid host galaxies with a selection-function treatment and reconstructed peculiar velocities (`which_run = "CH0"`), and includes JWST forecast mocks.

Paper: [arXiv:2509.09665](https://arxiv.org/abs/2509.09665).

## TRGB $H_0$ — candel-trgb

[candel-trgb](https://github.com/candel-cosmo/candel-trgb) is a two-rung ladder from EDD Tip of the Red Giant Branch distances and geometric anchors (`which_run = "EDD_TRGB"`), with mocks and posterior predictive checks.

Paper: [arXiv:2609.29996](https://arxiv.org/abs/2609.29996).

## Milky Way Cepheids — candel-mwcepheids

[candel-mwcepheids](https://github.com/candel-cosmo/candel-mwcepheids) models the Gaia--HST Milky Way Cepheid period--luminosity calibration with selection effects and physical priors (`which_run = "MWCepheids"`).

Paper: [arXiv:2603.09880](https://arxiv.org/abs/2603.09880).

## Megamasers — candel-maser

[candel-maser](https://github.com/candel-cosmo/candel-maser) fits spot-level warped-disk models to NGC 5765b, NGC 6264, NGC 6323, UGC 3789, CGCG 074-064 and NGC 4258, then combines the per-galaxy distances into a joint $H_0$ with peculiar velocities.
It runs through its own scripts rather than `main.py`; see [running inference](running.md#megamasers).

Paper: [arXiv:2609.17684](https://arxiv.org/abs/2609.17684).
