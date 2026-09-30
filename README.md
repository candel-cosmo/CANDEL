# CANDEL

**A GPU-accelerated, hierarchical Bayesian framework for the local distance ladder.**

[![Documentation](https://readthedocs.org/projects/candel/badge/?version=latest)](https://candel.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Powered by JAX](https://img.shields.io/badge/powered%20by-JAX-orange.svg)](https://github.com/jax-ml/jax)

CANDEL forward-models the local distance ladder end to end and infers cosmology directly from the data. A single likelihood spans megamaser disk geometry, Cepheid and TRGB calibrations, Type Ia supernovae, the Tully--Fisher relation, and the Fundamental Plane, and jointly constrains the Hubble constant $H_0$, the peculiar-velocity amplitude $\beta$, and external bulk flows $\mathbf{V}_\mathrm{ext}$. Inference is fully Bayesian, gradient-based, and runs on a single GPU or scales across a cluster.

**Documentation:** [candel.readthedocs.io](https://candel.readthedocs.io/en/latest/)

## Overview

CANDEL forward-models distance-indicator observables (e.g. magnitude, line width, velocity dispersion) and redshift while marginalising over latent variables such as distance and absolute magnitude. Distance is either marginalised numerically via Simpson integration or sampled explicitly; latent observables are marginalised analytically where Gaussian conjugacy allows (SN stretch and colour via the Tripp relation; Fundamental Plane velocity dispersion and surface brightness), and via Gauss--Hermite quadrature otherwise. Posterior sampling uses the No-U-Turn Sampler (NUTS) from [NumPyro](https://github.com/pyro-ppl/numpyro), with JAX providing automatic differentiation and JIT compilation throughout.

When a reconstructed density and velocity field is supplied, CANDEL jointly calibrates each distance indicator and the underlying velocity field (e.g. amplitude $\beta$ and external bulk flow $\mathbf{V}_\mathrm{ext}$). The external dipole can also be inferred without any reconstructed field. For peculiar-velocity inferences, the distance prior is modelled following the phenomenological approach of [Lavaux (2016)](https://arxiv.org/abs/1512.04534), which effectively accounts for selection effects; for $H_0$ inferences, a rigorous selection function treatment is used instead (see [Stiskalek et al. 2025](https://arxiv.org/abs/2509.09665)). Model comparison is supported via BIC/AIC, Laplace evidence, and the [harmonic](https://github.com/astro-informatics/harmonic) package.

CANDEL runs locally for small samples or scales to computing clusters with GPU support (one GPU per chain). It includes cluster submission helpers and batch job generation tools for launching large parameter-grid runs from a frozen copy of the code.

### Key capabilities
- **Full-ladder forward modelling** in JAX and NumPyro, from geometric maser anchors to peculiar-velocity tracers.
- **Joint field-and-calibration inference:** distance-indicator relations and the underlying density/velocity field are constrained simultaneously, rather than calibrated in separate steps.
- **Reduced sampler dimensionality:** latent observables are marginalised analytically wherever Gaussian conjugacy allows, with Gauss--Hermite quadrature otherwise.
- **Flexible galaxy-bias models:** linear ($1 + b_1 \delta$), quadratic ($1 + b_1 \delta + b_2 \delta^2$), power-law ($\rho^\alpha$), and double power-law.
- **Density-dependent velocity dispersion** $\sigma_v(\delta)$ via a sigmoid in log-density, separating underdense and overdense regions.
- **Redshift-to-real-space mapping** of observed redshifts given a calibrated velocity field.
- **Peculiar-velocity covariance matrices** from CAMB power spectra.
- **HPC-ready tooling:** batch config generation, queue submission scripts, GPU auto-detection, and precomputed line-of-sight field generation.

## Supported distance indicators and catalogues

### Peculiar-velocity models

These models work in units of $h^{-1}\,\mathrm{Mpc}$ (i.e. assume $h = 1$). Multiple catalogues can be analysed jointly via `JointPVModel`, with user-specified shared parameters across sub-models.

- **Tully--Fisher relation:** 2MTF, SFI++, CF4-TFR
- **Type Ia supernovae (SALT2):** LOSS, Foundation, Pantheon+
- **Fundamental Plane:** 6dFGS-FP, SDSS-FP

### $H_0$ inference

- **Cepheid-calibrated $H_0$:** 35 Cepheid host galaxies from SH0ES
- **Milky Way Cepheid calibration:** standalone Galactic Cepheid period-luminosity calibration via `model.which_run = "MWCepheids"`
- **TRGB-calibrated $H_0$:** Tip of the Red Giant Branch distances from EDD
- **Megamaser disk $H_0$:** spot-level warped disk fits for NGC 5765b, NGC 6264, NGC 6323, UGC 3789, CGCG 074-064, and NGC 4258 using the BlackJAX explicit MCMC sampler.

## Repository structure

CANDEL is a small core library plus one package per probe. The core library never
imports a probe package: each package registers a `candel.Probe` under the
`candel.probes` entry-point group, and the core scripts (`main.py`,
`generate_tasks.py`, field preparation) look the probe up from the config's
`model.which_run`.

```
candel/                    core library
  field/                   field loaders, LOS interpolation, field caches, 3D volume grids
  model/                   ModelBase/H0ModelBase, priors, quadrature, LOS and bias utilities
  inference/               run_inference (NUTS + L-BFGS start), evidence
  cosmo/, plotting/        cosmography and corner plots
  configs/                 shared config fragments (data/field paths, common priors)
  probe.py, tasks.py       probe registry and task-spec helpers

packages/
  candel-pv/               peculiar-velocity models (TFR, SN, FP, Pantheon+), redshift2real
  candel-ch0/              Cepheid-calibrated H0 (SH0ES hosts), JWST forecast mocks
  candel-trgb/             EDD TRGB two-rung H0, mocks and posterior predictive checks
  candel-mwcepheids/       Milky Way Cepheid calibration
  candel-maser/            megamaser disk model and the two-stage megamaser H0 pipeline
    each with <module>/, configs/, scripts/, papers/ and tests/

scripts/
  runs/                    main.py, generate_tasks.py, submit.sh
  preprocess/              LOS and 3D field-cache preparation
  BORG_fields/, H0_convergence/, sync/
```

## Running inference

All experiments are defined in TOML configuration files that specify data paths, model parameters, priors, and output locations.

**Distance-ladder and peculiar-velocity models** (the probe is chosen by `model.which_run`):
```bash
python scripts/runs/main.py --config path/to/config.toml
```

To generate a batch of configs from a named parameter grid (each package defines its grids in `specs.py`):
```bash
python scripts/runs/generate_tasks.py list
python scripts/runs/generate_tasks.py build test
bash scripts/runs/submit.sh -q QUEUE --batch 8 --parallel 4 -n 8 test
```

`--batch N` puts N tasks in one scheduler job and `--parallel P` runs P of them at once on a CPU job, which amortises queue waits for many short tasks.

**Megamaser disk model:**
```bash
python -m candel_maser.run_maser NGC5765b
bash packages/candel-maser/scripts/submit.sh -q cmbgpu --galaxy NGC5765b --sampler mcmc
```

### Inference methods

- **NUTS** (default for the non-maser distance-indicator models): No-U-Turn Sampler via NumPyro. Robust, gradient-based.
- **BlackJAX explicit MCMC:** dedicated megamaser disk sampler in `candel_maser.run_maser`.
- **L-BFGS initialisation:** multi-start L-BFGS-B optimisation of the posterior to initialise NUTS chains, set by `init_maxiter` and `init_num_starts` in the `[inference]` section of the TOML config.

### Adding a probe

A new analysis (for example an SN Ia distance ladder) is a new package under
`packages/`: a model and data loader built on the core, a `Probe` subclass
implementing `load_data` and `build_model` (plus `task_specs` and
`task_tag_parts` for batch runs), and an entry point
`[project.entry-points."candel.probes"]` in its `pyproject.toml`. After
`pip install -e packages/<name>`, `main.py` and `generate_tasks.py` pick it up
without changes to the core.

## Results

CANDEL underpins a series of recent analyses:

- **A 1.8 per cent measurement of $H_0$ from Cepheids alone**, using a rigorous selection-function treatment of the SH0ES calibration sample.
  Stiskalek et al. (2025), [arXiv:2509.09665](https://arxiv.org/abs/2509.09665)

- **No evidence for $H_0$ anisotropy** in Tully--Fisher or supernova distances, constraining directional departures from isotropic expansion.
  Stiskalek et al. (2025), [arXiv:2509.14997](https://arxiv.org/abs/2509.14997)

- **$S_8$ from Tully--Fisher, Fundamental Plane, and supernova distances agrees with Planck**, from joint calibration of the velocity field and distance indicators.
  Stiskalek (2025), [arXiv:2509.20235](https://arxiv.org/abs/2509.20235)

- **The Velocity Field Olympics:** a systematic comparison of velocity-field reconstructions against direct distance tracers.
  Stiskalek et al. (2025), [arXiv:2502.00121](https://arxiv.org/abs/2502.00121)

- **Forward-modelling Milky Way Cepheids:** selection effects and physical priors in the Gaia--HST calibration.
  Stiskalek et al. (2026), [arXiv:2603.09880](https://arxiv.org/abs/2603.09880)

- **A reanalysis of the megamaser Hubble constant**, from spot catalogues to peculiar velocities.
  Stiskalek & Desmond (2026), [arXiv:2609.17684](https://arxiv.org/abs/2609.17684)

- **Two-rung ladder:** $H_0$ from the Tip of the Red Giant Branch and geometric anchors alone.
  Stiskalek et al. (2026), [arXiv:2609.29996](https://arxiv.org/abs/2609.29996)

## Installation
```
git clone https://github.com/candel-cosmo/CANDEL.git
cd CANDEL

python -m venv venv_candel
source venv_candel/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
for pkg in packages/*/; do python -m pip install -e "$pkg"; done
```

Install only the probe packages you need; the core runs without any of them.

For learned harmonic-mean evidence estimates, also install [harmonic](https://github.com/astro-informatics/harmonic).

## Local configuration (`local_config.toml`)

Per-machine settings live in a `local_config.toml` file at the repository
root. This file is **not** versioned and must be created on each machine where
CANDEL is installed. Start from `example_local_config.toml`. It supplies
machine-specific paths and Python interpreters used by the run scripts.

A minimal `local_config.toml` looks like:

```toml
root_main    = "/path/to/CANDEL/"   # repo root (required)
root_data    = "/path/to/data/"     # optional, defaults to <root_main>/data
root_results = "/path/to/results/"  # optional, defaults to <root_main>/results

python_exec = "/path/to/venv_candel/bin/python"  # used by cluster helpers
```

Relative input paths in run-time TOML configs are resolved against `root_data`;
output paths such as `fname_output` are resolved against `root_results`.
Absolute paths are left unchanged. Cluster submission helpers may use
additional machine/module keys; see [`docs/configuration.rst`](docs/configuration.rst)
for the full configuration schema.

## Known issues

- **TODO:** some scripts and notebooks still hard-code machine-specific paths
  (e.g. `/mnt/users/...`, `/Users/...`) instead of resolving them through
  `local_config.toml`: `scripts/H0_convergence/posterior_selection_integral_subsample.{py,sh}`,
  `packages/candel-pv/scripts/load_zcosmo_posterior.py`, and many of the paper
  notebooks under `packages/*/papers/`. Adjust these paths before running them.

## Citation

If you use CANDEL, or find it useful, please cite the papers listed above.

## License

MIT License -- see [LICENSE](LICENSE) for details.
