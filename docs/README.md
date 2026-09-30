# CANDEL documentation

CANDEL is a GPU-accelerated hierarchical Bayesian framework for the local distance ladder and peculiar velocities, written in Python with JAX and NumPyro.
It forward-models distance-indicator observables and redshifts, from geometric megamaser anchors through Cepheids and the TRGB to peculiar-velocity tracers, and infers $H_0$, $S_8$ and velocity-field parameters directly from the data.

The code is split across several repositories in the [candel-cosmo](https://github.com/candel-cosmo) organisation: this core repository and one package per probe.
These pages describe how that code is organised and how to run it; the physics of each model is in the papers listed on the [probes](probes.md) page.

## Getting started

Clone the core and the probe packages you need side by side, then install them into one environment:

```bash
git clone https://github.com/candel-cosmo/CANDEL.git
git clone https://github.com/candel-cosmo/candel-pv.git
cd CANDEL
python -m venv venv_candel && source venv_candel/bin/activate
pip install -e .
pip install --no-deps -e ../candel-pv
cp example_local_config.toml local_config.toml   # then edit the paths
python scripts/runs/main.py --config ../candel-pv/configs/config.toml
```

Commands assume the CANDEL checkout as the working directory unless stated otherwise.

## User manual

| Guide | Contents |
| --- | --- |
| [Repositories](repositories.md) | The core and probe repositories, how they depend on each other, and where data and results live |
| [Installation](installation.md) | Environment, probe packages, GPU JAX, and `local_config.toml` |
| [Running inference](running.md) | `main.py`, batch grids with `generate_tasks.py`, cluster submission, and the megamaser runners |
| [Configuration](configuration.md) | Run TOML structure, config inheritance, path resolution, and priors |
| [Probes](probes.md) | What each probe package provides, its `which_run` value, configs and papers |
| [Development](development.md) | Core source layout, adding a probe, and tests |
