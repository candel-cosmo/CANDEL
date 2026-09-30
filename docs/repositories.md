[Documentation](README.md) / Repositories

# Repositories

CANDEL is a small core library plus one repository per probe, all under [candel-cosmo](https://github.com/candel-cosmo).

| Repository | Contents | `model.which_run` |
| --- | --- | --- |
| [CANDEL](https://github.com/candel-cosmo/CANDEL) | Core library, run and submission scripts, field preparation, these docs, and the shared `data/`, `results/` and `local_config.toml` | — |
| [candel-pv](https://github.com/candel-cosmo/candel-pv) | Peculiar-velocity models (TFR, SN, FP, Pantheon+), growth rate and $S_8$, redshift-to-real-space mapping | `PV` (or unset) |
| [candel-ch0](https://github.com/candel-cosmo/candel-ch0) | Cepheid-calibrated $H_0$ (SH0ES hosts), JWST forecast mocks | `CH0` |
| [candel-trgb](https://github.com/candel-cosmo/candel-trgb) | Two-rung $H_0$ from EDD TRGB distances, mocks and posterior predictive checks | `EDD_TRGB` |
| [candel-mwcepheids](https://github.com/candel-cosmo/candel-mwcepheids) | Milky Way Cepheid period--luminosity calibration | `MWCepheids` |
| [candel-maser](https://github.com/candel-cosmo/candel-maser) | Megamaser warped-disk model and the two-stage megamaser $H_0$ pipeline | own runners |

## How they fit together

**Dependencies point one way.**
Probe packages import `candel`; the core never imports a probe.
The core holds everything the probes share: inference (NUTS with an L-BFGS start, evidence), selection integrals, field loaders and line-of-sight interpolation, cosmography, priors, and plotting.

**Probes are discovered through entry points.**
Each probe package defines a subclass of `candel.Probe` with `load_data` and `build_model`, plus `task_specs` for batch runs, and registers it in its `pyproject.toml`:

```toml
[project.entry-points."candel.probes"]
CH0 = "candel_ch0.probe:CH0Probe"
```

`scripts/runs/main.py` reads `model.which_run` from the run config and calls `candel.get_probe(which_run).run(config)`.
Installing a package is all it takes to make its probe available; nothing in the core names it.
Peculiar-velocity runs set `which_run = "PV"` or leave it unset; both mean the same.
The megamaser package is the exception: it has its own runners (`python -m candel_maser.run_maser`) and uses the core as a library.

**There is one shared workspace.**
The probe repositories are cloned next to the core (`../candel-pv`, ...).
Input data, results and `local_config.toml` live only in the CANDEL checkout; probe code finds it through `candel.util.CANDEL_ROOT` in Python and `$CANDEL_ROOT`, defaulting to `../CANDEL`, in shell scripts.

**Every probe repository has the same layout.**

| Directory | Contents |
| --- | --- |
| `candel_<name>/` | The package: data loader, model, `probe.py`, `specs.py` |
| `configs/` | Run configurations for that probe |
| `scripts/` | Preprocessing, mocks and submission helpers |
| `papers/` | Scripts and notebooks behind each paper |
| `tests/` | `pytest` tests |

```
candel-cosmo/            any parent directory; the name is free
├── CANDEL/              core; data/, results/, local_config.toml
├── candel-pv/           ─┐
├── candel-ch0/           │ probe packages,
├── candel-trgb/          │ each `pip install -e`'d
├── candel-mwcepheids/    │ into CANDEL's environment
└── candel-maser/        ─┘
```
