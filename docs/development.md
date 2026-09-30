[Documentation](README.md) / Development

# Development

## Core layout

```
candel/                    core library
  field/                   field loaders, LOS interpolation, field caches, 3D volume grids
  model/                   ModelBase/H0ModelBase, priors, quadrature, LOS and bias utilities
  inference/               run_inference (NUTS + L-BFGS start), evidence
  cosmo/, plotting/        cosmography and corner plots
  configs/                 shared config fragments (data/field paths, common priors)
  probe.py, tasks.py       probe registry and task-spec helpers

scripts/
  runs/                    main.py, generate_tasks.py, submit.sh, freeze_candel.sh
  preprocess/              LOS and 3D field-cache preparation
  BORG_fields/             BORG/Manticore field generation and diagnostics
  H0_convergence/          selection-integral convergence checks
  sync/                    rsync helpers for the clusters
docs/                      this documentation (MkDocs)
```

Code that more than one probe needs belongs in the core; code that only one probe uses belongs in that probe's package.

## Adding a probe

A new analysis is a new repository with the standard [layout](repositories.md#how-they-fit-together):

1. Build a data loader and a model on the core (`candel.model.ModelBase` or `H0ModelBase`).
2. Subclass `candel.Probe`, set `which_run`, and implement `load_data` and `build_model`; add `task_specs` and `task_tag_parts` for batch runs.
3. Register it in `pyproject.toml`:

    ```toml
    [project.entry-points."candel.probes"]
    MyProbe = "candel_myprobe.probe:MyProbe"
    ```

4. `pip install --no-deps -e ../candel-myprobe`.

`main.py` and `generate_tasks.py` then pick it up without any change to the core.
Two installed probes claiming the same `which_run` is an error.

## Tests

Each repository runs its own tests with `pytest` from its root; the core's are in `tests/`.

## Documentation

These pages are Markdown built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and published on [Read the Docs](https://candel.readthedocs.io/en/latest/).
Preview them locally with:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```
