[Documentation](README.md) / Running inference

# Running inference

## A single run

Every run is one TOML config.
The config's `model.which_run` picks the probe; see [configuration](configuration.md) for the format.

```bash
python scripts/runs/main.py --config ../candel-ch0/configs/config_CH0.toml
```

Samples are written to HDF5 under `root_results`, at the config's `io.fname_output`.
Read them back with `candel.read_samples` and plot them with `candel.plotting.corner.plot_corner`.

## Batch grids

Parameter sweeps are named task specs defined in each probe's `specs.py`.
`generate_tasks.py` lists the specs of every installed probe and builds one config per grid point plus a task list:

```bash
python scripts/runs/generate_tasks.py list
python scripts/runs/generate_tasks.py show CH0_main
python scripts/runs/generate_tasks.py build CH0_main --dry-run
python scripts/runs/generate_tasks.py build CH0_main
```

Building writes the configs to `scripts/runs/generated_configs/` and the list to `scripts/runs/tasks_<spec>.txt`, one `<index> <config_path>` per line.

## Submission

`scripts/runs/submit.sh` submits a task list to the scheduler named by `machine` in `local_config.toml`, or runs it inline when `machine = "local"`:

```bash
bash scripts/runs/submit.sh -q QUEUE --batch 8 --parallel 4 -n 8 CH0_main
```

`--batch N` puts N tasks in one scheduler job, and `--parallel P` runs P of them at once on a CPU job, which amortises queue waits for many short tasks.
Run `bash scripts/runs/submit.sh --help` for the full option list.

`scripts/runs/freeze_candel.sh` copies the core, every installed probe module and `main.py` into a frozen install root.
With `use_frozen = true`, jobs run against that copy, so editing the source does not change jobs already in the queue.

## Field inputs

Runs that use a reconstructed density and velocity field read precomputed line-of-sight products.
`scripts/preprocess/prepare_field_inputs.py` builds them for the catalogues of a config; each probe tells it where its catalogue's sky positions are.

## Megamasers

The megamaser package has its own runners rather than a `which_run`:

```bash
python -m candel_maser.run_maser NGC5765b
bash ../candel-maser/scripts/submit.sh -q cmbgpu --galaxy NGC5765b --sampler mcmc
```

The per-galaxy disk fits come first (differential-evolution MAP, then BlackJAX MCMC), followed by the joint $H_0$ fit over the galaxies.
`candel-maser/docs/README.md` is the guide to its scripts.
