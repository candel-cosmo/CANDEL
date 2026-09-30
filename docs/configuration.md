[Documentation](README.md) / Configuration

# Configuration

## Structure

A run is defined by one TOML file.
Most configs inherit from a maintained base and override a few keys:

```toml
base = ["config.toml"]

[inference]
model = "TFRModel"
num_warmup = 500
num_samples = 1000
num_chains = 4

[io]
catalogue_name = "CF4_W1"
fname_output = "results/example/samples.hdf5"

[pv_model]
kind = "Vext"

[model]
which_selection = "none"
```

| Section | Purpose |
| --- | --- |
| `[model]` | Which probe (`which_run`), selection function, reconstruction, cosmology, and `[model.priors]` |
| `[inference]` | Sampler settings (`num_warmup`, `num_samples`, `num_chains`, `seed`, L-BFGS start via `init_maxiter` and `init_num_starts`), and the model class for PV runs |
| `[io]` | Catalogue names, input paths, and `fname_output` |
| `[pv_model]` | Peculiar-velocity options such as the distance prior and the flow model |

For joint peculiar-velocity fits, `inference.model` and `io.catalogue_name` take lists, one entry per catalogue.

## Inheritance

`base` fragments are resolved relative to the config file first, then against the shared fragments in `candel/configs/` (`config_paths.toml`, `config_priors.toml`).
Values merge with increasing precedence: `base` fragments, then `local_config.toml`, then the config itself.
Each probe keeps its maintained base configs in its own `configs/` directory.

## Paths

Relative input paths resolve against `root_data`, and output paths such as `fname_output` against `root_results`, both set in [`local_config.toml`](installation.md#local-configuration).
Absolute paths are used unchanged, so reusable run configs should not contain machine paths.

## Priors

Priors are set in `[model.priors]` by distribution name:

| Distribution | Parameters |
| --- | --- |
| `uniform` | `low`, `high` |
| `normal` | `mean`, `std` |
| `half_normal` | `std` |
| `log_normal` | `mean`, `std` of the underlying normal |
| `delta` | `value` (fixed parameter) |

Common defaults live in `candel/configs/config_priors.toml`.
