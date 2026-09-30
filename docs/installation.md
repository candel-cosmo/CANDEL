[Documentation](README.md) / Installation

# Installation

## Environment

Use Python 3.10 or newer.
Clone the core and any probe packages side by side, then install the core first and the probes on top of it:

```bash
git clone https://github.com/candel-cosmo/CANDEL.git
for pkg in candel-pv candel-ch0 candel-trgb candel-mwcepheids candel-maser; do
    git clone "https://github.com/candel-cosmo/$pkg.git"
done
cd CANDEL

python -m venv venv_candel
source venv_candel/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e .
for pkg in ../candel-*/; do python -m pip install --no-deps -e "$pkg"; done
```

The core runs without any probe package, so clone and install only the ones you need.
`--no-deps` stops pip from reinstalling `candel` from its git URL over the editable checkout.

For learned harmonic-mean evidence estimates, also install [harmonic](https://github.com/astro-informatics/harmonic).

## GPU

On Linux the core pins `jax`, `jaxlib` and the matching `jax-cuda12-plugin`, so a machine with NVIDIA drivers gets a GPU build directly.
Check that JAX sees the device:

```bash
python -c "import jax; print(jax.devices())"
```

Inference uses one GPU per chain when several are visible.

## Local configuration

Machine-specific settings live in `local_config.toml` at the root of the CANDEL checkout.
It is not versioned; start from `example_local_config.toml`:

```toml
root_main    = "/path/to/CANDEL/"   # repository root (required)
root_data    = "/path/to/data/"     # optional, defaults to <root_main>/data
root_results = "/path/to/results/"  # optional, defaults to <root_main>/results

python_exec = "/path/to/venv_candel/bin/python"  # used by the submission scripts
machine     = "local"                            # local, arc or glamdring
```

The submission scripts also read `use_frozen`, module lists and GPU library paths from it; the example file lists them.
Probe repositories do not have their own `local_config.toml`: they read the core's.
