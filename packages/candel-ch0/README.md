# candel-ch0

Cepheid-calibrated H0 forward model for CANDEL. A probe package for
[CANDEL](https://github.com/candel-cosmo/CANDEL): it imports the core `candel`
library and registers itself with it through the `candel.probes` entry point.

## Install

Clone this repository next to the CANDEL core and install both, core first:

```bash
git clone https://github.com/candel-cosmo/CANDEL.git
git clone https://github.com/candel-cosmo/candel-ch0.git
cd CANDEL
python -m venv venv_candel && source venv_candel/bin/activate
pip install -e .
pip install --no-deps -e ../candel-ch0
```

Data, results and the machine-local `local_config.toml` live in the CANDEL
checkout. Python code finds it through the installed `candel`
(`candel.util.CANDEL_ROOT`); shell scripts use `$CANDEL_ROOT`, defaulting to
`../CANDEL`.

## Layout

- `candel_ch0/` - the package
- `configs/` - run configurations
- `scripts/` - preprocessing, mocks and submission helpers
- `papers/` - scripts and notebooks behind each paper (CH0)
- `tests/` - tests (`pytest`)

## Run

```bash
python ../CANDEL/scripts/runs/main.py --config configs/<config>.toml
```

## License

MIT; see `LICENSE`.
