# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""
Registry of the probes that installed CANDEL packages provide.

A probe is one `model.which_run` value (``"PV"`` or unset for peculiar-velocity
runs, stored as ``None``)
together with its data loader, model, task specs and filename tags. Packages
register a `Probe` subclass under the ``candel.probes`` entry-point group, so
core scripts (`main.py`, `generate_tasks.py`, field preparation) never name a
probe themselves.
"""
from functools import lru_cache
from importlib.metadata import entry_points

# Explicit `model/which_run` of PV runs; the registry keys them under None.
PV_WHICH_RUN = "PV"


class Probe:
    """Base class for a probe registered by a CANDEL package."""

    # `model/which_run` value handled by this probe (None for PV runs).
    which_run = None
    # Whether the model normalises with the H0 3D selection volume.
    uses_h0_volume = False
    # H0 host LOS: catalogue name and config paths of its reconstruction
    # and LOS-file template.
    los_catalogue = None
    reconstruction_key = None
    los_file_key = None
    # Named sweeps for generate_tasks.py; config paths may be absolute.
    task_specs = {}

    def load_data(self, config_path):
        raise NotImplementedError

    def build_model(self, config_path, data):
        raise NotImplementedError

    def model_kwargs(self, model, data):
        """Keyword arguments passed to the NumPyro model."""
        return {}

    def run(self, config_path):
        """Load the data, build the model and run inference."""
        from .inference import run_inference
        data = self.load_data(config_path)
        model = self.build_model(config_path, data)
        run_inference(model, self.model_kwargs(model, data))

    def task_tag_parts(self, config):
        """Probe-specific components of a generated task filename."""
        return []

    def sky_positions(self, catalogue, config):
        """Return (RA, dec, LOS template) of `catalogue`, or None if unknown.

        Used to prepare LOS products for the catalogues this probe owns.
        """
        return None

    def h0_volume_field_key(self, config):
        """3D field product (density or velocity) the H0 volume needs."""
        return "density"


@lru_cache(maxsize=None)
def probes():
    """Installed probes keyed by their `which_run`."""
    out = {}
    for ep in entry_points(group="candel.probes"):
        probe = ep.load()()
        if probe.which_run in out:
            raise RuntimeError(
                f"Two installed probes handle which_run={probe.which_run!r}.")
        out[probe.which_run] = probe
    return out


def get_probe(which_run):
    """Return the installed probe for a `model/which_run` value."""
    if which_run == PV_WHICH_RUN:
        which_run = None
    try:
        return probes()[which_run]
    except KeyError:
        raise ValueError(
            f"No installed CANDEL package handles which_run={which_run!r}; "
            f"installed: {sorted(map(str, probes()))}. Install the package "
            "providing it (e.g. `pip install -e ../candel-<name>`).") from None


def task_specs():
    """Named task specs of every installed probe."""
    specs = {}
    for probe in probes().values():
        overlap = set(specs) & set(probe.task_specs)
        if overlap:
            raise RuntimeError(f"Task specs defined twice: {sorted(overlap)}")
        specs.update(probe.task_specs)
    return specs


def reconstruction_keys():
    """Config paths naming the H0 host reconstructions of installed probes."""
    return tuple(p.reconstruction_key for p in probes().values()
                 if p.reconstruction_key is not None)
