# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.
"""Small building blocks shared by the probes' task specs and tags."""


def is_active(value):
    """Check if value is active (not None or 'none')."""
    if value is None:
        return False
    return str(value).lower() != "none"


def is_delta_prior(prior):
    """Check if prior is a delta distribution."""
    return isinstance(prior, dict) and prior.get("dist") == "delta"


def is_manticore_los(value):
    return isinstance(value, str) and "manticore" in value.lower()


def tag_number(value):
    """Compact number formatting for generated filename tags."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def delta(value):
    return {"dist": "delta", "value": value}


def normal(loc, scale):
    return {"dist": "normal", "loc": loc, "scale": scale}


def nu_cz_student_t_prior():
    return {
        "dist": "truncated_normal",
        "low": 1.0,
        "high": 100.0,
        "mean": 30.0,
        "scale": 10.0,
    }


def with_root(root_output):
    return {"io/root_output": root_output}
