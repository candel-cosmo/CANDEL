# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.

from os.path import join

from h5py import File


def read_samples(root, fname, keys=None):
    fname = join(root, fname)

    with File(fname, "r") as f:
        if keys is None:
            keys = list(f["samples"].keys())
        elif isinstance(keys, str):
            keys = [keys]

        samples = {key: f[f"samples/{key}"][...] for key in keys}

    if isinstance(keys, list) and len(keys) == 1:
        return samples[keys[0]]
    return samples
