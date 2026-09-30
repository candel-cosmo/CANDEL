# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.

from h5py import File


def get_key_all(fnames, key):
    values = []
    for fname in fnames:
        with File(fname, 'r') as f:
            values.append(f["samples"][key][...])

    return values


def compute_S8_all(fnames, beta2cosmo):
    S8_list = []
    for fname in fnames:
        with File(fname, 'r') as f:
            beta = f["samples"]["beta"][...]
        S8 = beta2cosmo.compute_S8(beta)
        S8_list.append(S8)

    return S8_list


def compute_fsigma8_lin_all(fnames, beta2cosmo):
    fsigma8_lin_list = []
    for fname in fnames:
        with File(fname, 'r') as f:
            beta = f["samples"]["beta"][...]
        fsigma8_lin = beta2cosmo.compute_fsigma8_linear(beta)
        fsigma8_lin_list.append(fsigma8_lin)

    return fsigma8_lin_list


def replace_token_in_paths(files, token, replacement=""):
    """Replace or remove a token in simple (path, label) tuples."""
    new_files = []
    for path, label in files:
        new_path = path.replace(token, replacement if replacement else "")
        new_files.append((new_path, label))
    return new_files
