# Copyright (C) 2025 Richard Stiskalek
# Licensed under the MIT License; see LICENSE in the repository root.

from .loader import (                                                           # noqa
    BORGFieldLoader,                                                            # noqa
    BORGSPHFieldLoader,                                                         # noqa
    FIELD_METADATA,                                                             # noqa
    FieldMetadata,                                                              # noqa
    ManticoreLocalCOLA_FieldLoader,                                             # noqa
    ManticoreLocalSWIFT_FieldLoader,                                            # noqa
    UNKNOWN_FIELD_METADATA,                                                     # noqa
    available_mcmc_field_indices,                                               # noqa
    field_allows_raw_product_reads,                                             # noqa
    field_mas_directory,                                                        # noqa
    field_metadata,                                                             # noqa
    field_requires_cached_products,                                             # noqa
    name2field_loader,                                                          # noqa
    supported_field_names,                                                      # noqa
    )


_FIELD_INTERP_EXPORTS = {
    "interpolate_los_density_velocity",
    "apply_gaussian_smoothing",
    "prepare_los_geometry",
}


def __getattr__(name):
    if name in _FIELD_INTERP_EXPORTS:
        from . import field_interp
        value = getattr(field_interp, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
