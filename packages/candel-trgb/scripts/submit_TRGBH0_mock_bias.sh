#!/bin/bash -l
# TRGBH0 paper mock bias test (appendix "Mock validation").
#
# Thin wrapper around the --default-manticore preset: fiducial model WITHOUT
# the sky-exposure term. Mocks are drawn from a single Manticore-Local COLA PCS
# field (picked as master seed % 80), with the double power-law
# source-density bias, 4 Mpc/h density smoothing, the |b| >= 10 deg mask,
# the soft TRGB-magnitude window, and Student-t redshift noise; recovery
# uses the same model with the selection edge and width inferred. Injected
# truths are a paper-motivated point with beta fixed to unity in generation
# and recovery; see FIDUCIAL_MANTICORE_DEFAULTS in mock_TRGB.py.
#
# All submission options (queue, --gpu, --n-mocks, --single, --local, --dry,
# --field-index, ...) are forwarded, e.g.:
#   ./submit_TRGBH0_mock_bias.sh -q gpulong --n-mocks 100 --gpu-shards 10
#   ./submit_TRGBH0_mock_bias.sh --local --single --seed 42
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$HERE/mock_TRGB.sh" --default-manticore "$@"
