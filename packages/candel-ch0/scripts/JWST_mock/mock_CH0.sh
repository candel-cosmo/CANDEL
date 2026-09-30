#!/bin/bash -l
# Submit the CH0 JWST-forecast mock scenarios, one GPU job per scenario.
# The host pool must exist first: python scripts/JWST_mock/mock_CH0.py pool
#
# usage: mock_CH0.sh [-q QUEUE] [--seeds 0-9] [--dry] [SCENARIO ...]
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Core CANDEL checkout (data/, results/, local_config.toml); defaults
# to a sibling clone of candel-cosmo/CANDEL.
ROOT="${CANDEL_ROOT:-$(cd "$PKG_ROOT/../CANDEL" 2>/dev/null && pwd)}"
[[ -f "$ROOT/scripts/_submit_lib.sh" ]] || {
    echo "[ERROR] Set CANDEL_ROOT to the CANDEL core checkout." >&2; exit 1; }
# shellcheck source=../../../../scripts/_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

queue="gpulong"
seeds="0-9"
dry_flag=()
scenarios=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--queue) queue="$2"; shift 2 ;;
        --seeds) seeds="$2"; shift 2 ;;
        --dry) dry_flag=(--dry); shift ;;
        *) scenarios+=("$1"); shift ;;
    esac
done
if [[ ${#scenarios[@]} -eq 0 ]]; then
    mapfile -t scenarios < <(cd "$PKG_ROOT" && "$CANDEL_PYTHON" -c \
        "import sys; sys.path.insert(0, 'scripts/JWST_mock'); \
import mock_CH0; print('\n'.join(mock_CH0.SCENARIOS))")
fi

cd "$ROOT"
mkdir -p "$PKG_ROOT/scripts/JWST_mock/logs"
for sc in "${scenarios[@]}"; do
    echo "[mock_CH0] scenario $sc, seeds $seeds, queue $queue"
    submit_job --queue "$queue" --mem 3 --gpu --name "mock_CH0_$sc" \
        --logdir "$PKG_ROOT/scripts/JWST_mock/logs" "${dry_flag[@]}" -- \
        /usr/bin/env PYTHONPATH="$ROOT" \
        "$CANDEL_PYTHON" -u "$PKG_ROOT/scripts/JWST_mock/mock_CH0.py" run \
        --scenario "$sc" --seeds "$seeds"
done
