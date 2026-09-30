#!/bin/bash -l
# Submit the CH0 new-host forecast runs (see ch0_new_hosts.py), one GPU job
# per line of runs. Needs `new_hosts.npz` from the prep step and the
# CH0_JWST_forecast_sigv150 generated configs.
#
# usage: ch0_new_hosts.sh [-q QUEUE] [--time H] [--dry]
set -euo pipefail

ROOT="${CANDEL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
# shellcheck source=../../../../scripts/_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

case "$CANDEL_CLUSTER" in
    arc) queue="short" ;;
    *)   queue="optgpu" ;;
esac
time_flag=()
dry_flag=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|--queue) queue="$2"; shift 2 ;;
        --time) time_flag=(--time "$2"); shift 2 ;;
        --dry) dry_flag=(--dry); shift ;;
        *) echo "unknown argument $1"; exit 1 ;;
    esac
done

# Ten random realisations, fixed by seed.
mapfile -t fields < <("$CANDEL_PYTHON" -c \
    "import numpy as np; print('\n'.join(map(str, sorted(np.random.default_rng(0).choice(80, 10, replace=False)))))")

jobs=()
# All 80 realisations marginalised over.
for sc in HST_windows HST_plus5 JWST35_plus5 JWST36_plus5; do
    jobs+=("--scenario $sc")
done
for sc in HST_plus5 JWST35_plus5; do
    jobs+=("--scenario $sc --sigma-mu 0.1")
done
# Added rows at the far hosts' mean log P - 1, so that b_W enters them.
jobs+=("--scenario HST_plus5 --logP 0.6")
# One realisation for both the mock velocities and the inference.
for k in "${fields[@]}"; do
    line=""
    for sc in HST_windows HST_plus5 JWST35_plus5; do
        line+="--scenario $sc --field $k;"
    done
    jobs+=("$line")
done

cd "$ROOT"
mkdir -p "$ROOT/packages/candel-ch0/scripts/JWST_mock/logs"
for j in "${jobs[@]}"; do
    cmd=""
    IFS=';' read -ra runs <<< "$j"
    for r in "${runs[@]}"; do
        [[ -z "$r" ]] && continue
        cmd+="$CANDEL_PYTHON -u $ROOT/packages/candel-ch0/scripts/JWST_mock/ch0_new_hosts.py run $r; "
    done
    name="ch0_newhosts_$(echo "$j" | tr -cd '[:alnum:]_.' | cut -c1-40)"
    echo "[ch0_new_hosts] $j (queue $queue)"
    submit_job --queue "$queue" --mem 16 --gpu --name "$name" \
        "${time_flag[@]}" --logdir "$ROOT/packages/candel-ch0/scripts/JWST_mock/logs" "${dry_flag[@]}" -- \
        /usr/bin/env PYTHONPATH="$ROOT" bash -c "$cmd"
done
