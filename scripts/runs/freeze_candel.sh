#!/bin/bash -l
#
# Freeze the candel/ core, the probe packages' modules (packages/*/candel_*)
# and scripts/runs/main.py into a per-cluster install root
# (CANDEL_FROZEN_ROOT), so subsequent submissions run against a stable
# snapshot instead of the evolving source tree. The frozen root is first on
# PYTHONPATH, so the probe entry points resolve to the frozen modules.
#
# Paths and cluster identity come from local_config.toml via _submit_lib.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../_submit_lib.sh
source "$ROOT/scripts/_submit_lib.sh"

src_dir="$CANDEL_ROOT/candel"
main_script="$CANDEL_ROOT/scripts/runs/main.py"
frozen_dir="$CANDEL_FROZEN_ROOT"

echo "[INFO] Freezing candel, probe packages + main.py"
echo "[INFO] Cluster: $CANDEL_CLUSTER"
echo "[INFO] From:    $src_dir"
echo "[INFO] To:      $frozen_dir"

rm -rf "$frozen_dir"
mkdir -p "$frozen_dir"

rsync -a --exclude '__pycache__' --exclude '*.pyc' "$src_dir" "$frozen_dir"
for pkg_dir in "$CANDEL_ROOT"/packages/*/candel_*; do
    [[ -d "$pkg_dir" ]] || continue
    echo "[INFO] Package: $pkg_dir"
    rsync -a --exclude '__pycache__' --exclude '*.pyc' "$pkg_dir" "$frozen_dir"
done
cp "$main_script" "$frozen_dir/main.py"

echo "[INFO] Frozen structure:"
if command -v tree >/dev/null 2>&1; then
    tree -L 2 "$frozen_dir"
else
    echo "[INFO] (Skipping tree output: 'tree' not found)"
    ls -l "$frozen_dir"
fi
