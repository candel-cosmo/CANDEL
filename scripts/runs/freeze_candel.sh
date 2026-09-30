#!/bin/bash -l
#
# Freeze the candel/ core, every installed probe package module (candel_*,
# from its own repository) and scripts/runs/main.py into a per-cluster install root
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
# Installed probe packages, wherever their repositories are checked out.
while IFS= read -r pkg_dir; do
    echo "[INFO] Package: $pkg_dir"
    rsync -a --exclude '__pycache__' --exclude '*.pyc' "$pkg_dir" "$frozen_dir"
done < <("$CANDEL_PYTHON" -c '
from importlib.metadata import distributions
from importlib.util import find_spec
names = {d.metadata["Name"].lower().replace("-", "_") for d in distributions()}
for name in sorted(n for n in names if n.startswith("candel_")):
    spec = find_spec(name)
    if spec is not None and spec.submodule_search_locations:
        print(spec.submodule_search_locations[0])')
cp "$main_script" "$frozen_dir/main.py"

echo "[INFO] Frozen structure:"
if command -v tree >/dev/null 2>&1; then
    tree -L 2 "$frozen_dir"
else
    echo "[INFO] (Skipping tree output: 'tree' not found)"
    ls -l "$frozen_dir"
fi
