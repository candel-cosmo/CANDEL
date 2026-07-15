#!/bin/bash
# Rebuild the public, megamaser-free CANDEL snapshot from this PRIVATE repo and
# (optionally) force-push it to the public repo.
#
# Source of truth is ALWAYS this private repo (CANDEL-dev). Never edit the public
# repo or its clone directly -- changes there are clobbered by the next release.
#
# Usage:
#   release/make_public.sh [SRC_REF] [--check|--yes]
#     SRC_REF   commit/branch to snapshot (default: HEAD = current branch tip)
#     --check   build + verify + squash, then stop (no push, no prompt)
#     --yes     force-push without the interactive prompt
#     (default) build + verify + squash, then prompt before force-pushing
set -euo pipefail

usage() {
  cat <<'EOF'
make_public.sh -- rebuild the public, megamaser-free CANDEL snapshot from this
private repo and (optionally) force-push it to github.com/Richard-Sti/CANDEL.

Usage:
  release/make_public.sh [SRC_REF] [--check | --yes]
  release/make_public.sh -h | --help

Arguments:
  SRC_REF     commit/branch to snapshot (default: HEAD = current branch tip)

Modes:
  (default)   build + verify + squash, then CONFIRM before force-pushing
  --check     build + verify + squash, then stop (no push, no prompt)
  --yes       force-push without the confirmation prompt (for automation)

The public repo is a rolling single squashed commit; publishing FORCE-PUSHES
(overwrites its history). Source of truth is always this private repo -- never
edit the public repo directly. The release/ tooling is excluded from the
public snapshot and a gate aborts if it ever appears there.
EOF
}

SRC_REF="HEAD"
MODE="prompt"
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --check)   MODE="check" ;;
    --yes)     MODE="yes" ;;
    --*)       echo "make_public: unknown option '$arg'" >&2; usage; exit 2 ;;
    *)         SRC_REF="$arg" ;;
  esac
done

PRIV_ROOT="$(git rev-parse --show-toplevel)"
PATCH="$PRIV_ROOT/release/public_scrub.patch"
PUBLIC_URL="https://github.com/Richard-Sti/CANDEL.git"
WORK="$(mktemp -d)"
KEEP=""
trap '[ -n "$KEEP" ] || rm -rf "$WORK"' EXIT

echo "[make_public] source ref : $SRC_REF"
echo "[make_public] scratch    : $WORK/pub"

# Clean clone of committed content only (no working-tree / untracked leakage).
git clone --no-hardlinks --quiet "$PRIV_ROOT" "$WORK/pub"
cd "$WORK/pub"
git checkout --quiet "$SRC_REF"

# --- 2a. Wholesale removals (patterns; missing paths are no-ops) ---
git rm -r -q --ignore-unmatch \
  candel/model/maser_*.py candel/model/model_H0_maser.py \
  candel/pvdata/megamaser_data.py \
  scripts/megamaser \
  notebooks/paper_MMH0 \
  'tests/test_megamaser_*.py' tests/test_reid_chain_loader.py \
  tools/mcp \
  .codex AGENT_MEMORY.md AGENTS.md .mcp.json \
  local_config_backup.toml \
  release

# --- 2b. Content scrubs (README/setup/__init__/... ) via patch ---
if ! git apply --index "$PATCH"; then
  echo "[make_public] ERROR: scrub patch did not apply cleanly." >&2
  echo "  Source moved around the maser edits. Regenerate the patch:" >&2
  echo "    hand-remove maser refs in the affected files, then" >&2
  echo "    (from a clean checkout) 'git diff > release/public_scrub.patch'." >&2
  exit 1
fi

# --- 2d. Verification gates ---
fail=0
if git grep -i -qE 'maser|mmh0|\breid\b' -- '*.py' '*.md' '*.toml' '*.sh' '*.cfg' '*.txt' '*.gitignore'; then
  echo "[make_public] GATE FAIL: maser/mmh0/reid text remains in source:" >&2
  git grep -i -nE 'maser|mmh0|\breid\b' -- '*.py' '*.md' '*.toml' '*.sh' '*.cfg' '*.txt' '*.gitignore' >&2
  fail=1
fi
if git ls-files | grep -qi 'local_config_backup'; then
  echo "[make_public] GATE FAIL: local_config_backup.* present." >&2; fail=1
fi
if git ls-files | grep -qE '(^|/)release/|(^|/)make_public\.sh$|public_scrub\.patch$'; then
  echo "[make_public] GATE FAIL: release tooling must never reach the public tree:" >&2
  git ls-files | grep -E '(^|/)release/|make_public\.sh$|public_scrub\.patch$' >&2
  fail=1
fi
if git grep -nE 'BEGIN (RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|xox[baprs]-' -- . >/dev/null 2>&1; then
  echo "[make_public] GATE FAIL: possible secret in tree." >&2; fail=1
fi
[ "$fail" -eq 0 ] || { echo "[make_public] aborting -- gates failed."; exit 1; }
# NOTE: notebooks (*.ipynb) are NOT text-gated -- grep matches base64 image output.
# The only known incidental hit is spiral_arms.ipynb ("Reid 2019 masers", a MW
# spiral-arm citation). Eyeball new notebooks before a release.
echo "[make_public] gates passed: $(git ls-files | wc -l | tr -d ' ') files, maser-free."

# --- 2e. Squash into a single orphan commit ---
git add -A
git checkout -q --orphan public
git commit -q -m "Public release of CANDEL"
echo "[make_public] snapshot ready: 1 commit, $(git ls-files | wc -l | tr -d ' ') files."

do_push() {
  git -c credential."https://github.com".helper='!gh auth git-credential' \
      push --force "$PUBLIC_URL" public:master
  echo "[make_public] force-pushed to $PUBLIC_URL (master)."
}

case "$MODE" in
  check)
    echo "[make_public] --check: not pushing.";;
  yes)
    do_push;;
  *)
    echo
    echo "  ABOUT TO FORCE-PUSH (this OVERWRITES the public repo's history):"
    echo "    from : $WORK/pub  (branch 'public', 1 commit, $(git ls-files | wc -l | tr -d ' ') files)"
    echo "    to   : $PUBLIC_URL  ->  master"
    read -r -p "  Type 'yes' to confirm: " ans
    if [ "${ans:-}" = "yes" ]; then
      do_push
    else
      KEEP=1
      echo "[make_public] push cancelled. Inspect or push manually from:"
      echo "  cd $WORK/pub && git push --force $PUBLIC_URL public:master"
    fi;;
esac
