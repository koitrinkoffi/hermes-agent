#!/usr/bin/env bash
# Preflight checks for the hermes-stack upgrade. Verifies both repos are on the
# fork workflow (origin=fork, upstream=official), records pre-merge SHAs for
# rollback, and reports dirty trees. Writes state to $STATE_FILE.
#
# Exit 0 = ready to upgrade. Exit 1 = not ready (see messages).
set -uo pipefail

HERMES_REPO="${HERMES_REPO:-$HOME/.hermes/hermes-agent}"
CAMOFOX_REPO="${CAMOFOX_REPO:-$HOME/camofox}"
STATE_DIR="${STATE_DIR:-$HOME/.hermes/state}"
STATE_FILE="${STATE_FILE:-$STATE_DIR/hermes-stack-upgrade.state}"
mkdir -p "$STATE_DIR"

fail=0

check_repo() {
  local name="$1" dir="$2" mods_branch="$3"
  echo "── $name ($dir)"
  if [ ! -d "$dir/.git" ]; then echo "   ✗ not a git repo"; fail=1; return; fi
  local origin upstream
  origin="$(git -C "$dir" remote get-url origin 2>/dev/null || echo '')"
  upstream="$(git -C "$dir" remote get-url upstream 2>/dev/null || echo '')"
  echo "   origin   = ${origin:-<none>}"
  echo "   upstream = ${upstream:-<none>}"
  if [ -z "$upstream" ]; then echo "   ✗ no 'upstream' remote — run one-time fork setup"; fail=1; fi
  if echo "$origin" | grep -qiE 'NousResearch/hermes-agent|jo-inc/camofox-browser'; then
    echo "   ✗ origin still points at the official repo — should be your fork"; fail=1
  fi
  # Ensure the mods branch exists.
  if ! git -C "$dir" show-ref --verify --quiet "refs/heads/$mods_branch"; then
    echo "   ✗ mods branch '$mods_branch' not found"; fail=1
  fi
  # Report (but do not block on) a dirty tree — the agent decides.
  if [ -n "$(git -C "$dir" status --porcelain)" ]; then
    echo "   ⚠ working tree is dirty:"; git -C "$dir" status --short | sed 's/^/      /'
  fi
  local sha; sha="$(git -C "$dir" rev-parse HEAD)"
  echo "   HEAD = $sha ($(git -C "$dir" rev-parse --abbrev-ref HEAD))"
  echo "PREMERGE_${name}=$sha" >> "$STATE_FILE.tmp"
}

: > "$STATE_FILE.tmp"
echo "UPGRADE_STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATE_FILE.tmp"
check_repo HERMES "$HERMES_REPO" main
check_repo CAMOFOX "$CAMOFOX_REPO" hermes-mods
mv "$STATE_FILE.tmp" "$STATE_FILE"

echo
if [ "$fail" -eq 0 ]; then
  echo "✓ Preflight OK. Pre-merge SHAs saved to $STATE_FILE"
else
  echo "✗ Preflight found problems above. Resolve before upgrading."
fi
exit "$fail"
