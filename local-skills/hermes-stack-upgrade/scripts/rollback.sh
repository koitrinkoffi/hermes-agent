#!/usr/bin/env bash
# Roll both repos back to the pre-merge SHAs captured by preflight.sh, then
# restart services. Use when the smoke gate fails and you want the prior state.
# Does NOT touch the forks (no push), so your backups are unaffected.
set -uo pipefail

HERMES_REPO="${HERMES_REPO:-$HOME/.hermes/hermes-agent}"
CAMOFOX_REPO="${CAMOFOX_REPO:-$HOME/camofox}"
STATE_FILE="${STATE_FILE:-$HOME/.hermes/state/hermes-stack-upgrade.state}"

[ -f "$STATE_FILE" ] || { echo "✗ no state file at $STATE_FILE — cannot determine rollback targets"; exit 1; }
# shellcheck disable=SC1090
. "$STATE_FILE"

reset_to() {
  local dir="$1" sha="$2"
  [ -n "$sha" ] || { echo "  ✗ no SHA recorded for $dir"; return 1; }
  echo "→ $dir: git reset --hard $sha"
  git -C "$dir" merge --abort 2>/dev/null || true
  git -C "$dir" reset --hard "$sha" && echo "  ✓ restored"
}

reset_to "$HERMES_REPO" "${PREMERGE_HERMES:-}"
reset_to "$CAMOFOX_REPO" "${PREMERGE_CAMOFOX:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/restart_services.sh" || true
echo "✓ rollback complete (forks untouched)"
