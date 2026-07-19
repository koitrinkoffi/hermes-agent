#!/usr/bin/env bash
# Push the upgraded mods branch to the fork. Run ONLY after smoke.sh is green,
# so the backup always points at a known-good state. No force-push.
set -uo pipefail

HERMES_REPO="${HERMES_REPO:-$HOME/.hermes/hermes-agent}"

push() {
  local dir="$1" branch="$2"
  echo "→ pushing $branch from $dir to origin (fork)"
  git -C "$dir" checkout "$branch" >/dev/null 2>&1
  if git -C "$dir" push origin "$branch"; then
    echo "  ✓ pushed"
  else
    echo "  ✗ push failed (non-fast-forward? check the fork)"; return 1
  fi
}

push "$HERMES_REPO" hermes-mods
echo "✓ backup updated"
