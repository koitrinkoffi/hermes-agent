#!/usr/bin/env bash
# Deploy (or re-sync) this skill from its canonical home in the hermes fork
# (local-skills/) into ~/.hermes/skills/ so Hermes discovers it. Running this on
# every upgrade keeps the live copy in lock-step with the backed-up canonical.
set -uo pipefail

SRC="${1:-$HOME/.hermes/hermes-agent/local-skills/hermes-stack-upgrade}"
DEST="$HOME/.hermes/skills/maintenance/hermes-stack-upgrade"

[ -f "$SRC/SKILL.md" ] || { echo "✗ canonical skill not found at $SRC"; exit 1; }
mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -r "$SRC" "$DEST"
chmod +x "$DEST"/scripts/*.sh 2>/dev/null || true
echo "✓ deployed skill to $DEST"
