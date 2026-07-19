#!/usr/bin/env bash
# Restart the Hermes services (systemd --user units) so upgraded code goes live.
# The hermes gateway runs the editable install.
set -uo pipefail

restart() {
  local unit="$1"
  # Presence check reads the unit; `cat` needs no sudo.
  if systemctl --user cat "$unit" >/dev/null 2>&1; then
    echo "→ restarting $unit"
    systemctl --user restart "$unit" && echo "  ✓ $unit restarted" || echo "  ✗ $unit restart failed"
  else
    echo "  ℹ $unit not present, skipping"
  fi
}

restart hermes-gateway.service
# The dashboard is a separate long-lived process; it caches the tool registry at
# startup, so it must restart too or newly added tools won't appear in the UI.
restart hermes-dashboard.service

echo "→ checking hermes-gateway is active"
sleep 2
if systemctl --user is-active --quiet hermes-gateway.service; then
  echo "✓ hermes-gateway active"
else
  echo "✗ hermes-gateway not active after restart"
  exit 1
fi
