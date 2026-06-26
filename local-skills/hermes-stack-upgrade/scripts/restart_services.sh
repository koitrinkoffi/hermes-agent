#!/usr/bin/env bash
# Restart the services that run the upgraded code so the new code is live.
# camofox runs from its checkout; the hermes gateway runs the editable install.
set -uo pipefail

restart() {
  local unit="$1"
  if systemctl list-unit-files 2>/dev/null | grep -q "^${unit}"; then
    echo "→ restarting $unit"
    sudo systemctl restart "$unit" && echo "  ✓ $unit restarted" || echo "  ✗ $unit restart failed"
  else
    echo "  ℹ $unit not present, skipping"
  fi
}

restart camofox.service
restart hermes-gateway.service

# Give camofox a moment, then health-check it.
sleep 2
url="${CAMOFOX_URL:-http://localhost:9377}"
if curl -fsS -m 10 "$url/health" >/dev/null 2>&1; then
  echo "✓ camofox health OK ($url)"
else
  echo "✗ camofox health check failed ($url)"
  exit 1
fi
