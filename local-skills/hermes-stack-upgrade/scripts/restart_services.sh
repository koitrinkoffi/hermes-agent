#!/usr/bin/env bash
# Restart the local stack so upgraded code goes live.
# The Hermes services run as systemd --user units (no sudo); Lemonade is a
# SYSTEM unit (lemond.service, /opt/bin/lemond on :13305) and needs sudo.
#   camofox runs from its checkout; the hermes gateway runs the editable install.
set -uo pipefail

# restart <scope> <unit>   scope = user | system
restart() {
  local scope="$1" unit="$2"
  local -a sctl=(systemctl); [ "$scope" = user ] && sctl=(systemctl --user)
  # Presence check reads the unit; `cat` needs no sudo even for system units.
  if ! "${sctl[@]}" cat "$unit" >/dev/null 2>&1; then
    echo "  ℹ $unit not present, skipping"; return 0
  fi
  echo "→ restarting $unit ($scope)"
  if [ "$scope" = system ]; then
    # System unit -> sudo. Don't hang a non-interactive run on a password prompt:
    # use sudo only when it's passwordless or we have a TTY; otherwise warn.
    if sudo -n true 2>/dev/null || [ -t 0 ]; then
      sudo systemctl restart "$unit" && echo "  ✓ $unit restarted" || echo "  ✗ $unit restart failed"
    else
      echo "  ⚠ $unit needs sudo (non-interactive) — run: sudo systemctl restart $unit"
    fi
  else
    systemctl --user restart "$unit" && echo "  ✓ $unit restarted" || echo "  ✗ $unit restart failed"
  fi
}

# Lemonade (LLM/STT backend) first, so the gateway comes up against a live
# backend. Bounded, non-fatal readiness wait; gated on lemond being present.
if systemctl cat lemond.service >/dev/null 2>&1; then
  restart system lemond.service
  lem_url="${LEMOND_URL:-http://localhost:13305}"
  for i in $(seq 1 20); do
    if curl -fsS -m 3 "$lem_url/api/v1/health" >/dev/null 2>&1; then
      echo "✓ lemonade health OK ($lem_url)"; break
    fi
    sleep 1
    [ "$i" = 20 ] && echo "⚠ lemonade not ready after 20s ($lem_url) — continuing"
  done
else
  echo "  ℹ lemond.service not present, skipping"
fi

restart user camofox.service
restart user hermes-gateway.service
# The dashboard is a separate long-lived process; it caches the tool registry at
# startup, so it must restart too or newly added tools won't appear in the UI.
restart user hermes-dashboard.service

# Give camofox a moment, then health-check it (fatal — the one hard gate).
sleep 2
url="${CAMOFOX_URL:-http://localhost:9377}"
if curl -fsS -m 10 "$url/health" >/dev/null 2>&1; then
  echo "✓ camofox health OK ($url)"
else
  echo "✗ camofox health check failed ($url)"
  exit 1
fi
