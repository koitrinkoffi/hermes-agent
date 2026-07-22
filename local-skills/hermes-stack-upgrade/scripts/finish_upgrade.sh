#!/usr/bin/env bash
# finish_upgrade.sh — the deterministic TAIL of the stack upgrade (restart →
# smoke → deploy/rollback → notify), designed to run DETACHED so it survives the
# gateway restart it performs.
#
# Why this exists: the agent that drives the upgrade runs *inside*
# hermes-gateway (kanban dispatched in-gateway). Restarting that unit kills the
# agent — and because the unit is `KillMode=mixed`, the whole cgroup is SIGKILLed,
# so even a `nohup`/`setsid` child launched from the agent dies too. The fix is to
# launch this script as its OWN transient user unit:
#
#     systemd-run --user --collect --unit=hermes-upgrade-finish-<ts> \
#       bash ${HERMES_SKILL_DIR}/scripts/finish_upgrade.sh
#
# That transient unit lives in its own cgroup under user.slice, independent of
# the gateway, so it runs to completion on its own and pings the user with the
# verdict. None of the tail steps need the LLM — they are pure shell.
set -uo pipefail

HERMES_REPO="${HERMES_REPO:-$HOME/.hermes/hermes-agent}"
HERMES_PY="$HERMES_REPO/venv/bin/python"
STATE_DIR="$HOME/.hermes/state"
LOG="$STATE_DIR/stack-upgrade-finish.log"
STATUS="$STATE_DIR/stack-upgrade-last-result.txt"
mkdir -p "$STATE_DIR"

# --- Self-relocate to a temp copy of the scripts dir ------------------------- #
# The green path runs deploy_self.sh, which `rm -rf`s the LIVE skill dir — i.e.
# this very script and its siblings — while we are still running. Re-exec from a
# /tmp copy so the deploy can't pull the rug out from under us.
if [ "${_UPGRADE_FINISH_RELOCATED:-}" != "1" ]; then
  _src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _work="$(mktemp -d /tmp/hermes-upgrade-finish.XXXXXX)"
  cp -r "$_src_dir"/. "$_work"/
  export _UPGRADE_FINISH_RELOCATED=1 _UPGRADE_FINISH_WORK="$_work"
  exec bash "$_work/finish_upgrade.sh" "$@"
fi
SCRIPTS="$_UPGRADE_FINISH_WORK"
trap 'rm -rf "$SCRIPTS" 2>/dev/null || true' EXIT

exec >>"$LOG" 2>&1
echo "===== $(date -Is) finish_upgrade START (pid $$, scripts=$SCRIPTS) ====="

# Best-effort notify: always write the status file; Telegram ping on top.
# `hermes send` reuses the gateway's bot-token creds and needs no running
# gateway, so it works even if the restart below failed.
notify() {
  local msg="$1"
  printf '%s\n' "$msg" > "$STATUS"
  "$HERMES_PY" -m hermes_cli.main send --quiet --to telegram \
    --subject "[stack-upgrade]" "$msg" \
    || echo "(notify: hermes send failed — verdict is in $STATUS)"
}

# 1) Restart services so the merged code is live. This is the step that kills
#    the launching agent; we survive it (own cgroup).
if ! bash "$SCRIPTS/restart_services.sh"; then
  notify "🔴 gateway restart FAILED after merge — NOT deployed. See $LOG"
  exit 1
fi

# 2) Smoke gate → 3a green (back up + deploy self) / 3b red (rollback).
if bash "$SCRIPTS/smoke.sh"; then
  if bash "$SCRIPTS/push_backups.sh" && bash "$SCRIPTS/deploy_self.sh"; then
    notify "🟢 upgrade GREEN → backed up + deployed."
  else
    notify "🟡 upgrade smoke GREEN but backup/deploy failed — code is live, fork/live copy may lag. See $LOG"
  fi
else
  if bash "$SCRIPTS/rollback.sh"; then
    notify "🔴 smoke FAILED → rolled back to pre-merge state (fork untouched)."
  else
    notify "⛔ smoke FAILED and ROLLBACK ALSO FAILED — manual intervention needed. See $LOG"
  fi
fi

echo "===== $(date -Is) finish_upgrade DONE ====="
