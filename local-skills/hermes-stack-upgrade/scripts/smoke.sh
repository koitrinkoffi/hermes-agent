#!/usr/bin/env bash
# Success gate: prove browser automation still works end-to-end after the
# upgrade. Two layers, fail fast on the first failure.
#   1. hermes unit tests (fast, no browser)
#   2. live smoke: real agent-browser + Helium upload, through a THROWAWAY
#      session/profile (never touches the live persistent Hermes browser
#      session at ~/.hermes/browser_profile — see the persistent-browser
#      turn/close exemptions in project-hermes-agent-browser-migration memory;
#      this smoke test must not compete with or tear down that session).
set -uo pipefail

HERMES_REPO="${HERMES_REPO:-$HOME/.hermes/hermes-agent}"

echo "=== [1/2] hermes unit tests ==="
"$HERMES_REPO/venv/bin/python" -m pytest \
  "$HERMES_REPO/tests/tools/test_browser_tab_upload_download.py" -q || { echo "✗ unit tests failed"; exit 1; }

echo "=== [2/2] live smoke: agent-browser + Helium (throwaway profile) ==="
# Pull just the one var we need out of ~/.hermes/.env by hand rather than
# sourcing the whole file — it contains unquoted {placeholder} values (e.g.
# HERMES_LOCAL_STT_COMMAND) that are safe for python-dotenv but not for a
# literal `bash -c '. file'`.
ENV_FILE="$HOME/.hermes/.env"
if [ -f "$ENV_FILE" ]; then
  ab_path="$(grep -m1 '^AGENT_BROWSER_EXECUTABLE_PATH=' "$ENV_FILE" | cut -d= -f2-)"
  [ -n "$ab_path" ] && export AGENT_BROWSER_EXECUTABLE_PATH="$ab_path"
fi

AB="$HERMES_REPO/node_modules/agent-browser/bin/agent-browser-linux-x64"
[ -x "$AB" ] || AB="npx --prefix $HERMES_REPO agent-browser"

SITE="$(mktemp -d)"
SOCK="$(mktemp -d /tmp/hermes-smoke-sock.XXXXXX)"
PROFILE="$SITE/profile"
export AGENT_BROWSER_SOCKET_DIR="$SOCK"
session="smoke$$"
# Derive the port from our own PID so concurrent/retried runs don't collide,
# and pre-emptively clear it in case a prior failed run's http.server was
# orphaned (its cleanup trap didn't get to run, e.g. `kill -9` on this script).
PORT=$((20000 + $$ % 10000))
fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true

cat > "$SITE/index.html" <<'HTML'
<!DOCTYPE html><html><head><title>Smoke</title></head><body>
<h1>Smoke</h1><input type="file" id="file-input" multiple /><h2 id="result">no file</h2>
<script>document.getElementById('file-input').addEventListener('change',function(e){
var n=Array.prototype.map.call(e.target.files,function(f){return f.name;});
document.getElementById('result').textContent=e.target.files.length?('uploaded: '+n.join(', ')):'no file';});</script>
</body></html>
HTML
echo "smoke-$(date +%s)" > "$SITE/smoke_upload.txt"
# No subshell/cd here: `cmd &` inside `( cd x && cmd & )` can make `$!` refer to
# the wrapping subshell rather than cmd's own PID (bash-version-dependent), which
# left orphaned http.server processes on kill. --directory sidesteps the cd.
python3 -m http.server "$PORT" --directory "$SITE" >/dev/null 2>&1 &
echo $! > "$SITE/pid"
sleep 1

cleanup() {
  $AB --session "$session" --profile "$PROFILE" --json close >/dev/null 2>&1 || true
  kill "$(cat "$SITE/pid" 2>/dev/null)" 2>/dev/null || true
  rm -rf "$SITE" "$SOCK"
}
trap cleanup EXIT

open_result="$($AB --session "$session" --profile "$PROFILE" --json open "http://localhost:${PORT}/")"
echo "$open_result" | grep -q '"success":true' || { echo "✗ open failed: $open_result"; exit 1; }

upload_result="$($AB --session "$session" --profile "$PROFILE" --json upload 'input[type=file]' "$SITE/smoke_upload.txt")"
echo "$upload_result" | grep -q '"success":true' || { echo "✗ upload failed: $upload_result"; exit 1; }

sleep 1
text_result="$($AB --session "$session" --profile "$PROFILE" --json get text '#result')"
echo "$text_result" | grep -q "uploaded: smoke_upload.txt" || { echo "✗ uploaded filename not reflected in page: $text_result"; exit 1; }

echo "✓ live smoke passed"
echo
echo "✓✓ all smoke layers passed"
