#!/usr/bin/env bash
# Success gate: prove the browser_upload feature still works end-to-end after the
# upgrade. Three layers, fail fast on the first failure.
#   1. hermes unit tests (fast, no browser)
#   2. camofox e2e (real browser)
#   3. live smoke: real browser_upload through the RUNNING systemd camofox
#      (the only layer that catches hermes-client <-> camofox-server version skew)
set -uo pipefail

HERMES_REPO="${HERMES_REPO:-$HOME/.hermes/hermes-agent}"
CAMOFOX_REPO="${CAMOFOX_REPO:-$HOME/camofox}"

echo "=== [1/3] hermes unit tests ==="
"$HERMES_REPO/venv/bin/python" -m pytest \
  "$HERMES_REPO/tests/tools/test_browser_tab_upload_download.py" -q || { echo "✗ unit tests failed"; exit 1; }

echo "=== [2/3] camofox e2e (upload) ==="
( cd "$CAMOFOX_REPO" && NODE_OPTIONS='--experimental-vm-modules' \
    npx jest --config jest.config.e2e.cjs --runInBand --forceExit upload ) || { echo "✗ camofox e2e failed"; exit 1; }

echo "=== [3/3] live smoke through running camofox ==="
# Source CAMOFOX_URL / CAMOFOX_API_KEY from the running service's env file.
[ -f "$CAMOFOX_REPO/.env" ] && set -a && . "$CAMOFOX_REPO/.env" && set +a
export CAMOFOX_URL="${CAMOFOX_URL:-http://localhost:9377}"

SITE="$(mktemp -d)"
cat > "$SITE/index.html" <<'HTML'
<!DOCTYPE html><html><head><title>Smoke</title></head><body>
<h1>Smoke</h1><input type="file" id="file-input" multiple /><h2 id="result">no file</h2>
<script>document.getElementById('file-input').addEventListener('change',function(e){
var n=Array.prototype.map.call(e.target.files,function(f){return f.name;});
document.getElementById('result').textContent=e.target.files.length?('uploaded: '+n.join(', ')):'no file';});</script>
</body></html>
HTML
echo "smoke-$(date +%s)" > "$SITE/smoke_upload.txt"
( cd "$SITE" && python3 -m http.server 8799 >/dev/null 2>&1 & echo $! > "$SITE/pid" )
sleep 1

"$HERMES_REPO/venv/bin/python" - "$SITE" <<'PY'
import sys, json, time
site = sys.argv[1]
sys.path.insert(0, __import__("os").path.expanduser("~/.hermes/hermes-agent"))
import tools.browser_tool as bt
assert bt._is_camofox_mode(), "camofox mode not active"
tid = "stack-upgrade-smoke"
bt.browser_navigate("http://localhost:8799/", task_id=tid); time.sleep(1)
up = json.loads(bt.browser_upload(ref="input[type=file]", path=f"{site}/smoke_upload.txt", task_id=tid))
assert up.get("success"), f"upload failed: {up}"
time.sleep(1)
snap = bt.browser_snapshot(task_id=tid)
assert "uploaded: smoke_upload.txt" in snap, "uploaded filename not reflected in page"
try:
    from tools.browser_camofox import camofox_close; camofox_close(tid)
except Exception:
    pass
print("✓ live smoke passed")
PY
rc=$?
kill "$(cat "$SITE/pid" 2>/dev/null)" 2>/dev/null
rm -rf "$SITE"
[ "$rc" -eq 0 ] || { echo "✗ live smoke failed"; exit 1; }

echo
echo "✓✓ all smoke layers passed"
