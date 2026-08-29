#!/usr/bin/env python3
"""Detached lifecycle for the local Chromium (Helium) browser.

WHY THIS MODULE EXISTS
----------------------
Until 2026-08-29 the browser was a grandchild of the agent-browser daemon
(``daemon -> AppRun(/bin/sh) -> helium``), which made the browser's lifetime
Hermes's lifetime:

* ``_reap_orphaned_browser_sessions()`` tree-kills an orphaned daemon through
  ``ProcessRegistry._terminate_host_pid()``: SIGTERM to every descendant, then
  SIGKILL after a grace window.  Helium sat inside that descendant set.
* A SIGKILLed Chromium writes ``profile.exit_type = "Crashed"``, which is what
  produced the "restore pages?" bubble on every single restart.
* Any Hermes exit — clean ``atexit`` or a hard SIGKILL of the gateway — took
  the window with it, mid-task, with no way to resume.

This module inverts the ownership.  The browser is started through a double
fork so it is reparented to init immediately (PPID 1) and belongs to no Hermes
process tree; Chromium additionally moves itself into its own systemd scope, so
it is outside Hermes's cgroup too.  agent-browser then ATTACHES to it over CDP
(``--cdp``), and a CDP attach merely disconnects on ``close`` instead of killing
the browser (measured on agent-browser 0.33.0 and 0.35.1).

The browser therefore outlives Hermes by construction, and is closed only on
purpose — ``browser_close`` or the inactivity net — both of which go through CDP
``Browser.close`` so the profile records ``exit_type = "Normal"`` and the crash
bubble never appears again.

Measured shutdown semantics (2026-08-29, Helium 0.14.6.1):

    CDP Browser.close  -> process exits, exit_type "Normal"       (no bubble)
    SIGTERM            -> process exits, exit_type "SessionEnded" (no bubble)
    SIGKILL            -> process exits, exit_type "Crashed"      (bubble)

So SIGTERM is an acceptable last-resort fallback; SIGKILL never is, and this
module never sends one.

The module is deliberately free of Hermes-specific imports beyond an optional
config read, so ``~/.hermes/local-tools/hermes_browser.py`` can share it and
there is only ever one implementation of "is it up / start it / close it".
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HERMES_DIR = pathlib.Path.home() / ".hermes"
ENV_FILE = HERMES_DIR / ".env"

#: Flags agent-browser itself passes when it launches Chromium.  Replicated
#: verbatim so a browser started here is indistinguishable from one started by
#: the daemon — same feature set, same suppressed prompts, same profile
#: behaviour.  Captured from a live agent-browser 0.33.0 launch on 2026-08-29.
_BASE_FLAGS: Tuple[str, ...] = (
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-backgrounding-occluded-windows",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-hang-monitor",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-sync",
    "--disable-features=Translate",
    "--enable-features=NetworkService,NetworkServiceInProcess",
    "--metrics-recording-only",
    "--password-store=basic",
    "--use-mock-keychain",
)

#: Port 0 lets the kernel pick.  Chromium then writes the real port (and the
#: browser-target GUID) into ``<profile>/DevToolsActivePort``, which is our only
#: source of truth.  A fixed, predictable port on a profile that carries live
#: logins would be reachable by any local process; a random one is not, and the
#: discovery file is inside the profile we already own.
_DEBUG_PORT_FLAG = "--remote-debugging-port=0"

_ACTIVE_PORT_FILE = "DevToolsActivePort"

DEFAULT_LAUNCH_TIMEOUT = 45.0
DEFAULT_CLOSE_TIMEOUT = 20.0
_PROBE_TIMEOUT = 2.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _env_file_values() -> Dict[str, str]:
    """Parse ``~/.hermes/.env`` into a dict.

    ``AGENT_BROWSER_EXECUTABLE_PATH`` and ``AGENT_BROWSER_ARGS`` reach the
    browser today only because ``_build_browser_env()`` does an
    ``os.environ.copy()`` — they are not in ``_BROWSER_PASSTHROUGH_KEYS``.  A
    caller that did not inherit Hermes's environment (a cron worker, a bare
    ``python -c``) would otherwise launch a *bundled Chromium* with none of the
    anti-detection flags, which is exactly the class of drift
    ``local-tools/hermes_browser.py`` was written to stop.  So read the file
    directly as a fallback rather than trusting the ambient environment.
    """
    values: Dict[str, str] = {}
    try:
        raw = ENV_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _setting(name: str, default: str = "") -> str:
    """Environment first, then ``~/.hermes/.env``, then *default*."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return _env_file_values().get(name, default).strip()


def browser_executable() -> str:
    """Absolute path to the Chromium build to drive (Helium here).

    Empty string means "not configured" — the caller must not guess a browser,
    because silently falling back to a bundled Chromium loses the persistent
    profile and every login in it.
    """
    return _setting("AGENT_BROWSER_EXECUTABLE_PATH")


def extra_args() -> List[str]:
    """User-supplied Chromium flags from ``AGENT_BROWSER_ARGS``.

    agent-browser splits this on commas (not spaces), so a flag may itself
    contain spaces.  Keep the same convention or the anti-detection set drifts.
    """
    raw = _setting("AGENT_BROWSER_ARGS")
    return [part.strip() for part in raw.split(",") if part.strip()]


def local_browser_config() -> Dict[str, Any]:
    """Read ``browser.local`` (profile_dir / headed) with safe defaults.

    Mirrors ``browser_tool._get_local_browser_settings()`` but tolerates the
    absence of the Hermes config module, so this file stays importable from a
    plain interpreter.
    """
    config: Dict[str, Any] = {"profile_dir": None, "headed": False}
    try:
        from hermes_cli.config import read_raw_config

        block = (read_raw_config().get("browser", {}) or {}).get("local", {}) or {}
        profile_dir = str(block.get("profile_dir") or "").strip()
        if profile_dir:
            config["profile_dir"] = os.path.expanduser(profile_dir)
        config["headed"] = bool(block.get("headed", False))
    except Exception as exc:  # config unreadable — fall back to the known default
        logger.debug("browser.local config unreadable (%s); using defaults", exc)
    if not config["profile_dir"]:
        config["profile_dir"] = str(HERMES_DIR / "browser_profile")
    return config


def profile_dir() -> pathlib.Path:
    return pathlib.Path(local_browser_config()["profile_dir"])


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _http_json(url: str, timeout: float = _PROBE_TIMEOUT) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def _active_port(profile: pathlib.Path) -> Optional[int]:
    """First line of ``DevToolsActivePort``, or None.

    The file is NOT removed when Chromium is killed, so its presence proves
    nothing — every caller must follow it with :func:`probe`.
    """
    try:
        first = (profile / _ACTIVE_PORT_FILE).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[0]
        return int(first.strip())
    except (OSError, IndexError, ValueError):
        return None


def probe(profile: Optional[pathlib.Path] = None) -> Optional[str]:
    """Return the HTTP CDP base URL of a live browser on *profile*, else None.

    "Live" means the endpoint actually answered ``/json/version`` — a stale
    ``DevToolsActivePort`` left behind by a killed browser resolves to None,
    which is what makes relaunch-on-stale automatic instead of a special case.
    """
    profile = profile or profile_dir()
    port = _active_port(profile)
    if port is None:
        return None
    base = f"http://127.0.0.1:{port}"
    if _http_json(f"{base}/json/version") is None:
        return None
    return base


def is_running(profile: Optional[pathlib.Path] = None) -> bool:
    return probe(profile) is not None


def websocket_url(base: str) -> Optional[str]:
    payload = _http_json(f"{base}/json/version")
    if isinstance(payload, dict):
        url = payload.get("webSocketDebuggerUrl")
        if isinstance(url, str) and url:
            return url
    return None


def tab_fingerprint(base: Optional[str] = None) -> Optional[Tuple[str, ...]]:
    """Sorted tuple of open page URLs, or None when no browser is reachable.

    Used by the inactivity net to tell "nobody touched this browser" from
    "Koitrin has been using the window": if the fingerprint moved, the window
    is in human hands and the net must stand down rather than close a page
    someone is reading.
    """
    base = base or probe()
    if base is None:
        return None
    payload = _http_json(f"{base}/json/list")
    if not isinstance(payload, list):
        return ()
    return tuple(
        sorted(
            str(target.get("url", ""))
            for target in payload
            if isinstance(target, dict) and target.get("type") == "page"
        )
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

#: Variables a headed Chromium needs in order to reach the display server.
_GRAPHICAL_KEYS: Tuple[str, ...] = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
)


def _systemd_user_environment() -> Dict[str, str]:
    """Graphical session variables as ``systemd --user`` knows them."""
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    values: Dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def launch_env() -> Dict[str, str]:
    """``os.environ`` plus whatever graphical variables it is missing.

    A headed Chromium with no ``DISPLAY``/``WAYLAND_DISPLAY`` (and no
    ``XAUTHORITY`` under X11) dies before it writes ``DevToolsActivePort``, and
    the caller sees nothing but a launch timeout - which is a miserable way to
    learn that the problem is the environment.

    The gateway inherits these from its ``graphical-session.conf`` drop-in, so
    it never noticed.  A plain shell does not: cron jobs, SSH sessions and the
    assistant's own tool calls all start bare, and that is exactly where
    ``local-tools/hermes_browser.py`` runs.  Backfill from systemd's user
    environment, the authoritative source for this seat.
    """
    env = os.environ.copy()
    missing = [key for key in _GRAPHICAL_KEYS if not env.get(key)]
    if not missing:
        return env
    session = _systemd_user_environment()
    for key in missing:
        value = session.get(key)
        if value:
            env[key] = value
    return env


def _launch_argv(profile: pathlib.Path, headed: bool) -> List[str]:
    executable = browser_executable()
    if not executable:
        raise RuntimeError(
            "AGENT_BROWSER_EXECUTABLE_PATH is not set (checked the environment "
            f"and {ENV_FILE}). Refusing to launch a fallback browser: it would "
            "not carry the persistent profile or its logins."
        )
    argv = [executable, _DEBUG_PORT_FLAG, *_BASE_FLAGS, f"--user-data-dir={profile}"]
    if not headed:
        argv.append("--headless=new")
    argv.extend(extra_args())
    return argv


def launch_detached(
    profile: Optional[pathlib.Path] = None,
    headed: Optional[bool] = None,
) -> None:
    """Start the browser so that it is a child of init, not of this process.

    ``setsid --fork`` always forks: the process we spawn exits immediately and
    the browser it left behind is reparented to PID 1 within milliseconds.  A
    plain ``start_new_session=True`` would not do — the browser would stay a
    child of the gateway until the gateway died, and any tree-kill aimed at the
    gateway would still take it down.  We avoid ``os.fork()`` in-process on
    purpose: the gateway is multithreaded, where forking is a footgun.
    """
    config = local_browser_config()
    profile = profile or pathlib.Path(config["profile_dir"])
    headed = config["headed"] if headed is None else headed

    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Drop a stale discovery file so the launch wait cannot latch onto the
    # port of the previous, already-dead browser.
    try:
        (profile / _ACTIVE_PORT_FILE).unlink()
    except OSError:
        pass

    argv = ["setsid", "--fork", *_launch_argv(profile, headed)]
    logger.info("Launching detached browser: %s", shlex.join(argv))
    with open(os.devnull, "rb") as devnull_in, open(os.devnull, "wb") as devnull_out:
        proc = subprocess.Popen(
            argv,
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=devnull_out,
            start_new_session=True,
            close_fds=True,
            env=launch_env(),
        )
    # `setsid` itself exits at once; reap it so it does not linger as a zombie.
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.debug("setsid wrapper still alive after 10s; not waiting further")


def ensure_running(
    timeout: float = DEFAULT_LAUNCH_TIMEOUT,
    profile: Optional[pathlib.Path] = None,
    headed: Optional[bool] = None,
) -> Tuple[str, bool]:
    """Guarantee a reachable browser; return ``(cdp_http_base, attached)``.

    ``attached`` is True when we joined a browser that was already open — the
    signal the resume hint keys off, so the agent is told to triage the tabs it
    inherited instead of assuming a clean window.
    """
    profile = profile or profile_dir()
    base = probe(profile)
    if base is not None:
        return base, True

    launch_detached(profile, headed)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        base = probe(profile)
        if base is not None:
            return base, False
        time.sleep(0.4)

    env = launch_env()
    display = env.get("DISPLAY") or env.get("WAYLAND_DISPLAY") or "<none>"
    raise RuntimeError(
        f"browser did not expose a CDP endpoint within {timeout:.0f}s "
        f"(profile {profile}, headed={headed if headed is not None else local_browser_config()['headed']}, "
        f"display={display}). Check that {browser_executable() or '<unset>'} is "
        "executable, that a display is reachable when running headed, and that "
        "no other process holds the profile lock."
    )


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def _browser_pids(profile: pathlib.Path) -> List[int]:
    """Main browser processes bound to *profile* (renderers excluded)."""
    pids: List[int] = []
    try:
        import psutil
    except ImportError:
        return pids
    needle = f"--user-data-dir={profile}"
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline or needle not in cmdline:
                continue
            if any(arg.startswith("--type=") for arg in cmdline):
                continue  # renderer / GPU / utility child
            pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def _cdp_browser_close(base: str) -> bool:
    """Send CDP ``Browser.close``. Returns True when the command was accepted.

    Uses the *synchronous* websockets client on purpose: this runs from the
    inactivity thread and from tool handlers, neither of which may assume an
    asyncio loop is absent (``asyncio.run`` inside a thread that already has a
    running loop raises).
    """
    url = websocket_url(base)
    if not url:
        return False
    try:
        from websockets.sync.client import connect
    except ImportError:
        logger.debug("websockets sync client unavailable; falling back to signal")
        return False
    try:
        with connect(url, open_timeout=5, close_timeout=5) as socket:
            socket.send(json.dumps({"id": 1, "method": "Browser.close"}))
            try:
                socket.recv(timeout=5)
            except Exception:
                # The browser often tears the socket down before replying.
                pass
        return True
    except Exception as exc:
        logger.debug("CDP Browser.close failed: %s", exc)
        return False


def close_browser(
    timeout: float = DEFAULT_CLOSE_TIMEOUT,
    profile: Optional[pathlib.Path] = None,
) -> bool:
    """Close the browser gracefully. Returns True if nothing is left running.

    Order matters and is the whole point of this function:

    1. CDP ``Browser.close`` — writes ``exit_type: "Normal"``, no crash bubble.
    2. SIGTERM as a fallback — ``exit_type: "SessionEnded"``, still no bubble.
    3. Never SIGKILL. That is precisely what wrote ``"Crashed"`` and produced
       the restore prompt this whole design exists to eliminate.
    """
    profile = profile or profile_dir()
    base = probe(profile)
    if base is None:
        return not _browser_pids(profile)

    _cdp_browser_close(base)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe(profile) is None and not _browser_pids(profile):
            return True
        time.sleep(0.3)

    # Still there — escalate to SIGTERM only.
    import signal

    for pid in _browser_pids(profile):
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("Browser.close timed out; sent SIGTERM to pid %d", pid)
        except (OSError, ProcessLookupError, PermissionError):
            pass

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not _browser_pids(profile):
            return True
        time.sleep(0.3)

    logger.warning(
        "Browser on profile %s survived Browser.close and SIGTERM; leaving it "
        "alone (SIGKILL would write exit_type=Crashed).", profile,
    )
    return False
