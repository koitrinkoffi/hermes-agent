"""Single-GPU arbitration between interactive turns and cron runs.

LOCAL MOD (``hermes-mods`` branch) — no upstream equivalent. Signature string
for the upgrade skill's mod inventory: ``__hermes_gpu_arbiter__``.

WHY THIS EXISTS
---------------
This install serves every model from one local lemonade instance on a single
integrated GPU (Strix Halo, unified memory). lemonade will happily hold up to
``max_models`` per class resident at once and has no cross-model request
queue: two clients hitting two different models simply contend. The observed
symptom is an interactive Hermes turn slowing to a crawl because a cron job
woke up and started generating on the same — or another — model.

Upstream has no notion of "the GPU is a single shared resource", so this
module adds one. It is deliberately *advisory*: it changes WHO WAITS, never
what runs. Nothing here can lose work, and every path fails open — if any
probe raises, the caller proceeds exactly as upstream would.

THE CONTRACT
------------
1. A cron job that is about to make an LLM call must pass ``cron_gate()``:
   no interactive turn in flight, no human queued behind the arbiter, the
   session quiet for ``quiet_minutes``, lemonade not currently generating,
   and the global lease free.
2. An interactive turn takes the global lease BEFORE its own session turn
   lease (see ORDERING below) and shows the user what it is waiting for.
3. ``no_agent`` cron jobs bypass everything: they burn no GPU, and they are
   what re-drives a processor that was refused, so gating them would break
   the recovery loop rather than free anything.

ORDERING (deadlock avoidance)
-----------------------------
The global lease is ALWAYS acquired before the per-conversation session turn
lease, never after. Cron runs take only the global lease. As long as no path
ever takes session-then-global, no cycle can form. If you add a caller, keep
this order.

WHY THE LEASE TABLE IS REUSED
-----------------------------
``SessionDB.try_acquire_session_turn_lease`` already gives us exactly the
primitive we need — atomic cross-process acquisition, TTL expiry, reclamation
of leases whose holder PID is dead, and a polling wrapper that never holds a
SQLite write lock while waiting. Its key is opaque (a synthetic id that is not
a real session resolves to itself in ``_session_turn_lease_key_on_conn``), so
reserved keys give us a global mutex and a "human is queued" flag for free,
with no schema change to a table upstream owns.

DEFERRAL BOOKKEEPING lives in its own JSON file rather than on the job record,
because ``jobs.json`` is upstream-owned and rewritten wholesale on every edit.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Reserved lease keys. Must not collide with a real session id (session ids are
# uuid-shaped); the leading/trailing dunder makes that structurally impossible.
GLOBAL_LEASE_KEY = "__hermes_gpu_arbiter__"
WAITER_KEY = "__hermes_gpu_waiter__"

# Defaults. Every one is overridable under ``cron:`` in config.yaml.
_DEFAULT_QUIET_MINUTES = 10.0
_DEFAULT_INTERACTIVE_WAIT_SECONDS = 120.0
_DEFAULT_DEFER_ALERT_MINUTES = 60.0
_DEFAULT_HEALTH_URL = "http://localhost:13305/api/v1/health"
_DEFAULT_HEALTH_TIMEOUT = 2.0

# A cron run can legitimately last minutes (HERMES_CRON_TIMEOUT defaults to
# 600s), so the global lease TTL must outlive a normal run; it is refreshed
# while the run is alive and released in a finally block. The TTL only matters
# when a holder dies hard, and dead-PID reclamation usually beats it anyway.
_LEASE_TTL_SECONDS = 900.0
_LEASE_REFRESH_SECONDS = 60.0

# The waiter flag is a hint, not a lock: a short TTL means a client that dies
# mid-wait cannot starve the cron fleet for longer than one gate evaluation.
_WAITER_TTL_SECONDS = 30.0

VALID_GPU_POLICIES = ("defer", "always")

_state_file_lock = threading.Lock()
# Holder -> refresher, for the acquire/release pair used by run_agent.
_ACTIVE_INTERACTIVE_LEASES: Dict[str, "_HeldLease"] = {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


_CONFIG_CACHE_TTL_SECONDS = 5.0
_config_cache: Tuple[float, dict] = (0.0, {})


def _config(config: Optional[dict] = None) -> dict:
    """Resolved config, memoised for a few seconds.

    The gate and the turn prologue both ask for settings on every fire and
    every turn. ``load_config`` reads and parses YAML from disk each time, so
    an uncached read turns a courtesy check into measurable per-turn work.
    A 5s window is far shorter than any human edit-then-observe loop.
    """
    if isinstance(config, dict):
        return config
    global _config_cache
    now = time.monotonic()
    cached_at, cached = _config_cache
    if cached and (now - cached_at) < _CONFIG_CACHE_TTL_SECONDS:
        return cached
    try:
        from hermes_cli.config import load_config

        resolved = load_config() or {}
    except Exception:
        resolved = {}
    _config_cache = (now, resolved)
    return resolved


def _cron_config(config: Optional[dict] = None) -> dict:
    cfg = _config(config)
    section = cfg.get("cron")
    return section if isinstance(section, dict) else {}


def is_enabled(config: Optional[dict] = None) -> bool:
    """Master switch. Only a literal ``false`` disables the arbiter.

    Kill switch for a mod that sits on a critical path: setting
    ``cron.gpu_arbiter: false`` restores stock upstream behaviour without
    touching code, which is how you tell "our mod" from "not our mod" when
    something misbehaves after an upgrade.

    INERT UNDER PYTEST. Upstream's suite drives thousands of simulated turns
    and cron fires; arbitrating them serialises work that has no GPU behind
    it and makes upstream results depend on our mod's timing -- which would
    poison the one signal that tells us whether an upstream merge is sound.
    ``PYTEST_CURRENT_TEST`` is set by pytest itself for the duration of every
    test, so this cannot mask the mod in production. Tests that want to
    exercise the arbiter call its functions directly (they take an explicit
    ``config``) rather than relying on the ambient switch.
    """
    setting = _cron_config(config).get("gpu_arbiter", None)
    if setting is True:
        # An explicit opt-in wins everywhere, tests included -- that is how
        # this module's OWN tests exercise the gate.
        return True
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return setting is not False


def _float_setting(key: str, default: float, config: Optional[dict] = None) -> float:
    raw = _cron_config(config).get(key)
    try:
        if raw is None:
            return default
        value = float(raw)
        return value if value >= 0 else default
    except (TypeError, ValueError):
        logger.warning("Invalid cron.%s=%r; using %s", key, raw, default)
        return default


def quiet_seconds(config: Optional[dict] = None) -> float:
    return _float_setting("gpu_arbiter_quiet_minutes", _DEFAULT_QUIET_MINUTES, config) * 60.0


def interactive_wait_seconds(config: Optional[dict] = None) -> float:
    return _float_setting(
        "gpu_arbiter_interactive_wait_seconds", _DEFAULT_INTERACTIVE_WAIT_SECONDS, config
    )


def defer_alert_seconds(config: Optional[dict] = None) -> float:
    return _float_setting(
        "gpu_arbiter_defer_alert_minutes", _DEFAULT_DEFER_ALERT_MINUTES, config
    ) * 60.0


def _health_url(config: Optional[dict] = None) -> str:
    raw = _cron_config(config).get("gpu_arbiter_health_url")
    return str(raw).strip() if isinstance(raw, str) and raw.strip() else _DEFAULT_HEALTH_URL


# ---------------------------------------------------------------------------
# Per-job policy
# ---------------------------------------------------------------------------


def normalize_gpu_policy(value: Any) -> Optional[str]:
    """Validate a per-job GPU policy at the storage choke point.

    Mirrors ``cron.jobs._normalize_reasoning_effort``: ``None``/empty clears
    the pin (job follows ``cron.gpu_policy``), anything invalid raises so a
    fire-and-forget job can never persist a policy the gate cannot read.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in VALID_GPU_POLICIES:
        raise ValueError(
            f"Invalid gpu_policy {value!r}. Valid values: "
            f"{', '.join(VALID_GPU_POLICIES)} (empty string clears the pin)."
        )
    return text


def resolve_job_gpu_policy(job: dict, config: Optional[dict] = None) -> str:
    """Per-job pin → ``cron.gpu_policy`` → ``defer``."""
    pinned = job.get("gpu_policy") if isinstance(job, dict) else None
    if isinstance(pinned, str) and pinned.strip().lower() in VALID_GPU_POLICIES:
        return pinned.strip().lower()
    fleet = _cron_config(config).get("gpu_policy")
    if isinstance(fleet, str) and fleet.strip().lower() in VALID_GPU_POLICIES:
        return fleet.strip().lower()
    return "defer"


# ---------------------------------------------------------------------------
# State-database probes (read-only, fail-open)
# ---------------------------------------------------------------------------


def _state_db_path() -> Optional[Path]:
    try:
        from hermes_state import _default_db_path

        return Path(_default_db_path())
    except Exception:
        logger.debug("gpu_arbiter: cannot resolve state db path", exc_info=True)
        return None


@contextlib.contextmanager
def _readonly_conn():
    """Read-only connection to state.db.

    Reads go through a dedicated read-only handle rather than SessionDB's
    private ``_read_ctx`` so an upstream refactor of those internals cannot
    break the gate. WAL means these reads never block a writer.
    """
    path = _state_db_path()
    if path is None or not path.exists():
        yield None
        return
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        yield conn
    except Exception:
        logger.debug("gpu_arbiter: read-only state db open failed", exc_info=True)
        yield None
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def _holder_is_dead(holder: str) -> bool:
    """True when the holder's PID is provably gone.

    ``try_acquire_session_turn_lease`` already reclaims such a lease, so
    without this the GATE would be stricter than the ACQUIRE: a gateway
    SIGKILLed mid-run would leave a lease no live process owns, and every
    cron job would defer for the rest of the 900s TTL while an interactive
    turn walked straight past it. Same predicate as hermes_state uses, so the
    two can never disagree about who is dead.
    """
    try:
        from hermes_state import _compression_lock_holder_process_is_dead

        return bool(_compression_lock_holder_process_is_dead(holder))
    except Exception:
        return False  # any doubt: treat the lease as live


def _lease_holder(conn, key: str) -> Optional[str]:
    """Return the live holder of *key*, or None when free/expired/dead."""
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT holder, expires_at FROM session_turn_leases WHERE conversation_id = ?",
            (key,),
        ).fetchone()
    except Exception:
        return None
    if row is None:
        return None
    try:
        if float(row["expires_at"]) <= time.time():
            return None
    except (TypeError, ValueError):
        return None
    holder = str(row["holder"])
    return None if _holder_is_dead(holder) else holder


def _lease_held_by_this_process(conn) -> bool:
    """True when THIS process already owns the global lease.

    Reentrancy guard. A cron run holds the lease for its whole execution, and
    the agent it runs goes through the same interactive turn prologue as a
    user turn -- without this check that inner turn would queue behind its own
    parent, wait out the cap, and then run "overridden" for nothing. Holder
    tokens end in the owning PID precisely so this is decidable.
    """
    holder = _lease_holder(conn, GLOBAL_LEASE_KEY)
    return bool(holder and holder.endswith(f":pid={os.getpid()}"))


def _pretty_holder(holder: Optional[str]) -> str:
    """Turn ``cron:mailflow-process:pid=1234`` into ``mailflow-process``.

    Holder tokens are machine-shaped by necessity (the ``pid=`` suffix is what
    makes dead-holder reclamation work). They also end up in a message a human
    reads while waiting, so strip the plumbing before showing it.
    """
    if not holder:
        return "another run"
    parts = [p for p in str(holder).split(":") if p and not p.startswith("pid=")]
    if len(parts) >= 2:
        kind, label = parts[0], ":".join(parts[1:])
        return f"{label}" if kind == "cron" else f"another session ({label})"
    return parts[0] if parts else "another run"


def _any_turn_in_flight(conn) -> bool:
    """True when any real conversation holds an unexpired turn lease.

    Reserved arbiter keys are excluded: the global lease is held by whoever
    is running (possibly a cron job), and the waiter flag is checked
    separately, so counting either here would make the gate self-blocking.
    """
    if conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM session_turn_leases "
            "WHERE expires_at > ? AND conversation_id NOT IN (?, ?)",
            (time.time(), GLOBAL_LEASE_KEY, WAITER_KEY),
        ).fetchone()
    except Exception:
        return False
    return bool(row and int(row["n"]) > 0)


def _seconds_since_last_session_activity(conn) -> Optional[float]:
    """Age of the newest conversation message, in seconds.

    This is the quiet-period signal. The turn lease only covers a turn in
    flight; between two turns of a live back-and-forth it is released, and a
    cron that started in that gap would put the user behind it on their very
    next message.
    """
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT MAX(timestamp) AS t FROM messages").fetchone()
    except Exception:
        return None
    if row is None or row["t"] is None:
        return None
    try:
        return max(0.0, time.time() - float(row["t"]))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# lemonade probe
# ---------------------------------------------------------------------------


def _backend_busy(config: Optional[dict] = None) -> bool:
    """True when lemonade reports any loaded model mid-request.

    Secondary guard only. It catches every client the lease cannot see —
    Hindsight, a detached subagent, another tool — but it must never be the
    primary signal: an interactive turn spends long stretches idle between
    tool calls, and a cron admitted during one of those gaps is exactly the
    collision this module exists to prevent.

    Unreachable/garbled backend means "not busy": failing closed here would
    silently stop the whole cron fleet whenever lemonade restarts.
    """
    url = _health_url(config)
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=_DEFAULT_HEALTH_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        logger.debug("gpu_arbiter: backend health probe failed (%s)", url, exc_info=True)
        return False
    models = payload.get("all_models_loaded")
    if not isinstance(models, list):
        return False
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("is_busy") or entry.get("is_streaming"):
            return True
    return False


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class GateDecision:
    """Result of evaluating the cron gate. Falsy reason == admitted."""

    __slots__ = ("allowed", "reason")

    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.allowed

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"GateDecision(allowed={self.allowed!r}, reason={self.reason!r})"


def cron_gate(job: dict, config: Optional[dict] = None) -> GateDecision:
    """Decide whether *job* may start its LLM call now.

    Every failure mode admits the job: this is a courtesy, and a broken probe
    must not be able to stop the automation fleet.
    """
    try:
        if not is_enabled(config):
            return GateDecision(True, "")
        if not isinstance(job, dict):
            return GateDecision(True, "")
        if job.get("no_agent"):
            # No LLM call, no GPU. Gating these would also break the recovery
            # loop: the watchers ARE the mechanism that re-drives a refused
            # processor (see mailflow_watch.py / linkedin_watch.py).
            return GateDecision(True, "")
        if resolve_job_gpu_policy(job, config) == "always":
            return GateDecision(True, "")

        with _readonly_conn() as conn:
            if _any_turn_in_flight(conn):
                return GateDecision(False, "an interactive turn is in flight")
            waiter = _lease_holder(conn, WAITER_KEY)
            if waiter:
                return GateDecision(False, "a user turn is queued for the GPU")
            idle = _seconds_since_last_session_activity(conn)
            quiet = quiet_seconds(config)
            if idle is not None and idle < quiet:
                return GateDecision(
                    False,
                    f"session active {int(idle)}s ago (quiet period {int(quiet)}s)",
                )
            holder = _lease_holder(conn, GLOBAL_LEASE_KEY)
            if holder:
                return GateDecision(
                    False, f"the GPU is in use by {_pretty_holder(holder)}"
                )

        if _backend_busy(config):
            return GateDecision(False, "inference backend is busy")
        return GateDecision(True, "")
    except Exception:
        logger.warning("gpu_arbiter: gate evaluation failed; admitting", exc_info=True)
        return GateDecision(True, "")


# ---------------------------------------------------------------------------
# Lease acquisition
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _borrowed_or_shared_db(db: Any = None):
    """Yield *db* when the caller lent us one, else borrow from the registry.

    A lent handle is NEVER released here -- its owner's lifecycle rules.
    """
    if db is not None and callable(
        getattr(type(db), "try_acquire_session_turn_lease", None)
    ):
        yield db
        return
    with _shared_db() as borrowed:
        yield borrowed


@contextlib.contextmanager
def _shared_db():
    db = None
    try:
        from hermes_state_registry import acquire, release_or_close

        db = acquire()
        yield db
    except Exception:
        logger.debug("gpu_arbiter: shared session db unavailable", exc_info=True)
        yield None
    finally:
        if db is not None:
            with contextlib.suppress(Exception):
                from hermes_state_registry import release_or_close

                release_or_close(db)


def _holder_token(prefix: str, label: str) -> str:
    """Build a holder string of the form ``<prefix>:<label>:pid=<n>``.

    The ``pid=`` spelling is NOT cosmetic. ``hermes_state`` reclaims a lease
    whose holder PID is provably gone, but only for holders matching
    ``_COMPRESSION_LOCK_HOLDER_PID_RE`` (``(?:^|:)pid=(\\d+)(?::|$)``). A
    holder that merely ends in a bare number is unmatched, so a gateway
    SIGKILLed mid-run would strand the GPU lease for the full TTL and stall
    every cron job until it expired. Keep this shape.

    Labels are sanitised because a colon inside one would break the parse.

    Holder strings are also user-visible: the interactive wait notice prints
    the holder so the user learns WHICH run is in front of them.
    """
    safe = str(label or "?").replace(":", "-")
    return f"{prefix}:{safe}:pid={os.getpid()}"


class _HeldLease:
    """A held global lease with a background refresher.

    A cron run outlives the TTL comfortably, so the lease must be renewed
    while the work is alive; without that, a long mailflow triage would drop
    its own lease mid-run and let another job in.
    """

    def __init__(self, holder: str) -> None:
        self.holder = holder
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start_refresh(self) -> None:
        def _loop() -> None:
            while not self._stop.wait(_LEASE_REFRESH_SECONDS):
                with _shared_db() as db:
                    if db is None:
                        continue
                    try:
                        db.refresh_session_turn_lease(
                            GLOBAL_LEASE_KEY, self.holder, ttl_seconds=_LEASE_TTL_SECONDS
                        )
                    except Exception:
                        logger.debug("gpu_arbiter: lease refresh failed", exc_info=True)

        self._thread = threading.Thread(
            target=_loop, name="gpu-arbiter-refresh", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            with contextlib.suppress(Exception):
                thread.join(timeout=2.0)


@contextlib.contextmanager
def hold_global_lease(label: str, *, prefix: str = "cron"):
    """Try to take the global lease for the duration of the block.

    Yields True when the lease was taken (and will be released on exit),
    False when it was already held — the caller decides what to do about it,
    which for a cron job means standing down rather than queueing.
    """
    holder = _holder_token(prefix, label)
    acquired = False
    lease: Optional[_HeldLease] = None
    try:
        with _shared_db() as db:
            if db is not None:
                try:
                    acquired = bool(
                        db.try_acquire_session_turn_lease(
                            GLOBAL_LEASE_KEY, holder, ttl_seconds=_LEASE_TTL_SECONDS
                        )
                    )
                except Exception:
                    logger.debug("gpu_arbiter: lease acquire failed", exc_info=True)
                    acquired = False
        if acquired:
            lease = _HeldLease(holder)
            lease.start_refresh()
        yield acquired
    finally:
        if lease is not None:
            lease.stop()
        if acquired:
            with _shared_db() as db:
                if db is not None:
                    with contextlib.suppress(Exception):
                        db.release_session_turn_lease(GLOBAL_LEASE_KEY, holder)


def _set_waiter(db, holder: str) -> bool:
    """Assert (or renew) the "a human is queued" flag.

    Renewal is not optional: the flag's TTL is deliberately short so a client
    that dies mid-wait cannot starve the cron fleet, but a wait can run to the
    full interactive cap. ``try_acquire_session_turn_lease`` is
    INSERT-OR-IGNORE, so re-calling it while we already own the row does NOT
    push ``expires_at`` out -- without the explicit refresh the flag would
    lapse mid-wait and a cron job could slip in behind the user's back for
    the remainder of the wait.
    """
    try:
        if db.refresh_session_turn_lease(
            WAITER_KEY, holder, ttl_seconds=_WAITER_TTL_SECONDS
        ):
            return True
    except Exception:
        pass
    try:
        return bool(
            db.try_acquire_session_turn_lease(
                WAITER_KEY, holder, ttl_seconds=_WAITER_TTL_SECONDS
            )
        )
    except Exception:
        return False


def acquire_interactive(
    label: str,
    *,
    on_wait: Optional[Callable[[float, str], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    config: Optional[dict] = None,
    db: Any = None,
) -> Tuple[str, Optional[str]]:
    """Take the GPU for an interactive turn. Returns (outcome, holder).

    Outcomes: ``acquired`` (holder set, caller MUST release), ``inherited``
    (this process already holds it -- do nothing), ``override`` (proceeding
    without the lease), ``disabled``.

    Split from the ``interactive_slot`` context manager because run_agent's
    turn prologue and its finally block are hundreds of lines apart; a
    with-block cannot span them without restructuring upstream code, and the
    smaller the diff in that file the cheaper every future merge is.

    Semantics deliberately DIVERGE from the session turn lease: that one
    guards conversation integrity, so discarding the message on contention is
    correct there. This one is a performance courtesy, so failing closed
    would trade a slow answer for a lost question. The turn ALWAYS proceeds:

      * lease free                        -> proceed holding it
      * ``should_abort()`` (user Ctrl+C)  -> proceed immediately, no lease
      * wait cap exceeded                 -> proceed anyway, no lease

    While waiting, the WAITER flag is held so the cron gate can see a human
    is queued and stand down instead of grabbing the lease the instant the
    current job releases it -- without it every handover is a race the human
    loses about half the time.
    """
    if not is_enabled(config):
        return "disabled", None

    with _readonly_conn() as conn:
        if _lease_held_by_this_process(conn):
            return "inherited", None

    holder = _holder_token("turn", label)
    deadline = time.monotonic() + interactive_wait_seconds(config)
    started = time.monotonic()
    announced = False
    last_notice = 0.0
    acquired = False

    # Prefer the caller's own SessionDB (run_agent already holds one for the
    # turn). Falling back to the registry would open AND tear down a full
    # SessionDB -- schema checks included -- on every turn in any process that
    # holds no other reference, which is real per-turn work for a courtesy
    # check. One handle for the whole wait either way: the registry is
    # refcounted and tears a generation down on its FINAL release, so
    # acquiring once a second would churn the connection.
    with _borrowed_or_shared_db(db) as db:
        while db is not None:
            if should_abort is not None:
                try:
                    if should_abort():
                        break
                except Exception:
                    pass
            try:
                acquired = bool(
                    db.try_acquire_session_turn_lease(
                        GLOBAL_LEASE_KEY, holder, ttl_seconds=_LEASE_TTL_SECONDS
                    )
                )
            except Exception:
                logger.debug("gpu_arbiter: turn lease acquire failed", exc_info=True)
                break
            if acquired:
                break
            # Announce the human so the cron gate stands down instead of
            # racing us for the lease at the next handover.
            _set_waiter(db, holder)
            if time.monotonic() >= deadline:
                break
            with _readonly_conn() as conn:
                blocker = _pretty_holder(_lease_holder(conn, GLOBAL_LEASE_KEY))
            elapsed = time.monotonic() - started
            if on_wait is not None and (
                not announced or (elapsed - last_notice) >= 15.0
            ):
                announced = True
                last_notice = elapsed
                with contextlib.suppress(Exception):
                    on_wait(elapsed, blocker)
            time.sleep(1.0)

        # Bookkeeping stays INSIDE the with-block: when the caller lent us
        # their handle this is the only one we may use, and when we borrowed
        # one it is still checked out here. Doing it after the block left the
        # waiter flag set on the override path -- blocking cron for the rest
        # of its TTL after the human had already given up waiting.
        if not acquired and db is not None:
            with contextlib.suppress(Exception):
                db.release_session_turn_lease(WAITER_KEY, holder)

    if acquired:
        lease = _HeldLease(holder)
        lease.start_refresh()
        _ACTIVE_INTERACTIVE_LEASES[holder] = lease
        return "acquired", holder

    return "override", None


def release_interactive(holder: Optional[str], *, db: Any = None) -> None:
    """Release a lease taken by :func:`acquire_interactive`. Never raises."""
    if not holder:
        return
    lease = _ACTIVE_INTERACTIVE_LEASES.pop(holder, None)
    if lease is not None:
        with contextlib.suppress(Exception):
            lease.stop()
    with _borrowed_or_shared_db(db) as db:
        if db is not None:
            with contextlib.suppress(Exception):
                db.release_session_turn_lease(GLOBAL_LEASE_KEY, holder)
            with contextlib.suppress(Exception):
                db.release_session_turn_lease(WAITER_KEY, holder)


@contextlib.contextmanager
def interactive_slot(
    label: str,
    *,
    on_wait: Optional[Callable[[float, str], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    config: Optional[dict] = None,
):
    """Hold the GPU for an interactive turn, waiting politely for a cron run.

    Semantics deliberately DIVERGE from the session turn lease: that one
    guards conversation integrity, so failing closed (discarding the message)
    is correct there. This one is a performance courtesy, so failing closed
    would trade a slow answer for a lost question. The turn ALWAYS proceeds:

      * lease free                        -> proceed holding it
      * ``should_abort()`` (user Ctrl+C)  -> proceed immediately, no lease
      * wait cap exceeded                 -> proceed anyway, no lease

    ``on_wait(elapsed_seconds, holder)`` drives the user-visible notice.

    While waiting, the WAITER flag is held so the cron gate can see that a
    human is queued and stand down instead of grabbing the lease the instant
    the current job releases it — without it, every handover would be a race
    the human loses about half the time.
    """
    if not is_enabled(config):
        yield "disabled"
        return

    holder = _holder_token("turn", label)
    deadline = time.monotonic() + interactive_wait_seconds(config)
    acquired = False
    lease: Optional[_HeldLease] = None
    waiting_announced = False
    started = time.monotonic()
    last_notice = 0.0

    try:
        while True:
            if should_abort is not None:
                try:
                    if should_abort():
                        break
                except Exception:
                    pass
            with _shared_db() as db:
                if db is None:
                    break
                try:
                    acquired = bool(
                        db.try_acquire_session_turn_lease(
                            GLOBAL_LEASE_KEY, holder, ttl_seconds=_LEASE_TTL_SECONDS
                        )
                    )
                except Exception:
                    logger.debug("gpu_arbiter: turn lease acquire failed", exc_info=True)
                    break
                if acquired:
                    break
                # Not ours: announce ourselves so cron stands down, then wait.
                _set_waiter(db, holder)
            if time.monotonic() >= deadline:
                break
            with _readonly_conn() as conn:
                blocker = _pretty_holder(_lease_holder(conn, GLOBAL_LEASE_KEY))
            elapsed = time.monotonic() - started
            if on_wait is not None and (
                not waiting_announced or (elapsed - last_notice) >= 15.0
            ):
                waiting_announced = True
                last_notice = elapsed
                with contextlib.suppress(Exception):
                    on_wait(elapsed, blocker)
            time.sleep(1.0)

        if acquired:
            lease = _HeldLease(holder)
            lease.start_refresh()
            yield "acquired"
        else:
            yield "override"
    finally:
        if lease is not None:
            lease.stop()
        with _shared_db() as db:
            if db is not None:
                if acquired:
                    with contextlib.suppress(Exception):
                        db.release_session_turn_lease(GLOBAL_LEASE_KEY, holder)
                with contextlib.suppress(Exception):
                    db.release_session_turn_lease(WAITER_KEY, holder)


# ---------------------------------------------------------------------------
# Deferral bookkeeping and starvation alerting
# ---------------------------------------------------------------------------


def _state_path() -> Path:
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception:
        home = Path(os.path.expanduser("~/.hermes"))
    return home / "cron" / "gpu_arbiter_state.json"


def _load_state() -> Dict[str, Any]:
    path = _state_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        logger.debug("gpu_arbiter: unreadable state file", exc_info=True)
    return {}


def _save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        logger.debug("gpu_arbiter: cannot persist state", exc_info=True)


def record_deferral(job: dict, reason: str, config: Optional[dict] = None) -> Tuple[int, float, bool]:
    """Count one deferral. Returns (count, waited_seconds, should_alert).

    ``should_alert`` goes True once per starvation episode, when cumulative
    deferral time crosses ``cron.gpu_arbiter_defer_alert_minutes``.

    WHY THIS EXISTS AT ALL: a deferred cron job is silent by design — a
    refused processor delivers nothing and a quiet watcher tick prints
    nothing. If the gate ever wedges shut (a lease never released, a
    condition inverted by a bad merge), every job stops and NOTHING says so.
    An alert that does not depend on a successful run is the only thing that
    can break that silence.
    """
    job_id = str(job.get("id") or "?")
    now = time.time()
    should_alert = False
    with _state_file_lock:
        state = _load_state()
        entry = state.get(job_id)
        if not isinstance(entry, dict):
            entry = {"count": 0, "first_deferred_at": now, "alerted_at": None}
        entry["count"] = int(entry.get("count") or 0) + 1
        entry.setdefault("first_deferred_at", now)
        entry["last_deferred_at"] = now
        entry["last_reason"] = reason
        entry["name"] = job.get("name") or job_id
        waited = max(0.0, now - float(entry.get("first_deferred_at") or now))
        threshold = defer_alert_seconds(config)
        if threshold > 0 and waited >= threshold and not entry.get("alerted_at"):
            entry["alerted_at"] = now
            should_alert = True
        state[job_id] = entry
        _save_state(state)
    return int(entry["count"]), waited, should_alert


def deferral_summary(job_id: str) -> Optional[Tuple[int, float]]:
    """(count, waited_seconds) for a job currently in a deferral episode."""
    entry = _load_state().get(str(job_id))
    if not isinstance(entry, dict):
        return None
    count = int(entry.get("count") or 0)
    if count <= 0:
        return None
    first = float(entry.get("first_deferred_at") or time.time())
    return count, max(0.0, time.time() - first)


def clear_deferrals(job_id: str) -> Optional[Tuple[int, float]]:
    """End a job's deferral episode; returns what it went through, if any."""
    summary = deferral_summary(job_id)
    if summary is None:
        return None
    with _state_file_lock:
        state = _load_state()
        if str(job_id) in state:
            state.pop(str(job_id), None)
            _save_state(state)
    return summary


def format_deferral_note(count: int, waited_seconds: float) -> str:
    minutes = int(waited_seconds // 60)
    if minutes >= 1:
        return f"_(deferred {count}x, {minutes} min waiting for the GPU)_"
    return f"_(deferred {count}x, {int(waited_seconds)}s waiting for the GPU)_"


def format_starvation_alert(job: dict, count: int, waited_seconds: float, reason: str) -> str:
    minutes = int(waited_seconds // 60)
    name = job.get("name") or job.get("id") or "?"
    return (
        f"⚠️ Cron job `{name}` has been deferred {count} times over {minutes} min "
        f"and has not run.\n\nLast reason: {reason}\n\n"
        "If no interactive session has been active that long, the GPU arbiter "
        "may be stuck holding its lease. Disable it with "
        "`hermes config set cron.gpu_arbiter false` and report the incident."
    )
