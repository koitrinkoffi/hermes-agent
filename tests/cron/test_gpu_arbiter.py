"""Tests for the local GPU arbiter mod (``hermes-mods``).

These exist because the mod lives in files upstream rewrites constantly. When
a merge conflicts in ``cron/scheduler.py`` or ``run_agent.py`` and you resolve
it, this file is what tells you whether the resolution kept the behaviour.

The arbiter is inert under pytest by design (see ``is_enabled``), so every
test here passes an explicit ``config`` to opt in — that is the documented
escape hatch, not a workaround.
"""

import time

import pytest

from cron import gpu_arbiter as G

ON = {"cron": {"gpu_arbiter": True}}


class _FakeConn:
    """Minimal stand-in for the read-only state.db handle."""

    def __init__(self, leases=None, last_message_age=None):
        self._leases = leases or {}
        self._age = last_message_age

    def execute(self, sql, params=()):
        if "MAX(timestamp)" in sql:
            value = None if self._age is None else time.time() - self._age
            return _Result([{"t": value}])
        if "COUNT(*)" in sql:
            live = [
                k
                for k, (h, exp) in self._leases.items()
                if exp > time.time() and k not in (G.GLOBAL_LEASE_KEY, G.WAITER_KEY)
            ]
            return _Result([{"n": len(live)}])
        key = params[0]
        row = self._leases.get(key)
        if row is None:
            return _Result([])
        return _Result([{"holder": row[0], "expires_at": row[1]}])


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


@pytest.fixture
def gate_env(monkeypatch):
    """Drive the gate off an in-memory lease table with no backend probe."""
    state = {"leases": {}, "age": None, "busy": False}

    import contextlib

    @contextlib.contextmanager
    def _conn():
        yield _FakeConn(state["leases"], state["age"])

    monkeypatch.setattr(G, "_readonly_conn", _conn)
    monkeypatch.setattr(G, "_backend_busy", lambda config=None: state["busy"])
    return state


# ── policy grammar ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", [None, "", "   "])
def test_normalize_gpu_policy_clears(value):
    assert G.normalize_gpu_policy(value) is None


@pytest.mark.parametrize("value,expected", [("defer", "defer"), ("ALWAYS", "always")])
def test_normalize_gpu_policy_canonicalises(value, expected):
    assert G.normalize_gpu_policy(value) == expected


def test_normalize_gpu_policy_rejects_unknown():
    with pytest.raises(ValueError):
        G.normalize_gpu_policy("urgent")


def test_resolve_job_gpu_policy_precedence():
    assert G.resolve_job_gpu_policy({}, config={}) == "defer"
    assert G.resolve_job_gpu_policy({}, config={"cron": {"gpu_policy": "always"}}) == "always"
    # The per-job pin beats the fleet default.
    assert (
        G.resolve_job_gpu_policy(
            {"gpu_policy": "defer"}, config={"cron": {"gpu_policy": "always"}}
        )
        == "defer"
    )


# ── the switch ──────────────────────────────────────────────────────────────


def test_disabled_by_config_admits_everything(gate_env):
    gate_env["leases"][G.GLOBAL_LEASE_KEY] = ("cron:other:pid=1", time.time() + 60)
    off = {"cron": {"gpu_arbiter": False}}
    assert G.cron_gate({"id": "j", "name": "j"}, config=off).allowed


def test_inert_under_pytest_without_explicit_optin(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "yes")
    assert G.is_enabled(config={}) is False
    assert G.is_enabled(config=ON) is True


# ── the gate ────────────────────────────────────────────────────────────────


def test_gate_admits_on_an_idle_machine(gate_env):
    assert G.cron_gate({"id": "j", "name": "j"}, config=ON).allowed


def test_gate_defers_while_a_turn_is_in_flight(gate_env):
    gate_env["leases"]["some-conversation"] = ("pid=1:turn=x", time.time() + 60)
    decision = G.cron_gate({"id": "j", "name": "j"}, config=ON)
    assert not decision.allowed
    assert "interactive turn" in decision.reason


def test_gate_defers_while_a_human_is_queued(gate_env):
    gate_env["leases"][G.WAITER_KEY] = ("turn:cli:pid=1", time.time() + 30)
    decision = G.cron_gate({"id": "j", "name": "j"}, config=ON)
    assert not decision.allowed
    assert "queued" in decision.reason


def test_gate_defers_during_the_quiet_period(gate_env):
    gate_env["age"] = 60  # last message one minute ago
    cfg = {"cron": {"gpu_arbiter": True, "gpu_arbiter_quiet_minutes": 10}}
    decision = G.cron_gate({"id": "j", "name": "j"}, config=cfg)
    assert not decision.allowed
    assert "quiet period" in decision.reason


def test_gate_admits_once_the_session_has_gone_quiet(gate_env):
    gate_env["age"] = 3600
    cfg = {"cron": {"gpu_arbiter": True, "gpu_arbiter_quiet_minutes": 10}}
    assert G.cron_gate({"id": "j", "name": "j"}, config=cfg).allowed


def test_gate_defers_when_the_lease_is_taken(gate_env):
    gate_env["leases"][G.GLOBAL_LEASE_KEY] = ("cron:mailflow-process:pid=1", time.time() + 60)
    decision = G.cron_gate({"id": "j", "name": "j"}, config=ON)
    assert not decision.allowed
    assert "mailflow-process" in decision.reason


def test_gate_ignores_an_expired_lease(gate_env):
    gate_env["leases"][G.GLOBAL_LEASE_KEY] = ("cron:stale:pid=1", time.time() - 1)
    assert G.cron_gate({"id": "j", "name": "j"}, config=ON).allowed


def test_gate_defers_when_the_backend_is_busy(gate_env):
    gate_env["busy"] = True
    decision = G.cron_gate({"id": "j", "name": "j"}, config=ON)
    assert not decision.allowed
    assert "backend" in decision.reason


def test_no_agent_jobs_bypass_the_gate(gate_env):
    """Script-only jobs burn no GPU AND re-drive refused processors."""
    gate_env["leases"][G.GLOBAL_LEASE_KEY] = ("cron:other:pid=1", time.time() + 60)
    assert G.cron_gate({"id": "j", "name": "j", "no_agent": True}, config=ON).allowed


def test_always_policy_bypasses_the_gate(gate_env):
    gate_env["leases"][G.GLOBAL_LEASE_KEY] = ("cron:other:pid=1", time.time() + 60)
    job = {"id": "j", "name": "j", "gpu_policy": "always"}
    assert G.cron_gate(job, config=ON).allowed


def test_gate_fails_open_when_a_probe_raises(monkeypatch):
    """A broken probe must never be able to stop the automation fleet."""

    def _boom(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(G, "_readonly_conn", _boom)
    assert G.cron_gate({"id": "j", "name": "j"}, config=ON).allowed


# ── holder tokens ───────────────────────────────────────────────────────────


def test_holder_token_carries_a_reclaimable_pid():
    """hermes_state only reclaims dead holders spelled ``pid=<n>``."""
    from hermes_state import _COMPRESSION_LOCK_HOLDER_PID_RE

    token = G._holder_token("cron", "some-job")
    assert _COMPRESSION_LOCK_HOLDER_PID_RE.search(token) is not None


def test_holder_token_sanitises_colons():
    assert ":" not in G._holder_token("cron", "a:b").split(":")[1]


@pytest.mark.parametrize(
    "holder,expected",
    [
        ("cron:mailflow-process:pid=7", "mailflow-process"),
        ("turn:telegram:pid=7", "another session (telegram)"),
        (None, "another run"),
    ],
)
def test_pretty_holder(holder, expected):
    assert G._pretty_holder(holder) == expected


# ── deferral bookkeeping ────────────────────────────────────────────────────


def test_deferrals_accumulate_and_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_state_path", lambda: tmp_path / "arb.json")
    job = {"id": "job1", "name": "job1"}
    G.record_deferral(job, "busy")
    count, _ = G.deferral_summary("job1")
    assert count == 1
    G.record_deferral(job, "busy")
    count, _ = G.deferral_summary("job1")
    assert count == 2
    assert G.clear_deferrals("job1")[0] == 2
    assert G.deferral_summary("job1") is None


def test_starvation_alert_fires_once_per_episode(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "_state_path", lambda: tmp_path / "arb.json")
    cfg = {"cron": {"gpu_arbiter_defer_alert_minutes": 1 / 60.0}}  # one second
    job = {"id": "job2", "name": "job2"}
    _, _, alert = G.record_deferral(job, "busy", config=cfg)
    assert alert is False
    time.sleep(1.2)
    _, _, alert = G.record_deferral(job, "busy", config=cfg)
    assert alert is True, "the alert must not depend on a successful run"
    _, _, alert = G.record_deferral(job, "busy", config=cfg)
    assert alert is False, "one alert per starvation episode, not per tick"


def test_deferral_state_survives_an_unreadable_file(tmp_path, monkeypatch):
    path = tmp_path / "arb.json"
    path.write_text("not json at all", encoding="utf-8")
    monkeypatch.setattr(G, "_state_path", lambda: path)
    count, _, _ = G.record_deferral({"id": "job3", "name": "job3"}, "busy")
    assert count == 1


# ── interactive acquisition ─────────────────────────────────────────────────


class _FakeDB:
    """A SessionDB stand-in with just the three lease methods we use."""

    def __init__(self, held=None):
        self.rows = dict(held or {})

    def try_acquire_session_turn_lease(self, key, holder, *, ttl_seconds=300.0, **kw):
        current = self.rows.get(key)
        if current is None or current == holder:
            self.rows[key] = holder
            return True
        return False

    def refresh_session_turn_lease(self, key, holder, *, ttl_seconds=300.0):
        return self.rows.get(key) == holder

    def release_session_turn_lease(self, key, holder):
        if self.rows.get(key) == holder:
            del self.rows[key]


def test_interactive_acquires_when_free(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(G, "_lease_held_by_this_process", lambda conn: False)
    outcome, holder = G.acquire_interactive("cli", config=ON, db=db)
    assert outcome == "acquired"
    assert db.rows[G.GLOBAL_LEASE_KEY] == holder
    G.release_interactive(holder, db=db)
    assert G.GLOBAL_LEASE_KEY not in db.rows


def test_interactive_overrides_after_the_cap(monkeypatch):
    """The turn must ALWAYS proceed; a lost message is the wrong trade."""
    db = _FakeDB({G.GLOBAL_LEASE_KEY: "cron:other:pid=1"})
    monkeypatch.setattr(G, "_lease_held_by_this_process", lambda conn: False)
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    cfg = {"cron": {"gpu_arbiter": True, "gpu_arbiter_interactive_wait_seconds": 0}}
    outcome, holder = G.acquire_interactive("cli", config=cfg, db=db)
    assert outcome == "override"
    assert holder is None
    # The waiter flag must not be left behind for the cron gate to trip over.
    assert G.WAITER_KEY not in db.rows


def test_interactive_marks_a_waiter_while_blocked(monkeypatch):
    db = _FakeDB({G.GLOBAL_LEASE_KEY: "cron:other:pid=1"})
    monkeypatch.setattr(G, "_lease_held_by_this_process", lambda conn: False)
    seen = {}

    def _fake_sleep(_s):
        seen["waiter"] = db.rows.get(G.WAITER_KEY)
        raise KeyboardInterrupt  # break out of the wait loop deterministically

    monkeypatch.setattr(G.time, "sleep", _fake_sleep)
    cfg = {"cron": {"gpu_arbiter": True, "gpu_arbiter_interactive_wait_seconds": 60}}
    with pytest.raises(KeyboardInterrupt):
        G.acquire_interactive("cli", config=cfg, db=db)
    assert seen["waiter"] is not None, "a queued human must be visible to the gate"


def test_interactive_aborts_immediately_on_interrupt(monkeypatch):
    db = _FakeDB({G.GLOBAL_LEASE_KEY: "cron:other:pid=1"})
    monkeypatch.setattr(G, "_lease_held_by_this_process", lambda conn: False)
    cfg = {"cron": {"gpu_arbiter": True, "gpu_arbiter_interactive_wait_seconds": 600}}
    outcome, holder = G.acquire_interactive(
        "cli", config=cfg, db=db, should_abort=lambda: True
    )
    assert outcome == "override"
    assert holder is None


def test_interactive_is_reentrant_within_one_process(monkeypatch):
    """A cron run's own agent turn must not queue behind its parent."""
    monkeypatch.setattr(G, "_lease_held_by_this_process", lambda conn: True)
    outcome, holder = G.acquire_interactive("cli", config=ON, db=_FakeDB())
    assert outcome == "inherited"
    assert holder is None


def test_release_interactive_is_safe_without_a_holder():
    G.release_interactive(None)
    G.release_interactive("")


def test_gate_ignores_a_lease_whose_holder_is_dead(gate_env):
    """The gate must not be stricter than the acquire path.

    try_acquire_session_turn_lease reclaims a dead holder's lease, so a gate
    that still honoured it would defer every cron job for the rest of the TTL
    after a hard gateway kill, while interactive turns sailed past.
    """
    gate_env["leases"][G.GLOBAL_LEASE_KEY] = ("cron:killed:pid=999999", time.time() + 900)
    assert G.cron_gate({"id": "j", "name": "j"}, config=ON).allowed
