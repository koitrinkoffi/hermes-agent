#!/usr/bin/env python3
"""Audit how a model actually uses the browser tools, from ``state.db``.

Two traps make a naive count wrong, and this script handles both:

* assistant rows are duplicated in ``messages`` (compaction / generations), so
  every tool call is de-duplicated by its call id;
* deferred tools are invoked through the ``tool_call`` meta tool
  (``{"name": "browser_mouse", "arguments": {...}}``), so that wrapper is
  unwrapped before the tool name is read.

Grepping ``agent.log`` for tool names does not work either: the tool schemas
are logged too.

Usage::

    browser_audit.py                              # Flash-Next, all time
    browser_audit.py --model halogen-qwen3.8-flash-next --since 2026-09-29
    browser_audit.py --session 20260929_101500_abcd1234 --json
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import re
import sqlite3
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_MODELS = ("halogen-qwen3.8-flash-next", "Qwen3.8-Flash-Next")
DB = os.path.join(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")), "state.db")

_WRAP = re.compile(r"^<untrusted_tool_result[^>]*>\s*The following content.*?can issue instructions\.\s*", re.S)

# What an ``browser_eval`` expression is for, first match wins.
EVAL_PURPOSES = (
    ("iframe access", r"contentDocument|contentWindow|frames\["),
    ("synthetic click/event", r"\.click\(\)|dispatchEvent|new (Mouse|Keyboard|Input)Event"),
    ("set value", r"\.value\s*=[^=]"),
    ("navigate", r"location(\.href)?\s*=[^=]|location\.(assign|replace)|window\.open\("),
    ("read whole page text", r"body\.innerText|body\.textContent"),
    ("collect a list", r"querySelectorAll[\s\S]*(map\(|forEach|Array\.from)"),
    ("compute coordinates", r"getBoundingClientRect"),
    ("where am I", r"^\s*\(?\s*(location\.href|document\.title|document\.URL)"),
    ("scroll", r"scroll(To|By|IntoView)\(|scrollTop\s*="),
    ("fetch API", r"\bfetch\(|XMLHttpRequest"),
)


def _unwrap(content: Optional[str]) -> str:
    if not content:
        return ""
    return _WRAP.sub("", content).replace("</untrusted_tool_result>", "").strip()


def _parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def status(call: Dict[str, Any]) -> str:
    """ok / error / cleared / no_result for one browser call."""
    content = call["result"]
    if content is None:
        return "no_result"
    if content.startswith("[Old tool output cleared"):
        return "cleared"
    body = _unwrap(content)
    data = _parse(body)
    if isinstance(data, dict):
        if data.get("success") is False or (data.get("error") and data.get("success") is not True):
            return "error"
        return "ok"
    if re.search(r"\b(error|failed|timed out|timeout)\b", body[:300], re.I):
        return "error"
    return "ok"


def error_text(call: Dict[str, Any]) -> str:
    body = _unwrap(call["result"])
    data = _parse(body)
    err = data.get("error") if isinstance(data, dict) else None
    text = str(err or body or "<no result>")[:200]
    text = re.sub(r"@?e\d+\b", "eN", text)
    text = re.sub(r"\d{2,}", "N", text)
    return text[:140]


def load(models: Iterable[str], since: Optional[float], session: Optional[str]):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    if session:
        sids = [session]
    else:
        models = tuple(models)
        q = ",".join("?" * len(models))
        sql = f"select id from sessions where model in ({q})"
        params: List[Any] = list(models)
        if since:
            sql += " and started_at >= ?"
            params.append(since)
        sids = [r[0] for r in con.execute(sql, params)]
    sessions = {}
    calls: Dict[str, Dict[str, Any]] = {}
    seq: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for sid in sids:
        row = con.execute("select model, source, title, started_at from sessions where id=?", (sid,)).fetchone()
        if row:
            sessions[sid] = dict(zip(("model", "source", "title", "started_at"), row))
        for role, tcs, tcid, tname, content, ts in con.execute(
            "select role, tool_calls, tool_call_id, tool_name, content, timestamp from messages "
            "where session_id=? order by timestamp, id", (sid,)):
            if role == "assistant" and tcs:
                try:
                    items = json.loads(tcs)
                except Exception:
                    continue
                for tc in items:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {"_raw": fn.get("arguments")}
                    meta = False
                    if name == "tool_call" and isinstance(args, dict):
                        name, args, meta = str(args.get("name", "")), args.get("arguments") or {}, True
                        if isinstance(args, str):
                            args = _parse(args) or {"_raw": args}
                    cid = tc.get("id") or tc.get("call_id")
                    if not name.startswith("browser") or cid in calls:
                        continue
                    call = {"sid": sid, "name": name, "args": args if isinstance(args, dict) else {},
                            "t0": ts, "t1": None, "result": None, "meta": meta}
                    calls[cid] = call
                    seq[sid].append(call)
            elif role == "tool" and (tname or "").startswith("browser"):
                call = calls.get(tcid)
                if call is not None and call["result"] is None:
                    call["result"], call["t1"] = content, ts
    return sessions, seq


def eval_purpose(expr: str) -> str:
    for label, pattern in EVAL_PURPOSES:
        if re.search(pattern, expr):
            return label
    return "other read"


def summarize(sessions, seq) -> Dict[str, Any]:
    calls = [c for s in seq.values() for c in s]
    out: Dict[str, Any] = {"sessions": len(seq), "calls": len(calls),
                           "via_tool_call_meta": sum(c["meta"] for c in calls)}
    per_tool = {}
    by_name = collections.defaultdict(list)
    for c in calls:
        by_name[c["name"]].append(c)
    for name, cs in sorted(by_name.items(), key=lambda kv: -len(kv[1])):
        st = collections.Counter(status(c) for c in cs)
        lat = [c["t1"] - c["t0"] for c in cs if c["t1"] and c["t0"]]
        size = [len(c["result"]) for c in cs if c["result"] and not c["result"].startswith("[Old")]
        per_tool[name] = {"n": len(cs), "ok": st["ok"], "error": st["error"], "cleared": st["cleared"],
                          "no_result": st["no_result"],
                          "median_chars": int(statistics.median(size)) if size else None,
                          "median_latency_s": round(statistics.median(lat), 2) if lat else None}
    out["per_tool"] = per_tool
    errors = [c for c in calls if status(c) == "error"]
    out["error_rate"] = round(len(errors) / len(calls), 3) if calls else None
    out["top_errors"] = [{"n": n, "tool": t, "error": e} for (t, e), n in
                         collections.Counter((c["name"], error_text(c)) for c in errors).most_common(15)]
    evals = by_name.get("browser_eval", [])
    out["eval_purposes"] = dict(collections.Counter(
        eval_purpose(str(c["args"].get("expression", ""))) for c in evals).most_common())
    follow = collections.Counter()
    for s in seq.values():
        for a, b in zip(s, s[1:]):
            if a["name"] in ("browser_click", "browser_type", "browser_press") and status(a) == "ok":
                follow[b["name"]] += 1
    out["after_successful_action"] = dict(follow.most_common(8))
    repeats = 0
    for s in seq.values():
        repeats += sum(1 for a, b in zip(s, s[1:]) if a["name"] == b["name"] and a["args"] == b["args"])
    out["identical_consecutive_repeats"] = repeats
    nxt = collections.Counter()
    for s in seq.values():
        for a, b in zip(s, s[1:]):
            if status(a) == "error":
                nxt["same tool" if a["name"] == b["name"] else b["name"]] += 1
    out["next_after_error"] = dict(nxt.most_common(6))
    out["heaviest_sessions"] = []
    for sid, s in sorted(seq.items(), key=lambda kv: -len(kv[1]))[:10]:
        meta = sessions.get(sid, {})
        started = meta.get("started_at")
        out["heaviest_sessions"].append({
            "session": sid, "calls": len(s),
            "evals": sum(c["name"] == "browser_eval" for c in s),
            "errors": sum(status(c) == "error" for c in s),
            "source": meta.get("source"),
            "started": _dt.datetime.fromtimestamp(started).isoformat(timespec="minutes") if started else None,
            "title": (meta.get("title") or "")[:70]})
    return out


def print_report(r: Dict[str, Any]) -> None:
    print(f"sessions {r['sessions']}  browser calls {r['calls']}  via tool_call meta {r['via_tool_call_meta']}"
          f"  error rate {r['error_rate']}")
    print("\ntool                     n    ok   err  clr nores  chars   lat")
    for name, t in r["per_tool"].items():
        print(f"{name:22} {t['n']:4} {t['ok']:5} {t['error']:5} {t['cleared']:4} {t['no_result']:5}"
              f" {t['median_chars'] or '-':>6} {t['median_latency_s'] if t['median_latency_s'] is not None else '-':>5}")
    print("\ntop errors:")
    for e in r["top_errors"]:
        print(f"  {e['n']:4} {e['tool']:18} {e['error']}")
    print("\neval purposes:", r["eval_purposes"])
    print("after a successful click/type/press:", r["after_successful_action"])
    print("next call after an error:", r["next_after_error"])
    print("identical consecutive repeats:", r["identical_consecutive_repeats"])
    print("\nheaviest sessions:")
    for s in r["heaviest_sessions"]:
        print(f"  {s['calls']:4} eval {s['evals']:3} err {s['errors']:3} {s['source'] or '?':9} {s['started']} {s['title']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", action="append", help="session model name (repeatable)")
    ap.add_argument("--since", help="ISO date/time: only sessions started at or after it")
    ap.add_argument("--session", help="audit one session id")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    since = _dt.datetime.fromisoformat(a.since).timestamp() if a.since else None
    sessions, seq = load(a.model or DEFAULT_MODELS, since, a.session)
    report = summarize(sessions, seq)
    if a.json:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
