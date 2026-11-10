#!/usr/bin/env python3
"""Rebuild an opencode run's solve_out.txt from opencode's session database.

opencode (`opencode run --format json`) prints one JSON event per finished part of the session (step_start,
reasoning [with --thinking], text, tool_use, step_finish) and per session error. Between 2026-11-03 and 2026-11-05
agents/opencode/solve.sh piped those events through `tee`, which stopped copying to solve_out.txt when the
non-blocking pipe was full; the rest of that opencode session never reached solve_out.txt. opencode's database
(~/agent_logs/opencode/opencode.db in the result dir, from 2026-11-04 on) still holds every part.

fill   (default): keep every line of solve_out.txt as it is, drop a cut-off (unparsable) event line, and add an event
        for each part/error of the session that solve_out.txt lacks, at the part's last-update time. Writes --out.
--check: rebuild ALL events from the database and compare them with the events in solve_out.txt (for runs whose
        solve_out.txt is complete): same events, same parts, same order. Exits 1 on any difference.

usage: rebuild_solve_out.py --db-dir DIR --solve-out FILE [--reasoning] (--out FILE | --check) [--until ISO]
  --db-dir     directory with opencode.db (+ -wal/-shm); copied to a local temp dir first (/fast has no file locks)
  --reasoning  the run used `opencode run --thinking` (reasoning events were printed)
  --until      only compare events up to this UTC time (a database snapshot taken while the run went on)
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] ?")


def ts_str(ms: int) -> str:
    return datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts_ms(s: str) -> int:
    return int(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def read_solve_out(path: Path) -> tuple[list[tuple[int, str, dict | None]], int]:
    """Lines as (ms, raw line, event or None); also the number of unparsable event lines (cut off).

    A line without the timestamp prefix (e.g. the shell's "Terminated" when the time limit kills the sandbox) is
    kept as it is, at the previous line's time.
    """
    lines, broken, last_ms = [], 0, 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TS_RE.match(raw)
        if not m:
            lines.append((last_ms, raw, None))
            continue
        last_ms = ts_ms(m.group(1))
        body = raw[m.end():]
        event = None
        if body.startswith("{"):
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                broken += 1
                continue
        lines.append((last_ms, raw, event))
    return lines, broken


def db_events(db_dir: Path, session_id: str, reasoning: bool) -> list[tuple[int, dict]]:
    """The events `opencode run --format json [--thinking]` prints for the session, as (ms, event)."""
    tmp = Path(tempfile.mkdtemp(prefix="opencode_db_"))
    try:
        for f in db_dir.glob("opencode.db*"):
            shutil.copy2(f, tmp / f.name)
        con = sqlite3.connect(tmp / "opencode.db")
        row = con.execute("select parent_id from session where id = ?", (session_id,)).fetchone()
        if row is None:
            raise SystemExit(f"session {session_id} not in {db_dir}/opencode.db")
        if row[0] is not None:
            raise SystemExit(f"session {session_id} is a child session (parent {row[0]})")
        events = []
        for pid, mid, updated, data in con.execute(
                "select id, message_id, time_updated, data from part where session_id = ? order by id", (session_id,)):
            p = json.loads(data)
            part = {"id": pid, "messageID": mid, "sessionID": session_id, **p}
            kind = {"step-start": "step_start", "step-finish": "step_finish"}.get(p["type"])
            if p["type"] == "tool" and p["state"]["status"] in ("completed", "error"):
                kind = "tool_use"
            elif p["type"] == "text" and p.get("time", {}).get("end"):
                kind = "text"
            elif p["type"] == "reasoning" and reasoning and p.get("time", {}).get("end"):
                kind = "reasoning"
            if kind:
                events.append((updated, {"type": kind, "timestamp": updated, "sessionID": session_id, "part": part}))
        for mid, updated, data in con.execute(
                "select id, time_updated, data from message where session_id = ? order by id", (session_id,)):
            m = json.loads(data)
            if m.get("error"):
                events.append((updated, {"type": "error", "timestamp": updated, "sessionID": session_id, "error": m["error"]}))
        con.close()
    finally:
        shutil.rmtree(tmp)
    events.sort(key=lambda e: e[0])
    return events


def key(event: dict) -> tuple:
    if event["type"] == "error":
        return ("error", json.dumps(event["error"], sort_keys=True))
    return (event["type"], event["part"]["id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", type=Path, required=True)
    ap.add_argument("--solve-out", type=Path, required=True)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--until")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", type=Path)
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()

    lines, broken = read_solve_out(args.solve_out)
    orig = [(ms, e) for ms, _, e in lines if e is not None]
    sessions = {e["sessionID"] for _, e in orig if "sessionID" in e}
    if len(sessions) != 1:
        raise SystemExit(f"expected one session in {args.solve_out}, found {sessions}")
    session_id = sessions.pop()
    rebuilt = db_events(args.db_dir, session_id, args.reasoning)
    by_key = {key(e): e for _, e in rebuilt}
    orig_keys = {key(e) for _, e in orig}

    if args.check:
        until = ts_ms(args.until) if args.until else None
        if until is not None:
            rebuilt = [(ms, e) for ms, e in rebuilt if ms <= until]
            orig = [(ms, e) for ms, e in orig if ms <= until]
        missing = [key(e) for _, e in orig if key(e) not in by_key]
        extra = [k for k in (key(e) for _, e in rebuilt) if k not in orig_keys]
        extra_ms = [ms for ms, e in rebuilt if key(e) not in orig_keys]
        differ = []
        for _, e in orig:
            r = by_key.get(key(e))
            if r is None:
                continue
            a = e.get("part", e.get("error"))
            b = r.get("part", r.get("error"))
            if a != b:
                differ.append((key(e), sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))))
        order_o = [key(e) for _, e in orig if key(e) in by_key]
        pos = {k: i for i, k in enumerate(key(e) for _, e in rebuilt)}
        inversions = sum(1 for a, b in zip(order_o, order_o[1:]) if pos[a] > pos[b])
        lag = [abs(by_key[key(e)]["timestamp"] - ms) / 1000 for ms, e in orig if key(e) in by_key]
        print(f"session {session_id}: solve_out events={len(orig)} (cut-off lines {broken}), database events={len(rebuilt)}")
        print(f"  in solve_out but not rebuilt: {len(missing)} {missing[:5]}")
        span = f" between {ts_str(min(extra_ms))} and {ts_str(max(extra_ms))}" if extra_ms else ""
        print(f"  rebuilt but not in solve_out: {len(extra)}{span} {extra[:3]}")
        if extra_ms:
            # in a cut trace every lost event comes after the last event that reached solve_out.txt (prefix: seconds)
            last = max(ms for ms, _ in orig)
            print(f"  last event in solve_out at {ts_str(last)}; lost events all after it: {min(extra_ms) // 1000 >= last // 1000}")
        print(f"  same event, different content: {len(differ)} {differ[:5]}")
        print(f"  order: {inversions} of {max(len(order_o) - 1, 0)} neighbouring pairs swapped")
        print(f"  timestamp difference (s): max {max(lag, default=0):.1f}, events > 2 s apart: {sum(x > 2 for x in lag)}")
        sys.exit(1 if missing or extra or differ else 0)

    added = [(ms, e) for ms, e in rebuilt if key(e) not in orig_keys]
    # by second (the prefix's resolution); within a second the kept lines first, in file order, then the added events
    merged = [((ms // 1000, 0, i), raw) for i, (ms, raw, _) in enumerate(lines)]
    merged += [((ms // 1000, 1, ms), f"[{ts_str(ms)}] {json.dumps(e, separators=(',', ':'), ensure_ascii=False)}")
               for ms, e in added]
    merged.sort(key=lambda t: t[0])
    args.out.write_text("".join(raw + "\n" for _, raw in merged), encoding="utf-8")
    print(f"{args.out}: kept {len(lines)} lines of {args.solve_out} (dropped {broken} cut-off), "
          f"added {len(added)} events from the database ({sum(e['type'] == 'tool_use' for _, e in added)} tool calls)")


if __name__ == "__main__":
    main()
