#!/usr/bin/env python3
"""Kill the process holding the most memory before HTCondor holds the whole job for its memory use.

On the MPI-IS cluster HTCondor holds a job whose memory passes 90% of request_memory; the job, and with it the whole
agent run, is then lost. That happened to Gemini 4 Argon humaneval runs (2026-11-03/04, at 128 and at 256 GB): the agent
executed model-generated programs inside pool workers without any memory limit, and one sample allocated without end.
apptainer's --memory needs cgroups v2 (the nodes run v1) and the job's cgroup is not writable, so this watchdog runs
next to the agent's sandbox (src/run_task.sh) and, while the job's memory is above --fraction of request_memory,
SIGKILLs the job process with the most private memory, once per --interval. The agent sees that command die (exit 137)
and can carry on. Every kill goes to --log.

Memory is the job cgroup's resident memory (cgroup v1: total_rss + total_mapped_file + total_swap of memory.stat;
v2: anon + file_mapped + memory.swap.current): each page counted once, without the page cache. Summing the processes'
RSS would count pages that forked workers share several times.

usage: memory_watchdog.py --check                       # validate the setup, print it, exit 1 if unusable
       memory_watchdog.py --log FILE [--fraction 0.8] [--interval 0.5] [--limit-mb MB]
  --limit-mb  instead of request_memory from the job ad ($_CONDOR_JOB_AD); for tests
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import signal
import sys
import time
from pathlib import Path

# never killed: the sandbox's own machinery (it holds little memory, but a kill would end the whole sandbox)
PROTECTED_COMMS = {"starter", "starter-suid", "squashfuse_ll", "squashfuse", "fuse-overlayfs", "apptainer", "timeout"}


def job_cgroup() -> tuple[int, Path]:
    """(cgroup version, directory) of this process's memory cgroup, i.e. the job's."""
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if "memory" in controllers.split(","):
            return 1, Path("/sys/fs/cgroup/memory") / path.lstrip("/")
    for line in lines:
        if line.startswith("0::"):
            return 2, Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    raise SystemExit(f"memory_watchdog: no memory cgroup in /proc/self/cgroup: {lines}")


def resident_bytes(version: int, cg: Path) -> int:
    stat = dict(line.split() for line in (cg / "memory.stat").read_text().splitlines())
    if version == 1:
        return int(stat["total_rss"]) + int(stat["total_mapped_file"]) + int(stat.get("total_swap", 0))
    swap = cg / "memory.swap.current"
    return int(stat["anon"]) + int(stat["file_mapped"]) + (int(swap.read_text()) if swap.exists() else 0)


def job_pids(version: int, cg: Path) -> set[int]:
    pids: set[int] = set()
    for procs in cg.rglob("cgroup.procs"):
        pids |= {int(p) for p in procs.read_text().split()}
    return pids


def request_memory_mb() -> int:
    ad = os.environ.get("_CONDOR_JOB_AD")
    if not ad:
        raise SystemExit("memory_watchdog: $_CONDOR_JOB_AD is not set (not an HTCondor job); pass --limit-mb")
    m = re.search(r"^RequestMemory = (\d+)$", Path(ad).read_text(), re.MULTILINE)
    if not m:
        raise SystemExit(f"memory_watchdog: no numeric RequestMemory in {ad}")
    return int(m.group(1))


def private_bytes(pid: int) -> int:
    """Memory only this process holds (Private_Clean + Private_Dirty + Swap of smaps_rollup)."""
    total = 0
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in ("Private_Clean", "Private_Dirty", "Swap"):
            total += int(rest.split()[0]) * 1024
    return total


def describe(pid: int) -> str:
    comm = Path(f"/proc/{pid}/comm").read_text().strip()
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    cmd = cmd.replace("\n", "\\n")   # one log line per kill
    return f"{comm}: {cmd[:300]}"


def biggest_process(pids: set[int]) -> tuple[int, int] | None:
    """(pid, private bytes) of the job process with the most private memory, protected ones excluded."""
    best = None
    for pid in pids - {os.getpid()}:
        try:
            if Path(f"/proc/{pid}/comm").read_text().strip() in PROTECTED_COMMS:
                continue
            size = private_bytes(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError):   # exited meanwhile / not ours to kill
            continue
        if best is None or size > best[1]:
            best = (pid, size)
    return best


def gb(n: int) -> str:
    return f"{n / 1024**3:.1f} GB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--fraction", type=float, default=0.8)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--limit-mb", type=int)
    args = ap.parse_args()
    if not 0 < args.fraction < 0.9:
        raise SystemExit(f"memory_watchdog: --fraction must be in (0, 0.9) (HTCondor holds at 0.9), got {args.fraction}")

    version, cg = job_cgroup()
    limit_mb = args.limit_mb if args.limit_mb is not None else request_memory_mb()
    threshold = int(limit_mb * 1024**2 * args.fraction)
    used = resident_bytes(version, cg)
    setup = (f"cgroup v{version} {cg}, request_memory {limit_mb} MB, kills above {gb(threshold)} "
             f"({args.fraction:.0%}), now {gb(used)} in {len(job_pids(version, cg))} processes")
    if args.check:
        print(f"memory_watchdog: OK: {setup}")
        return
    if args.log is None:
        raise SystemExit("memory_watchdog: --log is required unless --check")

    def log(msg: str) -> None:
        with args.log.open("a") as f:
            f.write(f"[{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}\n")

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    log(f"started: {setup}")
    while True:
        used = resident_bytes(version, cg)
        if used > threshold:
            victim = biggest_process(job_pids(version, cg))
            if victim is None:
                log(f"job memory {gb(used)} > {gb(threshold)}, but no process to kill")
            else:
                pid, size = victim
                try:
                    what = describe(pid)
                    os.kill(pid, signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):   # exited meanwhile: measure again at once
                    continue
                log(f"job memory {gb(used)} > {gb(threshold)}: killed pid {pid} ({gb(size)} private) {what}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
