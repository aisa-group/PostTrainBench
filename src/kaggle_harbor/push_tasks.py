#!/usr/bin/env python3
"""Push generated PostTrainBench tasks to Kaggle Benchmarks.

**Dry-run by default.** Nothing is sent without `--yes`, because a push is
effectively permanent: slugs cannot be deleted ("Delete is not supported by
the server yet"), and a re-push mints a new versionId which orphans any
leaderboard mappings pointing at the old one.

    # see exactly what would be sent
    python src/kaggle_harbor/push_tasks.py \\
        --tasks ./tasks --defs jonathanouyang/ptb-task-defs/versions/3 \\
        --shards jonathanouyang/ptb-hf-cache-01/versions/1,...

    # actually send it
    ... --yes

Two things this exists to get right, both of which are silent when wrong:

* **`/tasks/push` returns HTTP 200 on validation failure.** The body carries
  `error`. Anything that only checks the status code reports success on a
  rejected task.
* **Every shard needs a `mounts[]` entry whose `mountPath` matches the
  `PTB_HF_CACHE_MOUNT` baked into the task.** A mismatch is not an error: the
  path simply is not there, `preflight.sh` logs a warning and skips it, and
  the run proceeds with a partial cache. This script reads the mount list back
  out of each `task.toml` and refuses to push if it disagrees with `--shards`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

PUSH_URL = "https://www.kaggle.com/api/v1/benchmarks/tasks/push"


def access_token() -> str:
    out = subprocess.run(["kaggle", "auth", "print-access-token"],
                         capture_output=True, text=True)
    token = out.stdout.strip()
    if out.returncode != 0 or not token:
        raise SystemExit(f"could not get a Kaggle access token: {out.stderr.strip()}")
    return token


def task_mounts(task_dir: Path) -> list[str]:
    """The mount paths this task's preflight will look for."""
    with open(task_dir / "task.toml", "rb") as f:
        cfg = tomllib.load(f)
    raw = cfg.get("environment", {}).get("env", {}).get("PTB_HF_CACHE_MOUNT", "")
    return [m for m in raw.split(":") if m]


def build_body(slug: str, defs: str, shards: list[str], mounts: list[str],
               agent_slug: str, n_attempts: int | None) -> dict:
    definition: dict = {
        "definitionSource": {"datasetVersionSlug": defs, "subPath": slug},
        "envVariables": [
            # KAGGLE_HARBOR_AGENT is reserved and rejected; the OVERRIDE key is
            # the documented escape hatch and the entrypoint prefers it
            # (Instructions.md:400-427). A value containing ':' also triggers
            # harbor_apply_custom_import_pythonpath, which is what makes the
            # vendored agents/ package importable.
            {"key": "KAGGLE_HARBOR_AGENT_OVERRIDE", "value": agent_slug},
        ],
    }
    if shards:
        definition["mounts"] = [
            {"datasetVersionSlug": s, "mountPath": m}
            for s, m in zip(shards, mounts)
        ]
    # Always emitted, and always 1 unless overridden.
    #
    # 1 is also the platform default, so this is a no-op on the wire — but an
    # implicit default is not a decision, and this one was decided: the team
    # leads settled it on 2026-09-01. Writing it down means a reader of the
    # push body sees the choice, and a change to it is a visible diff rather
    # than a silent shift in what the number means.
    #
    # Why not 3, when upstream averages 3 trials per cell: upstream's three
    # trials are three INDEPENDENT sweeps (commit.sh has no attempt loop — it
    # is re-run under a new EXPERIMENT_NAME, and scripts/utils.py groups the
    # resulting directories so scripts/aggregate.py can take mean/stddev per
    # cell). Kaggle's nAttempts is serial inside one session, so nAttempts=3
    # on a 10 h task is 30 h on one VM, past the session cap and past the 24 h
    # budget — and it would measure within-machine variance where upstream
    # measures across-machine, across-time variance. The faithful mechanism is
    # nAttempts=1 scheduled three times; aggregation happens afterwards.
    #
    # Not merely cosmetic, and NOT strictly a no-op. Server-side precedence is
    #   `request.NAttemptsNullable ?? userSuppliedJobConfig.NumAttempts ?? 1`
    # (CreateBenchmarkTaskVersionFromHarborKaggleDatasetsHandler.cs:204), and
    # the handler's own comment at :92 says "n_attempts wins over config.yaml,
    # so an explicit value here is final." Sending it PINS the value; omitting
    # it defers to a config.yaml we do not currently ship. Same result today,
    # different guarantee.
    #
    # Server validation is only `>= 1` (:84) plus ValidateAttemptsAndMetrics
    # (:102), which rejects attempts<=1 paired with aggregation metrics. We
    # ship no metrics, so 1 passes; and if metrics are ever added, this
    # fast-fails the push instead of silently producing a meaningless
    # aggregate.
    #
    # ⚠️ An earlier revision of this comment claimed the field "requires the
    # amw-attempts-and-metrics ops flag" and that emitting it was verified
    # safe without one. Neither was checked. Grepping the handlers finds no
    # ops-flag gate on NAttempts at all — only the two validations above. What
    # IS verified: a probe push of zz-hprobe-gemini with nAttempts=1 returned
    # HTTP 200 with an empty `error` body.
    definition["nAttempts"] = 1 if n_attempts is None else n_attempts
    return {"slug": slug, "definition": {"harborKaggleDatasets": definition}}


def push(body: dict, token: str) -> dict:
    req = urllib.request.Request(
        PUSH_URL, method="POST",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:400]}"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", type=Path, required=True,
                   help="The build_tasks.py --output directory.")
    p.add_argument("--defs", required=True,
                   help="owner/slug/versions/N of the definition dataset.")
    p.add_argument("--shards", default="",
                   help="Comma-separated owner/slug/versions/N for each HF "
                        "cache shard, in the same order as the task's "
                        "PTB_HF_CACHE_MOUNT list.")
    p.add_argument("--agent-slug", default="agents.ptb_harness:PtbHarness_codex",
                   help="Harbor agent import path for KAGGLE_HARBOR_AGENT_OVERRIDE.")
    p.add_argument("--n-attempts", type=int, default=None,
                   help="Trials per run. DEFAULT 1, and leave it there: "
                        "upstream's 3 trials are 3 independent sweeps, not 3 "
                        "attempts in one job, and Kaggle's nAttempts is serial "
                        "-- 3 attempts of a 10h task is 30h on one VM, past "
                        "the session cap. To reproduce upstream, schedule the "
                        "sweep three times at nAttempts=1 and aggregate after "
                        "(scripts/aggregate.py takes mean/stddev per cell).")
    p.add_argument("--only", action="append",
                   help="Push only these task directory names. Repeatable.")
    p.add_argument("--yes", action="store_true",
                   help="Actually send. Without it, nothing leaves this box.")
    args = p.parse_args()

    shards = [s.strip() for s in args.shards.split(",") if s.strip()]

    dirs = sorted(d for d in args.tasks.iterdir()
                  if d.is_dir() and (d / "task.toml").is_file())
    if args.only:
        wanted = set(args.only)
        dirs = [d for d in dirs if d.name in wanted]
        missing = wanted - {d.name for d in dirs}
        if missing:
            raise SystemExit(f"--only named tasks that do not exist: {sorted(missing)}")
    if not dirs:
        raise SystemExit(f"no task directories under {args.tasks}")

    # Refuse a smoke build outright — it swaps in lightweight images and
    # tolerates missing judge verdicts, so a pushed one would score nonsense.
    for d in dirs:
        with open(d / "task.toml", "rb") as f:
            meta = tomllib.load(f).get("metadata", {})
        if meta.get("smoke_build"):
            raise SystemExit(f"{d.name} is a --smoke build. Never push one.")
        if meta.get("skip_cuda_check"):
            print(f"  ! {d.name}: skip_cuda_check=true — validation build, "
                  f"numbers are not comparable to the published benchmark")

    # The mount list is baked into each task; --shards has to line up with it.
    for d in dirs:
        mounts = task_mounts(d)
        if len(mounts) != len(shards):
            raise SystemExit(
                f"{d.name} expects {len(mounts)} cache shard(s) "
                f"{mounts} but --shards gave {len(shards)}. A mismatch is "
                f"silent at run time: the path is simply absent, preflight "
                f"warns and skips, and the task runs with a partial cache."
            )

    token = access_token() if args.yes else ""
    failures = 0
    for d in dirs:
        body = build_body(d.name, args.defs, shards, task_mounts(d),
                          args.agent_slug, args.n_attempts)
        if not args.yes:
            print(f"\n--- {d.name} (DRY RUN) ---")
            print(json.dumps(body, indent=2))
            continue
        resp = push(body, token)
        # /tasks/push answers 200 with an `error` body on validation failure.
        err = resp.get("error") or resp.get("errorNullable")
        if err:
            failures += 1
            print(f"FAIL {d.name}: {err}")
        else:
            print(f"ok   {d.name}  -> {resp.get('url') or resp.get('slug')}")

    if not args.yes:
        print(f"\nDRY RUN — {len(dirs)} task(s) would be pushed. "
              f"Re-run with --yes to send.")
        return 0
    print(f"\n{len(dirs) - failures}/{len(dirs)} pushed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
