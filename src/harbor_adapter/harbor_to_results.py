#!/usr/bin/env python3
"""Export Harbor trials into the PostTrainBench results-directory layout.

Harbor's verifier reports the *pre-fallback* accuracy as the reward. The
baseline fallback for judge-flagged runs, aggregation, flagged-run review and
judge reruns all live in the condor-side tooling (scripts/collect.py,
scripts/find_flagged_runs.py, src/judges/run_judges.sh). This script writes
each Harbor trial as a result dir those tools accept unchanged:

    <results_dir>/<agent>_<agent_model>_<N>h[<experiment>]/<benchmark>_<Org>_<Model>_<run_id>/
        metrics.json                 verifier/metrics.json (accuracy, stderr)
        judgement_<id>.json          per-judge verdicts (gpt5_4, api, ptb_lookup, general)
        judge_output_<id>.json/.txt  raw + parsed judge traces (+ *_sanitized companions)
        solve_out.txt / solve_parsed.txt (+ *_sanitized)   the agent transcript
        prompt.txt                   the instruction the agent received
        time_taken.txt               agent phase duration, H:M:S
        cli_version.txt              agent CLI version (from harbor's agent_info)
        final_eval_<n>.txt           evaluation attempts
        system_monitor.log, output.log, error.log
        task/                        the agent's code snapshot
        harbor/                      result.json + config.json (provenance)
        final_model/                 only with --with-model (fetched from the run's Modal volume)

Naming follows src/run_task.sh: the method dir is <agent>_<agent_config>_<N>h
with '/', ':', '[', ']' in the agent model replaced by '_' (harbor agent
`claude-code` on `anthropic/claude-opus-4-8` for 1h ->
`claude-code_anthropic_claude-opus-4-8_1h`); the run dir is
<benchmark>_<model id with '/' -> '_'>_<run_id> with run_id = the trial's
start time as Unix seconds (an int, monotonic, so "latest run wins" and it
clears scripts/utils.py NEWER_JUDGES_MIN_RUN_ID).

Traces are (re)parsed on the host with src/trace_parsing/parse_trace.py so the
*_sanitized companions are produced against the real .env secrets.

Usage:
    python harbor_to_results.py jobs/gsm8k-1h-2                # a job dir (all trials)
    python harbor_to_results.py jobs/gsm8k-1h-2/<trial_dir>    # one trial
    python harbor_to_results.py jobs/* --experiment-name _harbor --with-model
    python harbor_to_results.py jobs/gsm8k-1h-2 --output /fast/me/ptb_results --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADAPTER_DIR.parent.parent
PARSE_TRACE = REPO_ROOT / "src" / "trace_parsing" / "parse_trace.py"

JUDGE_OUTPUT_IDS = ("gpt5_4", "api", "ptb_lookup", "general")


# --------------------------------------------------------------------------- helpers

def load_dotenv_value(name: str, default: str) -> str:
    """POST_TRAIN_BENCH_* value: environment first, then the repo .env."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == name:
                return os.path.expandvars(value.strip().strip('"').strip("'"))
    return default


def safe_name(value: str) -> str:
    """run_task.sh: tr '/:[]' '____'."""
    return re.sub(r"[/:\[\]]", "_", value)


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def find_trials(paths: list[Path]) -> list[Path]:
    """Accept job dirs and/or trial dirs."""
    trials = []
    for p in paths:
        p = p.resolve()
        if (p / "result.json").is_file() and (p / "config.json").is_file() and (p / "verifier").is_dir():
            trials.append(p)
            continue
        for sub in sorted(p.iterdir()) if p.is_dir() else []:
            if sub.is_dir() and (sub / "result.json").is_file() and (sub / "config.json").is_file():
                trials.append(sub)
    return trials


def resolve_task_dir(trial: Path, result: dict) -> Path | None:
    """The task dir the trial was generated from (for metadata.json + instruction.md)."""
    raw = (result.get("task_id") or {}).get("path") or result["config"]["task"]["path"]
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates += [ADAPTER_DIR / raw, Path.cwd() / raw, trial.parent.parent / raw]
    for c in candidates:
        if (c / "task.toml").is_file():
            return c.resolve()
    return None


def trial_metadata(trial: Path, result: dict, task_dir: Path | None) -> dict:
    """benchmark_id, model_id, num_hours — from the task's metadata.json, else parsed."""
    if task_dir and (task_dir / "tests" / "metadata.json").is_file():
        return json.loads((task_dir / "tests" / "metadata.json").read_text())
    # Fallback: posttrainbench-<benchmark>-<model_short>
    from adapter import MODELS  # noqa: E402  (same directory)
    name = result["task_name"]
    m = re.match(r"^(?:posttrainbench|smoke-transfer)-(?P<bench>[a-z0-9]+)-(?P<model>.+)$", name)
    if not m:
        raise ValueError(f"{trial}: cannot infer benchmark/model from task name {name!r}")
    model_key = m.group("model")
    if model_key not in MODELS:
        raise ValueError(f"{trial}: unknown model short name {model_key!r}")
    return {
        "benchmark_id": m.group("bench"),
        "model_id": MODELS[model_key].model_id,
        "num_hours": None,
    }


def run_parse_trace(agent_name: str, src: Path, dst: Path) -> None:
    """Parsed transcript + sanitized companions next to src and dst (needs REPO_ROOT/.env)."""
    if not PARSE_TRACE.is_file():
        print(f"    WARNING: {PARSE_TRACE} not found; skipping trace parsing")
        return
    proc = subprocess.run(
        [sys.executable, str(PARSE_TRACE), "--agent", agent_name, str(src), "-o", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 and not dst.is_file():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["?"]
        print(f"    WARNING: parse_trace.py failed for {src.name}: {tail[0]}")


# --------------------------------------------------------------------------- export

def export_trial(trial: Path, results_dir: Path, *, experiment_name: str,
                 with_model: bool, volume: str | None, overwrite: bool, dry_run: bool) -> Path | None:
    result = json.loads((trial / "result.json").read_text())
    config = json.loads((trial / "config.json").read_text())
    task_dir = resolve_task_dir(trial, result)
    meta = trial_metadata(trial, result, task_dir)

    agent_name = (result.get("agent_info") or {}).get("name") or config["agent"]["name"]
    agent_model = config["agent"].get("model_name") or (result.get("agent_info") or {}).get("model_info", {}).get("name") or "unknown"
    agent_version = (result.get("agent_info") or {}).get("version") or "unknown"

    started = parse_ts(result["started_at"])
    run_id = int(started.timestamp())
    num_hours = meta.get("num_hours")
    if num_hours is None and task_dir:
        m = re.search(r"^\[agent\]\s*\ntimeout_sec = ([0-9.]+)", (task_dir / "task.toml").read_text(), re.M)
        num_hours = int(float(m.group(1)) // 3600) if m else 0
    num_hours = num_hours or 0

    method_dir = f"{agent_name}_{safe_name(agent_model)}_{num_hours}h{experiment_name}"
    run_name = f"{meta['benchmark_id']}_{safe_name(meta['model_id'])}_{run_id}"
    out = results_dir / method_dir / run_name

    exc = result.get("exception_info")
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    print(f"{trial.name}")
    print(f"  -> {out}")
    print(f"     benchmark={meta['benchmark_id']} model={meta['model_id']} agent={agent_name}/{agent_model} "
          f"reward={reward} exception={exc['exception_type'] if exc else None}")
    if dry_run:
        return out
    if out.exists():
        if not overwrite:
            print("     exists, skipping (use --overwrite)")
            return out
        shutil.rmtree(out)
    out.mkdir(parents=True)

    v = trial / "verifier"
    art = trial / "artifacts" / "logs" / "artifacts"

    # metrics + judges (verbatim from the verifier)
    for name in ["metrics.json", "reward.txt"]:
        if (v / name).is_file():
            shutil.copy(v / name, out / name)
    for jid in JUDGE_OUTPUT_IDS:
        for name in [f"judgement_{jid}.json", f"judge_output_{jid}.json"]:
            if (v / name).is_file():
                shutil.copy(v / name, out / name)
        if (out / f"judge_output_{jid}.json").is_file():
            run_parse_trace("codex", out / f"judge_output_{jid}.json", out / f"judge_output_{jid}.txt")
    for f in sorted(v.glob("final_eval_*.txt")):
        shutil.copy(f, out / f.name)

    # agent transcript: harbor's /logs/agent/<agent>.txt, else the largest staged log
    raw = trial / "agent" / f"{agent_name}.txt"
    if not (raw.is_file() and raw.stat().st_size > 0):
        staged = sorted((art / "agent_logs").glob("*.txt"), key=lambda p: p.stat().st_size, reverse=True) if (art / "agent_logs").is_dir() else []
        raw = staged[0] if staged else None
    if raw:
        shutil.copy(raw, out / "solve_out.txt")
        run_parse_trace(agent_name, out / "solve_out.txt", out / "solve_parsed.txt")
    else:
        print("     WARNING: no agent transcript found")

    # prompt, timing, cli version, monitors
    if task_dir and (task_dir / "instruction.md").is_file():
        shutil.copy(task_dir / "instruction.md", out / "prompt.txt")
    ae = result.get("agent_execution") or {}
    if ae.get("started_at") and ae.get("finished_at"):
        secs = int((parse_ts(ae["finished_at"]) - parse_ts(ae["started_at"])).total_seconds())
        (out / "time_taken.txt").write_text(hms(secs) + "\n")
    (out / "cli_version.txt").write_text(
        f"binary: {agent_name}\npackage: harbor:{agent_name}\npath: harbor\n"
        f"version: {agent_version}\nupdate: harbor-managed\n"
        f"recorded_at: {result.get('finished_at', '')}\n"
    )
    if (trial / "agent" / "system_monitor.log").is_file():
        shutil.copy(trial / "agent" / "system_monitor.log", out / "system_monitor.log")
    if (v / "test-stdout.txt").is_file():
        shutil.copy(v / "test-stdout.txt", out / "output.log")
    if (trial / "trial.log").is_file():
        shutil.copy(trial / "trial.log", out / "error.log")

    # code snapshot -> task/
    if (art / "workspace").is_dir():
        shutil.copytree(art / "workspace", out / "task")
    elif (trial / "artifacts" / "workspace").is_dir():   # pre-snapshot layout
        shutil.copytree(trial / "artifacts" / "workspace", out / "task",
                        ignore=shutil.ignore_patterns("final_model"))
    else:
        print("     WARNING: no code snapshot in artifacts")

    # provenance
    (out / "harbor").mkdir()
    shutil.copy(trial / "result.json", out / "harbor" / "result.json")
    shutil.copy(trial / "config.json", out / "harbor" / "config.json")

    # weights (optional): fetched from the run's Modal volume
    if with_model:
        vol = volume or default_volume_name(trial, result)
        harbor_py = Path(shutil.which("harbor") or "").resolve().parent / "python"
        modal = [str(harbor_py), "-m", "modal"] if harbor_py.is_file() else ["modal"]
        print(f"     fetching final_model from volume {vol} ...")
        proc = subprocess.run(modal + ["volume", "get", vol, "/", str(out / "final_model")],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"     WARNING: volume fetch failed: {(proc.stderr or proc.stdout).strip()[-200:]}")
    return out


def default_volume_name(trial: Path, result: dict) -> str:
    """run_modal_task.sh convention: ptb-<job-name>-<task short name>."""
    job = trial.parent.name
    short = re.sub(r"^posttrainbench-", "", result["task_name"])
    return re.sub(r"[^A-Za-z0-9._-]", "-", f"ptb-{job}-{short}")[:64]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path, help="harbor job dir(s) and/or trial dir(s)")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="results dir (default: POST_TRAIN_BENCH_RESULTS_DIR from env/.env, else ./results)")
    ap.add_argument("--experiment-name", default="",
                    help="suffix appended to the method dir, like POST_TRAIN_BENCH_EXPERIMENT_NAME (e.g. _harbor)")
    ap.add_argument("--with-model", action="store_true", help="also fetch final_model from the run's Modal volume")
    ap.add_argument("--volume", default=None, help="volume name for --with-model (default: run_modal_task.sh convention)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (REPO_ROOT / ".env").is_file():
        print("NOTE: no .env in the repo root — traces are parsed, but the *_sanitized companions "
              "(API-key redaction, src/trace_parsing/sanitize_trace.py) are skipped. "
              "`cp example.env .env` and fill in the keys to produce them.\n")
    results_dir = args.output or Path(load_dotenv_value("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
    if not results_dir.is_absolute():
        results_dir = (REPO_ROOT / results_dir).resolve()
    trials = find_trials(args.paths)
    if not trials:
        sys.exit("no harbor trials found under the given paths")
    print(f"results dir: {results_dir}\n")
    for trial in trials:
        export_trial(trial, results_dir, experiment_name=args.experiment_name,
                     with_model=args.with_model, volume=args.volume,
                     overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
