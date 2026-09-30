#!/usr/bin/env python3
"""Reruns the final evaluation of many result dirs in parallel: one GPU job per (run, seed) instead of one job per run
that evaluates its seeds one after the other (scripts/rerun_eval_n_times.sh). The slowest evaluations (the seeds of a
sampling gsm8k model) take about 2 h instead of 6 h.

Each seed runs `src/eval/run_final_eval.sh --single-seed` (job: scripts/rerun_final_eval_seed.sh), i.e. its own full
max-tokens cascade from stage 0; unlike run_task.sh's sequential evaluation, a later seed does not start at the stage
where the first seed succeeded, and a failed first seed does not skip the others. The seeds are the final evaluation's
defaults (run_final_eval.sh --default-seeds), and only the first one for a model that vLLM decodes greedily, as in
run_final_eval.sh. The job that finishes last averages the seeds that succeeded (src/utils/aggregate_seed_metrics.py)
into metrics_averaged.json; with no seed succeeded it writes reruns/no_seed_succeeded.txt instead.

A cell is one result dir <method>/<benchmark>_<model>_<cluster_id>. Its outputs go to <cell>/reruns/ (plan.json, per
seed: metrics_seed<S>.json, result_seed<S>.txt, final_eval_seed<S>_<N>.txt, inspect_logs/seed<S>/, seed<S>_jobs.txt)
and <cell>/metrics_averaged.json, never to metrics.json. The cell is the result dir itself (needs write access), or,
with --out-root, <out-root>/<method>/<run>/ with final_model as a symlink to the result dir's (for result dirs of
other users). Existing files are never overwritten: a cell that already has reruns/ or metrics_averaged.json is
skipped by submit.

Usage (from anywhere; runs from the repo it lives in, on the login node):
  scripts/rerun_final_eval_parallel.py submit --bid 50 [--out-root DIR] [--dry-run] SELECTION
  scripts/rerun_final_eval_parallel.py status [--out-root DIR] [-v] SELECTION
  scripts/rerun_final_eval_parallel.py retry --bid 50 [--out-root DIR] SELECTION
  scripts/rerun_final_eval_parallel.py aggregate [--if-complete] CELL_DIR...
SELECTION is run dirs as arguments and/or --results-root ROOT with --methods METHOD... and/or --agents AGENT...
(names from HARDCODED_AGENT_MAP in scripts/utils.py), which take the latest run of each (benchmark, model), the one
scripts/collect.py aggregates. --benchmarks B... restricts either. retry resubmits the seeds whose job died without a
result (and seeds planned but never submitted), after moving their files to reruns/failed_attempts/.

The jobs run the code of this checkout when they start: do not edit it while jobs are queued, or run from a copy
(e.g. `git archive`; a copy that is no git checkout needs a GIT_STATE.txt saying what it is, recorded in plan.json).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from utils import HARDCODED_AGENT_MAP, HARDCODED_BENCHMARKS, load_dotenv, walk_latest_runs  # noqa: E402

SUB_FILE = os.path.join(SCRIPT_DIR, "rerun_final_eval_parallel.sub")
CONDOR_LOG_DIR = os.path.join(REPO_ROOT, "logs", "rerun_parallel")
ENV_KEYS = ("POST_TRAIN_BENCH_CONTAINERS_DIR", "HF_HOME", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
CONDOR_STATES = {"1": "queued", "2": "running", "5": "held"}


# ---------------------------------------------------------------------------
# Selection and cells
# ---------------------------------------------------------------------------

def task_of(run_dir: str) -> str:
    return os.path.basename(run_dir.rstrip("/")).split("_")[0]


def select_runs(args: argparse.Namespace) -> list[str]:
    runs = [os.path.abspath(p) for p in args.run_dirs]
    for run in runs:
        if not os.path.isdir(run) or task_of(run) not in HARDCODED_BENCHMARKS:
            raise ValueError(f"{run} is not a result dir of a scored benchmark ({', '.join(HARDCODED_BENCHMARKS)})")
    methods = list(args.methods or [])
    for agent in args.agents or []:
        if agent not in HARDCODED_AGENT_MAP:
            raise KeyError(f"agent {agent!r} not in HARDCODED_AGENT_MAP")
        methods += HARDCODED_AGENT_MAP[agent]
    if methods and not args.results_root:
        raise ValueError("--methods/--agents need --results-root")
    for method in methods:
        method_path = os.path.join(args.results_root, method)
        if not os.path.isdir(method_path):
            raise FileNotFoundError(method_path)
        runs += [run["path"] for (bench, _), run in walk_latest_runs(method_path).items()
                 if bench in HARDCODED_BENCHMARKS]
    if not runs:
        raise ValueError("no runs selected: give run dirs or --results-root with --methods/--agents")
    order = args.benchmarks or HARDCODED_BENCHMARKS
    runs = [r for r in runs if task_of(r) in order]
    if len(set(runs)) != len(runs):
        raise ValueError("a run is selected twice")
    return sorted(runs, key=lambda r: (order.index(task_of(r)), r))


def cell_dir(run_dir: str, out_root: str | None) -> str:
    if out_root is None:
        return run_dir
    method, run = os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir)
    return os.path.join(os.path.abspath(out_root), method, run)


def model_problem(model_dir: str) -> str | None:
    """Why model_dir cannot be evaluated, or None."""
    if not os.path.isdir(model_dir):
        return "no final_model/"
    names = os.listdir(model_dir)
    if not any(n.endswith((".safetensors", ".bin")) for n in names):
        return f"no weight files in final_model/ ({', '.join(sorted(names)[:3])})"
    for root, _, files in os.walk(model_dir):
        for name in files:
            if not os.access(os.path.join(root, name), os.R_OK):
                return f"unreadable: {os.path.join(root, name)}"
    return None


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def eval_env() -> dict[str, str]:
    env = dict(os.environ)
    dotenv = load_dotenv()
    for key in ENV_KEYS:
        if key in dotenv and key not in env:
            env[key] = dotenv[key]
    return env


def default_seeds(task: str, env: dict[str, str]) -> tuple[list[int], bool]:
    """(the task's default final-eval seeds, whether its seed only drives sampling)."""
    out = subprocess.run(["bash", "src/eval/run_final_eval.sh", "--default-seeds", task], cwd=REPO_ROOT, env=env,
                         check=True, capture_output=True, text=True).stdout.split("\n")
    return [int(s) for s in out[0].split()], out[1].strip() == "1"


def decodings(model_dirs: list[str], env: dict[str, str]) -> dict[str, list]:
    """{model_dir: [mode, temperature]} from src/utils/default_temperature.py --batch-output, in the eval container."""
    if not model_dirs:
        return {}
    container = os.path.join(env["POST_TRAIN_BENCH_CONTAINERS_DIR"], "vllm_debug.sif")
    os.makedirs(CONDOR_LOG_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CONDOR_LOG_DIR) as tmp:
        out = os.path.join(tmp, "decodings.json")
        proc = subprocess.run(["apptainer", "exec", "--env", "PYTHONNOUSERSITE=1", "--bind", f"{REPO_ROOT}:{REPO_ROOT}",
                               "--bind", f"{tmp}:{tmp}", "--pwd", REPO_ROOT, container, "python",
                               "src/utils/default_temperature.py", "--batch-output", out] + model_dirs,
                              cwd=REPO_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"default_temperature.py failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
        return json.load(open(out))


def plan_seeds(runs: list[str], env: dict[str, str]) -> dict[str, tuple[list[int], list]]:
    """{run_dir: (seeds, decoding)} as run_final_eval.sh would pick them."""
    task_seeds = {t: default_seeds(t, env) for t in sorted({task_of(r) for r in runs})}
    need_check = [os.path.realpath(os.path.join(r, "final_model")) for r in runs
                  if task_seeds[task_of(r)][1] and len(task_seeds[task_of(r)][0]) > 1]
    dec = decodings(need_check, env)
    plans = {}
    for run in runs:
        seeds, seed_only_samples = task_seeds[task_of(run)]
        model = os.path.realpath(os.path.join(run, "final_model"))
        decoding = dec.get(model, ["not checked", None])
        if decoding[0] == "greedy":
            seeds = seeds[:1]
        elif decoding[0] == "unknown":
            print(f"WARNING: default temperature of {model} unknown ({decoding[1]}); all seeds", file=sys.stderr)
        plans[run] = (seeds, decoding)
    return plans


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def queued_seeds() -> dict[tuple[str, int], str]:
    out = subprocess.run(["condor_q", os.environ["USER"], "-constraint", "PtbRerunCell =!= undefined", "-af:t",
                          "PtbRerunCell", "PtbRerunSeed", "JobStatus"],
                         check=True, capture_output=True, text=True, timeout=300).stdout
    queued = {}
    for line in out.splitlines():
        cell, seed, status = line.split("\t")
        key = (cell, int(seed))
        if key in queued:
            raise RuntimeError(f"{cell} seed {seed} is in the queue twice")
        queued[key] = CONDOR_STATES.get(status, f"condor_status_{status}")
    return queued


def seed_state(cell: str, seed: int, queued: dict[tuple[str, int], str]) -> str:
    reruns = os.path.join(cell, "reruns")
    result = os.path.join(reruns, f"result_seed{seed}.txt")
    if os.path.exists(result):
        return "failed" if open(result).read().strip() == "failed" else "done"
    if (cell, seed) in queued:
        return queued[(cell, seed)]
    if glob.glob(os.path.join(reruns, f"*_seed{seed}[._]*")) or os.path.exists(os.path.join(reruns, "inspect_logs",
                                                                                            f"seed{seed}")):
        return "job_failed"
    return "pending"


def cell_state(run: str, cell: str, queued: dict[tuple[str, int], str]) -> tuple[str, dict[int, str]]:
    if os.path.exists(os.path.join(cell, "metrics_averaged.json")):
        return "done", {}
    plan_path = os.path.join(cell, "reruns", "plan.json")
    if not os.path.exists(plan_path):
        problem = model_problem(os.path.join(run, "final_model"))
        return ("cannot_run: " + problem if problem else "not_planned"), {}
    if os.path.exists(os.path.join(cell, "reruns", "no_seed_succeeded.txt")):
        return "no_seed_succeeded", {}
    seeds = {s: seed_state(cell, s, queued) for s in json.load(open(plan_path))["seeds"]}
    return "in_progress", seeds


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def git_state() -> dict[str, object]:
    """The code the jobs run: the commit, or for a copy that is no git checkout (e.g. a `git archive` export), the
    GIT_STATE.txt it must hold (what it was made from)."""
    if not os.path.exists(os.path.join(REPO_ROOT, ".git")):
        state_file = os.path.join(REPO_ROOT, "GIT_STATE.txt")
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"{REPO_ROOT} is no git checkout and has no GIT_STATE.txt saying what code it is")
        return {"export": open(state_file).read().strip(), "uncommitted_changes": False}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, check=True,
                           capture_output=True, text=True).stdout.strip() != ""
    return {"commit": commit, "uncommitted_changes": dirty}


def submit_jobs(jobs: list[tuple[str, int]], bid: int, label: str) -> None:
    if not 0 < bid <= 1000:
        raise ValueError(f"bid {bid} out of range")
    os.makedirs(CONDOR_LOG_DIR, exist_ok=True)
    job_list = os.path.join(CONDOR_LOG_DIR, f"{datetime.datetime.now():%Y%m%dT%H%M%S}_{label}_jobs.txt")
    if os.path.exists(job_list):
        raise FileExistsError(job_list)
    with open(job_list, "w") as f:
        f.writelines(f"{cell} {seed}\n" for cell, seed in jobs)
    subprocess.run(["condor_submit_bid", str(bid), "-a", f"job_list={job_list}", SUB_FILE], cwd=REPO_ROOT, check=True)
    print(f"submitted {len(jobs)} jobs (list: {job_list})")


def cmd_submit(args: argparse.Namespace) -> None:
    runs = select_runs(args)
    skipped = collections.Counter()
    todo = []
    for run in runs:
        cell = cell_dir(run, args.out_root)
        problem = model_problem(os.path.join(run, "final_model"))
        if problem:
            skipped[f"cannot run ({problem.split(':')[0].split(' (')[0]})"] += 1
            print(f"skip {run}: {problem}")
        elif any(os.path.lexists(os.path.join(cell, n)) for n in ("reruns", "metrics_averaged.json")):
            skipped["already planned or done (see status/retry)"] += 1
        elif args.out_root is None and not os.access(run, os.W_OK):
            raise PermissionError(f"{run} is not writable; use --out-root")
        else:
            todo.append(run)
    plans = plan_seeds(todo, eval_env())
    jobs = [(cell_dir(run, args.out_root), seed) for run in todo for seed in plans[run][0]]
    per_bench = collections.Counter(task_of(r) for r in todo)
    print(f"{len(runs)} runs selected; {len(todo)} to evaluate {dict(per_bench)} as {len(jobs)} seed jobs; "
          f"skipped: {dict(skipped) or 'none'}")
    if args.dry_run or not jobs:
        return
    code = git_state()
    if code["uncommitted_changes"]:
        print(f"WARNING: {REPO_ROOT} has uncommitted changes; the jobs run them", file=sys.stderr)
    for run in todo:
        cell = cell_dir(run, args.out_root)
        if args.out_root is not None:
            os.makedirs(cell, exist_ok=True)
            link = os.path.join(cell, "final_model")
            target = os.path.join(run, "final_model")
            if os.path.lexists(link):
                if os.readlink(link) != target:
                    raise RuntimeError(f"{link} does not point to {target}")
            else:
                os.symlink(target, link)
        os.mkdir(os.path.join(cell, "reruns"))
        seeds, decoding = plans[run]
        plan = {"task": task_of(run), "seeds": seeds, "decoding": decoding, "source_run_dir": run,
                "code": {"repo": REPO_ROOT, **code}, "created": datetime.datetime.now().isoformat(timespec="seconds")}
        with open(os.path.join(cell, "reruns", "plan.json"), "w") as f:
            json.dump(plan, f, indent=2)
    submit_jobs(jobs, args.bid, "submit")


def cmd_status(args: argparse.Namespace) -> None:
    queued = queued_seeds()
    cells = collections.Counter()
    seeds = collections.Counter()
    attention = []
    for run in select_runs(args):
        cell = cell_dir(run, args.out_root)
        state, seed_states = cell_state(run, cell, queued)
        cells[(task_of(run), state.split(":")[0])] += 1
        for seed, s in seed_states.items():
            seeds[s] += 1
            if s in ("job_failed", "pending", "held"):
                attention.append(f"{s:10} {cell} seed {seed}")
        if args.verbose and state not in ("done", "in_progress"):
            print(f"{state:20} {cell}")
    states = sorted({s for _, s in cells})
    print("benchmark".ljust(12) + "".join(s.rjust(19) for s in states))
    for bench in HARDCODED_BENCHMARKS:
        if any(b == bench for b, _ in cells):
            print(bench.ljust(12) + "".join(str(cells[(bench, s)]).rjust(19) for s in states))
    print("seeds of cells in progress: " + (", ".join(f"{s} {n}" for s, n in sorted(seeds.items())) or "none"))
    if attention:
        print(f"\nneed retry or attention ({len(attention)}):")
        print("\n".join("  " + a for a in attention))


def cmd_retry(args: argparse.Namespace) -> None:
    queued = queued_seeds()
    jobs = []
    stamp = f"{datetime.datetime.now():%Y%m%dT%H%M%S}"
    for run in select_runs(args):
        cell = cell_dir(run, args.out_root)
        state, seed_states = cell_state(run, cell, queued)
        if state != "in_progress":
            continue
        if all(s in ("done", "failed") for s in seed_states.values()):
            aggregate(cell, if_complete=False)
            continue
        for seed, s in seed_states.items():
            if s not in ("job_failed", "pending"):
                continue
            reruns = os.path.join(cell, "reruns")
            leftovers = glob.glob(os.path.join(reruns, f"*_seed{seed}[._]*"))
            inspect_dir = os.path.join(reruns, "inspect_logs", f"seed{seed}")
            if os.path.exists(inspect_dir):
                leftovers.append(inspect_dir)
            if leftovers:
                attempt = os.path.join(reruns, "failed_attempts", f"{stamp}_seed{seed}_retry")
                os.makedirs(attempt)
                for path in leftovers:
                    shutil.move(path, attempt)
            jobs.append((cell, seed))
    print(f"{len(jobs)} seeds to resubmit")
    if jobs:
        submit_jobs(jobs, args.bid, "retry")


def aggregate(cell: str, if_complete: bool) -> None:
    reruns = os.path.join(cell, "reruns")
    plan = json.load(open(os.path.join(reruns, "plan.json")))
    output = os.path.join(cell, "metrics_averaged.json")
    results = {}
    for seed in plan["seeds"]:
        path = os.path.join(reruns, f"result_seed{seed}.txt")
        if not os.path.exists(path):
            if if_complete:
                return
            raise FileNotFoundError(f"{path}: seed {seed} has no result yet")
        results[seed] = open(path).read().split()
    try:
        os.mkdir(os.path.join(reruns, ".aggregate_lock"))  # atomic: only one of the last seeds' jobs aggregates
    except FileExistsError:
        if if_complete:
            return
        raise
    if os.path.exists(output):
        raise FileExistsError(output)
    succeeded = [(seed, r[1]) for seed, r in results.items() if r[0] == "stage"]
    if not succeeded:
        with open(os.path.join(reruns, "no_seed_succeeded.txt"), "w") as f:
            f.write(f"all {len(results)} seeds failed at every stage: {sorted(results)}\n")
        print(f"{cell}: no seed succeeded")
        return
    seed_args = []
    for seed, stage in succeeded:
        seed_args += ["--seed-result", str(seed), stage, os.path.join(reruns, f"metrics_seed{seed}.json")]
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "src", "utils", "aggregate_seed_metrics.py"),
                    "--output", output, "--num-seeds", str(len(results))] + seed_args, check=True)
    print(f"wrote {output}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    for cell in args.cells:
        aggregate(os.path.abspath(cell), args.if_complete)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("submit", "status", "retry"):
        p = sub.add_parser(name)
        p.add_argument("run_dirs", nargs="*")
        p.add_argument("--results-root")
        p.add_argument("--methods", nargs="+")
        p.add_argument("--agents", nargs="+")
        p.add_argument("--benchmarks", nargs="+", choices=HARDCODED_BENCHMARKS)
        p.add_argument("--out-root", help="write cells to <out-root>/<method>/<run>/ instead of the result dirs")
        if name in ("submit", "retry"):
            p.add_argument("--bid", type=int, required=True)
        if name == "submit":
            p.add_argument("--dry-run", action="store_true")
        if name == "status":
            p.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("aggregate")
    p.add_argument("--if-complete", action="store_true", help="do nothing unless every seed has a result")
    p.add_argument("cells", nargs="+")
    args = parser.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "retry": cmd_retry, "aggregate": cmd_aggregate}[args.cmd](args)


if __name__ == "__main__":
    main()
