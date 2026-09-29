#!/usr/bin/env python3
"""Audit official HumanEval final-eval logs for samples scored correct only because the process exited before check().

The upstream inspect_evals scorer counted any exit code 0 as correct. This re-runs every sample that a log marks
correct through the upstream verification program plus a line that prints a random token after check(). A sample is
flagged `early_exit` when that program exits 0 without printing the token. `passes_without_exit` then says whether
the answer passes check() once its own exits are swallowed: False means the logged score was inflated by that sample,
True means it was untested but correct anyway.

Final-eval logs are the inspect logs of run_task.sh's official evaluation. They are written to
<submitting checkout>/src/eval/tasks/humaneval/logs/ and named in the run dir's final_eval*.txt ("Log: logs/..."), or
in evaluation/final_eval_seed*_*.txt for runs with the seeded final evaluation (one log per seed).

Steps:
  1. list   (login node)       find each humaneval run dir's final-eval logs; writes a TSV
  2. check  (compute node, in the eval container, since it executes model-generated code): submit audit.sub
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import resource
import secrets
import signal
import subprocess
import tempfile

VERIFY_TIMEOUT = 30
FINAL_EVAL_LOG_RE = re.compile(r"Log: (logs/\S+\.json)")
MISSING = "MISSING:"


def cmd_list(args: argparse.Namespace) -> None:
    rows = []
    run_dirs = []
    for results_dir in args.results_dirs:
        assert os.path.isdir(results_dir), results_dir
        run_dirs += sorted(glob.glob(os.path.join(results_dir, "*", "humaneval_*")))
    for run_dir in run_dirs:
        final_eval_txts = (glob.glob(os.path.join(run_dir, "final_eval*.txt"))
                           + glob.glob(os.path.join(run_dir, "evaluation", "final_eval*.txt")))
        for final_eval_txt in sorted(final_eval_txts):
            with open(final_eval_txt, errors="replace") as f:
                referenced = FINAL_EVAL_LOG_RE.findall(f.read())
            for rel in referenced:
                found = [os.path.join(d, rel) for d in args.eval_dirs if os.path.isfile(os.path.join(d, rel))]
                assert len(found) <= 1, found
                rows.append((run_dir, found[0] if found else MISSING + rel))
    with open(args.out, "w") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    with_log = {r[0] for r in rows if not r[1].startswith(MISSING)}
    scored = {d for d in run_dirs if os.path.isfile(os.path.join(d, "metrics.json"))}
    unchecked_out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "unchecked_scored_runs.txt")
    with open(unchecked_out, "w") as f:
        f.writelines(os.path.normpath(d) + "\n" for d in sorted(scored - with_log))
    n_found = sum(1 for r in rows if not r[1].startswith(MISSING))
    print(f"humaneval run dirs: {len(run_dirs)}, scored: {len(scored)}")
    print(f"final-eval logs: {n_found} found (in {len(with_log)} run dirs), {len(rows) - n_found} referenced but missing")
    print(f"scored run dirs without a surviving final-eval log: {len(scored - with_log)} -> {unchecked_out}")
    print(f"wrote {args.out}")


def build_verification_code(prompt: str, answer: str, test: str, entry_point: str, token: str) -> str:
    """The upstream inspect_evals verification program, plus a line that prints `token` once check() has returned."""
    return "".join([prompt, answer, "\n", test, "\n", f"check({entry_point})", "\n", f"print({token!r})", "\n"])


def build_exit_neutralized_code(prompt: str, answer: str, test: str, entry_point: str, token: str) -> str:
    """Like build_verification_code(), but exits raised while the answer's own module-level code runs
    (unittest.main(), exit(), os._exit(), ...) are swallowed, so check() still runs."""
    return "".join([
        "import os as _ptb_os, sys as _ptb_sys\n",
        "_ptb_os._exit = _ptb_sys.exit\n",
        "try:\n",
        f"    exec(compile({prompt + answer!r}, '<answer>', 'exec'))\n",
        "except SystemExit:\n",
        "    pass\n",
        test, "\n", f"check({entry_point})", "\n", f"print({token!r})", "\n",
    ])


def run_python(code: str, cwd: str) -> tuple[int | None, str]:
    """Returns (returncode, stdout); returncode None means timeout."""
    def limits() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))

    proc = subprocess.Popen(
        ["python", "-c", code], cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, start_new_session=True, preexec_fn=limits,
    )
    try:
        stdout, _ = proc.communicate(timeout=VERIFY_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return None, ""
    # Reap anything the program left running in its session.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return proc.returncode, stdout.decode(errors="replace")


def resolve(sample: dict, value: str) -> str:
    if value.startswith("attachment://"):
        return sample["attachments"][value[len("attachment://"):]]
    return value


def check_log(path: str, cwd: str) -> dict:
    with open(path) as f:
        log = json.load(f)
    summary = {"status": log["status"], "n_samples": 0, "n_correct": 0, "flagged": []}
    for sample in log["samples"]:
        (score,) = sample["scores"].values()
        answer = resolve(sample, score["answer"])
        summary["n_samples"] += 1
        if score["value"] != "C":
            continue
        summary["n_correct"] += 1
        prompt, test, entry_point = (resolve(sample, sample["metadata"][k]) for k in ("prompt", "test", "entry_point"))
        token = secrets.token_hex(16)
        fixed_rc, fixed_stdout = run_python(build_verification_code(prompt, answer, test, entry_point, token), cwd)
        if fixed_rc == 0 and token in fixed_stdout:
            continue
        upstream_rc, _ = run_python("".join([prompt, answer, "\n", test, "\n", f"check({entry_point})"]), cwd)
        flag = {"id": sample["id"], "epoch": sample["epoch"], "answer": answer, "fixed_rc": fixed_rc,
                "upstream_rc": upstream_rc, "stdout_tail": fixed_stdout[-500:],
                "kind": "early_exit" if fixed_rc == 0 and upstream_rc == 0 else "fails_on_rerun"}
        if flag["kind"] == "early_exit":
            token = secrets.token_hex(16)
            rc, stdout = run_python(build_exit_neutralized_code(prompt, answer, test, entry_point, token), cwd)
            flag["passes_without_exit"] = rc == 0 and token in stdout
        summary["flagged"].append(flag)
    logged_accuracy = log["results"]["scores"][0]["metrics"]["accuracy"]["value"]
    assert abs(logged_accuracy - summary["n_correct"] / summary["n_samples"]) < 1e-9, (path, logged_accuracy)
    return summary


def cmd_check(args: argparse.Namespace) -> None:
    with open(args.log_list) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory() as cwd:
        for i, (run_dir, path) in enumerate(r for r in rows if not r[1].startswith(MISSING)):
            record = {"run_dir": os.path.normpath(run_dir), "log": path, **check_log(path, cwd)}
            records.append(record)
            print(f"[{i + 1}] flagged={len(record['flagged'])} {path}", flush=True)

    affected = sorted({r["run_dir"] for r in records if any(f["kind"] == "early_exit" for f in r["flagged"])})

    with open(os.path.join(args.out_dir, "results.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "per_log.tsv"), "w") as f:
        f.write("run_dir\tlog\tn_samples\tn_correct\tn_early_exit\tn_inflated\tn_fails_on_rerun\n")
        for r in records:
            early = [fl for fl in r["flagged"] if fl["kind"] == "early_exit"]
            n_inflated = sum(1 for fl in early if not fl["passes_without_exit"])
            n_rerun_fail = sum(1 for fl in r["flagged"] if fl["kind"] == "fails_on_rerun")
            f.write(f"{r['run_dir']}\t{r['log']}\t{r['n_samples']}\t{r['n_correct']}\t{len(early)}\t{n_inflated}"
                    f"\t{n_rerun_fail}\n")
    with open(os.path.join(args.out_dir, "affected_runs.txt"), "w") as f:
        f.writelines(d + "\n" for d in affected)

    print(f"final-eval logs checked: {len(records)} (from {len({r['run_dir'] for r in records})} run dirs)")
    print(f"flagged samples: {sum(len(r['flagged']) for r in records)}")
    print(f"affected run dirs: {len(affected)} -> {args.out_dir}/affected_runs.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--results-dirs", nargs="+", required=True)
    p.add_argument("--eval-dirs", nargs="+", required=True,
                   help="<checkout>/src/eval/tasks/humaneval dirs that 'Log: logs/...' paths resolve against")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--log-list", required=True)
    p.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    {"list": cmd_list, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
