#!/usr/bin/env python3
"""Check that the Harbor adapter still matches the condor pipeline.

The adapter reads most of PostTrainBench straight from the repo at task
generation (prompt, benchmarks, evaluate.py, test sets, the judges' code and
prompts, trace parsers, baselines). What it cannot read, it mirrors, and those
mirrors drift silently when condor changes. This script compares each mirror
with its condor source and exits 1 on any mismatch:

  images      agent Dockerfile vs the agent .def, verifier Dockerfile vs the
              eval .def (run_final_eval.sh) and the judge .def (judge_lib.sh)
  env         constant --env values run_task.sh / run_final_eval.sh /
              judge_lib.sh pass to the agent, eval and judge containers vs the
              Dockerfiles' ENV
  eval        test.sh runs condor's src/eval/run_final_eval.sh (seeds, retry
              cascade, seed average) with EVAL_RUNTIME=local, no ladder of its own;
              humaneval's answers run in with_answer_sandbox_local.sh, which keeps
              every isolation property of the apptainer answer sandbox
  judges      test.sh runs condor's run_all_judges (no re-implementation);
              the verdict fields test.sh writes vs the judge prompts;
              the exporter's multi-run dir vs scripts/utils.py
  launch      agents/claude/solve.sh env and flags vs run_modal_task.sh, and
              update_agent_cli.sh's npm packages vs the wrapper's
  timer       condor's and Harbor's timer.sh print the same
  scope       scored benchmarks and expected models vs the adapter
  generate    every task generates; its shell scripts and task.toml parse; the
              verifier slice has every src/eval module its final-eval scripts
              import (per_sample_seed.py, exact_numeric_match.py, ...)

No GPU, API keys or network needed. Missing (gitignored) test_data.json files
are replaced by placeholders for the duration of the run and removed again.

Usage (Python >= 3.11 for tomllib):
    python3 src/harbor_adapter/check_parity.py
    uv run --no-project --python 3.12 python src/harbor_adapter/check_parity.py
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.version_info < (3, 11):
    sys.exit("check_parity.py needs Python >= 3.11 (tomllib); e.g. "
             "uv run --no-project --python 3.12 python src/harbor_adapter/check_parity.py")
import tomllib  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "src" / "harbor_adapter"
AGENT_DF = ADAPTER / "template" / "environment" / "Dockerfile"
VERIFIER_DF = ADAPTER / "template" / "tests" / "Dockerfile"
TEST_SH = ADAPTER / "template" / "tests" / "test.sh"
WRAPPER = ADAPTER / "run_modal_task.sh"
EXPORTER = ADAPTER / "harbor_to_results.py"
RUN_TASK = REPO / "src" / "run_task.sh"
FINAL_EVAL = REPO / "src" / "eval" / "run_final_eval.sh"
LOCAL_SANDBOX = REPO / "src" / "eval" / "tasks" / "humaneval" / "with_answer_sandbox_local.sh"
JUDGES = REPO / "src" / "judges"
JUDGE_LIB = JUDGES / "judge_lib.sh"
UTILS_PY = REPO / "scripts" / "utils.py"
CLAUDE_SOLVE = REPO / "agents" / "claude" / "solve.sh"
UPDATE_CLI = REPO / "src" / "utils" / "update_agent_cli.sh"
CREATE_TIMER = REPO / "src" / "utils" / "create_timer.sh"
# condor picks the agent container per run (.env POST_TRAIN_BENCH_CONTAINER_NAME);
# this is the one the Harbor agent image mirrors (README "Known gotchas").
AGENT_DEF = REPO / "containers" / "opus_5.def"

FAILURES: list[str] = []


def fail(check: str, msg: str) -> None:
    FAILURES.append(f"[{check}] {msg}")


def expect_equal(check: str, what: str, condor, harbor, condor_src: str, harbor_src: str) -> None:
    if condor != harbor:
        fail(check, f"{what}: condor {condor!r} ({condor_src}) != harbor {harbor!r} ({harbor_src})")


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def shell_function(text: str, name: str) -> str:
    """Body of `name() { ... }` in a bash script (up to the first line that is `}`)."""
    m = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.S | re.M)
    if not m:
        raise SystemExit(f"check_parity.py: function {name}() not found — update the checker")
    return m.group(1)


def source_judge_lib(expr: str) -> str:
    """Evaluate a bash expression after sourcing judge_lib.sh (it only defines things)."""
    return subprocess.run(["bash", "-c", f'source "$1" > /dev/null 2>&1 && echo {expr}', "_", str(JUDGE_LIB)],
                          capture_output=True, text=True, check=True).stdout.strip()


# ----------------------------------------------------------------- images

def image_spec(path: Path) -> dict:
    """The build inputs that must agree between a condor .def and a Dockerfile."""
    t = path.read_text()
    one = lambda rx: (m.group(1) if (m := re.search(rx, t, re.M)) else None)  # noqa: E731
    npm = dict(re.findall(r"((?:@[a-z0-9-]+/)?[a-z0-9][a-z0-9.-]*)@(\d[0-9A-Za-z.+-]*)", t))
    return {
        "base image": one(r"^\s*(?:From:|FROM)\s*(\S+)"),
        "vllm": one(r"vllm==([0-9][\w.]*)"),
        "torch backend": one(r"--torch-backend=(\w+)"),
        "flash-attn": one(r"flash-attn==([0-9][\w.]*)"),
        "inspect_evals commit": one(r"git checkout ([0-9a-f]{7,40})"),
        "inspect_ai repo": (one(r"git clone (\S*inspect_ai_vllm_stdout\S*)") or "").removesuffix(".git"),
        "requirements-direct.txt": "-r /opt/requirements-direct.txt" in t,
        "npm": npm,
    }


def expected_torch_backend(base_image: str | None, def_backend: str | None) -> str | None:
    """condor builds with --torch-backend=auto, which resolves to the CUDA build of
    the base image's driver (nvidia/cuda:12.9.1-... -> cu129). Modal has no GPU at
    build time, so Harbor must spell that value out."""
    if def_backend != "auto":
        return def_backend
    m = re.search(r"cuda:(\d+)\.(\d+)", base_image or "")
    return f"cu{m.group(1)}{m.group(2)}" if m else None


def check_ml_stack(check: str, condor_def: Path, dockerfile: Path) -> None:
    c, h = image_spec(condor_def), image_spec(dockerfile)
    for key in ("base image", "vllm", "flash-attn", "inspect_evals commit", "inspect_ai repo", "requirements-direct.txt"):
        expect_equal(check, key, c[key], h[key], rel(condor_def), rel(dockerfile))
    want = expected_torch_backend(c["base image"], c["torch backend"])
    if h["torch backend"] != want:
        fail(check, f"torch backend: {rel(dockerfile)} has {h['torch backend']!r}, condor's "
                    f"{c['torch backend']!r} on {c['base image']} means {want!r}")


def check_images() -> None:
    check_ml_stack("images/agent", AGENT_DEF, AGENT_DF)
    c, h = image_spec(AGENT_DEF)["npm"], image_spec(AGENT_DF)["npm"]
    expect_equal("images/agent", "agent CLI pins (npm)", c, h, rel(AGENT_DEF), rel(AGENT_DF))

    eval_sif = re.search(r'^\s*EVAL_CONTAINER="\$\{POST_TRAIN_BENCH_CONTAINERS_DIR\}/(\w+)\.sif"', FINAL_EVAL.read_text(), re.M)
    if not eval_sif:
        fail("images/verifier", f"cannot find the eval container in {rel(FINAL_EVAL)} (EVAL_CONTAINER=...<name>.sif)")
    else:
        check_ml_stack("images/verifier", REPO / "containers" / f"{eval_sif.group(1)}.def", VERIFIER_DF)

    judge_def = REPO / "containers" / source_judge_lib('"${JUDGE_CONTAINER%.sif}.def"')
    expect_equal("images/verifier", "judge codex (@openai/codex)",
                 image_spec(judge_def)["npm"].get("@openai/codex"), image_spec(VERIFIER_DF)["npm"].get("@openai/codex"),
                 rel(judge_def), rel(VERIFIER_DF))


# -------------------------------------------------------------------- env

def literal_envs(text: str) -> list[tuple[str, str]]:
    """Every constant `--env NAME="value"` (skips "$VAR" values and empty strings). A list,
    not a dict: a variable set in several places must match in each of them."""
    return re.findall(r'--env ([A-Z_][A-Z0-9_]*)="([^"$]+)"', text)


def dockerfile_env(path: Path) -> dict[str, str]:
    return {k: v for k, v in re.findall(r'^ENV ([A-Z_][A-Z0-9_]*)="?([^"\n]*)"?$', path.read_text(), re.M)}


def check_env() -> None:
    rt, jl = RUN_TASK.read_text(), JUDGE_LIB.read_text()
    agent_env = dockerfile_env(AGENT_DF)
    for k, v in literal_envs(shell_function(rt, "solve_task")):
        expect_equal("env/agent", k, v, agent_env.get(k), f"{rel(RUN_TASK)} solve_task", rel(AGENT_DF))
    gpus = str(tomllib.loads((ADAPTER / "template" / "task.toml").read_text())["environment"]["gpus"])
    expect_equal("env/agent", "NUM_GPUS (condor passes the job's GPU count)", gpus, agent_env.get("NUM_GPUS"),
                 "template/task.toml [environment] gpus", rel(AGENT_DF))

    fe = FINAL_EVAL.read_text()
    judge_block = re.search(r"^JUDGE_EXTRA_APPTAINER_ARGS=\((.*?)^\)", rt, re.S | re.M)
    verifier_src = (shell_function(fe, "run_evaluation") + shell_function(fe, "probe_default_temperature")
                    + (judge_block.group(1) if judge_block else "") + shell_function(jl, "run_judge_exec"))
    verifier_env = dockerfile_env(VERIFIER_DF)
    for k, v in literal_envs(verifier_src):
        expect_equal("env/verifier", k, v, verifier_env.get(k),
                     f"{rel(FINAL_EVAL)} / {rel(RUN_TASK)} / {rel(JUDGE_LIB)}", rel(VERIFIER_DF))


# ------------------------------------------------------------------- eval

def code_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def check_eval() -> None:
    ts, fe = code_lines(TEST_SH.read_text()), FINAL_EVAL.read_text()
    # the evaluation itself (<task> <model_dir> ...), not just the --check at job start
    if not re.search(r'^[^#\n]*src/eval/run_final_eval\.sh "\$\{EVALUATION_TASK\}" "[^"]*final_model"', RUN_TASK.read_text(), re.M):
        fail("eval", f"{rel(RUN_TASK)} no longer evaluates via src/eval/run_final_eval.sh; the Harbor verifier relies on it")
    if 'EVAL_RUNTIME="${EVAL_RUNTIME:-apptainer}"' not in fe or '"${EVAL_RUNTIME}" = "local"' not in fe:
        fail("eval", f"{rel(FINAL_EVAL)} lost its EVAL_RUNTIME=local path, which the Harbor verifier runs")
    if not re.search(r"EVAL_RUNTIME=local\b.*\n?.*run_final_eval\.sh", ts):
        fail("eval", f"{rel(TEST_SH)} must run the final evaluation via EVAL_RUNTIME=local ... run_final_eval.sh")
    for own in ("run_evaluation_with_retry", "--max-tokens", "--max-new-tokens", "evaluate.py", "evaluate_final_eval.py"):
        if own in ts:
            fail("eval", f"{rel(TEST_SH)} evaluates on its own (`{own}`); use condor's run_final_eval.sh")

    # humaneval's answer sandbox under EVAL_RUNTIME=local: present, used, and as isolated as the apptainer one
    # (with_answer_sandbox.sh: no network, own PID namespace, none of the host's files, clean environment).
    if not LOCAL_SANDBOX.is_file():
        fail("eval", f"{rel(LOCAL_SANDBOX)} is missing; humaneval cannot be evaluated with EVAL_RUNTIME=local")
    else:
        if 'answer_sandbox=(bash "${REPO_ROOT}/src/eval/tasks/humaneval/with_answer_sandbox_local.sh")' not in code_lines(fe):
            fail("eval", f"{rel(FINAL_EVAL)} no longer runs humaneval through {LOCAL_SANDBOX.name} with EVAL_RUNTIME=local")
        sbx = code_lines(LOCAL_SANDBOX.read_text())
        for flag, what in (("--user", "user namespace"), ("--net", "no network"), ("--pid", "own PID namespace"),
                           ("--mount", "own mount namespace"), ('exec chroot "$ROOT"', "allow-list root"),
                           ('mount -o remount,bind,ro "$ROOT/$d"', "read-only system dirs"),
                           ('mount -o remount,bind,ro "$ROOT/answer_sandbox.py"', "read-only answer server"),
                           ('exec env -i PATH="$PATH"', "clean environment for the sandbox"),
                           ("/usr/bin/env -i", "clean environment for the answer server"),
                           ("--reuid=65534", "drops root"), ("answer_sandbox.py serve", "condor's answer server")):
            if flag not in sbx:
                fail("eval", f"{rel(LOCAL_SANDBOX)} lost `{flag}` ({what}); it must stay as isolated as "
                             f"with_answer_sandbox.sh")


# ----------------------------------------------------------------- judges

def check_judges() -> None:
    ts = TEST_SH.read_text()
    code = code_lines(ts)
    for needed in ('source "$JUDGES_DIR/judge_lib.sh"', "prepare_judge_sandbox ", "setup_judge_codex_auth ",
                   "run_all_judges "):
        if needed not in ts:
            fail("judges", f"{rel(TEST_SH)} must use condor's judge phase; missing `{needed.strip()}`")
    for forbidden in ("codex exec", "--search", "model_reasoning_effort"):
        if forbidden in code:
            fail("judges", f"{rel(TEST_SH)} invokes codex itself (`{forbidden}`); judges must run via run_all_judges")
    if "run_all_judges " not in RUN_TASK.read_text():
        fail("judges", f"{rel(RUN_TASK)} no longer calls run_all_judges; the Harbor verifier relies on it")

    # The local judge runner must drop root before codex runs: the verifier
    # container is root, and a prompt-injected judge with root could write the
    # verifier's metrics.json, the baked eval scripts or the model volume's
    # files (condor's apptainer sandbox rules that out structurally).
    lib = JUDGE_LIB.read_text()
    local_exec = lib[lib.index("run_judge_exec_local()"):]
    for needed in ("drop_privs=(setpriv --reuid=65534 --regid=65534 --clear-groups --inh-caps=-all)",
                   'chown -R -h 65534:65534 "$job_dir"',
                   'cannot drop root'):
        if needed not in local_exec:
            fail("judges", f"{rel(JUDGE_LIB)} run_judge_exec_local no longer drops root (`{needed}` missing); "
                           "a root judge could overwrite the verifier's metrics, eval scripts or model volume")
    if local_exec.index("drop_privs=(setpriv") > local_exec.index('"${JUDGE_CODEX_ARGS[@]}"'):
        fail("judges", f"{rel(JUDGE_LIB)} run_judge_exec_local: the privilege drop must come before the codex call")

    # verdict fields test.sh writes when no model is submitted vs the judge prompts
    written = {jid: set(re.findall(r'"(\w+)"', fields))
               for jid, fields in re.findall(r'"(\w+)": \(([^)]*)\)', ts)}
    prompts = {}
    for judge in source_judge_lib('"${ALL_JUDGES[*]}"').split():
        conf = (JUDGES / judge / "judge.conf").read_text()
        jid = re.search(r'^JUDGE_OUTPUT_ID="([^"]+)"', conf, re.M).group(1)
        pfile = re.search(r'^JUDGE_PROMPT_FILE="([^"]+)"', conf, re.M).group(1)
        prompts[jid] = set(re.findall(r'"(contamination|disallowed_\w+|general_anomaly)"',
                                      (JUDGES / judge / pfile).read_text()))
    expect_equal("judges", "no-model verdict fields per judge", prompts, written,
                 "src/judges/*/judge.conf + prompts", f"{rel(TEST_SH)} write_no_model_results")

    def str_const(path: Path, name: str):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return ast.literal_eval(node.value)
        return None
    expect_equal("judges", "MULTI_RUN_DIRNAME", str_const(UTILS_PY, "MULTI_RUN_DIRNAME"),
                 str_const(EXPORTER, "MULTI_RUN_DIRNAME"), rel(UTILS_PY), rel(EXPORTER))


# ----------------------------------------------------------------- launch

# claude flags harbor's claude-code agent always passes itself.
HARBOR_BUILTIN_CLAUDE_FLAGS = {"--print", "--verbose", "--model", "--output-format", "--dangerously-skip-permissions"}


def check_launch() -> None:
    solve, wrapper = CLAUDE_SOLVE.read_text(), WRAPPER.read_text()
    for name, value in re.findall(r'^export ([A-Z_]+)="([^"]*)"', solve, re.M):
        if name == "CLAUDE_CODE_EFFORT_LEVEL":
            ok = f'EFFORT="{value}"' in wrapper
        elif name == "BASH_MAX_TIMEOUT_MS":
            ok = f"BASH_MAX_TIMEOUT_MS={value}" in wrapper
        else:
            fail("launch", f"{rel(CLAUDE_SOLVE)} exports {name}={value!r}, which has no Harbor mapping — "
                           f"add it to run_modal_task.sh and to this check")
            continue
        if not ok:
            fail("launch", f"{rel(CLAUDE_SOLVE)} sets {name}={value!r}; {rel(WRAPPER)} does not default to it")

    cmd = re.search(r"claude --print(.*?)(?:\n\s*\n|\Z)", solve, re.S).group(0).replace("\\\n", " ")
    tokens = cmd.split()
    for i, tok in enumerate(tokens):
        if not tok.startswith("--") or tok in HARBOR_BUILTIN_CLAUDE_FLAGS:
            continue
        if tok == "--thinking-display":
            if f'THINKING_DISPLAY="{tokens[i + 1]}"' not in wrapper:
                fail("launch", f"{rel(CLAUDE_SOLVE)} passes --thinking-display {tokens[i + 1]}; "
                               f"{rel(WRAPPER)} does not default to it")
        else:
            fail("launch", f"{rel(CLAUDE_SOLVE)} passes {tok}, which has no Harbor mapping — "
                           f"add it to run_modal_task.sh and to this check")

    condor_pkgs = set(re.findall(r'\)\s+PKG="([^"]+)"', UPDATE_CLI.read_text()))
    harbor_pkgs = set(re.findall(r'CLI_PKG="([^"]+)"', wrapper)) - {""}
    expect_equal("launch", "agent CLI npm packages", condor_pkgs, harbor_pkgs, rel(UPDATE_CLI), rel(WRAPPER))


# ------------------------------------------------------------------ timer

def check_timer(adapter_mod) -> None:
    """Both timers on the same fake clock (a `date` shim answering `+%s`): started at T0,
    read at fixed elapsed times."""
    t0 = 1_800_000_000
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        shim = d / "bin"
        shim.mkdir()
        (shim / "date").write_text('#!/bin/sh\n[ "$1" = "+%s" ] && { echo "$FAKE_NOW"; exit 0; }\nexec /bin/date "$@"\n')
        (shim / "date").chmod(0o755)
        env = lambda now: {"PATH": f"{shim}:/usr/bin:/bin", "FAKE_NOW": str(now)}  # noqa: E731
        for hours in (1, 10):
            condor_timer = d / f"condor_{hours}.sh"
            subprocess.run(["bash", str(CREATE_TIMER), str(hours), str(condor_timer)], check=True, env=env(t0))
            start = d / f"start_{hours}"
            start.write_text(f"{t0}\n")
            out = d / f"harbor_{hours}"
            out.mkdir()
            adapter_mod.PostTrainBenchAdapter(output_dir=d, num_hours=hours).generate_timer_sh(out)
            harbor_timer = out / "timer.sh"
            harbor_timer.write_text(harbor_timer.read_text().replace('START_FILE="/timer_start"', f'START_FILE="{start}"'))
            for elapsed in (0, 600, hours * 3600 - 59, hours * 3600, hours * 3600 + 1):
                c = subprocess.run(["bash", str(condor_timer)], capture_output=True, text=True, env=env(t0 + elapsed)).stdout
                h = subprocess.run(["bash", str(harbor_timer)], capture_output=True, text=True, env=env(t0 + elapsed)).stdout
                expect_equal("timer", f"output {elapsed} s into {hours} h", c, h, rel(CREATE_TIMER), "adapter.py generate_timer_sh")


# ------------------------------------------------------------------ scope

def check_scope(adapter_mod) -> None:
    consts = {t.id: node.value for node in ast.parse(UTILS_PY.read_text()).body if isinstance(node, ast.Assign)
              for t in node.targets if isinstance(t, ast.Name)}
    scored = set(ast.literal_eval(consts["HARDCODED_BENCHMARKS"]))
    missing = scored - set(adapter_mod.BENCHMARKS)
    if missing:
        fail("scope", f"scored benchmarks with no Harbor task (adapter SKIP_BENCHMARKS or no info.json): {sorted(missing)}")
    expect_equal("scope", "`all` benchmarks", sorted(scored & set(adapter_mod.BENCHMARKS)),
                 sorted(adapter_mod.DEFAULT_BENCHMARKS), rel(UTILS_PY), "adapter.py DEFAULT_BENCHMARKS")
    expected_models = set(ast.literal_eval(consts["EXPECTED_MODELS"]))
    harbor_models = {m.model_id.split("/")[-1] for m in adapter_mod.MODELS.values()}
    expect_equal("scope", "base models", sorted(expected_models), sorted(harbor_models),
                 f"{rel(UTILS_PY)} EXPECTED_MODELS", "adapter.py MODELS")


# --------------------------------------------------------------- generate

def imported_modules(text: str) -> set[str]:
    """Top-level module names a Python source imports (import X / from X import ...)."""
    mods = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            mods.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return mods


def check_generate(adapter_mod) -> None:
    with tempfile.TemporaryDirectory() as d:
        gen = adapter_mod.PostTrainBenchAdapter(output_dir=Path(d), num_hours=1)
        for bench in adapter_mod.BENCHMARKS:
            for model in adapter_mod.MODELS:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        task = gen.generate_task(bench, model)
                except Exception as e:  # noqa: BLE001 — report and keep going
                    fail("generate", f"{bench} x {model}: {type(e).__name__}: {e}")
                    continue
                for script in ("tests/test.sh", "environment/ptb_collect.sh", "environment/entrypoint.sh",
                               "environment/timer.sh"):
                    r = subprocess.run(["bash", "-n", str(task / script)], capture_output=True, text=True)
                    if r.returncode:
                        fail("generate", f"{task.name}/{script}: bash -n: {r.stderr.strip()}")
                try:
                    tomllib.loads((task / "task.toml").read_text())
                except tomllib.TOMLDecodeError as e:
                    fail("generate", f"{task.name}/task.toml: {e}")
                if not (task / "instruction.md").read_text().strip():
                    fail("generate", f"{task.name}/instruction.md is empty")
                ptb = task / "tests" / "ptb"
                for needed in ("src/eval/run_final_eval.sh", f"src/eval/tasks/{bench}/evaluate_final_eval.py",
                               f"src/eval/tasks/{bench}/test_data.json", "src/eval/templates",
                               "src/utils/aggregate_seed_metrics.py", "src/utils/default_temperature.py",
                               "src/judges/judge_lib.sh"):
                    if not (ptb / needed).exists():
                        fail("generate", f"{task.name}: verifier slice lacks ptb/{needed}")
                if bench == "humaneval" and not (ptb / "src/eval/tasks/humaneval/with_answer_sandbox_local.sh").exists():
                    fail("generate", f"{task.name}: verifier slice lacks humaneval's with_answer_sandbox_local.sh")
                # The final-eval scripts import shared modules from src/eval/ (sys.path two levels up from the
                # task dir). One missing from the slice fails every evaluation at import time, and the reward
                # silently falls back to the baseline.
                for script_py in sorted((ptb / "src/eval/tasks" / bench).glob("*.py")):
                    for mod in sorted(imported_modules(script_py.read_text())):
                        if (REPO / "src/eval" / f"{mod}.py").is_file() and not (ptb / "src/eval" / f"{mod}.py").is_file():
                            fail("generate", f"{task.name}: {script_py.name} imports `{mod}` but the verifier "
                                             f"slice lacks ptb/src/eval/{mod}.py")
                # Build contexts are uploaded to Modal on every image build: no local junk (gitignored eval
                # logs, caches), and nothing near the size of a stray log dump.
                for ctx in ("tests", "environment"):
                    files = [f for f in (task / ctx).rglob("*") if f.is_file()]
                    junk = [f for f in files if {"logs", "__pycache__"} & set(f.relative_to(task / ctx).parts)]
                    size = sum(f.stat().st_size for f in files)
                    if junk:
                        fail("generate", f"{task.name}/{ctx}: contains local junk, e.g. {junk[0].relative_to(task)}")
                    if size > 200 * 1024 * 1024:
                        fail("generate", f"{task.name}/{ctx}: build context is {size / 2**20:.0f} MiB (> 200 MiB)")


# ------------------------------------------------------------------- main

def main() -> int:
    sys.path.insert(0, str(ADAPTER))
    placeholders = []
    try:
        import adapter as adapter_mod
        for bench in adapter_mod.BENCHMARKS:
            p = adapter_mod.TASKS_ROOT / bench / "test_data.json"
            if not p.exists():  # gitignored; CI has none. Never touch real ones.
                p.write_text(json.dumps([{"id": 0, "question": "placeholder", "answer": "placeholder"}]))
                placeholders.append(p)
        checks = [("images", check_images), ("env", check_env), ("eval", check_eval), ("judges", check_judges),
                  ("launch", check_launch), ("timer", lambda: check_timer(adapter_mod)),
                  ("scope", lambda: check_scope(adapter_mod)), ("generate", lambda: check_generate(adapter_mod))]
        for name, fn in checks:
            before = len(FAILURES)
            fn()
            print(f"{'FAIL' if len(FAILURES) > before else 'ok  '} {name}")
    finally:
        for p in placeholders:
            p.unlink(missing_ok=True)
    if FAILURES:
        print(f"\n{len(FAILURES)} condor/Harbor mismatch(es):")
        for f in FAILURES:
            print(f"  - {f}")
        print("\nFix the Harbor side (or the condor side) so they agree, or update this check if the "
              "difference is intended (and document it in src/harbor_adapter/README.md).")
        return 1
    print("\ncondor and Harbor agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
