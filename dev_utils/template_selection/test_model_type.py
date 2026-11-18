"""Regression test for model_type() in the final-eval scripts (src/eval/tasks/*/evaluate*_final_eval.py): which chat
template a model gets.

Until 2026-11-17 model_type() searched the WHOLE model path for a family name before reading config.json. The final
evaluation runs on <results>/<agent>_<agent_config>_.../<benchmark>_<base model>_<cluster id>/final_model, so the
qwen_code agent with agent_config qwen/qwen4-max gave every gemma and SmolLM3 cell the qwen3 template (gemma never
stopped generating: humaneval 0.030 instead of the agent-measured 0.48). Now a local model dir goes by the
`architectures` in its config.json, and only a non-local Hugging Face id goes by its name. The agent-facing evaluate.py /
evaluate_openrouter.py keep the old rule on purpose: the agent runs them on local paths (final_model, work/...), which
name no family, so they already went by config.json (481 of 481 logged calls in that grid had the right template).

The function is taken from each file with ast (no import of inspect_ai/vllm) and run on synthetic cases. With
--results-root, the version from git HEAD (or --old-rev) and the current one also run on every <root>/*/*/final_model
that still has a config.json, and every dir where they disagree is listed with the cell's base model.

Run from the repo root:
    python3 dev_utils/template_selection/test_model_type.py
    python3 dev_utils/template_selection/test_model_type.py --results-root /fast/brank/auto-posttrain/results
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import subprocess
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FAMILIES = ("qwen", "llama", "gemma", "smollm")


def load_model_type(source: str, origin: str):
    """The model_type function of a script's source, compiled on its own with os and json in scope."""
    tree = ast.parse(source, filename=origin)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "model_type"]
    if len(funcs) != 1:
        raise SystemExit(f"{origin}: expected one model_type(), found {len(funcs)}")
    namespace = {"os": os, "json": json}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), origin, "exec"), namespace)
    return namespace["model_type"]


def run(fn, model_path: str) -> str:
    try:
        return fn(types.SimpleNamespace(model_path=model_path))
    except Exception as exc:  # an expected outcome in some cases; compared like a family name
        return f"<{type(exc).__name__}>"


def synthetic_cases(tmp: Path) -> list[tuple[str, str]]:
    def model_dir(rel: str, config: dict | None) -> str:
        d = tmp / rel
        d.mkdir(parents=True)
        if config is not None:
            (d / "config.json").write_text(json.dumps(config))
        return str(d)

    grid = "results/qwen_code_qwen_qwen4-max_xhigh_10h_run1"
    gemma = model_dir(f"{grid}/humaneval_google_gemma-3-4b-pt_1/final_model", {"architectures": ["Gemma3ForCausalLM"]})
    return [
        # the bug: the agent_config in the result path named qwen for gemma and SmolLM3 cells
        (gemma, "gemma"),
        (gemma + "/", "gemma"),
        (model_dir(f"{grid}/gsm8k_google_gemma-3-4b-pt_2/final_model",
                   {"architectures": ["Gemma3ForConditionalGeneration"]}), "gemma"),
        (model_dir(f"{grid}/gsm8k_HuggingFaceTB_SmolLM3-3B-Base_3/final_model",
                   {"architectures": ["SmolLM3ForCausalLM"]}), "smollm"),
        (model_dir(f"{grid}/gsm8k_Qwen_Qwen3-4B-Base_4/final_model", {"architectures": ["Qwen3ForCausalLM"]}), "qwen"),
        # Hugging Face ids go by their name
        ("Qwen/Qwen3-4B-Base", "qwen"),
        ("google/gemma-3-4b-pt", "gemma"),
        ("HuggingFaceTB/SmolLM3-3B-Base", "smollm"),
        ("meta-llama/Llama-3.2-3B", "llama"),
        # what vLLM cannot load fails here too, loudly
        ("someone/some-model", "<ValueError>"),
        (model_dir("results/x_10h/gsm8k_google_gemma-3-4b-pt_5/final_model", None), "<FileNotFoundError>"),
        (model_dir("results/x_10h/gsm8k_google_gemma-3-4b-pt_6/final_model", {"model_type": "gemma3_text"}),
         "<KeyError>"),
        (model_dir("results/x_10h/gsm8k_google_gemma-3-4b-pt_7/final_model", {"architectures": ["MistralForCausalLM"]}),
         "<ValueError>"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", help="compare the old and new model_type on <root>/*/*/final_model")
    parser.add_argument("--old-rev", default="HEAD", help="git revision of the old version (default HEAD)")
    args = parser.parse_args()

    files = sorted(glob.glob(str(REPO / "src/eval/tasks/*/evaluate*_final_eval.py")))
    files = [p for p in files if "def model_type(args) -> str:" in Path(p).read_text()]
    if len(files) != 8:
        raise SystemExit(f"expected 8 final-eval scripts with model_type(), found {len(files)}")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        cases = synthetic_cases(Path(tmp))
        for f in files:
            fn = load_model_type(Path(f).read_text(), f)
            for path, want in cases:
                got = run(fn, path)
                if got != want:
                    failures += 1
                    print(f"FAIL {os.path.relpath(f, REPO)}: {path!r} -> {got}, expected {want}")
    print(f"synthetic cases: {len(files)} scripts x {len(cases)} cases, {failures} failures")

    if args.results_root:
        old, new = {}, {}
        for f in files:
            rel = os.path.relpath(f, REPO)
            old_src = subprocess.run(["git", "show", f"{args.old_rev}:{rel}"], cwd=REPO, capture_output=True, text=True,
                                     check=True).stdout
            old[rel], new[rel] = load_model_type(old_src, f"{args.old_rev}:{rel}"), load_model_type(Path(f).read_text(), rel)
        all_dirs = sorted(glob.glob(os.path.join(args.results_root, "*", "*", "final_model")))
        dirs = [d for d in all_dirs if os.path.exists(os.path.join(d, "config.json"))]
        changed = {}
        for d in dirs:
            cell = os.path.basename(os.path.dirname(d)).lower()
            base = next((fam for fam in FAMILIES if fam in cell), "?")
            results = {(run(old[rel], d), run(new[rel], d)) for rel in old}
            if len({n for _, n in results}) != 1:
                failures += 1
                print(f"FAIL the 8 scripts disagree on {d}: {results}")
            o, n = sorted(results)[0]
            if n != base:
                failures += 1
                print(f"FAIL {d}: new {n}, base model {base}")
            if o != n:
                changed[d] = (o, n, base)
        print(f"real runs: {len(dirs)} final_model dirs with config.json under {args.results_root} "
              f"({len(all_dirs) - len(dirs)} without, cleaned up after their evaluation); {len(changed)} change template:")
        for d, (o, n, base) in changed.items():
            print(f"  {o:7s} -> {n:7s} (base model: {base}) {os.path.relpath(d, args.results_root)}")
    if failures:
        raise SystemExit(f"{failures} failures")
    print("PASS")


if __name__ == "__main__":
    main()
