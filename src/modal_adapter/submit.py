#!/usr/bin/env python3
"""Submit PostTrainBench runs to the deployed Modal app.

Modal analogue of src/commit_utils/commit.sh: one spawned run_agent call per
(model, eval task) cell. The calls run server-side on Modal, so this script
(and your machine) can exit immediately after submitting.

Example:
    python src/modal_adapter/submit.py \
        --agent claude --agent-config claude-opus-4-5 \
        --eval gsm8k --model Qwen/Qwen3-1.7B-Base --num-hours 1

Requires `modal deploy src/modal_adapter/app.py` first. Every submission is
appended to modal_runs.jsonl at the repo root for status.py.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import (  # noqa: E402
    APP_NAME,
    HF_RO_MOUNT,
    MAX_NUM_HOURS,
    RESULTS_MOUNT,
    build_spec,
    eval_dir_rel,
    hf_cache,
    keys_secret,
    results,
)

LEDGER = Path(__file__).resolve().parent.parent.parent / "modal_runs.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="agent dir under agents/, e.g. claude, codex, gemini, opencode")
    parser.add_argument("--agent-config", required=True, help="model passed to the agent CLI, e.g. claude-opus-4-5")
    parser.add_argument("--eval", dest="evals", action="append", required=True,
                        help="task id under src/eval/tasks/ (repeatable)")
    parser.add_argument("--model", dest="models", action="append", required=True,
                        help="model to post-train, e.g. Qwen/Qwen3-4B-Base (repeatable)")
    parser.add_argument("--num-hours", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--experiment-name", default="",
                        help="suffix for the method dir, like POST_TRAIN_BENCH_EXPERIMENT_NAME")
    parser.add_argument("--prompt-variant", default="prompt",
                        help="prompt template name under src/eval/general/ (POST_TRAIN_BENCH_PROMPT)")
    parser.add_argument("--dry-run", action="store_true", help="print what would be submitted and exit")
    args = parser.parse_args()

    if args.num_hours > MAX_NUM_HOURS:
        sys.exit(
            f"error: --num-hours {args.num_hours} exceeds the Modal adapter limit of {MAX_NUM_HOURS}h "
            "(single Modal Function calls are capped at 24h)."
        )

    run_agent = modal.Function.from_name(APP_NAME, "run_agent")
    if args.num_gpus > 1:
        # with_options replaces volumes/secrets rather than extending them,
        # so they must be re-specified alongside the resource overrides.
        run_agent = run_agent.with_options(
            gpu=f"H100:{args.num_gpus}",
            cpu=min(8 * args.num_gpus, 64),
            memory=min(65536 * args.num_gpus, 262144),
            volumes={
                HF_RO_MOUNT: hf_cache.with_mount_options(read_only=True),
                RESULTS_MOUNT: results,
            },
            secrets=[keys_secret],
        )

    run_id_base = int(time.time())
    submitted = []
    i = 0
    for model in args.models:
        for eval_task in args.evals:
            spec = build_spec(
                eval_task=eval_task,
                agent=args.agent,
                agent_config=args.agent_config,
                model_to_train=model,
                run_id=run_id_base + i,
                num_hours=args.num_hours,
                num_gpus=args.num_gpus,
                experiment_name=args.experiment_name,
                prompt_variant=args.prompt_variant,
            )
            i += 1
            if args.dry_run:
                print(f"would submit: {eval_dir_rel(spec)}")
                continue
            call = run_agent.spawn(spec)
            record = {
                "run_id": spec["run_id"],
                "call_id": call.object_id,
                "eval_dir": eval_dir_rel(spec),
                "spec": spec,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(LEDGER, "a") as f:
                f.write(json.dumps(record) + "\n")
            submitted.append(record)
            print(f"submitted {eval_dir_rel(spec)}  (call {call.object_id})")

    if submitted:
        print(f"\n{len(submitted)} run(s) submitted; ledger: {LEDGER}")
        print("Watch progress: python src/modal_adapter/status.py  |  modal app logs posttrainbench")


if __name__ == "__main__":
    main()
