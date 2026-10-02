# How the Harbor Adapter Works

Internals of the tasks that src/harbor_adapter generates. To run the benchmark, see the
[README](README.md); for the intended deviations from condor, see its
[Differences from condor](README.md#differences-from-condor) section.

## Task layout

```
posttrainbench-gsm8k-qwen3-1.7b/
├── task.toml              # GPU/CPU/RAM, timeouts, env vars, verifier.collect hook
├── instruction.md         # agent prompt, rendered by src/eval/general/get_prompt.py
├── environment/           # AGENT image build context (COPY . -> /home/agent/workspace)
│   ├── Dockerfile, .dockerignore, requirements-direct.txt
│   ├── entrypoint.sh, system_monitor.sh, ptb_collect.sh  # -> /usr/local/bin (not in the workspace)
│   ├── contamination_check.py, test_data.json            # -> /home/agent/ (agent self-decontamination)
│   ├── evaluate.py, templates/, timer.sh, metadata.json  # the agent's workspace
│   └── evaluation_code/                                  # (arenahardwriting, healthbench)
└── tests/                 # VERIFIER image build context, baked into /tests
    ├── Dockerfile, requirements-direct.txt, entrypoint.sh, system_monitor.sh
    ├── test.sh            # verifier: condor's judge phase, then its final evaluation
    ├── metadata.json      # adds baseline_accuracy (verifier only)
    └── ptb/               # repo slice: src/judges, src/trace_parsing, src/eval/run_final_eval.sh,
                           # src/eval/tasks/<id>/ (final-eval scripts, test set), templates, src/utils helpers
```

Condor's own get_prompt.py renders the prompt. The timer counts down from /timer_start, which the
task.toml healthcheck writes immediately before the agent starts.

## Model hand-off: shared Modal volume

The verifier runs in a separate sandbox, so the agent cannot change the evaluation, the judges or
the installed packages. The trained weights must cross the sandboxes. Harbor's artifact transfer
cannot carry them: Modal limits file downloads to 5 GiB, and every base model is larger. So
task.toml does this:

1. The launcher creates a Modal volume. Modal mounts it at /mnt/ptb_final_model in both sandboxes.
2. The agent writes final_model/ in its workspace, as on condor. Its instructions do not mention
   the volume.
3. After the agent exits, the verifier.collect hook runs ptb_collect.sh. The hook copies
   final_model/ to the volume. It also stages a size-limited snapshot of the workspace code and
   the agent transcript under /logs/artifacts, which harbor carries to the host and the verifier.
4. The verifier reads the model from the volume (PTB_MODEL_DIR).

Two warnings. Do not add an artifacts entry for the whole workspace: agents leave multi-GB
directories behind, and the transfer breaks on them. Do not mount the volume inside the workspace:
Modal shows nested mounts as symlinks, harbor's pre-upload cleanup deletes them, and an agent that
replaces final_model would then silently write off the volume.

## Verifier: judges, evaluation, reward

test.sh runs condor's judge phase (run_all_judges from src/judges/judge_lib.sh, the same function
run_task.sh calls), then condor's final evaluation, in run_task.sh's order. The judge tree, the
trace parsers and the benchmark's test set are baked into the verifier image under /tests/ptb. A
judge change on condor arrives when you generate the tasks again.

| Judge | Output id | Verdict | Runs |
|---|---|---|---|
| data_contamination_judge | gpt5_4 | contamination, disallowed_model | 3 (best-of-3: slots 2 and 3 in judgement_multi_runs/; scoring takes the per-field majority) |
| api_usage_judge | api | disallowed_api_usage | 1 |
| ptb_lookup_judge | ptb_lookup | disallowed_ptb_lookup | 1 |
| general_judge | general | general_anomaly | 1 |

All judges run gpt-5.6-terra at xhigh on codex 0.144.5 (defaults in judge_lib.sh, overridable per
judge.conf). The only Harbor-specific part is JUDGE_RUNTIME=local: codex runs directly in the
verifier image, as user nobody, with OPENAI_API_KEY, and with a 3000 s limit per run
(PTB_JUDGE_TIMEOUT_SEC). Each judge sees the condor sandbox layout: the code snapshot as its task
dir (final_model is a symlink to the volume), the traces, the test set, the checker tools and the
final model config. A judge that produces no verdict is a warning, not a failure.

The evaluation is condor's src/eval/run_final_eval.sh, run with EVAL_RUNTIME=local. It runs the
benchmark's evaluate_final_eval.py (never shown to the agent) once per fixed seed: 5 seeds for
aime2025, 3 for gpqamain, gsm8k and humaneval, 1 for arenahardwriting, healthbench and models that
decode greedily. Each seed goes through the max-tokens retry cascade. The reward is the mean
accuracy of the seeds that succeeded. The outputs go to /logs/verifier/: metrics.json (the seed
mean), evaluation/ (per-seed metrics, attempt logs, inspect_logs/), reward.txt, the judge files
and the traces. If the first seed fails at every stage, there is no metrics.json, and the reward
is the base model's zero-shot score. That is also how scripts/collect.py scores such a run.

If final_model is missing or has no config.json, the verifier skips the judges and the evaluation.
It writes the four verdicts unflagged, with a justification such as "No final model submitted",
and uses the baseline score (scripts/baselines.json, baked into the verifier's metadata.json) as
the reward. metrics.json keeps the reason in its error field.
