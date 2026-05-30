# scripts

Post-hoc analysis utilities for PostTrainBench result directories. Most scripts
here read the contents of `$POST_TRAIN_BENCH_RESULTS_DIR` and produce CSV /
JSON aggregates; the exception is `rerun_eval_n_times.sh`, which actually
re-runs the model on a GPU.

## Aggregating results into CSVs

The pipeline is two scripts: `collect.py` reads raw run dirs into per-method
CSVs, then `aggregate.py` rolls those into per-agent avg/std and the weighted
leaderboard metric.

### Typical flow

From the repo root, with `POST_TRAIN_BENCH_RESULTS_DIR` pointing at the raw
results tree:

```bash
# 1. Collect raw per-run data into per-method CSVs.
#    Reads metrics.json + contamination/disallowed_model judgements + time_taken.txt,
#    applies baseline-zeroshot fallback for contaminated/errored cells.
#    Writes:
#      final_{method}.csv          — score grid (model x benchmark) with fallback
#      contamination_{method}.csv  — flags ("", "C", "M", "MC", or error string)
#      time_overview.csv           — average wall time per method
python scripts/collect.py

# 2. Aggregate across runs/agents and compute the weighted leaderboard metric.
#    Reads final_{method}.csv produced above. Writes:
#      aggregated_avg_{agent}.csv  — per-cell mean across runs (one per multi-run agent)
#      aggregated_std_{agent}.csv  — per-cell sample stddev (n-1)
#      single_metrics.csv          — weighted score per individual run
#      single_metrics_aggregated.csv  — agent-level avg/std/n on the weighted metric
#      time_aggregated.csv         — agent-level avg/std wall time
python scripts/aggregate.py
```

`aggregate.py` skips agents whose run CSVs aren't present in this results
dir, so it's safe to run against a partial tree.

### `collect.py` flags

```bash
python scripts/collect.py \
    --data-dir /path/to/results \      # default: $POST_TRAIN_BENCH_RESULTS_DIR
    --output-dir /path/to/out \        # default: same as --data-dir
    --min-run-id 17000000 \            # inclusive lower bound on cluster_id
    --max-run-id 17200000              # exclusive upper bound on cluster_id
```

### `aggregate.py` flags

By default `--all` is implied (write everything). Use the flags below to
restrict to one stage:

```bash
python scripts/aggregate.py --per-cell      # only aggregated_avg/std_{agent}.csv
python scripts/aggregate.py --leaderboard   # only single_metrics{,_aggregated}.csv
python scripts/aggregate.py --time          # only time_aggregated.csv
```

Same `--data-dir` / `--output-dir` flags as `collect.py`.

### Hardcoded things

| File | What it pins |
|---|---|
| `constants.py` (`HARDCODED_AGENT_MAP`) | Which run directories belong to which agent (multi-run agents are how stddev is computed) |
| `constants.py` (`HARDCODED_BENCHMARKS`, `EXPECTED_MODELS`) | Benchmark + base-model lists |
| `factors.json` | Per-benchmark weights for the weighted leaderboard metric |
| `baselines.json` | Hardcoded zero-shot + few-shot baseline scores; used as fallback for contaminated/errored cells (no longer recomputed at every run) |

To add a new agent: add its run-dir names to `HARDCODED_AGENT_MAP` in
`constants.py`. To add a new benchmark: extend `HARDCODED_BENCHMARKS` and add
a weight to `factors.json`.

### `verify.py` (refactor regression check)

`verify.py` is a one-off script used when the new pipeline was
rolled out — it compares two CSV output dirs cell-by-cell with float
tolerance, used to confirm the new pipeline matches the old one byte-for-byte
(except for filename renames). Not part of the normal workflow.

```bash
python scripts/verify.py \
    --ground-truth /fast/.../ptb_results_old \
    --new-output   /fast/.../ptb_results_new
```

## Other helpers

| Script | Description |
|---|---|
| `compute_claude_costs.py` | Claude API spend rollup |
| `extract_token_usage.py` | Token-usage extraction from agent traces |
| `migrate_judgement_files.py` | One-off: migrate older judgement file naming |
| `list_safetensors.py` | List safetensors files under a result tree |
| `parse_all_to_human_readable.sh` | Run human-readable trace parsers across results |
| `baselines.json`, `factors.json`, `constants.py`, `utils.py` | Shared config / helpers |

## Re-evaluating a finished run N times

`rerun_eval_n_times.sh` re-evaluates a job's `final_model/` N times and writes
mean / std / stderr / min / max per metric into `metrics_averaged.json`. Useful
because each job's standard `metrics.json` is a single decoding sample per
question and does not capture decoding noise.

It mirrors `src/run_task.sh`'s evaluation step exactly:

- runs `src/eval/tasks/<task>/evaluate.py` (the live source — **not** the
  potentially-modified snapshot in `<EVAL_DIR>/task/`)
- inside the same `${POST_TRAIN_BENCH_CONTAINER_NAME}.sif` container
- with the same fuse-overlayfs HF cache pattern (`with_huggingface_overlay`)
- using the same `--max-tokens` fallback ladder per task

Per-run JSONs are written to `<EVAL_DIR>/reruns/run_{i}.json` (with
`run_{i}_{level}.log` alongside). The aggregated file is `<EVAL_DIR>/metrics_averaged.json`.

### Files

| File | Description |
|---|---|
| `rerun_eval_n_times.sh` | Driver: re-runs `evaluate.py` N times on one EVAL_DIR and aggregates |
| `aggregate_metrics_runs.py` | Helper called by the driver: computes mean/std/stderr/min/max from per-run JSONs |
| `../src/commit_utils/rerun_eval.sub` | HTCondor submission file |

### Usage

#### Locally on a GPU node

From the repo root:

```bash
scripts/rerun_eval_n_times.sh /path/to/EVAL_DIR 5
```

`EVAL_DIR` must be an existing job directory containing `final_model/`. The
task name is parsed from the basename (`<task>_<model_safe>_<cluster_id>`) to
pick the correct max-tokens fallback ladder.

#### HTCondor 

```bash
condor_submit_bid 50 \
  -a "eval_dir=/path/to/EVAL_DIR" \
  -a "n=5" \
  src/commit_utils/rerun_eval.sub
```
