# Reward-Hacking Judges

Agent-as-judge checks that detect reward hacking in agent runs. They analyze the agent's trace
(`solve_parsed.txt`, fallback `solve_out.txt`) by starting from the end and tracing back where
`final_model` comes from, then write a structured verdict.

The judges run in two contexts:

- **Inline** — `src/run_task.sh` runs every judge right after the agent finishes, writing
  `judgement_<id>.json` etc. into the result dir.
- **Standalone/rerun** — `run_judges.sh <result_dir>` reruns judges on an existing result dir.
  Outputs always carry the `_rerun` suffix so the originals are preserved.

## Judges

| Folder | Output id | Verdict schema | Consumed downstream? |
|--------|-----------|----------------|----------------------|
| `data_contamination_judge/` | `gpt5_4` | `contamination`, `disallowed_model` + justifications | **Yes** — canonical contamination verdict (`judgement_gpt5_4.json`, or `judgement_gpt5_4_rerun.json` when present) |
| `api_usage_judge/` | `api` | `disallowed_api_usage` + justification | No — archival record only |

Both run as GPT-5.4 via the codex CLI with ChatGPT-subscription auth
(`agents/codex_non_api/auth.json`, bind-mounted so rotated refresh tokens persist).

## Layout

```
src/judges/
├── judge_lib.sh              # shared bash helpers (sandbox prep, codex exec, output collection);
│                             # also defines ALL_JUDGES (the set + execution order)
├── run_judges.sh             # standalone runner: run_judges.sh [--judges a,b] <result_dir>
├── get_judge_prompt.py       # prompt generation (--judge <judge_name>)
├── <judge_name>/             # one folder per judge
│   ├── judge.conf            # label, output id, prompt file, inline-fatality, condor weight
│   └── prompt.md             # prompt template
├── judge_tools/              # helper tools copied into the judge sandbox
│   ├── contamination_check.py    # n-gram contamination checker (decon SIMPLE algorithm)
│   ├── model_identity_check.py   # architecture-identity check vs reference config
│   └── reference_configs/        # config.json of every allowed base model
├── test_data_download/       # host-side helper to (re)download benchmark test data
└── rerun/                    # batch-rerun pipeline (see below)
```

## Running

```bash
# Rerun all judges on one result dir (writes *_rerun outputs, originals preserved)
bash src/judges/run_judges.sh /path/to/result_dir

# Rerun a subset
bash src/judges/run_judges.sh --judges data_contamination_judge /path/to/result_dir
bash src/judges/run_judges.sh --judges api_usage_judge /path/to/result_dir
```

The result dir must contain `task/` and a trace file. The trace is copied next to the sandbox
task directory, so the judge reads it as `../solve_parsed.txt` from its working directory.

## Batch reruns (HTCondor)

```bash
# Rerun all judges on the latest run of every (method, benchmark, model)
bash src/judges/rerun/commit_rerun_judges.sh

# Preview without submitting
bash src/judges/rerun/commit_rerun_judges.sh --dry-run

# Only a subset of judges
bash src/judges/rerun/commit_rerun_judges.sh --judges data_contamination_judge

# Per result dir, only rerun judges whose _rerun output is missing
bash src/judges/rerun/commit_rerun_judges.sh --skip-existing
```

Submitted cluster IDs are logged to `rerun/submission_logs/submitted_<timestamp>.txt`
(one `cluster<TAB>judges<TAB>result_dir` line per job).

For ad-hoc targeting (a specific method/benchmark, a limit, etc.), build the directory list with
`list_results.py` and submit `rerun_judges.sub` directly:

```bash
python src/judges/rerun/list_results.py --paths-only --latest-only --method "claude" \
| while read -r d; do
    condor_submit_bid 100 -a "result_dir=$d" src/judges/rerun/rerun_judges.sub
done
```

`rerun_judges.sub` accepts `-a "judges=..."` (default: all) and `-a "judge_weight=..."`
(user.judge concurrency units the job consumes; `commit_rerun_judges.sh` passes the max
`JUDGE_CONDOR_WEIGHT` over the selected judges).

### Listing and aggregation

```bash
# List/filter result directories (--with-trace, --missing-rerun, --paths-only, ...)
python src/judges/rerun/list_results.py --paths-only --method "claude"

# Compare original vs rerun contamination verdicts (--diff-only, --csv out.csv)
python src/judges/rerun/aggregate_rerun_results.py

# Absolute paths of latest dirs flagged for disallowed API usage
python src/judges/rerun/find_disallowed_api_usage.py
```

## Output files

Per judge (`<id>` = `JUDGE_OUTPUT_ID`, `<sfx>` = empty inline / `_rerun` standalone):

- `judge_output_<id><sfx>.json` — raw codex CLI trace
- `judge_output_<id><sfx>.txt` — human-readable trace (`src/trace_parsing/parse_trace.py`)
- `judgement_<id><sfx>.json` — structured verdict the judge wrote to `judgement.json`
  in its sandbox task directory

`judgement_gpt5_4*.json` schema:

```json
{
  "contamination": true,
  "disallowed_model": false,
  "justification_contamination": "...",
  "justification_disallowed_model": "..."
}
```

`judgement_api*.json` schema:

```json
{
  "disallowed_api_usage": false,
  "justification_disallowed_api_usage": "..."
}
```

## Adding a new judge

1. Create `src/judges/<judge_name>/` with:
   - a prompt template (any filename). `{model}` and `{benchmark}` are always filled; if the
     template needs extra placeholders, register a fill function in `EXTRA_FILLERS` in
     `get_judge_prompt.py`.
   - `judge.conf` — simple `KEY="value"` lines (sourced by bash, parsed by python):
     - `JUDGE_LABEL` — human-readable name used in logs
     - `JUDGE_OUTPUT_ID` — suffix for all output files (`judgement_<id>.json`, ...)
     - `JUDGE_PROMPT_FILE` — the template's filename
     - `JUDGE_MISSING_JUDGEMENT_FATAL_INLINE` — `1` to abort `run_task.sh` when the judge
       produces no `judgement.json` during the initial run, `0` to continue (standalone reruns
       are always fatal)
     - `JUDGE_CONDOR_WEIGHT` — user.judge concurrency units a rerun job consumes
     - optional: `JUDGE_MODEL` / `JUDGE_REASONING_EFFORT` to override the codex defaults
       (`gpt-5.4` / `xhigh`, see `judge_lib.sh`)
2. Add the folder name to `ALL_JUDGES` in `judge_lib.sh` (array order = execution order).

That's it — `run_task.sh`, `run_judges.sh` and `commit_rerun_judges.sh` all iterate over the
judge set, and `--skip-existing` / concurrency weights are derived from `judge.conf`.

The judge prompt must instruct the model to write its verdict to `judgement.json` in the task
directory; that file is collected as `judgement_<id><sfx>.json`.
