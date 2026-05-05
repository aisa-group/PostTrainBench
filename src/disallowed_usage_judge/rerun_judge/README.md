# Rerun Judge

Re-run the disallowed-usage / contamination judge on existing result
directories without re-running the agent or eval. Useful when the judge step
in `src/run_task.sh` failed mid-run (e.g. API quota hit) and the run dir is
missing `contamination_judgement.txt` / `disallowed_model_judgement.txt`.

The judge invocation mirrors `src/run_task.sh` exactly — single GPT-5.1-Codex
call via the `CODEX_API_KEY` API path. All outputs are suffixed with `_rerun`
so the originals (if any) are preserved.

## Files

| File | Description |
|---|---|
| `../run_judge.sh` | Standalone script: run judge on one result dir |
| `utils.py` | Shared dir-walking / parsing / judgement-loading helpers |
| `list_results.py` | List + filter result directories |
| `aggregate_rerun_results.py` | Compare original vs rerun judgements |
| `rerun_single.sh` | Thin wrapper over `run_judge.sh` (for HTCondor) |
| `commit_rerun_judge.sh` | Submit HTCondor jobs |
| `rerun_judge.sub` | HTCondor submission file |

## Usage

### Single directory

```bash
bash src/disallowed_usage_judge/run_judge.sh /path/to/result_dir
```

Writes:
- `contamination_judgement_rerun.txt`
- `disallowed_model_judgement_rerun.txt`
- `judge_output_rerun.json`
- `judge_output_rerun.txt`

### Listing candidates

```bash
# Every result dir
python src/disallowed_usage_judge/rerun_judge/list_results.py

# Only dirs where the original judge step failed
python src/disallowed_usage_judge/rerun_judge/list_results.py --only-missing-judgement

# Just paths, ready for piping
python src/disallowed_usage_judge/rerun_judge/list_results.py \
    --only-missing-judgement --paths-only
```

### Submit HTCondor jobs

```bash
# All dirs missing original judgement, latest run per method/model/benchmark
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh \
    --only-missing-judgement --latest-only

# Filter by method
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh \
    --method "codex_non_api_xhigh_reprompt_gpt-5.5"

# Skip dirs that already have rerun output
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh --skip-existing

# Preview without submitting
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh \
    --only-missing-judgement --dry-run
```

### Aggregate / diff

```bash
# Plain summary
python src/disallowed_usage_judge/rerun_judge/aggregate_rerun_results.py

# Only show dirs where the rerun changed the verdict
python src/disallowed_usage_judge/rerun_judge/aggregate_rerun_results.py --diff-only

# Only show dirs where the rerun filled a previously-missing judgement
python src/disallowed_usage_judge/rerun_judge/aggregate_rerun_results.py --filled-only

# Export to CSV
python src/disallowed_usage_judge/rerun_judge/aggregate_rerun_results.py --csv rerun.csv
```

## Adopting rerun results

`scripts/collect.py` reads `contamination_judgement.txt` /
`disallowed_model_judgement.txt`, not the `_rerun` variants. Once you're happy
with the rerun output, copy the files over (or symlink) so collect.py picks
them up:

```bash
cp result_dir/contamination_judgement_rerun.txt    result_dir/contamination_judgement.txt
cp result_dir/disallowed_model_judgement_rerun.txt result_dir/disallowed_model_judgement.txt
```

(A `--prefer-rerun` flag in `collect.py` would be the cleaner long-term
option.)
