# Rerun Judge Pipeline

Reruns the disallowed usage judge on existing result directories. The judge analyzes `solve_trace.txt` by starting from the end and tracing back where `final_model` comes from.

## Architecture

The rerun functionality is integrated into the main judge:
- `../prompt.txt` contains trace analysis instructions
- `../get_judge_prompt.py` generates prompts with trace support
- `../run_judge.sh` is the unified script for running/rerunning the judge

This directory contains orchestration scripts for batch reruns.

## Files

| File | Description |
|------|-------------|
| `utils.py` | Shared utilities for directory listing and parsing |
| `list_results.py` | List and filter result directories |
| `aggregate_rerun_results.py` | Aggregate and compare results |
| `rerun_single.sh` | Run judge on a single result directory (wrapper for `run_judge.sh`) |
| `commit_all_gpt_only.sh` | Submit HTCondor rerun jobs for the latest run of every method |
| `commit_gpt_contamination_only.sh` | Same, but only reruns the GPT-5.4 contamination judge |
| `rerun_judge.sub` | HTCondor submission file |

## Usage

### Run judge on a single directory

```bash
# Run judge (overwrites existing judgements)
bash src/disallowed_usage_judge/run_judge.sh /path/to/result_dir

# Rerun judge (saves with _rerun suffix, preserves originals)
bash src/disallowed_usage_judge/run_judge.sh --rerun /path/to/result_dir

# Or use the wrapper script
bash src/disallowed_usage_judge/rerun_judge/rerun_single.sh /path/to/result_dir
```

### Submit HTCondor jobs

```bash
# Rerun both GPT-5.4 judges on the latest run of every method
./src/disallowed_usage_judge/rerun_judge/commit_all_gpt_only.sh

# Preview which directories would be submitted, without submitting
./src/disallowed_usage_judge/rerun_judge/commit_all_gpt_only.sh --dry-run

# Skip directories whose _rerun outputs already exist (per judge)
./src/disallowed_usage_judge/rerun_judge/commit_all_gpt_only.sh --skip-existing

# Only rerun the GPT-5.4 contamination judge (skip the API judge)
./src/disallowed_usage_judge/rerun_judge/commit_gpt_contamination_only.sh
```

For ad-hoc targeting (a specific method/benchmark, a limit, etc.), use
`list_results.py` to build the directory list and submit `rerun_judge.sub`
directly — e.g.:

```bash
python src/disallowed_usage_judge/rerun_judge/list_results.py \
    --paths-only --latest-only --method "claude" \
| while read -r d; do
    condor_submit_bid 100 -a "result_dir=$d" \
        src/disallowed_usage_judge/rerun_judge/rerun_judge.sub
done
```

### List and filter results

```bash
# List all directories
python src/disallowed_usage_judge/rerun_judge/list_results.py

# Show trace file info
python src/disallowed_usage_judge/rerun_judge/list_results.py --with-trace

# List directories missing rerun judgement
python src/disallowed_usage_judge/rerun_judge/list_results.py --missing-rerun

# Get just paths (for piping)
python src/disallowed_usage_judge/rerun_judge/list_results.py --paths-only --method "claude"
```

### Aggregate results

```bash
# Show summary
python src/disallowed_usage_judge/rerun_judge/aggregate_rerun_results.py

# Only show changed judgements
python src/disallowed_usage_judge/rerun_judge/aggregate_rerun_results.py --diff-only

# Export to CSV
python src/disallowed_usage_judge/rerun_judge/aggregate_rerun_results.py --csv results.csv
```

## Output Files

When running with `--rerun` flag, new files are created in each result directory (originals preserved):
- `judgement_gpt5_4_rerun.json` (canonical contamination verdict)
- `judgement_api_rerun.json` (API-usage verdict; archival, not consumed downstream)
- `judge_output_gpt5_4_rerun.{json,txt}` (raw judge transcripts)

When running without `--rerun`, the original files are overwritten:
- `judgement_gpt5_4.json`
- `judgement_api.json`
- `judge_output_gpt5_4.{json,txt}`

The `judgement_gpt5_4*.json` contamination file has the schema:

```json
{
  "contamination": true,
  "disallowed_model": false,
  "justification_contamination": "...",
  "justification_disallowed_model": "..."
}
```

## Trace File

The judge expects `solve_parsed.txt` in the result directory. It copies this to the parent of the task directory (accessible as `../solve_parsed.txt` from the judge's working directory).
