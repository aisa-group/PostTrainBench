# Rerun Judge Pipeline

Reruns the disallowed usage judge with the trace file included in the prompt. The judge is instructed to analyze `solve_trace.txt` by starting from the end and tracing back where `final_model` comes from.

## Files

| File | Description |
|------|-------------|
| `prompt_with_trace.txt` | Judge prompt template with trace analysis instructions |
| `utils.py` | Shared utilities for directory listing and parsing |
| `get_judge_prompt_with_trace.py` | Generate the full prompt |
| `list_results.py` | List and filter result directories |
| `aggregate_rerun_results.py` | Aggregate and compare results |
| `rerun_single.sh` | Run judge on a single result directory |
| `rerun_all.sh` | Run judge on all directories (local) |
| `commit_rerun_judge.sh` | Submit HTCondor jobs |
| `rerun_judge.sub` | HTCondor submission file |

## Usage

### Submit HTCondor jobs

```bash
# Submit jobs for all result directories
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh

# Filter by method
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh --method "claude"

# Filter by benchmark
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh --benchmark "aime"

# Skip directories that already have rerun judgements
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh --skip-existing

# Limit to first 10 directories
./src/disallowed_usage_judge/rerun_judge/commit_rerun_judge.sh --limit 10
```

### Run locally

```bash
# Run on a single directory
./src/disallowed_usage_judge/rerun_judge/rerun_single.sh /path/to/result_dir

# Run on all directories sequentially
./src/disallowed_usage_judge/rerun_judge/rerun_all.sh

# Run with filters
./src/disallowed_usage_judge/rerun_judge/rerun_all.sh --method "claude" --benchmark "aime"

# Run 4 in parallel
./src/disallowed_usage_judge/rerun_judge/rerun_all.sh --parallel 4 --skip-existing
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

New files created in each result directory (originals preserved):
- `contamination_judgement_rerun.txt`
- `disallowed_model_judgement_rerun.txt`

## Trace Selection

The pipeline copies `solve_parsed.txt` (preferred) or `solve_out.txt` into the task directory as `solve_trace.txt` for the judge to analyze.
