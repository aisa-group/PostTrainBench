## Tips for running the benchmark
These comments are mainly for internal use.

You can use `experiment_name` to set experiment names for the benchmark, e.g. to distinguish experiments where you test and ones which are for actual results.
E.g. you can pass `-a "experiment_name=_testing"` when submitting jobs with `condor_submit_bid`.

Some useful scripts after running experiments:
- `dev_utils/list_cuda_not_avl.py` lists runs where the cuda check failed (those runs need to be rerun)
- `dev_utils/runs_no_metrics.py` lists runs where metrics.json was not produced, also try `--all` to make this list more inclusive. Sometimes final evaluation needs to be rerun.
- `dev_utils/contamination_list.py` to see runs where contamination occured (sometimes useful to check if the judge works correctly).

### Runs without metrics
Workflow:
Run `python dev_utils/runs_no_metrics.py`

Rerun evaluation for the runs here with e.g. (note that the eval parameter needs to be adjusted):
```
condor_submit_bid 50 -a "eval=arenahardwriting" -a "eval_dir=/path/to/results/claude_claude-sonnet-4-5_10h_v3/arenahardwriting_Qwen_Qwen3-4B-Base_16775298" dev_utils/test_evaluation/single_evaluation.sub
```

Things to be aware of:
- gemma3-4B is often a bit more tricky to prepare, because the evaluation will complain that no image processor is there. The agent should add this, so in case they didn't do this, it is fine that no `metrics.json` was produced.
- failures happen because the gpu memory is too little. Also here this is a mistake on part of the agent. But we can run the evaluation again with `dev_utils/test_evaluation/single_evaluation.sub` and `dev_utils/test_evaluation/single_evaluation_less_mem.sub`. We should try our best to evaluate the model, but if it fails after repeated retrys, we also ok that no `metrics.json` was produced. The agent will receive the base model score for this case.

For runs for which no metrics are produced, but for which this is fine (e.g. because they forgot to add the image processor to gemma or repeated final evals brought no results), add them to this environment variable:
```
export POST_TRAIN_BENCH_NO_METRICS_IS_FINE="/path/to/run1:/path/to/run2:/path/to/run3"
```
Then they will not be shown when running `dev_utils/runs_no_metrics.py`.

### Double Check the Judge
It is good to double check if the judge worked correctly.
For this you can use the `dev_utils/contamination_list.py` script to list flagged runs, or `python scripts/find_flagged_runs.py --judge data_contamination_judge --justification` to also see the judge's reasoning.
Both list the *effective* verdict that `scripts/collect.py` scores by (see `resolve_judgement` in `scripts/utils.py`): a manual override (`judgement_gpt5_4_manual.json`) if present, else the per-field majority of the three contamination-judge runs (`judgement_gpt5_4_rerun.json` or `judgement_gpt5_4.json`, plus `judgement_multi_runs/judgement_gpt5_4_run{2,3}.json`), else the single verdict.
Then look at the judge traces (`judge_output_gpt5_4*.txt`, also under `judgement_multi_runs/`) to see the judge's reasoning and potentially at `task/` to see the output code of the agent.

If the judge was wrong, do **not** edit the judge's verdict files: with three runs, flipping one of them is outvoted by the other two. Record a manual override instead, which ranks above every judge output: create `judgement_gpt5_4_manual.json` in the run directory (next to the judge files) with both boolean fields set to the corrected verdict, e.g.
```
{
  "contamination": false,
  "disallowed_model": false,
  "justification_contamination": "Manual override by <you>, <date>: overlap is with the train split only",
  "justification_disallowed_model": "Manual override by <you>, <date>: unchanged"
}
```
Both `contamination` and `disallowed_model` are required and must be JSON booleans (`collect.py` raises otherwise); the override replaces the whole verdict, so copy the unchanged field from the current effective verdict. Delete the file to go back to the judge verdict.

For all runs which you went over, add them to the `POST_TRAIN_BENCH_CONTAMINATION_CORRECT` environment variable which is build up like this:
```
export POST_TRAIN_BENCH_CONTAMINATION_CORRECT="/path/to/run1:/path/to/run2:/path/to/run3"
```
Things to be aware of:
- The judge is often right when bfcl contamination is flagged
- The judge is often wrong when gsm8k contamination is flagged

### Debugging
For debugging the final evaluation (=evaluation of the model checkpoint produced by the agent), use `dev_utils/test_evaluation/run_only_evaluation.sh`.
Internally, or on HTCondor, this can be used via `dev_utils/test_evaluation/single_evaluation.sub`.
Also it is best to run this with 
```
export POST_TRAIN_BENCH_JOB_SCHEDULER="vllm_debug"
```
to get vllm logs in the output.

## For our internal cluster (MPI)
### Env var
Set these in your `.env` file:
```
POST_TRAIN_BENCH_JOB_SCHEDULER="htcondor_mpi-is"
POST_TRAIN_BENCH_CONTAINERS_DIR="/fast/username/ptb_containers"
POST_TRAIN_BENCH_RESULTS_DIR="/fast/username/ptb_results"
```
Substitute "username" by your username.
You will need to move your containers there after this.

### Gemini issues
Gemini sometimes runs into issues like "API Error: exception TypeError: fetch failed sending request".
This likely is a result of running to many jobs at once.
You can find such jobs with the `dev_utils/api_error_list.py` script.

You can use the submission file `src/commit_utils/single_task_gemini.sub` instead of `src/commit_utils/single_task.sub`, to only have 8 gemini jobs running at once (even if you submit more).

### Huggingface Cache
If you point your huggingface cache to some subdir of `/fast`, set `HF_HOME` to a subfolder of fast and then download the cache in the following way:

First build the soft-file-locking container via
```
bash containers/build_container.sh soft_file_locking
```
Then download the huggingface cache inside this container.
You can start a shell with the container by calling
```
bash dev_utils/shell.sh soft_file_locking
```
and call the following inside the container
```
bash containers/download_hf_cache/download_hf_cache.sh
```