## Tips on running the benchmark
Some useful scripts after running experiments:
- `dev_utils/list_cuda_not_avl.py` lists runs where the cuda check failed (those runs need to be rerun)
- `dev_utils/runs_no_metrics.py` lists runs where metrics.json was not produced, also try `--all` to make this list more inclusive. Sometimes final evaluation needs to be rerun.
- `dev_utils/contamination_list.py` to see runs where contamination occured (sometimes useful to check if the judge works correctly).

## For our internal cluster (MPI)
#### Gemmini issues
Gemini sometimes runs into issues like "API Error: exception TypeError: fetch failed sending request".
This likely is a result of running to many jobs at once.
You can find such jobs with the `dev_utils/api_error_list.py` script.

You can use the submission file `src/commit_utils/single_task_gemini.sub` instead of `src/commit_utils/single_task.sub`, to only have 8 gemini jobs running at once (even if you submit more).

#### Huggingface Cache
If you point your huggingface cache to some subdir of `/fast`, first build the soft-file-locking container via
```
bash containers/build_container.sh soft_file_locking
```
Then download the huggingface cache inside this container.
You can start a shell with the container by calling
```
bash dev_utils/shell.sh soft_file_locking
```