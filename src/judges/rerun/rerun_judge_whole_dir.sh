TAG="user.judge_rerun_${USER}"
MAX_CONCURRENT=5
TOKENS=$(( 10000 / MAX_CONCURRENT ))   # 1000 tokens per job → 10 concurrent

POST_TRAIN_BENCH_RESULTS_DIR=/fast/hbhatnagar/ptb_results \
python src/disallowed_usage_judge/rerun_judge/list_results.py \
    --paths-only \
    --latest-only \
    --method "claude_non_api_max_claude-fable-5_1m__10h_run2" \
| while read -r d; do
    echo "Submitting: $d"
    condor_submit_bid 100 \
        -a "result_dir=$d" \
        -a "concurrency_limits=${TAG}:${TOKENS}" \
        src/disallowed_usage_judge/rerun_judge/rerun_judge.sub
done    