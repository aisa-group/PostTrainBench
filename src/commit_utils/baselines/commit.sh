#!/bin/bash
bash src/commit_utils/set_env_vars.sh

models=(
    "google/gemma-3-4b-it"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-1.7B"
    "HuggingFaceTB/SmolLM3-3B"
    # 
    # base models
    "google/gemma-3-4b-pt"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-1.7B-Base"
    "HuggingFaceTB/SmolLM3-3B-Base"
)

evals=(
    "aime2025"
    "arenahardwriting"
    "bfcl"
    "gpqamain"
    "gsm8k"
    "humaneval"
)

for model in "${models[@]}"; do
    for eval in "${evals[@]}"; do
        echo $model on $eval
        condor_submit_bid 25 -a "eval=$eval" -a "model=$model" src/commit_utils/baselines/baseline_cluster.sub
        # sleep 30
    done
done
