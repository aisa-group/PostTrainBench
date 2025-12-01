#!/bin/bash

export POST_TRAIN_BENCH_CONTAINER_NAME="standard"
export POST_TRAIN_BENCH_PROMPT="prompt"

models=(
    "google/gemma-3-4b-pt"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-1.7B-Base"
    "HuggingFaceTB/SmolLM3-3B-Base"
)

evals=(
    "aime2025"
    "bfcl"
    "gpqamain"
    "gsm8k"
    "humaneval"
)
for model in "${models[@]}"; do
    for eval in "${evals[@]}"; do
        echo ""
        echo $model on $eval
        condor_submit_bid 100 -a "agent=codex" -a "agent_config=gpt-5.1-codex" -a "eval=$eval" -a "model_to_train=$model" src/commit_utils/single_task.sub
        condor_submit_bid 100 -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" src/commit_utils/single_task.sub
    done
done
