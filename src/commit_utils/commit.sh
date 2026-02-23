#!/bin/bash
source src/commit_utils/set_env_vars.sh

models=(
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
    "healthbench"
)
export POST_TRAIN_BENCH_EXPERIMENT_NAME="_run3"
for model in "${models[@]}"; do
    for eval in "${evals[@]}"; do
        echo ""
        echo $model on $eval
        if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
            # Proprietary (API)
            condor_submit_bid 100 -a "agent=codex" -a "agent_config=gpt-5.1-codex-max" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=codex" -a "agent_config=gpt-5.3-codex" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=claude" -a "agent_config=claude-opus-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=claude" -a "agent_config=claude-opus-4-6" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=qwen3max" -a "agent_config=qwen3-max-2026-01-23" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            # Proprietary (Subscription plan)
            condor_submit_bid 100 -a "agent=codex_non_api" -a "agent_config=gpt-5.3-codex" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=claude_non_api" -a "agent_config=claude-opus-4-6" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 150 -a "agent=claude_non_api" -a "agent_config=claude-sonnet-4-6" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=1" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=gemini" -a "agent_config=models/gemini-3-pro-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "agent=gemini" -a "agent_config=models/gemini-3-flash-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 150 -a "agent=gemini" -a "agent_config=models/gemini-3.1-pro-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            # OpenCode 
            condor_submit_bid 50 -a "agent=opencode" -a "agent_config=anthropic/claude-opus-4-5" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=opencode" -a "agent_config=opencode/gpt-5.1-codex-max" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50 -a "agent=opencode" -a "agent_config=opencode/kimi-k2-thinking" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 50 -a "agent=opencode" -a "agent_config=opencode/glm-4.7-free" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 500 -a "agent=opencode" -a "agent_config=opencode/gemini-3-pro" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 50 -a "agent=opencode" -a "agent_config=opencode/minimax-m2.1-free" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 50 -a "agent=glm5" -a "agent_config=glm-5" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 100 -a "agent=opencode" -a "agent_config=opencode/minimax-m2.5-free" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 100 -a "agent=opencode" -a "agent_config=zai/glm-5" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 100 -a "agent=opencode" -a "agent_config=opencode/kimi-k2.5" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 50 -a "agent=opencode" -a "agent_config=opencode/glm-5" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            condor_submit_bid 150 -a "agent=opencode" -a "agent_config=opencode/gemini-3.1-pro" -a "eval=$eval" -a "model_to_train=$model" "num_hours=10" src/commit_utils/single_task.sub 
            sleep 10
        elif [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor" ]; then
            condor_submit_bid -a "agent=codex" -a "agent_config=gpt-5.1-codex-max" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=codex" -a "agent_config=gpt-5.3-codex" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-opus-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-opus-4-6" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=claude" -a "agent_config=claude-sonnet-4-5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=1" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=gemini" -a "agent_config=models/gemini-3-pro-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "agent=gemini" -a "agent_config=models/gemini-3-flash-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            sleep 20
        else
            echo ERROR: job scheduler "${POST_TRAIN_BENCH_JOB_SCHEDULER}" is not supported.
        fi
    done
done
