#!/bin/bash
source src/commit_utils/set_env_vars.sh

# Sweep template. Every submission line below is active by
# default — comment out what you don't want to run in a given session
# (agents, benchmarks, or base models).

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
EXPERIMENT_NAME="_run2"

# Concurrency caps: HTCondor gives each `user.<tag>` resource 10 000 tokens.
# Each job consumes `10000 / max_concurrent`, so 1250 tokens ⇒ 8 concurrent jobs.
# One tag per model family so Claude and Codex caps are independent.
CLAUDE_CONCURRENCY="user.${USER}_claude:1250"
CODEX_CONCURRENCY="user.${USER}_codex:1250"

for model in "${models[@]}"; do
    for eval in "${evals[@]}"; do
        echo ""
        echo $model on $eval
        if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then

            # ---------- Proprietary (API) ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex"    -a "agent_config=gpt-5.1-codex-max"       -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex"    -a "agent_config=gpt-5.3-codex"           -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude"   -a "agent_config=claude-sonnet-4-5"       -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude"   -a "agent_config=claude-opus-4-5"         -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude"   -a "agent_config=claude-opus-4-6"         -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=qwen3max" -a "agent_config=qwen3-max-2026-01-23"    -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub

            # ---------- Proprietary (subscription plan) ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api"        -a "agent_config=gpt-5.3-codex"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_high"   -a "agent_config=gpt-5.2"              -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_high"   -a "agent_config=gpt-5.3-codex"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_high"   -a "agent_config=gpt-5.4"              -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_xhigh"  -a "agent_config=gpt-5.3-codex"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_xhigh"  -a "agent_config=gpt-5.5"              -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" -a "concurrency_limits=$CODEX_CONCURRENCY" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_max"    -a "agent_config=gpt-5.6-sol"          -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 150 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api"       -a "agent_config=claude-sonnet-4-6"    -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api"       -a "agent_config=claude-opus-4-6"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api"       -a "agent_config=claude-opus-4-6[1m]"  -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api"       -a "agent_config=claude-opus-4-7"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" -a "concurrency_limits=$CLAUDE_CONCURRENCY" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api"       -a "agent_config=claude-opus-5"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task_opus5.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api_max"   -a "agent_config=claude-opus-4-6"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api_max"   -a "agent_config=claude-opus-4-8"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api_max"   -a "agent_config=claude-fable-5[1m]"   -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub

            # ---------- Reprompted variants (push the agent when it gives up early) ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_high_reprompt"  -a "agent_config=gpt-5.4" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex_non_api_xhigh_reprompt" -a "agent_config=gpt-5.5" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" -a "concurrency_limits=$CODEX_CONCURRENCY" src/commit_utils/single_task.sub

            # ---------- Multi-GPU (need more CPUs + RAM: 512 GB to be safe) ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api" -a "agent_config=claude-opus-4-6[1m]" -a "eval=$eval" -a "model_to_train=$model" -a "num_gpus=8" -a "num_hours=50" -a "request_memory=524288" -a "request_cpus=128" src/commit_utils/single_task.sub
            condor_submit_bid 500 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude_non_api" -a "agent_config=claude-opus-4-6[1m]" -a "eval=$eval" -a "model_to_train=$model" -a "num_gpus=8" -a "num_hours=50" src/commit_utils/single_task.sub

            # ---------- Gemini ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=gemini" -a "agent_config=models/gemini-3-pro-preview"    -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=gemini" -a "agent_config=models/gemini-3-flash-preview"  -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 150 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=gemini" -a "agent_config=models/gemini-3.1-pro-preview"  -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub

            # ---------- Cursor CLI (Grok, via subscription auth) ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=cursor_cli" -a "agent_config=cursor-grok-4.5-high" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub

            # ---------- Kimi (Moonshot) via anthropic-compatible endpoint ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=kimi_claude" -a "agent_config=kismet-0715[1m]" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task_kimi.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=kimi_claude" -a "agent_config=biui-0724"       -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task_kimi.sub

            # ---------- Z.ai GLM (via anthropic-compatible glmx wrapper) ----------
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=glmx" -a "agent_config=glm-x-preview[1m]" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=glmx" -a "agent_config=glm-5.2[1m]"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub

            # ---------- OpenCode (multi-provider aggregator; agent_config = <provider>/<model>) ----------
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=anthropic/claude-opus-4-5"  -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/gpt-5.1-codex-max" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/kimi-k2-thinking"  -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/kimi-k2.5"         -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/glm-4.7-free"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/glm-5"             -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=zai/glm-5"                  -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/minimax-m2.1-free" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/minimax-m2.5-free" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid 500 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/gemini-3-pro"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            # gemini-3.1-pro via opencode currently needs the vllm_debug container
            condor_submit_bid 150 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=opencode/gemini-3.1-pro"    -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task_vllm_debug.sub
            # muse-spark uses the dedicated muse.sub
            condor_submit_bid 100 -a "experiment_name=$EXPERIMENT_NAME" -a "agent=opencode" -a "agent_config=meta/muse-spark-1.1"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task_muse.sub
            condor_submit_bid 50  -a "experiment_name=$EXPERIMENT_NAME" -a "agent=glm5"     -a "agent_config=glm-5"                      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub

            sleep 10
        elif [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor" ]; then
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex"  -a "agent_config=gpt-5.1-codex-max"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=codex"  -a "agent_config=gpt-5.3-codex"          -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude" -a "agent_config=claude-sonnet-4-5"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude" -a "agent_config=claude-opus-4-5"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude" -a "agent_config=claude-opus-4-6"        -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=claude" -a "agent_config=claude-sonnet-4-5"      -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=1"  src/commit_utils/single_task.sub
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=gemini" -a "agent_config=models/gemini-3-pro-preview"   -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            condor_submit_bid -a "experiment_name=$EXPERIMENT_NAME" -a "agent=gemini" -a "agent_config=models/gemini-3-flash-preview" -a "eval=$eval" -a "model_to_train=$model" -a "num_hours=10" src/commit_utils/single_task.sub
            sleep 20
        else
            echo ERROR: job scheduler "${POST_TRAIN_BENCH_JOB_SCHEDULER}" is not supported.
        fi
    done
done
