#!/bin/bash

NUM_HOURS="$1"
PROMPT="$2"
AGENT_CONFIG="$3" # e.g. claude-sonnet-4-5
JOB_DIR="$4"
JOB_TMP="$5"

apptainer exec \
    --nv \
    -c \
    --env PATH="/home/ben/.local/bin:$PATH" \
    --env HF_HOME="${HF_HOME_NEW}" \
    --env ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    --env BASH_MAX_TIMEOUT_MS="36000000" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --bind "${JOB_TMP}:/tmp" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json --dangerously-skip-permissions "$PROMPT"
