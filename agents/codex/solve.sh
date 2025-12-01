#!/bin/bash

NUM_HOURS="$1"
PROMPT="$2"
AGENT_CONFIG="$3" # e.g. gpt-5.1-codex
JOB_DIR="$4"
JOB_TMP="$5"

apptainer exec \
    --nv \
    -c \
    --env PATH="/home/ben/.local/bin:$PATH" \
    --env HF_HOME="${HF_HOME_NEW}" \
    --env CODEX_API_KEY="${OPENAI_API_KEY}" \
    --env CODEX_API_KEY="${OPENAI_API_KEY}" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --bind "${JOB_TMP}:/tmp" \
    --home "${JOB_DIR}:/home/ben" \
    --pwd "/home/ben/task" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${POST_TRAIN_BENCH_CONTAINER_NAME}.sif" timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" codex --search exec --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"
