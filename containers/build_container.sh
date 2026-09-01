#!/bin/bash
container="${1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Pull POST_TRAIN_BENCH_CONTAINERS_DIR from .env without invoking set_env_vars.sh
# (its module-loading block fails on nodes without tclsh). An already-exported
# value takes precedence; otherwise fall back to the repo-local containers dir.
if [ -z "$POST_TRAIN_BENCH_CONTAINERS_DIR" ] && [ -f "$ENV_FILE" ]; then
    POST_TRAIN_BENCH_CONTAINERS_DIR="$(grep -E '^POST_TRAIN_BENCH_CONTAINERS_DIR=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"')"
fi
export POST_TRAIN_BENCH_CONTAINERS_DIR=${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}
export APPTAINER_BIND=""

apptainer build "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${container}.sif" "containers/${container}.def"
