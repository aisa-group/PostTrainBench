#!/bin/bash
# Runs `audit.py check` inside the eval container (it executes model-generated code).
# Usage: run_check.sh <log_list> <out_dir> <container.sif> <ro_dir>...
#   <ro_dir>...  directories holding the run dirs and logs; bind-mounted read-only
set -euo pipefail

LOG_LIST="$1"
OUT_DIR="$2"
SIF="$3"
shift 3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$OUT_DIR"

BINDS=(--bind "$REPO_ROOT:$REPO_ROOT:ro" --bind "$(dirname "$LOG_LIST"):$(dirname "$LOG_LIST"):ro")
for d in "$@"; do
    BINDS+=(--bind "$d:$d:ro")
done
# Last, so a read-only parent bind above cannot hide it.
BINDS+=(--bind "$OUT_DIR:$OUT_DIR")

apptainer exec -c --cleanenv "${BINDS[@]}" --pwd "$REPO_ROOT" "$SIF" \
    python dev_utils/humaneval_early_exit/audit.py check --log-list "$LOG_LIST" --out-dir "$OUT_DIR"
