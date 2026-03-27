SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: .env file not found at $ENV_FILE" >&2
    echo "Copy example.env to .env and fill in your values." >&2
    exit 1
fi

# Export all variables from .env, without overriding already-set env vars
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    var_name="${line%%=*}"
    current_value="$(eval echo "\${$var_name:-}")"

    if [ -z "$current_value" ] || [ "$current_value" = "UNDEFINED" ]; then
        eval "export $line"
    fi
done < "$ENV_FILE"

export HF_HOME_NEW="/home/ben/hf_cache"

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    source /etc/profile.d/modules.sh
fi

export PYTHONNOUSERSITE=1

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi