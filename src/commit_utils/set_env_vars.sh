export HF_HOME_NEW="/home/ben/hf_cache"
source /etc/profile.d/modules.sh

# Helper function: sets variable to default if unset or "UNDEFINED"
set_default() {
    local var_name="${1:-}"
    local default_value="${2:-}"
    local current_value
    eval "current_value=\"\${$var_name:-}\""
    
    if [ -z "$current_value" ] || [ "$current_value" = "UNDEFINED" ]; then
        export "$var_name"="$default_value"
    fi
}

set_default HF_HOME "$HOME/.cache/huggingface"
set_default POST_TRAIN_BENCH_RESULTS_DIR "results"
set_default POST_TRAIN_BENCH_CONTAINERS_DIR "containers"
set_default POST_TRAIN_BENCH_CONTAINER_NAME "standard"
set_default POST_TRAIN_BENCH_PROMPT "prompt"
set_default POST_TRAIN_BENCH_JOB_SCHEDULER "htcondor"
set_default POST_TRAIN_BENCH_EXPERIMENT_NAME ""

export PYTHONNOUSERSITE=1

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi