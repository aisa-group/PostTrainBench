#!/bin/bash
#
# Re-submit condor jobs for a list of result cell paths.
#
# For each cell path like
#   /fast/hbhatnagar/ptb_results/{method}/{benchmark}_{model}_{cluster_id}
# parses the method dir + cell name to reconstruct the -a flags commit.sh
# would use (agent, agent_config, eval, model_to_train, num_hours,
# experiment_name, num_gpus) and calls condor_submit_bid.
#
# Handy for re-running C-flagged cells surfaced by list_contaminated_reruns.sh
# without touching commit.sh.
#
# Usage:
#   bash src/commit_utils/rerun_cells.sh [flags] <cell_path> [<cell_path>...]
#   cat cells.txt | bash src/commit_utils/rerun_cells.sh [flags]
#
# Flags:
#   --dry-run              Print the commands without submitting
#   --bid <n>              condor bid amount (default: 100)
#   --num-hours <n>        Override hours (default: parsed from method dir)
#   --experiment-suffix <s>
#                          Append to experiment_name so re-runs land in a
#                          separate method dir. Example: --experiment-suffix _v2
#                          turns _run2 into _run2_v2.
#   --concurrency-limits <l>
#                          Pass through as -a concurrency_limits=<l>
#   --sub-file <name>      Override the .sub file (basename under
#                          src/commit_utils/, e.g. single_task_muse.sub)
#
# Sub-file auto-detection:
#   agent=kimi_claude       → single_task_kimi.sub
#   agent=gemini            → single_task_gemini.sub
#   agent=opencode + config contains "muse-spark"
#                           → single_task_muse.sub
#   everything else         → single_task.sub

set -euo pipefail

DRY_RUN=""; BID=100; HOURS_OVERRIDE=""; EXP_SUFFIX=""
CONCURRENCY=""; SUB_OVERRIDE=""
CELLS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=1; shift ;;
        --bid) BID="$2"; shift 2 ;;
        --num-hours) HOURS_OVERRIDE="$2"; shift 2 ;;
        --experiment-suffix) EXP_SUFFIX="$2"; shift 2 ;;
        --concurrency-limits) CONCURRENCY="$2"; shift 2 ;;
        --sub-file) SUB_OVERRIDE="$2"; shift 2 ;;
        -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) CELLS+=("$1"); shift ;;
    esac
done

# Read from stdin if no CLI paths.
# `|| [ -n "$line" ]` catches the last line when the file has no trailing
# newline (common with copy-paste). Only ASCII CR is stripped (Windows line
# endings); we don't try to strip other whitespace, since valid paths never
# start with space and any surrounding whitespace would be a user error worth
# surfacing rather than silently trimming.
if [ ${#CELLS[@]} -eq 0 ] && [ ! -t 0 ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"   # strip CR from Windows line endings
        line="${line%/}"        # strip trailing slash
        [ -z "$line" ] && continue
        CELLS+=("$line")
    done
fi

if [ ${#CELLS[@]} -eq 0 ]; then
    echo "ERROR: no cell paths given (via args or stdin). Try --help." >&2
    exit 1
fi

[ -d agents ] || { echo "ERROR: run from repo root (agents/ not found)" >&2; exit 1; }
ALL_AGENTS=($(ls agents 2>/dev/null))

# Benchmarks known to the framework — must match src/eval/tasks/ subdirs
BENCHMARKS="aime2025 aime2026 arenahardwriting bfcl gpqamain gsm8k humaneval healthbench"

parse_and_submit() {
    local cell_path="${1%/}"
    local cell_name method_path method_name
    cell_name=$(basename "$cell_path")
    method_path=$(dirname "$cell_path")
    method_name=$(basename "$method_path")

    # Cell: {benchmark}_{model}_{cluster_id}
    if [[ ! $cell_name =~ ^(.+)_([0-9]+)$ ]]; then
        echo "ERROR [$cell_name]: doesn't match {bench}_{model}_{cid} pattern" >&2
        return 1
    fi
    local bench_model="${BASH_REMATCH[1]}"
    local old_cluster="${BASH_REMATCH[2]}"

    # Split into benchmark + model (benchmark is first known token; model can
    # contain underscores like HuggingFaceTB_SmolLM3-3B-Base)
    local benchmark="" model_dirname=""
    for b in $BENCHMARKS; do
        if [[ $bench_model == ${b}_* ]]; then
            benchmark="$b"
            model_dirname="${bench_model#${b}_}"
            break
        fi
    done
    if [ -z "$benchmark" ]; then
        echo "ERROR [$cell_name]: benchmark not recognised in '$bench_model'" >&2
        return 1
    fi

    # Model in dir: first underscore is the org/name separator (base models
    # never have '_' inside the model name — they use hyphens)
    local model="${model_dirname/_//}"

    # Method: {agent}_{config[_1m_]}_{N}h[_suffix]
    if [[ ! $method_name =~ ^(.+)_([0-9]+)h(_.+)?$ ]]; then
        echo "ERROR [$method_name]: no _{N}h_ separator" >&2
        return 1
    fi
    local prefix="${BASH_REMATCH[1]}" hours="${BASH_REMATCH[2]}" suffix="${BASH_REMATCH[3]:-}"

    # Longest-agent-name match in prefix
    local agent=""
    for candidate in "${ALL_AGENTS[@]}"; do
        [[ $prefix == ${candidate}_* ]] || continue
        [ ${#candidate} -gt ${#agent} ] && agent="$candidate"
    done
    if [ -z "$agent" ]; then
        echo "ERROR [$method_name]: no matching agent in agents/ for prefix '$prefix'" >&2
        return 1
    fi

    local config="${prefix#${agent}_}"
    # Reverse the dirname mangling: trailing `_1m_` in dir → literal `[1m]`
    [[ $config == *_1m_ ]] && config="${config%_1m_}[1m]"

    # Per-agent config remaps for cases where a method dir was renamed to a
    # display-friendly name but the agent's actual endpoint expects a different
    # config token. Add cases here as they come up.
    case "$agent" in
        glmx)
            # Submit as `glm-5.2` (the endpoint name). Dirs from earlier runs
            # may say `glm-5.2-preview` — user will rename those separately.
            # agents/glmx/solve.sh exports ANTHROPIC_MODEL from AGENT_CONFIG
            # verbatim, so this is where the actual model name is chosen.
            case "$config" in
                glm-5.2-preview*) config="${config/glm-5.2-preview/glm-5.2}" ;;
            esac
            ;;
        opencode)
            # opencode's real agent_config is {provider}/{model} (e.g.
            # opencode/gemini-3.1-pro, anthropic/claude-opus-4-5), but the dir
            # name encodes the slash as an underscore. Reverse the first `_`
            # back to `/`. Model tokens never contain `_` (they use hyphens),
            # so this is unambiguous.
            config="${config/_//}"
            ;;
    esac

    # Suffix: extract num_gpus if present, keep the rest as experiment_name
    local num_gpus="1" experiment_name="$suffix"
    if [[ $suffix =~ _([0-9]+)gpu ]]; then
        num_gpus="${BASH_REMATCH[1]}"
        experiment_name="${suffix//_${num_gpus}gpu/}"
    fi
    [ -n "$EXP_SUFFIX" ] && experiment_name="${experiment_name}${EXP_SUFFIX}"

    local use_hours="${HOURS_OVERRIDE:-$hours}"

    # .sub file selection
    local sub_file
    if [ -n "$SUB_OVERRIDE" ]; then
        sub_file="src/commit_utils/${SUB_OVERRIDE}"
    else
        case "$agent" in
            kimi_claude) sub_file="src/commit_utils/single_task_kimi.sub" ;;
            gemini)      sub_file="src/commit_utils/single_task_gemini.sub" ;;
            opencode)
                # opencode originally shipped alongside vllm_debug — gemini-3.1-pro
                # and the older opencode/glm5/kimi-k2.5/minimax-m2.5 configs were all
                # baked into vllm_debug.sif (opencode-ai@1.1.59). Faithful re-runs of
                # those should keep the same container. Newer opencode configs default
                # to the current container in .env (muse_1_1 → opencode-ai@1.17.18).
                if [[ $config == *muse-spark* ]]; then
                    sub_file="src/commit_utils/single_task_muse.sub"
                elif [[ $config == *gemini-3.1-pro* ]]; then
                    sub_file="src/commit_utils/single_task_vllm_debug.sub"
                else
                    sub_file="src/commit_utils/single_task.sub"
                fi ;;
            *) sub_file="src/commit_utils/single_task.sub" ;;
        esac
    fi
    if [ ! -f "$sub_file" ]; then
        echo "ERROR [$method_name]: .sub file not found: $sub_file" >&2
        return 1
    fi

    # Build condor args
    local args=(
        -a "experiment_name=$experiment_name"
        -a "agent=$agent"
        -a "agent_config=$config"
        -a "eval=$benchmark"
        -a "model_to_train=$model"
        -a "num_hours=$use_hours"
    )
    [ "$num_gpus" != "1" ] && args+=(-a "num_gpus=$num_gpus")
    [ -n "$CONCURRENCY" ] && args+=(-a "concurrency_limits=$CONCURRENCY")

    printf '\n%s\n' "-----------------------------------------------------------"
    printf '  cell:       %s\n' "$cell_path"
    printf '  agent:      %s\n  config:     %s\n' "$agent" "$config"
    printf '  benchmark:  %s\n  model:      %s\n' "$benchmark" "$model"
    printf '  hours:      %s\n  expname:    %s\n  num_gpus:   %s\n' "$use_hours" "$experiment_name" "$num_gpus"
    printf '  sub_file:   %s\n' "$sub_file"

    if [ -n "$DRY_RUN" ]; then
        printf '  [DRY-RUN]   condor_submit_bid %s' "$BID"
        for a in "${args[@]}"; do printf ' %s' "$a"; done
        printf ' %s\n' "$sub_file"
        return 0
    fi

    local out
    out="$(condor_submit_bid "$BID" "${args[@]}" "$sub_file" 2>&1)"
    printf '%s\n' "$out" | tail -3
    if ! echo "$out" | grep -q 'submitted to cluster'; then
        echo "  ERROR: submission may have failed" >&2
        return 1
    fi
}

n_ok=0; n_bad=0
for cell in "${CELLS[@]}"; do
    if parse_and_submit "$cell"; then
        n_ok=$((n_ok+1))
    else
        n_bad=$((n_bad+1))
    fi
done

echo
echo "==========================================================="
echo "Processed ${#CELLS[@]} cells: $n_ok $([ -n "$DRY_RUN" ] && echo would-submit || echo submitted), $n_bad failed"
[ -n "$DRY_RUN" ] && echo "(dry-run — drop --dry-run to actually submit)"
echo "==========================================================="
