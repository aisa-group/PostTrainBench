#!/bin/bash

shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AGENTS_DIR="${PROJECT_DIR}/agents"

export POST_TRAIN_BENCH_RESULTS_DIR=${POST_TRAIN_BENCH_RESULTS_DIR:-results}

# Find all agents that have a human_readable_trace.py
for agent_dir in "${AGENTS_DIR}"/*/; do
    agent_name=$(basename "$agent_dir")
    parser="${agent_dir}human_readable_trace.py"

    if [ ! -f "$parser" ]; then
        continue
    fi

    echo "=== Processing agent: ${agent_name} ==="

    # Find results directories matching this agent's prefix
    for results_dir in "${POST_TRAIN_BENCH_RESULTS_DIR}/${agent_name}"*/; do
        if [ -d "$results_dir" ]; then
            for subdir in "$results_dir"*/; do
                if [ -d "$subdir" ]; then
                    input_file="${subdir}/solve_out.txt"
                    output_file="${subdir}/solve_parsed.txt"

                    if [ -f "$input_file" ]; then
                        echo "Processing ${subdir}"
                        python3 "$parser" "$input_file" -o "$output_file"
                    fi
                fi
            done
        fi
    done
done
