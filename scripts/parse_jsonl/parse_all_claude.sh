#!/bin/bash

shopt -s nullglob

export POST_TRAIN_BENCH_RESULTS_DIR=${POST_TRAIN_BENCH_RESULTS_DIR:-results}

for claude_dir in "${POST_TRAIN_BENCH_RESULTS_DIR}"/claude*/; do
    if [ -d "$claude_dir" ]; then
        for subdir in "$claude_dir"*/; do
            if [ -d "$subdir" ]; then
                echo "Processing ${subdir}"
                python3 scripts/parse_jsonl/claude_code_parse_jsonl.py ${subdir}/solve_out.txt -o ${subdir}/solve_parsed.txt
            fi
        done
    fi
done