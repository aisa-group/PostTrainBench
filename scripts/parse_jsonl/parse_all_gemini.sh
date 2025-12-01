#!/bin/bash

shopt -s nullglob

export POST_TRAIN_BENCH_RESULTS_DIR=${POST_TRAIN_BENCH_RESULTS_DIR:-results}

for gemini_dir in "${POST_TRAIN_BENCH_RESULTS_DIR}"/gemini*/; do
    if [ -d "$gemini_dir" ]; then
        for subdir in "$gemini_dir"*/; do
            if [ -d "$subdir" ]; then
                echo "Processing ${subdir}"
                python3 scripts/parse_jsonl/gemini_parse_jsonl.py ${subdir}/solve_out.txt -o ${subdir}/solve_parsed.txt
            fi
        done
    fi
done