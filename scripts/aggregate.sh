#!/bin/bash
source src/commit_utils/set_env_vars.sh

echo "==============================="
echo "Aggregating method results..."
python scripts/aggregate_methods.py
echo "==============================="
echo "Aggregating time results..."
python scripts/aggregate_time_baselines.py
echo "==============================="
echo "Aggregating contamination results..."
python scripts/aggregate_contamination.py

python scripts/aggregate_time.py
sleep 1
python scripts/aggregate_final.py
sleep 1
python scripts/aggregate_summary.py \
    opencode_anthropic_claude-opus-4-5_10h \
    opencode_opencode_big-pickle_10h \
    opencode_opencode_gemini-3-pro_10h \
    opencode_opencode_glm-4.7-free_10h \
    opencode_opencode_gpt-5.1-codex-max_10h \
    opencode_opencode_kimi-k2-thinking_10h \
    opencode_opencode_minimax-m2.1-free_10h \
    qwen3max_qwen3-max-2026-01-23_10h

# python scripts/aggregate_together.py \
#     opencode_anthropic_claude-opus-4-5_10h \
#     opencode_opencode_big-pickle_10h \
#     opencode_opencode_gemini-3-pro_10h \
#     opencode_opencode_glm-4.7-free_10h \
#     opencode_opencode_gpt-5.1-codex-max_10h \
#     opencode_opencode_kimi-k2-thinking_10h \
#     opencode_opencode_minimax-m2.1-free_10h