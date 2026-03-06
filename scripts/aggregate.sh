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
    claude_claude-opus-4-6_10h_run1_old_container \
    claude_claude-opus-4-6_10h_run2 \
    claude_claude-opus-4-6_10h_run3 \
    codex_non_api_gpt-5.3-codex_10h_run1 \
    codex_non_api_gpt-5.3-codex_10h_run2 \
    codex_non_api_gpt-5.3-codex_10h_run3 \
    opencode_opencode_glm-5_10h_run2 \
    opencode_opencode_kimi-k2.5_10h_run2 \
    opencode_opencode_minimax-m2.5-free_10h_run2 \
    opencode_zai_glm-5_10h_run2 \
    # opencode_anthropic_claude-opus-4-5_10h \
    # opencode_opencode_big-pickle_10h \
    # opencode_opencode_gemini-3-pro_10h \
    # opencode_opencode_glm-4.7-free_10h \
    # opencode_opencode_gpt-5.1-codex-max_10h \
    # opencode_opencode_kimi-k2-thinking_10h \
    # opencode_opencode_minimax-m2.1-free_10h \
    # qwen3max_qwen3-max-2026-01-23_10h

# python scripts/aggregate_together.py \
#     opencode_anthropic_claude-opus-4-5_10h \
#     opencode_opencode_big-pickle_10h \
#     opencode_opencode_gemini-3-pro_10h \
#     opencode_opencode_glm-4.7-free_10h \
#     opencode_opencode_gpt-5.1-codex-max_10h \
#     opencode_opencode_kimi-k2-thinking_10h \
#     opencode_opencode_minimax-m2.1-free_10h