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

# python scripts/aggregate_summary.py claude_claude-sonnet-4-5 claude_claude-opus-4-5 codex_gpt-5.1-codex-max codex_gpt-5.2 gemini_models_gemini-3-pro-preview 
# python scripts/aggregate_summary.py claude_claude-opus-4-5_1h_v4 claude_claude-opus-4-5_2h_v4 claude_claude-opus-4-5_5h_v4 codex_gpt-5.1-codex-max_1h_v4 codex_gpt-5.1-codex-max_2h_v4 codex_gpt-5.1-codex-max_5h_v4 gemini_models_gemini-3-pro-preview_1h_v4 gemini_models_gemini-3-pro-preview_2h_v4 gemini_models_gemini-3-pro-preview_5h_v4 codex_gpt-5.1-codex-max_10h_v4_seed1 codex_gpt-5.1-codex-max_10h_v4_seed2 codex_gpt-5.2_10h_v4
# python scripts/aggregate_together.py claude_claude-sonnet-4-5 claude_claude-opus-4-5 codex_gpt-5.1-codex-max codex_gpt-5.2 gemini_models_gemini-3-pro-preview 

python scripts/aggregate_summary.py claude_claude-haiku-4-5_10h_v6 claude_claude-opus-4-5_10h_final_v3 claude_claude-opus-4-5_10h_v5 claude_claude-opus-4-5_10h_v6_seed1 claude_claude-opus-4-5_1h_v4 claude_claude-opus-4-5_2h_v4 claude_claude-opus-4-5_5h_v4 claude_claude-sonnet-4-5_10h_final_v3 codex_gpt-5.1-codex-max_10h_final_v3 codex_gpt-5.1-codex-max_10h_v4_seed1 codex_gpt-5.1-codex-max_10h_v4_seed2 codex_gpt-5.1-codex-max_1h_v4 codex_gpt-5.1-codex-max_2h_v4 codex_gpt-5.1-codex-max_5h_v4 codex_gpt-5.2-codex_10h_v6 codex_gpt-5.2-codex_10h_v6_seed1 codex_gpt-5.2-codex_10h_v6_seed2 codex_gpt-5.2_10h_v4 codex_gpt-5.2_10h_v6_seed1 codex_gpt-5.2_10h_v6_seed2 codexhigh_gpt-5.1-codex-max_10h_v7 codexhigh_gpt-5.1-codex-max_10h_v7_seed1 codexhigh_gpt-5.1-codex_10h_v7 codexhigh_gpt-5.1-codex_10h_v7_seed1 codexlow_gpt-5.1-codex-max_10h_v7 codexlow_gpt-5.1-codex-max_10h_v7_seed1 codexlow_gpt-5.1-codex_10h_v7 codexlow_gpt-5.1-codex_10h_v7_seed1 gemini_models_gemini-3-pro-preview_10h_final_v3 gemini_models_gemini-3-pro-preview_10h_v5 gemini_models_gemini-3-pro-preview_10h_v6_seed1 gemini_models_gemini-3-pro-preview_1h_v4 gemini_models_gemini-3-pro-preview_2h_v4 gemini_models_gemini-3-pro-preview_5h_v4

# python scripts/aggregate_together.py claude_claude-opus-4-5_1h_v4 claude_claude-opus-4-5_2h_v4 claude_claude-opus-4-5_5h_v4 codex_gpt-5.1-codex-max_1h_v4 codex_gpt-5.1-codex-max_2h_v4 codex_gpt-5.1-codex-max_5h_v4 gemini_models_gemini-3-pro-preview_1h_v4 gemini_models_gemini-3-pro-preview_2h_v4 gemini_models_gemini-3-pro-preview_5h_v4 codex_gpt-5.1-codex-max_10h_v4_seed1 codex_gpt-5.1-codex-max_10h_v4_seed2 codex_gpt-5.2_10h_v