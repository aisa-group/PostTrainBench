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
sleep 1
python scripts/aggregate_final.py
sleep 1
python scripts/aggregate_summary.py claude_claude-sonnet-4-5_final_final codex_gpt-5.1-codex_final_final