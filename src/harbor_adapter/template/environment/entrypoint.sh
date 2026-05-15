#!/bin/bash
# PostTrainBench container entrypoint.
#
# Runs as PID 1 so its stdout is what Modal/Harbor stream live to the
# sandbox dashboard. We:
#   1. background `tail -F` of the agent and verifier log files so their
#      output flows through PID 1 stdout in real time
#   2. start a system monitor daemon that writes GPU/CPU/memory stats to
#      /logs/agent/system_monitor.log every 60s (parity with condor's
#      src/utils/system_monitor.sh)
#
# The timer start file (/timer_start) is *not* written here — it is
# written by the task.toml healthcheck, which runs immediately before
# the agent launches. That gives the timer the tightest possible
# alignment with actual agent start time.

set -e

mkdir -p /logs/agent /logs/verifier

# Pre-create well-known log files so `tail -F` can attach to them
# immediately, before the agent or verifier creates them.
#
# Agent side: Harbor's installed agents tee stdout into /logs/agent/<name>.txt
# (we cover the four we care about today: claude-code, codex, gemini, opencode).
#
# Verifier side: tests/test.sh tees evaluate.py output into
# /logs/verifier/final_eval_${N}.txt for each retry phase, plus Harbor itself
# writes /logs/verifier/test-stdout.txt for the test.sh process.
touch /logs/agent/claude-code.txt \
      /logs/agent/codex.txt \
      /logs/agent/gemini.txt \
      /logs/agent/opencode.txt \
      /logs/verifier/test-stdout.txt \
      /logs/verifier/final_eval_1.txt \
      /logs/verifier/final_eval_2.txt \
      /logs/verifier/final_eval_3.txt \
      /logs/verifier/contamination_judgement.txt \
      /logs/verifier/disallowed_model_judgement.txt

# Stream every agent and verifier .txt log into PID 1's stdout. -q suppresses
# the "==> file <==" headers tail prints between files so the stream reads
# like a single transcript. system_monitor.log is intentionally NOT included
# (.log extension) — it's for postmortem analysis, not live streaming.
tail -F -q /logs/agent/*.txt /logs/verifier/*.txt &

# Background system monitor (parity with condor pipeline). Logs to
# /logs/agent/system_monitor.log so Harbor downloads it with the rest of
# the agent dir at trial end.
/usr/local/bin/system_monitor.sh &

# Keep the sandbox alive for sandbox.exec calls.
exec sleep infinity
