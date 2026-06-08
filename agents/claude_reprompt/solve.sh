#!/bin/bash

export BASH_MAX_TIMEOUT_MS="36000000"

MIN_REMAINING_MINUTES=30

# Auto-update the CLI harness to the latest release and record its version.
bash /home/ben/update_agent_cli.sh claude

claude --print --verbose --model "$AGENT_CONFIG" --output-format stream-json \
    --dangerously-skip-permissions "$PROMPT"

# Re-prompt loop: if the agent finishes early, resume the session
while true; do
    TIMER_OUTPUT=$(bash timer.sh 2>/dev/null)
    if echo "$TIMER_OUTPUT" | grep -q "expired"; then
        break
    fi

    REMAINING_HOURS=$(echo "$TIMER_OUTPUT" | grep -oP '^\d+(?=:)')
    REMAINING_MINS=$(echo "$TIMER_OUTPUT" | grep -oP '(?<=:)\d+')
    TOTAL_REMAINING_MINS=$(( REMAINING_HOURS * 60 + REMAINING_MINS ))

    if [ "$TOTAL_REMAINING_MINS" -lt "$MIN_REMAINING_MINUTES" ]; then
        break
    fi

    CONTINUATION_PROMPT="You still have ${REMAINING_HOURS}h ${REMAINING_MINS}m remaining. Please continue improving your result and maximize performance."

    claude --print --verbose --continue --model "$AGENT_CONFIG" --output-format stream-json \
        --dangerously-skip-permissions "$CONTINUATION_PROMPT"
done
