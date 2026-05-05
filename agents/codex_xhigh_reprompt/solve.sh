#!/bin/bash
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Set reasoning effort to xhigh (prepend to ensure precedence)
file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "xhigh"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

MIN_REMAINING_MINUTES=30

codex --search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$PROMPT"

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

    codex --search exec resume --last --json -c model_reasoning_summary=detailed --skip-git-repo-check --yolo --model "$AGENT_CONFIG" "$CONTINUATION_PROMPT"
done
