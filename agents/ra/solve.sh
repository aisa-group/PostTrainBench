#!/bin/bash

ENABLE_SEARCH=""
if [ -n "${TAVILY_API_KEY:-}" ]; then
    ENABLE_SEARCH="--enable-search"
fi

ra $ENABLE_SEARCH --exec --stream-json --model "$AGENT_CONFIG" "$PROMPT"
