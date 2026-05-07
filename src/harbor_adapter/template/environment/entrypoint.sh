#!/usr/bin/env bash
# entrypoint.sh — Container entrypoint. Starts the timer daemon in the
# background and then execs whatever command Harbor (or docker run) passes.

/home/agent/timer.sh >/dev/null 2>&1 &

# Harbor/Modal creates the sandbox first and execs commands into it later.
# If no explicit command is provided, keep the sandbox alive.
if [ "$#" -eq 0 ]; then
    exec tail -f /dev/null
fi

exec "$@"
