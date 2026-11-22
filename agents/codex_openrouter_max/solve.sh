#!/bin/bash
# Codex CLI at max reasoning effort on an OpenAI model, paid per token through OpenRouter (key
# OPENROUTER_API_KEY_OPENAI), with codex's web search through the ChatGPT subscription.
# agent_config: codex's model slug, e.g. gpt-6.2-astra; the OpenRouter model is openai/<agent_config>.
#
# codex talks only to a local relay (or_relay.py, written below), configured as its model provider. The relay holds
# both credentials and sends each upstream only its own:
# - model turns (POST /responses) go to OpenRouter, with OpenRouter's provider routing added to the body (codex cannot
#   add body fields): only OpenAI's own standard endpoint, no fallbacks. OpenRouter also serves these models from Azure
#   and Bedrock, and codex's encrypted reasoning items and encrypted search results only decrypt at OpenAI; OpenAI's
#   "fast" endpoint (twice the price) would otherwise also match "openai", so it is ignored explicitly.
# - web searches (POST /alpha/search: codex's "standalone" web search, the only one codex offers for the GPT-6
#   models; OpenRouter has no such endpoint) go to the ChatGPT backend with the subscription's access token. The token
#   comes from agents/codex_non_api/auth.json (the judges' subscription; the auth.json symlink in this directory makes
#   run_task.sh bind-mount it at ~/.codex/auth.json). The relay reads it for every search and never refreshes it:
#   other jobs and the judges rotate it, and a second refresher would race them.
# The sandbox shares the node's network with other jobs, so the relay listens on a random free port and accepts only
# requests with a random per-run token, which codex sends as its bearer token. The port range 21000-28999 avoids the
# ephemeral range and common defaults of training tools (torch.distributed 29400/29500, SGLang 30000).
# codex runs with its own CODEX_HOME (no subscription token there) and without the OpenRouter key in its environment,
# so neither credential is in the environment of the agent's commands. Both stay readable for a determined agent (the
# key in /proc/<pid>/environ of solve.sh, its relay loop and the processes that started it, as for every agent; the
# token in ~/.codex/auth.json, as for the codex_non_api agents).

set -u

TASK_DIR=/home/ben/task
if [ -z "${OPENROUTER_API_KEY_OPENAI:-}" ]; then
    echo "ERROR: OPENROUTER_API_KEY_OPENAI is not set (agents/codex_openrouter_max/api_keys.json allows it; is it in .env?)"
    exit 1
fi
# Only the relay gets the key (on stdin). Unset before anything else runs, so that the CLI install, codex and the
# agent's commands do not have it in their environment.
OPENROUTER_KEY="$OPENROUTER_API_KEY_OPENAI"
unset OPENROUTER_API_KEY_OPENAI
AUTH_JSON="/home/ben/.codex/auth.json"
if [ ! -s "$AUTH_JSON" ]; then
    echo "ERROR: $AUTH_JSON is missing or empty (agents/codex_openrouter_max/auth.json should link to ../codex_non_api/auth.json)"
    exit 1
fi

OPENROUTER_MODEL="openai/${AGENT_CONFIG}"
RELAY_DIR="/home/ben/.codex_relay"
RELAY_LOG="/home/ben/agent_logs/or_relay.log"
export CODEX_HOME="/home/ben/.codex_openrouter"
mkdir -p "$RELAY_DIR" /home/ben/agent_logs/codex_sessions "$CODEX_HOME"

# The model must exist on OpenRouter with reasoning effort max and an endpoint of OpenAI itself.
python3 -I - "$OPENROUTER_MODEL" <<'EOF' || exit 1
import json
import sys
import time
import urllib.request


def get(url: str) -> dict:
    for attempt in (1, 2, 3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, json.JSONDecodeError) as e:
            if attempt == 3:
                raise
            print(f"OpenRouter check: {url} failed ({e!r}), retrying in 20 s")
            time.sleep(20)
    raise AssertionError("unreachable")


model = sys.argv[1]
models = get("https://openrouter.ai/api/v1/models")["data"]
matches = [m for m in models if m["id"] == model]
if not matches:
    sys.exit(f"ERROR: OpenRouter has no model {model}")
efforts = (matches[0].get("reasoning") or {}).get("supported_efforts") or []
if "max" not in efforts:
    sys.exit(f"ERROR: OpenRouter's {model} does not support reasoning effort max (supported: {efforts})")
url = f"https://openrouter.ai/api/v1/models/{model}/endpoints"
tags = [e.get("tag") for e in get(url)["data"]["endpoints"]]
if "openai" not in tags:
    sys.exit(f"ERROR: OpenRouter's {model} has no endpoint of OpenAI itself (endpoints: {tags})")
print(f"OpenRouter {model}: reasoning efforts {efforts}, endpoints {tags}")
EOF

# codex knows gpt-6.2-astra from 0.166.0 on. Pinned to the newest release when this agent was added (2026-11-18);
# overrides CODEX_CLI_VERSION from .env.
export CODEX_CLI_VERSION="0.170.0"
# Auto-update the CLI harness to the pinned release and record its version.
bash /home/ben/update_agent_cli.sh codex || exit 1

cat > "$RELAY_DIR/or_relay.py" <<'PYEOF'
#!/usr/bin/env python3
"""Local relay between the codex CLI and its two upstreams (see agents/codex_openrouter_max/solve.sh).

Every request must carry the per-run relay token as its bearer token; others get 401. Routes (anything else is
rejected and logged):
- GET /relay/health -> 200 (solve.sh's readiness check: with the token, only this run's relay answers 200).
- POST /api/v1/responses -> https://openrouter.ai/api/v1/responses with the OpenRouter key; the model slug is mapped to
  OpenRouter's id and the body gets PROVIDER (OpenRouter's provider routing).
- POST /api/v1/alpha/search -> https://chatgpt.com/backend-api/codex/alpha/search with the ChatGPT subscription's
  access token and account id from auth.json, read for every request (never refreshed here).
Responses are streamed back as they arrive. One log line per request (route, status, bytes, seconds; for OpenRouter
also the generation id and the usage of the response).
Usage: or_relay.py <port> <codex model slug> <OpenRouter model id> <auth.json> <log file> [<debug dir>]
stdin: the OpenRouter key on the first line, the relay token on the second. With a debug dir, every request body and
response is saved there.
"""
import hmac
import http.client
import http.server
import itertools
import json
import os
import ssl
import sys
import threading
import time
import traceback
import urllib.parse

# "openai" alone also matches OpenAI's "fast" endpoint (twice the price): OpenRouter's routing report for a request with
# order ["openai/fast"] showed it removed by the ignore filter only (2026-11-18).
PROVIDER = {"order": ["openai"], "allow_fallbacks": False, "ignore": ["openai/fast", "openai/ultrafast"]}
# route name -> (upstream host, upstream path, socket timeout in seconds)
UPSTREAMS = {
    "openrouter": ("openrouter.ai", "/api/v1/responses", 1800),
    "chatgpt_search": ("chatgpt.com", "/backend-api/codex/alpha/search", 300),
}
ROUTES = {"/api/v1/responses": "openrouter", "/api/v1/alpha/search": "chatgpt_search"}
HEALTH_PATH = "/relay/health"
# Hop-by-hop headers, headers http.client sets itself, and credentials (the relay sets those per upstream).
# accept-encoding is dropped so that upstreams answer uncompressed.
DROP_REQUEST_HEADERS = {"connection", "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding",
                        "upgrade", "host", "content-length", "accept-encoding", "authorization", "chatgpt-account-id"}
DROP_RESPONSE_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "trailer", "upgrade"}
# Server-sent events that end an OpenRouter response; their data is logged (generation id, status, usage).
FINAL_EVENTS = (b'"type":"response.completed"', b'"type":"response.failed"', b'"type":"response.incomplete"',
                b'"type":"error"')

PORT = int(sys.argv[1])
CODEX_MODEL = sys.argv[2]
OPENROUTER_MODEL = sys.argv[3]
AUTH_JSON = sys.argv[4]
LOG_FILE = sys.argv[5]
DEBUG_DIR = sys.argv[6] if len(sys.argv) > 6 else None
OPENROUTER_KEY = sys.stdin.readline().strip()
RELAY_TOKEN = sys.stdin.readline().strip()
if not OPENROUTER_KEY.startswith("sk-or-"):
    sys.exit("or_relay: expected an OpenRouter key (sk-or-...) on the first line of stdin")
if len(RELAY_TOKEN) < 32:
    sys.exit("or_relay: expected the relay token (at least 32 characters) on the second line of stdin")
EXPECTED_AUTHORIZATION = f"Bearer {RELAY_TOKEN}".encode()
SSL_CONTEXT = ssl.create_default_context()
# Head nodes reach the internet only through an HTTP proxy (used for tests); the agent sandbox has no proxy variables.
HTTPS_PROXY = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
LOG_LOCK = threading.Lock()
COUNTER = itertools.count(1)


def log(msg: str) -> None:
    # A log that cannot be written (full disk, deleted ~/agent_logs) must not stop the relay: without it, codex cannot
    # reach the model, and the agent cannot clean up.
    try:
        with LOG_LOCK, open(LOG_FILE, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")
    except OSError:
        pass


def upstream_connection(host: str, timeout: int) -> http.client.HTTPSConnection:
    if not HTTPS_PROXY:
        return http.client.HTTPSConnection(host, timeout=timeout, context=SSL_CONTEXT)
    proxy = urllib.parse.urlsplit(HTTPS_PROXY)
    conn = http.client.HTTPSConnection(proxy.hostname, proxy.port, timeout=timeout, context=SSL_CONTEXT)
    conn.set_tunnel(host, 443)
    return conn


def chatgpt_credentials() -> tuple[str, str]:
    # codex rewrites auth.json in place when it refreshes the token (judges, other jobs), so a read can catch a
    # half-written file: read again once after a second.
    for attempt in (1, 2):
        try:
            with open(AUTH_JSON) as f:
                tokens = json.load(f)["tokens"]
            return tokens["access_token"], tokens["account_id"]
        except (json.JSONDecodeError, KeyError):
            if attempt == 2:
                raise
            time.sleep(1)
    raise AssertionError("unreachable")


def final_event_summary(line: bytes) -> str:
    # Only feeds the log; a malformed event must not break the stream that codex is reading.
    try:
        event = json.loads(line[len(b"data: "):])
    except json.JSONDecodeError as e:
        return f"UNPARSEABLE final event ({e}): {line[:300]!r}"
    response = event.get("response")
    if response is None:
        return f"event={json.dumps(event)[:500]}"
    usage = response.get("usage") or {}
    summary = (f"gen={response.get('id')} status={response.get('status')} in={usage.get('input_tokens')} "
               f"cached={(usage.get('input_tokens_details') or {}).get('cached_tokens')} "
               f"out={usage.get('output_tokens')} "
               f"reasoning={(usage.get('output_tokens_details') or {}).get('reasoning_tokens')} "
               f"cost={usage.get('cost')}")
    if response.get("error") or response.get("incomplete_details"):
        summary += f" error={json.dumps(response.get('error'))[:300]} incomplete={response.get('incomplete_details')}"
    return summary


class RelayHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_request(self, code="-", size="-"):
        pass  # every request gets its own log line from the handlers

    def log_error(self, format, *args):  # noqa: A002 (signature of BaseHTTPRequestHandler)
        # Malformed requests and unsupported methods (501), which never reach the handlers.
        log(f"HTTP ERROR from {self.client_address[0]}: {format % args}")

    def authorized(self, n: int) -> bool:
        given = (self.headers.get("Authorization") or "").encode()
        if hmac.compare_digest(given, EXPECTED_AUTHORIZATION):
            return True
        log(f"#{n} {self.command} {self.path} REJECTED: missing or wrong relay token (from {self.client_address[0]})")
        self.send_error_json(401, "missing or wrong relay token")
        return False

    def do_GET(self):
        n = next(COUNTER)
        if not self.authorized(n):
            return
        if self.path == HEALTH_PATH:
            self.send_json(200, {"relay": "ok"})
            return
        log(f"#{n} GET {self.path} REJECTED: no such route")
        self.send_error_json(404, f"no route for GET {self.path}")

    def do_POST(self):
        n = next(COUNTER)
        self.response_started = False
        try:
            body = self.read_body()
            if self.authorized(n):
                self.relay(n, body)
        except Exception:
            log(f"#{n} POST {self.path} RELAY ERROR\n{traceback.format_exc()}")
            if not self.response_started:
                self.send_error_json(502, "relay error, see or_relay.log")

    def read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip(), 16)
                if size == 0:
                    self.rfile.readline()
                    return b"".join(chunks)
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
        return self.rfile.read(int(self.headers["Content-Length"]))

    def send_json(self, status: int, obj: dict) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json(status, {"error": {"message": f"or_relay: {message}", "code": status}})

    def relay(self, n: int, body: bytes) -> None:
        start = time.time()
        route = ROUTES.get(self.path)
        if route is None:
            log(f"#{n} POST {self.path} REJECTED: no such route")
            self.send_error_json(404, f"no route for POST {self.path}")
            return
        if self.headers.get("Content-Encoding"):
            log(f"#{n} POST {self.path} REJECTED: compressed body ({self.headers['Content-Encoding']})")
            self.send_error_json(400, f"compressed request bodies are not supported ({self.headers['Content-Encoding']})")
            return
        data = json.loads(body)
        if data.get("model") != CODEX_MODEL:
            log(f"#{n} POST {self.path} REJECTED: model {data.get('model')!r}, this relay serves only {CODEX_MODEL!r}")
            self.send_error_json(400, f"model {data.get('model')!r} is not served, only {CODEX_MODEL!r}")
            return
        headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP_REQUEST_HEADERS}
        if route == "openrouter":
            data["model"] = OPENROUTER_MODEL
            data["provider"] = PROVIDER
            headers["Authorization"] = f"Bearer {OPENROUTER_KEY}"
            desc = (f"openrouter model={OPENROUTER_MODEL} reasoning={json.dumps(data.get('reasoning'))} "
                    f"input_items={len(data.get('input') or [])}")
        else:
            access_token, account_id = chatgpt_credentials()
            headers["Authorization"] = f"Bearer {access_token}"
            headers["ChatGPT-Account-Id"] = account_id
            desc = f"chatgpt_search commands={json.dumps(data.get('commands'))[:400]}"
        body = json.dumps(data).encode()
        if DEBUG_DIR:
            with open(f"{DEBUG_DIR}/req_{n:05d}_{route}.json", "w") as f:
                json.dump({"headers": {k: v for k, v in headers.items() if k.lower() != "authorization"},
                           "body": data}, f, indent=1)
        host, path, timeout = UPSTREAMS[route]
        conn = upstream_connection(host, timeout)
        try:
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                log(f"#{n} {desc} UPSTREAM ERROR before the response: {e!r}")
                self.send_error_json(502, f"upstream error: {e!r}")
                return
            self.response_started = True
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in DROP_RESPONSE_HEADERS:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            nbytes = 0
            head = b""
            pending = b""
            finals = []
            dump = open(f"{DEBUG_DIR}/resp_{n:05d}_{route}.txt", "wb") if DEBUG_DIR else None
            try:
                while True:
                    try:
                        chunk = resp.read1(65536)
                    except (OSError, http.client.HTTPException) as e:
                        log(f"#{n} {desc} status={resp.status} UPSTREAM ERROR after {nbytes} bytes, "
                            f"{time.time() - start:.0f}s: {e!r}")
                        return
                    if not chunk:
                        break
                    nbytes += len(chunk)
                    if len(head) < 2000:
                        head += chunk[:2000 - len(head)]
                    if dump:
                        dump.write(chunk)
                    if route == "openrouter":
                        lines = (pending + chunk).split(b"\n")
                        pending = lines.pop()
                        for line in lines:
                            if line.startswith(b"data: ") and any(m in line[:300] for m in FINAL_EVENTS):
                                finals.append(final_event_summary(line.rstrip(b"\r")))
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except OSError as e:
                        log(f"#{n} {desc} status={resp.status} CLIENT GONE after {nbytes} bytes, "
                            f"{time.time() - start:.0f}s: {e!r}")
                        return
            finally:
                if dump:
                    dump.close()
        finally:
            conn.close()
        line = f"#{n} {desc} status={resp.status} bytes={nbytes} {time.time() - start:.0f}s"
        if finals:
            line += " " + " | ".join(finals)
        if resp.status != 200:
            line += f" body={head.decode('utf-8', 'replace')}"
        log(line)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    server = Server(("127.0.0.1", PORT), RelayHandler)
    log(f"or_relay pid {os.getpid()} listening on 127.0.0.1:{PORT}: {CODEX_MODEL} -> OpenRouter {OPENROUTER_MODEL} "
        f"provider={json.dumps(PROVIDER)}, web search -> ChatGPT subscription ({AUTH_JSON})")
    server.serve_forever()
PYEOF

# A random token (the relay rejects requests without it) and a random free port (see the top of this file).
RELAY_TOKEN=$(python3 -I -c 'import secrets; print(secrets.token_hex(24))') || exit 1
pick_port() {
    python3 -I - <<'EOF'
import random
import socket
import sys

for _ in range(200):
    port = random.randrange(21000, 29000)
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    sys.exit(0)
sys.exit("ERROR: no free port found in 21000-28999")
EOF
}

# If the relay dies (e.g. the agent kills python processes), the loop restarts it and codex retries the request.
relay_loop() {
    while true; do
        printf '%s\n%s\n' "$OPENROUTER_KEY" "$RELAY_TOKEN" | python3 -I "$RELAY_DIR/or_relay.py" "$RELAY_PORT" \
            "$AGENT_CONFIG" "$OPENROUTER_MODEL" "$AUTH_JSON" "$RELAY_LOG"
        relay_status=$?
        echo "$(date -u +%FT%TZ) or_relay exited with status ${relay_status}, restarting in 2 s"
        sleep 2
    done
}
# Readiness: only this run's relay answers the health check with this token (the token goes on stdin, not into a
# command line).
relay_answers() {
    printf '%s' "$RELAY_TOKEN" | python3 -I -c '
import sys, urllib.request
token = sys.stdin.read()
request = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/relay/health", headers={"Authorization": f"Bearer {token}"})
urllib.request.urlopen(request, timeout=2)' "$RELAY_PORT" 2>/dev/null
}
RELAY_LOOP_PID=""
trap '[ -n "$RELAY_LOOP_PID" ] && kill "$RELAY_LOOP_PID" 2>/dev/null; pkill -f "$RELAY_DIR/or_relay.py"' EXIT
relay_up=0
# A port can be taken between the check and the relay's bind: then try again with another port.
for attempt in 1 2 3; do
    RELAY_PORT=$(pick_port) || exit 1
    # The loop writes only to the relay log, so it never holds solve.sh's stdout open.
    relay_loop >> "$RELAY_LOG" 2>&1 &
    RELAY_LOOP_PID=$!
    for _ in $(seq 30); do
        if relay_answers; then
            relay_up=1
            break
        fi
        sleep 1
    done
    [ "$relay_up" = 1 ] && break
    echo "or_relay did not answer on port ${RELAY_PORT} (attempt ${attempt}/3)"
    kill "$RELAY_LOOP_PID"
    pkill -f "$RELAY_DIR/or_relay.py"
done
if [ "$relay_up" != 1 ]; then
    echo "ERROR: or_relay did not start; its log:"
    cat "$RELAY_LOG"
    exit 1
fi

# CODEX_HOME: the container's codex config (project trust, shell environment) plus the relay as model provider.
# Top-level keys must come before the tables of the copied config. stream_idle_timeout_ms: codex's default (5 min)
# could abort and resend a turn during a long silent reasoning phase at max effort.
{
    printf 'model_reasoning_effort = "max"\n'
    printf 'model_provider = "openrouter_relay"\n\n'
    cat /home/ben/.codex/config.toml
    cat <<EOF

[model_providers.openrouter_relay]
name = "OpenRouter via local relay"
base_url = "http://127.0.0.1:${RELAY_PORT}/api/v1"
experimental_bearer_token = "${RELAY_TOKEN}"
wire_api = "responses"
supports_standalone_web_search = true
request_max_retries = 10
stream_max_retries = 10
stream_idle_timeout_ms = 900000
EOF
} > "$CODEX_HOME/config.toml"
# codex's session rollouts (every request item and event) go straight to ~/agent_logs, which run_task.sh keeps.
ln -s /home/ben/agent_logs/codex_sessions "$CODEX_HOME/sessions"

# A codex session that ends with an error (e.g. an OpenAI outage longer than codex's retries; there are no fallback
# providers) is resumed with a short note while time is left. Before each resume the loop waits, 1, 2, 4, ... minutes
# (at most 30), so that 10 resumes in a row span about 3 hours of outage; a session that ran for 30 minutes or more
# made progress, and the count starts again. A session that ends normally is not resumed, as for the other codex
# agents. The loop uses absolute paths and its own copy of timer.sh: on 2026-11-06 an opencode agent's tests deleted
# /home/ben/task.
TIMER="$HOME/solve_timer.sh"
cp "$TASK_DIR/timer.sh" "$TIMER" || exit 1
LAST_RUN="$HOME/codex_last_run.jsonl"
MAX_RESUMES=10
CODEX_ARGS=(--json -c model_reasoning_summary=detailed --skip-git-repo-check --model "$AGENT_CONFIG")

START=$(date +%s)
printf '%s' "$PROMPT" | codex --search exec "${CODEX_ARGS[@]}" --yolo | tee "$LAST_RUN"
CODEX_EXIT=${PIPESTATUS[1]}
RUN_SECONDS=$(( $(date +%s) - START ))
THREAD_ID=$(python3 -I -c '
import json, sys
for line in open(sys.argv[1]):
    if line.startswith("{\"type\":\"thread.started\""):
        print(json.loads(line)["thread_id"])
        break' "$LAST_RUN")

resumes=0
while [ "$CODEX_EXIT" -ne 0 ]; do
    [ "$RUN_SECONDS" -ge 1800 ] && resumes=0
    resumes=$((resumes + 1))
    if [ "$resumes" -gt "$MAX_RESUMES" ]; then
        echo "=== solve.sh: ${MAX_RESUMES} resumes in a row ended with an error within 30 minutes; not resuming again"
        break
    fi
    if [ -z "$THREAD_ID" ]; then
        echo "ERROR: codex exited with status ${CODEX_EXIT}, and its output has no thread id to resume"
        exit 1
    fi
    WAIT_MINUTES=$(( 1 << (resumes - 1) ))
    [ "$WAIT_MINUTES" -gt 30 ] && WAIT_MINUTES=30
    echo "=== solve.sh: codex exited with status ${CODEX_EXIT}; resume ${resumes}/${MAX_RESUMES} in ${WAIT_MINUTES} min"
    sleep "$((WAIT_MINUTES * 60))"
    TIMER_OUTPUT=$(bash "$TIMER")
    echo "$TIMER_OUTPUT" | grep -q "expired" && break
    REMAINING=$(echo "$TIMER_OUTPUT" | tail -n 1)
    if ! [[ "$REMAINING" =~ ^[0-9]+:[0-9][0-9]$ ]]; then
        echo "ERROR: $TIMER printed no remaining time (H:MM): $TIMER_OUTPUT; not resuming"
        exit 1
    fi
    cd "$TASK_DIR" || { echo "ERROR: $TASK_DIR does not exist; not resuming"; exit 1; }
    MESSAGE="Your session ended with an error (codex exit status ${CODEX_EXIT}), but ${REMAINING%%:*}h ${REMAINING#*:}m"
    MESSAGE+=" of the time limit remain. Processes that codex ran for you may have been stopped when the session ended;"
    MESSAGE+=" check what is still running. Continue working on the task."
    echo "=== solve.sh: resume ${resumes}/${MAX_RESUMES} of codex thread ${THREAD_ID} with the message: ${MESSAGE}"
    START=$(date +%s)
    printf '%s' "$MESSAGE" | codex --search exec resume "${CODEX_ARGS[@]}" --dangerously-bypass-approvals-and-sandbox \
        "$THREAD_ID" - | tee "$LAST_RUN"
    CODEX_EXIT=${PIPESTATUS[1]}
    RUN_SECONDS=$(( $(date +%s) - START ))
done
exit "$CODEX_EXIT"
