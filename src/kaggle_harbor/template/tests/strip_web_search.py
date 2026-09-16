#!/usr/bin/env python3
"""A loopback shim that removes codex's `web_search` tool from every request.

DIAGNOSTIC ONLY. This is not part of a faithful run.

Why it has to exist at all. Measured in prod (run 866861), with the token the
platform mints for the run:

    POST /responses  no tools                 -> 200
    POST /responses  one function tool        -> 200
    POST /responses  tools:[{web_search}]     -> 403
                     "Google Search Grounding is not enabled for your account"
    POST /chat/completions  function tool     -> 200

So function/tool calling is fine and only the provider-NATIVE web_search tool
is refused. codex advertises web_search on every single request and there is no
config that removes it: `--search` / no `--search`, `tools.web_search=false`,
`tools.web_search_request=false`, `tools.enabled_tools=[]` and
`experimental_supported_tools=[]` all produce the identical 11-tool list
(verified locally by pointing codex at a recording HTTP server). The tool list
is not user-configurable in codex 0.137.0.

The real fix is the feature flag MODEL_PROXY_NATIVE_TOOL_ACCESS (4831), which
is what puts `allow_model_native_tools` on the minted token
(Kaggle.Services.Shared/ModelProxy/Utils/TokenUtil.cs:151-154). Until that is
granted, this shim is the only way to find out whether ANYTHING ELSE in the
judge path works. It answers exactly the question that was asked of it: with
web search stripped from the judges and models, does the rest of the judge
path work and judge correctly?

A run using this shim is NOT 1:1 with upstream and its reward must be reported
as a pipeline proof, not as a benchmark result.
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import urllib.error
import urllib.request

UPSTREAM = os.environ["PTB_SHIM_UPSTREAM"].rstrip("/")
PORT = int(os.environ.get("PTB_SHIM_PORT", "8931"))

# codex names it "web_search"; OpenAI has also shipped dated variants
# ("web_search_preview", "web_search_2025_08_26"). Prefix-match so a rename
# does not silently reintroduce the 403.
BLOCKED_PREFIX = "web_search"

_stripped = 0
_lock = threading.Lock()


def strip_tools(body: bytes) -> bytes:
    """Drop web_search from `tools`. Anything unparseable passes through."""
    global _stripped
    try:
        payload = json.loads(body)
    except Exception:
        return body
    if not isinstance(payload, dict):
        return body
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return body
    kept = []
    dropped = 0
    for t in tools:
        name = t.get("type") if isinstance(t, dict) else None
        if isinstance(name, str) and name.startswith(BLOCKED_PREFIX):
            dropped += 1
            continue
        kept.append(t)
    if not dropped:
        return body
    payload["tools"] = kept
    with _lock:
        _stripped += dropped
    return json.dumps(payload).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _relay(self, method: str) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else None
        if body:
            body = strip_tools(body)

        headers = {}
        for k, v in self.headers.items():
            # Hop-by-hop and length headers are recomputed below.
            if k.lower() in ("host", "content-length", "connection",
                             "transfer-encoding", "accept-encoding"):
                continue
            headers[k] = v
        # No compression: the response is copied through verbatim and codex
        # parses SSE, so a gzip body would have to be decoded here first.
        headers["Accept-Encoding"] = "identity"
        if body is not None:
            headers["Content-Length"] = str(len(body))

        url = UPSTREAM + self.path
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
            status, out_headers = resp.status, resp.headers
        except urllib.error.HTTPError as e:
            resp, status, out_headers = e, e.code, e.headers
        except Exception as e:                       # connection-level failure
            msg = json.dumps({"error": {"message": f"shim upstream error: {e}"}}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        self.send_response(status)
        for k, v in out_headers.items():
            if k.lower() in ("transfer-encoding", "content-length", "connection",
                             "content-encoding"):
                continue
            self.send_header(k, v)
        # Length is unknown for SSE, so always chunk and stream.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")

    def log_message(self, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    srv = Server(("127.0.0.1", PORT), Handler)
    print(f"[shim] 127.0.0.1:{PORT} -> {UPSTREAM} (stripping {BLOCKED_PREFIX}*)",
          file=sys.stderr, flush=True)
    srv.serve_forever()
