#!/usr/bin/env python3
"""Runs HumanEval answers apart from the tests, for the final-evaluation scorer in evaluate_final_eval.py.

The scorer's checker process holds the HumanEval test and runs check(); only this module's connect() proxy stands in
for the model's function. The model's code runs in a separate apptainer container, started by with_answer_sandbox.sh
without network, without the host filesystems and in its own PID namespace, so it cannot reach the tests. Each call of
the proxy is forwarded to that container, and the result comes back as repr() parsed with ast.literal_eval, so the
checker only ever sees plain built-in values: no objects with a custom __eq__, and no code of the answer.

Inside the sandbox container:
    python answer_sandbox.py serve SOCKET_PATH
        The server, PID 1 of the container. Per connection it starts a fresh Python process (`run`) for one sample,
        and it exits once its stdin reaches EOF, which ends the container with every answer process in it. As PID 1
        it cannot be killed by the answer processes: the kernel drops signals that other members of a PID namespace
        send to its init, unless the init has a handler for them.
    python answer_sandbox.py run FD
        One sample's answer process, speaking the protocol below on the connected socket FD.

Protocol, each message a 4-byte big-endian length followed by UTF-8 JSON:
    checker -> answer   {"op": "load", "code": prompt + answer, "entry_point": name}
    answer  -> checker  {"loaded": true}  |  {"load_error": traceback}
    checker -> answer   {"op": "call", "args": repr(args), "kwargs": repr(kwargs)}
    answer  -> checker  {"value": repr(result)}  |  {"raised": traceback}
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import traceback
from typing import Any

# The checker exits with this code when the answer sandbox itself fails (unreachable socket, or a test argument that
# cannot be sent). The scorer raises on it instead of scoring the sample, so an infrastructure failure never counts as
# a wrong answer. Nothing the answer does leads to it.
HARNESS_ERROR_EXIT_CODE = 3

MAX_MESSAGE_BYTES = 64 * 1024 * 1024

# Hard limit on one answer process. Longer than evaluate_final_eval.py's VERIFY_TIMEOUT (30 s), so the checker's own
# timeout decides the score; this only reaps answer processes that outlive their checker.
ANSWER_PROCESS_TIMEOUT_SECONDS = 60


class AnswerError(Exception):
    """The answer process failed, exited, or broke the protocol. In the checker it fails the sample."""


def send(sock: socket.socket, message: dict[str, Any]) -> None:
    body = json.dumps(message).encode("utf-8")
    if len(body) > MAX_MESSAGE_BYTES:
        raise AnswerError(f"message of {len(body)} bytes exceeds the {MAX_MESSAGE_BYTES}-byte limit")
    sock.sendall(len(body).to_bytes(4, "big") + body)


def _read_exactly(sock: socket.socket, n: int) -> bytes | None:
    """n bytes from sock, or None if the peer closed the connection before sending any."""
    chunks = []
    received = 0
    while received < n:
        try:
            chunk = sock.recv(min(n - received, 1 << 20))
        except OSError as err:
            raise AnswerError(f"connection failed: {err}") from err
        if not chunk:
            if received == 0:
                return None
            raise AnswerError(f"connection closed after {received} of {n} bytes")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv(sock: socket.socket) -> dict[str, Any] | None:
    """The next message, or None if the peer closed the connection."""
    header = _read_exactly(sock, 4)
    if header is None:
        return None
    length = int.from_bytes(header, "big")
    if length > MAX_MESSAGE_BYTES:
        raise AnswerError(f"message of {length} bytes exceeds the {MAX_MESSAGE_BYTES}-byte limit")
    body = _read_exactly(sock, length)
    if body is None:
        raise AnswerError("connection closed before the message body")
    try:
        message = json.loads(body.decode("utf-8"))
    except ValueError as err:
        raise AnswerError(f"malformed message: {err}") from err
    if not isinstance(message, dict):
        raise AnswerError(f"malformed message: {message!r:.200}")
    return message


# ---------------------------------------------------------------------------
# Checker side (eval container)
# ---------------------------------------------------------------------------


def _harness_error(reason: str) -> None:
    print(f"answer sandbox failure: {reason}", file=sys.stderr, flush=True)
    sys.exit(HARNESS_ERROR_EXIT_CODE)


def _encode(value: Any) -> str:
    """repr(value), after checking that ast.literal_eval gives the value back (the answer process parses it so)."""
    encoded = repr(value)
    try:
        decoded = ast.literal_eval(encoded)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        decoded = None
    if type(decoded) is not type(value) or decoded != value:
        _harness_error(f"the test passed a value that cannot be sent to the answer process: {encoded:.200}")
    return encoded


class Candidate:
    """Stands in for the model's function in check(candidate): each call runs in the answer process."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        send(self._sock, {"op": "call", "args": _encode(args), "kwargs": _encode(kwargs)})
        reply = recv(self._sock)
        if reply is None:
            raise AnswerError("the answer process exited during the call")
        if set(reply) == {"raised"}:
            raise AnswerError(f"the answer raised:\n{reply['raised']}")
        if set(reply) != {"value"} or not isinstance(reply["value"], str):
            raise AnswerError(f"unexpected reply: {reply!r:.200}")
        try:
            return ast.literal_eval(reply["value"])
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as err:
            raise AnswerError(f"the answer returned something other than a plain value: {reply['value']:.200}") from err


def connect(socket_path: str, code: str, entry_point: str) -> Candidate:
    """Loads code (prompt + answer) into a fresh answer process and returns the proxy for its entry_point."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(socket_path)
    except OSError as err:
        _harness_error(f"cannot connect to {socket_path}: {err}")
    send(sock, {"op": "load", "code": code, "entry_point": entry_point})
    reply = recv(sock)
    if reply is None:
        raise AnswerError("the answer process exited while loading the answer")
    if set(reply) == {"load_error"}:
        raise AnswerError(f"loading the answer failed:\n{reply['load_error']}")
    if reply != {"loaded": True}:
        raise AnswerError(f"unexpected reply: {reply!r:.200}")
    return Candidate(sock)


# ---------------------------------------------------------------------------
# Sandbox side (answer container)
# ---------------------------------------------------------------------------


def plain(value: Any) -> Any:
    """value as built-in types, so that its repr() parses with ast.literal_eval.

    Only avoids false negatives for correct answers that return e.g. a numpy scalar, a namedtuple or an IntEnum, all of
    which compare equal to the built-in value. Anything else is returned as is; the checker rejects its repr().
    """
    if value is None or type(value) in (bool, int, float, complex, str, bytes):
        return value
    if type(value).__module__ == "numpy" and getattr(value, "ndim", None) == 0:
        return plain(value.item())
    for base in (bool, int, float, complex, str, bytes):
        if isinstance(value, base):
            return base(value)
    if isinstance(value, tuple):
        return tuple(plain(v) for v in value)
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {plain(v) for v in value}
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    return value


def run_answer(fd: int) -> None:
    sock = socket.socket(fileno=fd)
    message = recv(sock)
    if message is None or message.get("op") != "load":
        return

    # Like upstream's `python -c prompt+answer+test`: the answer runs as __main__, so e.g. an
    # `if __name__ == "__main__":` block runs, and an exit there ends the process before any call.
    namespace: dict[str, Any] = {"__name__": "__main__", "__builtins__": builtins}
    try:
        exec(compile(message["code"], "<answer>", "exec"), namespace)
        function = namespace[message["entry_point"]]
    except Exception:
        send(sock, {"load_error": traceback.format_exc()})
        return
    send(sock, {"loaded": True})

    while True:
        message = recv(sock)
        if message is None:
            return
        try:
            result = function(*ast.literal_eval(message["args"]), **ast.literal_eval(message["kwargs"]))
            reply = {"value": repr(plain(result))}
        except Exception:
            reply = {"raised": traceback.format_exc()}
        send(sock, reply)


def _run_sample(conn: socket.socket) -> None:
    workdir = tempfile.mkdtemp(prefix="sample_")
    process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "run", str(conn.fileno())],
        pass_fds=(conn.fileno(),),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=workdir,
        start_new_session=True,
    )
    conn.close()
    try:
        process.wait(timeout=ANSWER_PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    # Kill whatever the answer left running in its session. The group is usually gone already.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()
    # The answer may have left files it made undeletable; they vanish with the container's tmpfs anyway.
    shutil.rmtree(workdir, ignore_errors=True)


def serve(socket_path: str) -> None:
    # Python's default SIGINT handler would let answer processes interrupt PID 1; without handlers, the kernel drops
    # every signal they send it.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(128)

    def exit_on_stdin_eof() -> None:
        sys.stdin.buffer.read()
        # As PID 1, exiting ends the PID namespace: the kernel kills every answer process left in it.
        os._exit(0)

    threading.Thread(target=exit_on_stdin_eof, daemon=True).start()
    while True:
        conn, _ = listener.accept()
        threading.Thread(target=_run_sample, args=(conn,), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("socket_path")
    p = sub.add_parser("run")
    p.add_argument("fd", type=int)
    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.socket_path)
    else:
        run_answer(args.fd)


if __name__ == "__main__":
    main()
