#!/usr/bin/env python3
"""
Preflight for the Hugging Face token run_modal_task.sh hands to the harbor sandboxes.

Condor serves every gated Hub asset the agent may touch from its pre-populated HF_HOME
overlay (containers/download_hf_cache/resources.json is that cache's manifest). Harbor
sandboxes start with an empty cache, so the agent and the verifier fetch from the Hub
with HF_TOKEN instead. Before a run starts, this script verifies against the Hub that
HF_TOKEN (read from the environment):

  1. is a valid user access token that is READ-ONLY — role `read`, or fine-grained with
     nothing but `*.read` permissions. The agent can read the token from its sandbox
     env, so a write token must never be handed in.
  2. can access every repo listed in gated_hf_resources.json: the gated (or private)
     entries of resources.json a harbor agent must be able to reach, so it has the same
     data reach as a condor agent (as of 2026-09-08: google/gemma-3-4b-pt and
     Idavidrein/gpqa, which gpqamain's evaluate.py loads in both sandboxes).

Any failure is an error (exit 1) with the fix spelled out; nothing is checked softly.

Usage:
    HF_TOKEN=hf_... python3 check_hf_token.py            # run_modal_task.sh does this
    HF_TOKEN=hf_... python3 check_hf_token.py --refresh  # regenerate gated_hf_resources.json

`--refresh` asks the Hub for the gating status of every resources.json entry (one
metadata call per repo, public data), rewrites gated_hf_resources.json with ALL gated
entries, then runs the checks. The committed list is curated by hand (entries a harbor
agent does not need were dropped), so after a refresh review the diff before committing.
The default mode trusts gated_hf_resources.json as committed and never looks at
resources.json.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

HUB = "https://huggingface.co"
HERE = Path(__file__).resolve().parent
RESOURCES_JSON = HERE.parents[1] / "containers" / "download_hf_cache" / "resources.json"
GATED_JSON = HERE / "gated_hf_resources.json"
TOKEN_SETTINGS = f"{HUB}/settings/tokens"
REPO_KINDS = ("models", "datasets")


def die(msg: str) -> None:
    print(f"check_hf_token: {msg}", file=sys.stderr)
    sys.exit(1)


def hub_get(path: str, token: str) -> tuple[int, str, str | None, str | None]:
    """GET {HUB}{path} with the token. Returns (status, body text, X-Error-Code,
    X-Error-Message). Only HTTP error statuses are caught — they are the signal here
    (401/403/404); anything else propagates."""
    req = urllib.request.Request(HUB + path, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode(), None, None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(), e.headers.get("X-Error-Code"), e.headers.get("X-Error-Message")


def check_token_role(token: str) -> tuple[str, str]:
    """Returns (hub user name, role); dies unless the token is valid and read-only."""
    status, body, code, msg = hub_get("/api/whoami-v2", token)
    if status != 200:
        die(f"the Hub rejected HF_TOKEN ({status} {code or ''} {msg or ''}). Create a read-only "
            f"user access token at {TOKEN_SETTINGS} and set it as HF_TOKEN.")
    who = json.loads(body)
    auth = who["auth"]
    if auth["type"] != "access_token":
        die(f"HF_TOKEN is a {auth['type']!r} credential; use a read-only user access token from {TOKEN_SETTINGS}")
    user = who["name"]
    role = auth["accessToken"]["role"]
    read_only_advice = (f"The agent can read HF_TOKEN from its sandbox environment, so hand it a read-only "
                        f"token: at {TOKEN_SETTINGS} create one of type 'Read' (or fine-grained with only "
                        f"read permissions plus 'Read access to contents of all public gated repos you can "
                        f"access'), put it in .env as HF_TOKEN, and keep this token out of the harbor flow.")
    if role == "write":
        die(f"HF_TOKEN ('{auth['accessToken']['displayName']}' of user {user}) has WRITE access. {read_only_advice}")
    if role == "fineGrained":
        fine = auth["accessToken"]["fineGrained"]
        permissions = list(fine["global"]) + [p for scope in fine["scoped"] for p in scope["permissions"]]
        non_read = sorted({p for p in permissions if not p.endswith(".read")})
        if non_read:
            die(f"HF_TOKEN ('{auth['accessToken']['displayName']}' of user {user}) is fine-grained but "
                f"grants non-read permissions {non_read}. {read_only_advice}")
        return user, "fineGrained (read-only)"
    if role != "read":
        die(f"unexpected HF token role {role!r} in the whoami-v2 response; expected read, write or fineGrained")
    return user, role


def refresh_gated(token: str) -> dict:
    """Rewrite gated_hf_resources.json from the Hub's gating metadata of every resources.json entry."""
    resources = json.loads(RESOURCES_JSON.read_text())
    items = [("models", m) for m in resources["models"]] + [("datasets", d["dataset"]) for d in resources["datasets"]]

    def probe(kind: str, repo: str) -> tuple[str, str, int, str, str | None]:
        status, body, code, _ = hub_get(f"/api/{kind}/{repo}?expand[]=gated&expand[]=private", token)
        return kind, repo, status, body, code

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda item: probe(*item), items))
    gated = {kind: {} for kind in REPO_KINDS}
    missing = []
    for kind, repo, status, body, code in results:
        if status == 404 and code == "RepoNotFound":
            missing.append(f"{kind}/{repo}")   # gone from the Hub: nobody can fetch it, gated or not
            continue
        if status != 200:
            die(f"metadata lookup for {kind}/{repo} failed: {status} {code}")
        info = json.loads(body)
        if info["private"]:
            gated[kind][repo] = "private"
        elif info["gated"]:            # False, or the gating mode "auto" / "manual"
            gated[kind][repo] = info["gated"]
    out = {
        "_generated_by": f"{Path(__file__).name} --refresh on {date.today().isoformat()}: the gated/private "
                         f"entries of {RESOURCES_JSON.relative_to(HERE.parents[1])} (value = gating mode)",
        **gated,
    }
    GATED_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {GATED_JSON}: {len(items)} resources probed, "
          f"{len(gated['models'])} gated models, {len(gated['datasets'])} gated datasets"
          + (f"; {len(missing)} no longer on the Hub (ignored): {', '.join(missing)}" if missing else ""))
    return out


def check_gated_access(token: str, user: str, gated: dict) -> int:
    """Dies unless the token passes the Hub's auth-check for every gated repo; returns the count."""
    items = [(kind, repo, mode) for kind in REPO_KINDS for repo, mode in gated[kind].items()]

    def probe(kind: str, repo: str, mode: str) -> tuple[str, str, str, int, str | None, str | None]:
        status, _, code, msg = hub_get(f"/api/{kind}/{repo}/auth-check", token)
        return kind, repo, mode, status, code, msg

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda item: probe(*item), items))
    denied = [r for r in results if r[3] != 200]
    if denied:
        lines = [f"  {kind}/{repo} (gated: {mode}) -> {status} {code}: {msg}"
                 for kind, repo, mode, status, code, msg in denied]
        die(f"HF_TOKEN (user {user}) cannot access {len(denied)} of the {len(items)} gated resources.json repos "
            f"that condor's HF cache holds:\n" + "\n".join(lines) + "\n"
            f"Accept each repo's conditions at the URL in its message while logged in as {user} ('auto' "
            f"grants instantly, 'manual' waits for the repo owner), or use a token of an account that has access.")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true",
                        help=f"regenerate {GATED_JSON.name} from the Hub before checking")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        die("HF_TOKEN is not set")

    user, role = check_token_role(token)
    gated = refresh_gated(token) if args.refresh else json.loads(GATED_JSON.read_text())
    n = check_gated_access(token, user, gated)
    print(f"hf token: user {user}, role {role}; access to all {n} gated resources.json repos "
          f"({len(gated['models'])} models, {len(gated['datasets'])} datasets) verified")


if __name__ == "__main__":
    main()
