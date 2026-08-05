#!/usr/bin/env python3
"""
Copy a sanitized snapshot of each run from results/ to collected_results/,
suitable for downstream parsing by the ptb_traces_viewer build.py.

Layout produced per run:
    <experiment>/<run>/
        solve_out.txt               # raw stream-JSON trace (NOT solve_parsed.txt)
        metrics.json                # if present
        metrics_averaged.json       # if present
        judgement_gpt5_4.json       # canonical contamination verdict (rerun preferred; if present)
        judge_output_gpt5_4.{json,txt} # GPT-5.4 contamination-judge trace (rerun preferred; if present)
        error.log                   # if present
        time_taken.txt              # if present
        system_monitor.log          # if present
        task/                       # workspace, text-only, weights/caches stripped
            ...
"""
import argparse
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

# Force line buffering on stdout so streaming progress survives pipes, tee,
# tmux, condor interactive shells, etc. (`flush=True` on print() is not
# always enough — some shells aggressively block-buffer the parent pipe.)
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

# Repo root holds the canonical .env. extract_traces.py lives in dev_utils/,
# exactly one level below the root.
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv() -> dict[str, str]:
    """Parse the repo-root .env into a dict.

    Project convention: tooling reads POST_TRAIN_BENCH_* (and, here, the
    API-key secrets) straight from .env rather than from the exported
    environment or by sourcing set_env_vars.sh (whose module-load block fails
    on compute nodes without tclsh)."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        raise RuntimeError(f".env file not found at {env_file}")
    values: dict[str, str] = {}
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


# Named API-key variables read from .env; their literal values are redacted
# from copied traces. Supplemented at runtime by the secrets file pointed to by
# POST_TRAIN_BENCH_SANITIZATION_SECRETS (see get_api_keys). Names absent from
# .env are skipped — this machine simply doesn't use that provider.
API_KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "BEN_HF_TOKEN",
    "HARDIK_HF_TOKEN",
    "OPENCODE_API_KEY",
    "ZAI_API_KEY",
    "DASHSCOPE_API_KEY"

]

API_KEY_PATTERNS = [
    "sk-proj",      # OpenAI project keys
    "sk-ant",       # Anthropic keys
    "AIzaSy",       # Google/Gemini keys
    # "sk-",        # Generic OpenAI keys - too broad (matches "mask-in", etc). Covered by sk-proj/sk-ant.
    # "hf_",        # HuggingFace tokens - too broad (matches hf_cache, hf_home etc). Actual tokens redacted via env vars.
    # not needed
    # AWS
    # "AKIA",         # AWS access key IDs
    # GitHub
    # "ghp_",         # GitHub personal access tokens
    # "gho_",         # GitHub OAuth tokens
    # "ghs_",         # GitHub app installation tokens
    # "ghr_",         # GitHub refresh tokens
    # # GitLab
    # "glpat-",       # GitLab personal access tokens
    # AI services
    "sk-or-",       # OpenRouter
    # "r8_",          # Replicate
    # "xai-",         # xAI/Grok
    # "nvapi-",       # NVIDIA
    # Slack
    # "xoxb-",        # Slack bot tokens
    # "xoxp-",        # Slack user tokens
    # Stripe
    # "sk_live_",     # Stripe live secret keys
    # "sk_test_",     # Stripe test secret keys
    # "whsec_",       # Stripe webhook secrets
    # Other
    # "SG.",          # SendGrid API keys
]


# Workspace filtering — mirrors the lists in ptb_traces_viewer/build.py so
# the on-disk layout post-extract matches what the viewer expects to inline.
WORKSPACE_SKIP_DIRS = {
    "final_model", "sft_output", "sft_output_v2", "checkpoints",
    "__pycache__", ".git", "node_modules", "wandb",
    "huggingface_cache", ".cache",
}
WORKSPACE_SKIP_EXTS = {
    ".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".onnx",
    ".tar", ".gz", ".zip", ".bz2", ".xz",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    ".so", ".pyc",
    # Numerical / columnar formats — never text, would fail _is_text_bytes
    # after a wasted stat + 4 KB read on Lustre. Skip by extension to dodge
    # both ops; matters when task/ has many of these (training dumps).
    ".npy", ".npz", ".parquet", ".arrow", ".feather",
}
# Hard cap on a single workspace file. The viewer can only inline up to
# 256 KB per file anyway; anything above ~2 MB is almost certainly a
# training-data jsonl or eval dump that's not interesting to scroll. We
# still list these in the viewer as "too_large" if you leave them on
# disk, but it's not worth sanitizing 10+ MB through regex.
WORKSPACE_MAX_BYTES = 2 * 1024 * 1024


def load_sanitization_secrets(dotenv: dict[str, str]) -> list[str]:
    """Read extra secrets to redact from the file named by
    POST_TRAIN_BENCH_SANITIZATION_SECRETS in .env.

    Each non-empty line that does not start with '#' is one secret (an API key)
    to redact. Returns [] when the variable is unset/empty. When it *is* set,
    the file must exist — a configured-but-missing path is a misconfiguration
    and crashes rather than silently redacting nothing.

    Duplicate lines are dropped (order preserved) since the same key is often
    appended more than once over time; get_api_keys dedupes again across all
    sources, but deduping here keeps the reported file-secret count honest."""
    path_str = dotenv.get("POST_TRAIN_BENCH_SANITIZATION_SECRETS", "")
    if not path_str:
        return []
    secrets_path = Path(path_str)
    if not secrets_path.is_absolute():
        secrets_path = REPO_ROOT / secrets_path
    if not secrets_path.exists():
        raise RuntimeError(
            f"POST_TRAIN_BENCH_SANITIZATION_SECRETS points to a missing file: "
            f"{secrets_path}"
        )
    secrets = []
    seen = set()
    for raw in secrets_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line in seen:
            continue
        seen.add(line)
        secrets.append(line)
    return secrets


def get_api_keys(dotenv: dict[str, str]) -> list[str]:
    """Collect the literal secret strings to redact from copied traces.

    Two sources, both resolved from .env (never the ambient environment):
      1. The named variables in API_KEY_ENV_VARS, for those present and
         non-empty in .env. Absent names are skipped (reported, not silently
         dropped) since not every machine uses every provider.
      2. The newline-delimited file named by POST_TRAIN_BENCH_SANITIZATION_SECRETS.
    Duplicates are removed while preserving order."""
    keys: list[str] = []
    present, absent = [], []
    for var in API_KEY_ENV_VARS:
        value = dotenv.get(var, "")
        if value:
            keys.append(value)
            present.append(var)
        else:
            absent.append(var)

    file_secrets = load_sanitization_secrets(dotenv)
    keys.extend(file_secrets)

    seen = set()
    deduped = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            deduped.append(k)

    summary = f"Redaction secrets: {len(present)} named .env var(s)"
    if absent:
        summary += f" ({len(absent)} absent: {', '.join(absent)})"
    summary += (
        f", {len(file_secrets)} from "
        f"{dotenv.get('POST_TRAIN_BENCH_SANITIZATION_SECRETS') or '<no secrets file>'}"
        f" → {len(deduped)} unique"
    )
    print(summary)
    return deduped

_warnings = []

def warn_if_api_key_in_content(content: str, prefix: str, src_path: str = "") -> None:
    if prefix in content:
        idx = content.index(prefix)
        start = max(0, idx - 50)
        end = min(len(content), idx + 50)
        context = content[start:end].replace('\n', '\\n')
        _warnings.append({
            "pattern": prefix,
            "file": src_path,
            "context": context,
        })

# Hard-redaction patterns for keys whose literal value we may NOT have in
# the local env. Each matches a structural prefix + N chars of base62-ish
# suffix. The threshold is high enough that "sk-proj" appearing in prose
# never triggers, but low enough that partial leaks (e.g. an agent
# debug-printing `start: sk-proj-i-nGplGD5b34`) still get redacted.
GENERIC_KEY_RES: list[tuple[str, re.Pattern]] = [
    # OpenAI project keys (real keys are 100+ chars; threshold 12 catches
    # the debug-print partial leaks seen in arenahard/healthbench traces).
    ('sk-proj-',  re.compile(r'sk-proj-[A-Za-z0-9_\-]{12,}')),
    # Anthropic in all flavors:
    #   - sk-ant-api##-  : API keys
    #   - sk-ant-admin##-: admin keys
    #   - sk-ant-oat##-  : CLAUDE_CODE_OAUTH_TOKEN — this is the big one,
    #                      missed by the old api|admin-only regex and the
    #                      reason real keys slipped into traces from
    #                      `env`-style dumps.
    #   - sk-ant-sid##-  : session IDs (defensive, future-proofing)
    ('sk-ant-',   re.compile(r'sk-ant-(?:api|admin|oat|sid)[0-9]{2,}-[A-Za-z0-9_\-]{12,}')),
    ('AIzaSy',    re.compile(r'AIzaSy[A-Za-z0-9_\-]{20,}')),                  # Google / Gemini
    ('sk-or-',    re.compile(r'sk-or-[vV][0-9]-[A-Za-z0-9_\-]{20,}')),        # OpenRouter
    ('hf_',       re.compile(r'hf_[A-Za-z0-9]{20,}')),                        # Hugging Face
    # OpenCode tokens: `sk-` + 40+ base62-ish chars. Negative lookahead skips
    # sk-proj-/sk-ant-/sk-or- which are already covered above (defensive
    # against the loop order — Pass 2 stops being idempotent otherwise).
    ('sk-',       re.compile(r'sk-(?!proj-|ant-|or-)[A-Za-z0-9_\-]{40,}')),
    # Z.ai tokens: `<32-hex>.<16+ base62>`. No useful fixed prefix, so the
    # cheap `prefix in content` guard is the lowest-info char (`.`) — that
    # short-circuit is functionally a no-op but keeps the loop shape uniform.
    ('.',         re.compile(r'\b[a-f0-9]{32}\.[A-Za-z0-9_\-]{12,}')),
]

# Pass 2: combined alternation regex over the full GENERIC_KEY_RES list.
# `re.sub` over a single combined pattern walks the text once; the previous
# per-pattern loop walked it up to N times. Since one of the prefix guards is
# `.` (matches every file), the loop body always ran the regex even for clean
# files — so the combined form is a strict win, no guard needed.
_GENERIC_KEY_COMBINED = re.compile(
    '|'.join(f'(?:{r.pattern})' for _, r in GENERIC_KEY_RES)
)

# Pass 1b: truncation regexes. Built per-process at startup from the env-key
# values, so they live in module globals rather than constants. The combined
# form sits next to the per-pattern list because the prefix guard IS useful
# here — random 10-char base62 strings essentially never collide with prose,
# so most files short-circuit out without running the regex at all.
_KEY_TRUNC_RES: list[tuple[str, re.Pattern]] = []
_KEY_TRUNC_PREFIXES: tuple[str, ...] = ()
_KEY_TRUNC_COMBINED: "re.Pattern | None" = None


def _build_key_regexes(api_keys: list[str]) -> None:
    global _KEY_TRUNC_RES, _KEY_TRUNC_PREFIXES, _KEY_TRUNC_COMBINED
    _KEY_TRUNC_RES = [
        (k[:10], re.compile(re.escape(k[:10]) + r'[A-Za-z0-9_\-]+'))
        for k in api_keys if len(k) >= 10
    ]
    _KEY_TRUNC_PREFIXES = tuple(p for p, _ in _KEY_TRUNC_RES)
    _KEY_TRUNC_COMBINED = (
        re.compile('|'.join(f'(?:{r.pattern})' for _, r in _KEY_TRUNC_RES))
        if _KEY_TRUNC_RES else None
    )


def sanitize_content(content: str, api_keys: list[str], src_path: str = "") -> str:
    """Replace any API keys found in content with a placeholder.

    Three passes, in order of confidence:
      1. Literal known key values from env vars (highest confidence).
      2. Generic structural patterns for known key shapes — handles keys
         whose values are not in the local env (different account, old run).
      3. Manual-review warnings for the bare prefix, in case anything
         slipped through (e.g. unusual delimiters, partial keys).

    Pass 1b + Pass 2 each use one combined alternation regex rather than a
    per-pattern loop, so each is at most one full-text sweep instead of N.
    """
    # Pass 1: literal known values
    for key in api_keys:
        if key and key in content:
            content = content.replace(key, "<omitted-api-key>")

    # Pass 1b: truncation combined sweep (guard short-circuits clean files)
    if _KEY_TRUNC_COMBINED is not None and any(p in content for p in _KEY_TRUNC_PREFIXES):
        content = _KEY_TRUNC_COMBINED.sub('<omitted-api-key>', content)

    # Pass 2: generic key-shape combined sweep — always runs (the ZAI pattern
    # has no useful fixed prefix, so a guard would never short-circuit anyway)
    content = _GENERIC_KEY_COMBINED.sub('<omitted-api-key>', content)

    # Pass 3: warn on residual prefixes (manual review)
    for pattern in API_KEY_PATTERNS:
        warn_if_api_key_in_content(content, pattern, src_path)

    return content


def prefer_sanitized(src: Path) -> Path:
    """Resolve a source path to its `<stem>_sanitized<ext>` sibling if one
    exists, else the path itself.

    parse_trace.py emits a sanitized companion (env-secret values already
    redacted) next to each raw/parsed trace and judge-output file. We prefer
    it as the source but still run it through our own sanitizer on copy.
    The _sanitized suffix is removed in the dest filename."""
    suffix = src.suffix
    stem = src.name[: -len(suffix)] if suffix else src.name
    sanitized = src.with_name(f"{stem}_sanitized{suffix}")
    return sanitized if sanitized.exists() else src


def copy_file_sanitized(src: Path, dest: Path, api_keys: list[str],
                        preserve_mtime: bool = True) -> bool:
    """Copy a file, sanitizing API keys from its content.

    Pass `preserve_mtime=False` for the per-run anchor file (solve_out.txt)
    so the incremental-skip check can compare dest.mtime against src.mtime
    meaningfully — copystat would otherwise pin dest.mtime to src.mtime and
    the `dest >= src` test would never strictly distinguish a cached extract
    from a stale one.
    """
    content = src.read_text(encoding="utf-8")
    sanitized = sanitize_content(content, api_keys, src_path=str(src))
    dest.write_text(sanitized, encoding="utf-8")
    if preserve_mtime:
        shutil.copystat(src, dest)
    return content != sanitized


def extract_model_name(dir_name: str) -> str:
    return dir_name


def get_latest_subdirs(input_dir: Path) -> list[Path]:
    """
    Group subdirectories by their prefix (everything before the last _<id>)
    and return only the one with the highest numeric ID for each group.
    """
    grouped = defaultdict(list)

    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue

        name = subdir.name
        parts = name.rsplit('_', 1)

        if len(parts) == 2 and parts[1].isdigit():
            prefix, id_str = parts
            grouped[prefix].append((int(id_str), subdir))
        else:
            # No numeric ID, treat the whole name as unique
            grouped[name].append((0, subdir))

    # For each group, keep only the one with the highest ID
    latest = []
    for prefix, entries in grouped.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        latest.append(entries[0][1])

    return latest


# ---------------------------------------------------------------------------
# Per-subdir worker (used in both serial and pool modes)
# ---------------------------------------------------------------------------

# Cached at module load — used by the incremental-skip check.
_SCRIPT_MTIME = Path(__file__).stat().st_mtime

# Set per-process by `_pool_init` (also set by main() for the serial path).
_WORKER_API_KEYS: list[str] = []


def _pool_init(api_keys: list[str]) -> None:
    """ProcessPoolExecutor initializer — set up per-worker state once.
    Also called by main() before the serial path so `_WORKER_API_KEYS` and
    the truncation regexes are populated in the parent too."""
    global _WORKER_API_KEYS
    _WORKER_API_KEYS = api_keys
    _build_key_regexes(api_keys)


def _process_subdir(subdir_str: str, dest_dir_str: str, force: bool) -> dict:
    """Sanitize-and-copy one task subdir. Returns a result dict the parent
    aggregates into counters and warnings."""
    subdir = Path(subdir_str)
    dest_dir = Path(dest_dir_str)
    # The trace must be solve_out.txt — the JSONL form the viewer parses.
    # solve_parsed.txt is human-readable plaintext and is NOT a valid input
    # for build.py. Prefer the sanitized companion if parse_trace.py wrote one.
    src_file = prefer_sanitized(subdir / "solve_out.txt")

    if not src_file.exists():
        return {"status": "missing", "name": subdir.name}

    # Incremental skip: if dest/solve_out.txt is newer than both the source
    # and this script, the cached extract is still valid. --force overrides.
    if not force:
        dest_solve = dest_dir / "solve_out.txt"
        if dest_solve.exists():
            try:
                dest_mtime = dest_solve.stat().st_mtime
                src_mtime = src_file.stat().st_mtime
                if dest_mtime >= src_mtime and dest_mtime >= _SCRIPT_MTIME:
                    return {"status": "skipped", "name": subdir.name}
            except OSError:
                pass

    # Reset the per-worker warnings list so we only return THIS task's hits.
    global _warnings
    _warnings = []

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_solve = dest_dir / "solve_out.txt"
    # solve_out.txt is the anchor used by the cache-hit check on the next run,
    # so its mtime needs to reflect WHEN-EXTRACTED, not the source's mtime.
    was_sanitized = copy_file_sanitized(src_file, dest_solve, _WORKER_API_KEYS,
                                        preserve_mtime=False)

    # Run-level metadata — copy if present, otherwise just skip
    # (no placeholder text, which previously broke downstream JSON
    # parsing and verdict detection).
    # NB: solve_parsed.txt is also shipped for direct human reading;
    # the viewer ignores it, but it's a useful escape hatch.
    for fname in (
        'solve_parsed.txt',
        'metrics.json',
        'metrics_averaged.json',
        'error.log',
        'time_taken.txt',
        'system_monitor.log',
    ):
        copy_other_files(subdir, dest_dir, fname, api_keys=_WORKER_API_KEYS, optional=True)

    # Judge artifacts — canonical GPT-5.4 contamination verdict + trace,
    # rerun variant preferred over the base. The third-party-API-usage judge
    # (judgement_api*.json) is NOT extracted even though scoring now consumes
    # it (flag => baseline fallback): it is known to flip on rerun (non-
    # deterministic false positives), so the viewer doesn't surface it. The
    # PTB-lookup judge (judgement_ptb_lookup*.json) is archival and likewise
    # not extracted.
    copy_preferring_rerun(subdir, dest_dir, 'judgement_gpt5_4.json',   api_keys=_WORKER_API_KEYS)
    copy_preferring_rerun(subdir, dest_dir, 'judge_output_gpt5_4.json', api_keys=_WORKER_API_KEYS)
    copy_preferring_rerun(subdir, dest_dir, 'judge_output_gpt5_4.txt',  api_keys=_WORKER_API_KEYS)

    # Legacy v1.0 judge artifacts (pre-revamp benchmark version) — copied
    # as fallback so old cells that were never re-judged still show a
    # verdict in the viewer. build.py prefers the _gpt5_4 files when both
    # eras coexist. Optional — no-ops on cells that were only ever
    # v1.1-judged.
    copy_preferring_rerun(subdir, dest_dir, 'contamination_judgement.txt',   api_keys=_WORKER_API_KEYS)
    copy_preferring_rerun(subdir, dest_dir, 'disallowed_model_judgement.txt', api_keys=_WORKER_API_KEYS)
    copy_preferring_rerun(subdir, dest_dir, 'judge_output.json',              api_keys=_WORKER_API_KEYS)

    # Workspace — copy text files only, skip weights/checkpoints/caches.
    ws_count = copy_workspace(subdir / "task", dest_dir / "task", _WORKER_API_KEYS)

    return {
        "status": "ok",
        "name": subdir.name,
        "sanitized": was_sanitized,
        "ws_count": ws_count,
        "warnings": list(_warnings),
    }


def _format_ok_line(name: str, sanitized: bool, ws_count: int) -> str:
    tag_bits = []
    if sanitized: tag_bits.append("sanitized")
    if ws_count:  tag_bits.append(f"task:{ws_count}")
    tag = f"  [{', '.join(tag_bits)}]" if tag_bits else ""
    return f"  OK: {name}{tag}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize and copy run directories to collected_results/ for the trace viewer."
    )
    parser.add_argument(
        "input_dirs",
        nargs="*",
        help="Input directory names (relative to POST_TRAIN_BENCH_RESULTS_DIR) "
             "to process. Ignored when --all-experiments is given."
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Process every subdirectory of POST_TRAIN_BENCH_RESULTS_DIR (subject "
             "to --exclude). Useful for a full corpus extract without listing "
             "experiments by hand."
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="NAME",
        help="Experiment directory names to skip. Only meaningful with "
             "--all-experiments. Example: --exclude baseline baseline_zeroshot"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Copy all runs per task, not just the latest seed (default: latest only)."
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=min(8, (os.cpu_count() or 4)),
        help="Number of parallel workers (default: min(8, cpu_count)). Use 1 for serial."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract every run even if the cached output is newer than its inputs."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Override the output directory (default: <results_base>/../collected_results)."
    )
    args = parser.parse_args()

    dotenv = load_dotenv()
    results_base = Path(dotenv.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
    default_output = results_base / ".." / "collected_results"
    output_base = Path(args.output) if args.output else default_output

    # Resolve which experiments to process. Either explicit names or
    # --all-experiments + exclusions, but not both.
    excluded = set(args.exclude or [])
    if args.all_experiments:
        if args.input_dirs:
            parser.error("Pass either input_dirs OR --all-experiments, not both.")
        if not results_base.is_dir():
            parser.error(f"results_base does not exist: {results_base}")
        input_dir_names = sorted(
            d.name for d in results_base.iterdir()
            if d.is_dir() and d.name not in excluded
        )
        if not input_dir_names:
            parser.error(f"No experiment directories found under {results_base} "
                         f"after exclusions ({sorted(excluded) or 'none'}).")
        print(f"Processing {len(input_dir_names)} experiments from {results_base}")
        if excluded:
            print(f"  (skipping {len(excluded)}: {', '.join(sorted(excluded))})")
    elif args.input_dirs:
        input_dir_names = args.input_dirs
        if excluded:
            print(f"NOTE: --exclude is ignored when input_dirs are passed explicitly "
                  f"(excluded names: {', '.join(sorted(excluded))})")
    else:
        parser.error("Pass one or more input_dirs, or use --all-experiments.")

    api_keys = get_api_keys(dotenv)
    # Populate both `_WORKER_API_KEYS` and the truncation regexes in the parent.
    # Required for the serial (--jobs 1) path, and harmless in pool mode where
    # spawn-mode workers re-import this module and re-init via `_pool_init`.
    _pool_init(api_keys)

    # Build the full task list up front. One task = one task subdir; the pool
    # then parallelises across subdirs (NOT just across experiments), so a
    # single big experiment doesn't dominate the wall clock.
    tasks: list[tuple[str, str, str]] = []     # (subdir_str, dest_dir_str, experiment_name)
    for input_dir_name in input_dir_names:
        input_dir = results_base / input_dir_name
        if not input_dir.is_dir():
            print(f"  SKIP: {input_dir} does not exist")
            continue
        model_dir = output_base / extract_model_name(input_dir_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        if args.all:
            subdirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
        else:
            subdirs = sorted(get_latest_subdirs(input_dir))
        for subdir in subdirs:
            tasks.append((str(subdir), str(model_dir / subdir.name), input_dir_name))

    if not tasks:
        print("No tasks to process.")
        return

    print(f"\nProcessing {len(tasks)} task subdirs across {len(input_dir_names)} experiment(s) "
          f"with {args.jobs} worker(s){' [force]' if args.force else ''}\n")

    copied = sanitized = missing = skipped = 0
    all_warnings: list[dict] = []
    total = len(tasks)
    done = 0

    def _consume(result: dict, exp_name: str) -> None:
        nonlocal copied, sanitized, missing, skipped, done
        status = result["status"]
        name = result["name"]
        if status == "ok":
            copied += 1
            if result.get("sanitized"): sanitized += 1
            line = _format_ok_line(name, result.get("sanitized", False), result.get("ws_count", 0))
            all_warnings.extend(result.get("warnings", []))
        elif status == "skipped":
            skipped += 1
            line = f"  -- cached: {name}"
        elif status == "missing":
            missing += 1
            line = f"  MISS: {name} (no solve_out.txt)"
        else:
            line = f"  ?? {name}"
        done += 1
        # Stream: prefix with [done/total] and the experiment, so out-of-order
        # completions stay legible. flush=True so the output appears live even
        # under buffering (tee, nohup, etc.).
        print(f"  [{done}/{total}] [{exp_name}] {line.lstrip()}", flush=True)

    if args.jobs <= 1:
        # Serial path — same code, no pool overhead. Useful for debugging.
        for subdir_s, dest_s, exp_name in tasks:
            _consume(_process_subdir(subdir_s, dest_s, args.force), exp_name)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_pool_init,
                                 initargs=(api_keys,)) as pool:
            futs = {
                pool.submit(_process_subdir, s, d, args.force): exp
                for s, d, exp in tasks
            }
            # Heartbeat between header and first completion. On cold Lustre
            # the first task can take 30+ s; without this you can't tell the
            # pool is alive vs hung.
            print(f"  queued {len(futs)} tasks, waiting for first completion…",
                  flush=True)
            for fut in as_completed(futs):
                _consume(fut.result(), futs[fut])

    # Summary
    print(f"{'='*60}")
    print(f"Done: {copied} copied, {sanitized} sanitized, {skipped} cache-hit, {missing} missing")
    print(f"Output: {output_base}")

    if all_warnings:
        print(f"\n--- {len(all_warnings)} pattern warnings (review manually) ---")
        for w in all_warnings:
            print(f"  [{w['pattern']}] {w['file']}")
            print(f"    ...{w['context']}...")

def copy_other_files(subdir, dest_dir, filename, dest_filename=None, api_keys=None, optional=False):
    """Copy a file with sanitization. Missing files are simply skipped —
    don't write placeholder text, since the downstream parser would
    misinterpret it (e.g. a placeholder in contamination_judgement.txt
    would be treated as a contamination flag)."""
    if dest_filename is None:
        dest_filename = filename
    if api_keys is None:
        api_keys = []
    src = prefer_sanitized(subdir / filename)
    dest = dest_dir / dest_filename
    if src.exists():
        copy_file_sanitized(src, dest, api_keys)
    elif not optional:
        # Hard-required missing — surface it, but don't fabricate content.
        print(f"  WARN: required file {filename} missing for {subdir.name}")


def copy_preferring_rerun(subdir, dest_dir, base_filename, api_keys):
    """Copy a judge artifact, preferring the `_rerun` variant when present.

    The rerun pipeline writes `<stem>_rerun<ext>` alongside the original.
    The rerun is the corrected output, so it wins; either way the file is
    written to the destination under the canonical non-rerun name so
    downstream consumers don't need to know a rerun happened. Within each
    candidate the `_sanitized` companion is preferred (see prefer_sanitized).
    Everything is optional — nothing is copied if no candidate exists."""
    base = Path(base_filename)
    rerun_filename = f"{base.stem}_rerun{base.suffix}"
    # rerun supersedes base; prefer_sanitized picks the redacted companion
    # within whichever candidate we land on.
    for candidate in (subdir / rerun_filename, subdir / base_filename):
        src = prefer_sanitized(candidate)
        if src.exists():
            copy_file_sanitized(src, dest_dir / base_filename, api_keys)
            return


def _is_text_bytes(head: bytes) -> bool:
    """Same heuristic as before, just takes already-read bytes so the caller
    can read the file once and reuse the prefix for both this check and the
    sanitize step. Reject NUL bytes or low printable-ratio in the first 4 KB."""
    if not head:
        return True
    sample = head[:4096]
    if b"\x00" in sample:
        return False
    printable = sum(1 for b in sample if b in (9, 10, 13) or 32 <= b < 127)
    return printable / len(sample) > 0.85


def copy_workspace(src_root: Path, dest_root: Path, api_keys: list[str]) -> int:
    """Mirror `task/` into the output, copying only text files and skipping
    known-binary extensions / weight or cache directories. Each text file
    is API-key-sanitized. Returns the number of files copied.

    Uses os.walk with in-place dirname pruning so we never *descend* into
    weight/cache trees (e.g. final_model/, sft_output/) — rglob walks them
    fully and then we'd just discard the results. Big speedup when runs
    have GB-scale checkpoint dirs.

    Per-file: one open(), one read. Originally each text file was opened
    twice (once for a 4 KB peek, again for the full read by
    copy_file_sanitized). On Lustre that doubled the MDS round-trips —
    by far the dominant cost when task/ has hundreds of small files.
    """
    if not src_root.is_dir():
        return 0
    count = 0
    src_root_str = str(src_root)
    for dirpath, dirnames, filenames in os.walk(src_root_str):
        # Mutate in place — os.walk respects this for descent.
        dirnames[:] = [d for d in dirnames if d not in WORKSPACE_SKIP_DIRS]
        for fn in filenames:
            src = Path(dirpath) / fn
            if src.suffix.lower() in WORKSPACE_SKIP_EXTS:
                continue
            try:
                size = src.stat().st_size
            except OSError:
                continue
            if size > WORKSPACE_MAX_BYTES:
                continue            # too big to be useful in the viewer
            # Single open + read. Cheaper on Lustre than peek-then-reopen.
            try:
                data = src.read_bytes()
            except OSError:
                continue
            if not _is_text_bytes(data):
                continue
            try:
                content = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            # Match read_text()'s default universal-newlines behavior, which
            # the old per-file-open path got for free. Without this, files
            # containing carriage returns (e.g. tqdm-style progress logs)
            # produce different bytes than prior runs did.
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            sanitized = sanitize_content(content, api_keys, src_path=str(src))
            rel = src.relative_to(src_root)
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.write_text(sanitized, encoding="utf-8")
                shutil.copystat(src, dest)
                count += 1
            except OSError as e:
                print(f"  WARN: could not write {rel} ({e})")
    return count


if __name__ == "__main__":
    main()
