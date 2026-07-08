#!/usr/bin/env python3
"""
Copy a sanitized snapshot of each run from results/ to collected_results/,
suitable for downstream parsing by the ptb_traces_viewer build.py.

Layout produced per run:
    <experiment>/<run>/
        solve_out.txt               # raw stream-JSON trace (NOT solve_parsed.txt)
        metrics.json                # if present
        metrics_averaged.json       # if present
        contamination_judgement.txt # if present
        disallowed_model_judgement.txt
        judge_output.json           # if present  (Codex-style judge agent trace)
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

# Constants - modify these as needed
RESULTS_BASE = Path(os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_BASE, "../collected_results")

# API key environment variables to check and redact
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
    # Numerical / columnar formats — never text, would fail _looks_like_text
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


def get_api_keys() -> list[str]:
    """Get API key values from environment variables (non-empty only)."""
    keys = []
    for var in API_KEY_ENV_VARS:
        if var not in os.environ:
            raise ValueError(f"Expected environment variable not set: {var}")
        value = os.environ[var]
        keys.append(value)

    return keys

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
_KEY_TRUNC_COMBINED: re.Pattern | None = None


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
    """ProcessPoolExecutor initializer — set up per-worker state once."""
    global _WORKER_API_KEYS
    _WORKER_API_KEYS = api_keys
    _build_key_regexes(api_keys)


def _process_subdir(subdir_str: str, dest_dir_str: str, force: bool) -> dict:
    """Sanitize-and-copy one task subdir. Returns a result dict the parent
    aggregates into counters and warnings."""
    subdir = Path(subdir_str)
    dest_dir = Path(dest_dir_str)
    src_file = subdir / "solve_out.txt"

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

    for fname in (
        'solve_parsed.txt',
        'metrics.json',
        'metrics_averaged.json',
        'contamination_judgement.txt',
        'disallowed_model_judgement.txt',
        'judge_output.json',
        'error.log',
        'time_taken.txt',
        'system_monitor.log',
    ):
        copy_other_files(subdir, dest_dir, fname, api_keys=_WORKER_API_KEYS, optional=True)

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
        help="Input directory names (relative to RESULTS_BASE) to process. "
             "Ignored when --all-experiments is given."
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Process every subdirectory of RESULTS_BASE (subject to --exclude). "
             "Useful for a full corpus extract without listing experiments by hand."
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
        help=f"Override the output directory (default: {DEFAULT_OUTPUT_DIR})."
    )
    args = parser.parse_args()

    # Resolve which experiments to process. Either explicit names or
    # --all-experiments + exclusions, but not both.
    excluded = set(args.exclude or [])
    if args.all_experiments:
        if args.input_dirs:
            parser.error("Pass either input_dirs OR --all-experiments, not both.")
        if not RESULTS_BASE.is_dir():
            parser.error(f"RESULTS_BASE does not exist: {RESULTS_BASE}")
        input_dir_names = sorted(
            d.name for d in RESULTS_BASE.iterdir()
            if d.is_dir() and d.name not in excluded
        )
        if not input_dir_names:
            parser.error(f"No experiment directories found under {RESULTS_BASE} "
                         f"after exclusions ({sorted(excluded) or 'none'}).")
        print(f"Processing {len(input_dir_names)} experiments from {RESULTS_BASE}")
        if excluded:
            print(f"  (skipping {len(excluded)}: {', '.join(sorted(excluded))})")
    elif args.input_dirs:
        input_dir_names = args.input_dirs
        if excluded:
            print(f"NOTE: --exclude is ignored when input_dirs are passed explicitly "
                  f"(excluded names: {', '.join(sorted(excluded))})")
    else:
        parser.error("Pass one or more input_dirs, or use --all-experiments.")

    output_base = Path(args.output if args.output else DEFAULT_OUTPUT_DIR)
    api_keys = get_api_keys()
    # The parent also needs the truncation regexes for the serial path AND
    # because spawn-mode workers re-import this module without inherited state.
    _build_key_regexes(api_keys)

    # Build the full task list up front. One task = one task subdir; the pool
    # then parallelises across subdirs (NOT just across experiments), so a
    # single big experiment doesn't dominate the wall clock.
    tasks: list[tuple[str, str, str]] = []     # (subdir_str, dest_dir_str, experiment_name)
    for input_dir_name in input_dir_names:
        input_dir = RESULTS_BASE / input_dir_name
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
    src = subdir / filename
    dest = dest_dir / dest_filename
    if src.exists():
        copy_file_sanitized(src, dest, api_keys)
    elif not optional:
        # Hard-required missing — surface it, but don't fabricate content.
        print(f"  WARN: required file {filename} missing for {subdir.name}")


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
