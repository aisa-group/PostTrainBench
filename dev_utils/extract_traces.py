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
from pathlib import Path
from collections import defaultdict

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
]

# Pre-compile per-key truncation regexes (was being recompiled per file).
_KEY_TRUNC_RES: list[tuple[str, re.Pattern]] = []

def _build_key_regexes(api_keys: list[str]) -> None:
    global _KEY_TRUNC_RES
    _KEY_TRUNC_RES = []
    for k in api_keys:
        if len(k) >= 10:
            _KEY_TRUNC_RES.append((k[:10], re.compile(re.escape(k[:10]) + r'[A-Za-z0-9_\-]+')))


def sanitize_content(content: str, api_keys: list[str], src_path: str = "") -> str:
    """Replace any API keys found in content with a placeholder.

    Three passes, in order of confidence:
      1. Literal known key values from env vars (highest confidence).
      2. Generic structural patterns for known key shapes — handles keys
         whose values are not in the local env (different account, old run).
      3. Manual-review warnings for the bare prefix, in case anything
         slipped through (e.g. unusual delimiters, partial keys).

    A cheap `prefix in content` short-circuits each pass — most files have
    no keys, so the regex never runs.
    """
    # Pass 1: literal known values
    for key in api_keys:
        if key and key in content:
            content = content.replace(key, "<omitted-api-key>")

    # Pass 1b: truncation fallbacks for those known values
    for prefix, regex in _KEY_TRUNC_RES:
        if prefix in content:
            content = regex.sub('<omitted-api-key>', content)

    # Pass 2: generic key-shape redaction (catches keys we don't have)
    for prefix, regex in GENERIC_KEY_RES:
        if prefix in content:
            content = regex.sub('<omitted-api-key>', content)

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


def copy_file_sanitized(src: Path, dest: Path, api_keys: list[str]) -> None:
    """Copy a file, sanitizing API keys from its content."""
    content = src.read_text(encoding="utf-8")
    sanitized = sanitize_content(content, api_keys, src_path=str(src))
    dest.write_text(sanitized, encoding="utf-8")
    # Preserve file metadata
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


def main():
    parser = argparse.ArgumentParser(
        description="Copy solve_parsed.txt (or solve_out.txt fallback) from result directories to a new organized structure."
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
    args = parser.parse_args()

    dotenv = load_dotenv()
    results_base = Path(dotenv.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
    output_base = results_base / ".." / "collected_results"

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
    _build_key_regexes(api_keys)   # compile truncation patterns once, not per file

    copied_count = 0
    sanitized_count = 0
    missing_count = 0

    for input_dir_name in input_dir_names:
        input_dir = results_base / input_dir_name

        if not input_dir.is_dir():
            print(f"  SKIP: {input_dir} does not exist")
            continue

        model_name = extract_model_name(input_dir_name)
        model_dir = output_base / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{input_dir_name}]")

        # Iterate over subdirectories (latest per task by default, all with --all)
        if args.all:
            subdirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
        else:
            subdirs = sorted(get_latest_subdirs(input_dir))
        for subdir in subdirs:
            # The trace must be solve_out.txt — the JSONL form the viewer
            # parses. solve_parsed.txt is human-readable plaintext and is
            # NOT a valid input for build.py.
            src_file = prefer_sanitized(subdir / "solve_out.txt")
            if not src_file.exists():
                print(f"  MISS: {subdir.name} (no solve_out.txt)")
                missing_count += 1
                continue

            task_name = subdir.name
            dest_dir = model_dir / task_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_file = dest_dir / "solve_out.txt"
            was_sanitized = copy_file_sanitized(src_file, dest_file, api_keys)
            if was_sanitized:
                sanitized_count += 1

            # Run-level metadata — copy if present, otherwise just skip
            # (no placeholder text, which previously broke downstream JSON
            # parsing and verdict detection).
            # NB: solve_parsed.txt is also shipped for direct human reading;
            # the viewer ignores it, but it's a useful escape hatch.
            # The canonical contamination verdict in this branch's two-judge
            # scheme is judgement_gpt5_4.json (or _rerun). We also ship the
            # GPT-5.4 contamination-judge trace so the viewer can show *why* a
            # run was flagged. The third-party-API-usage judge
            # (judgement_api*.json) is deliberately NOT extracted: it is
            # archival, not consumed by scoring, and known to flip on rerun
            # (non-deterministic false positives). The old single-judge files
            # (contamination_judgement.txt / judge_output.json) no longer exist.
            copy_other_files(subdir, dest_dir, 'solve_parsed.txt', api_keys=api_keys, optional=True)
            copy_other_files(subdir, dest_dir, 'metrics.json', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'metrics_averaged.json', api_keys=api_keys, optional=True)
            # Rerun output supersedes the original (it is the corrected
            # verdict/trace) and is written under the canonical non-rerun name.
            copy_preferring_rerun(subdir, dest_dir, 'judgement_gpt5_4.json', api_keys=api_keys)
            copy_preferring_rerun(subdir, dest_dir, 'judge_output_gpt5_4.json', api_keys=api_keys)
            copy_preferring_rerun(subdir, dest_dir, 'judge_output_gpt5_4.txt', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'error.log', api_keys=api_keys, optional=True)
            copy_other_files(subdir, dest_dir, 'time_taken.txt', api_keys=api_keys)
            copy_other_files(subdir, dest_dir, 'system_monitor.log', api_keys=api_keys, optional=True)

            # Workspace — copy text files only, skip weights/checkpoints/caches.
            ws_count = copy_workspace(subdir / "task", dest_dir / "task", api_keys)

            tag_bits = []
            if was_sanitized: tag_bits.append("sanitized")
            if ws_count:      tag_bits.append(f"task:{ws_count}")
            tag = f"  [{', '.join(tag_bits)}]" if tag_bits else ""
            print(f"  OK: {subdir.name}{tag}")
            copied_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {copied_count} copied, {sanitized_count} sanitized, {missing_count} missing")
    print(f"Output: {output_base}")

    if _warnings:
        print(f"\n--- {len(_warnings)} pattern warnings (review manually) ---")
        for w in _warnings:
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


def _looks_like_text(path: Path) -> bool:
    """Heuristic — read 4KB; reject anything with NUL bytes or low-print ratio."""
    try:
        head = path.open('rb').read(4096)
    except OSError:
        return False
    if not head:
        return True
    if b"\x00" in head:
        return False
    printable = sum(1 for b in head if b in (9, 10, 13) or 32 <= b < 127)
    return printable / len(head) > 0.85


def copy_workspace(src_root: Path, dest_root: Path, api_keys: list[str]) -> int:
    """Mirror `task/` into the output, copying only text files and skipping
    known-binary extensions / weight or cache directories. Each text file
    is API-key-sanitized. Returns the number of files copied.

    Uses os.walk with in-place dirname pruning so we never *descend* into
    weight/cache trees (e.g. final_model/, sft_output/) — rglob walks them
    fully and then we'd just discard the results. Big speedup when runs
    have GB-scale checkpoint dirs."""
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
            if not _looks_like_text(src):
                continue
            rel = src.relative_to(src_root)
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                copy_file_sanitized(src, dest, api_keys)
                count += 1
            except (OSError, UnicodeDecodeError) as e:
                print(f"  WARN: could not copy {rel} ({e})")
    return count


if __name__ == "__main__":
    main()