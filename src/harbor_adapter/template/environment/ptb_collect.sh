#!/bin/bash
# PostTrainBench post-agent collection. Installed at /usr/local/bin/ptb_collect.sh
# in the AGENT image and run by harbor as the `[[verifier.collect]] service="main"`
# hook in task.toml: after the agent process exits, before the agent container
# is stopped. Harbor treats hook failures as warnings (trial continues), so
# each step below is independent and reports loudly.
#
#   1. Trained weights -> shared Modal volume (mounted at $PTB_VOLUME_DIR in
#      both the agent and the verifier sandbox). This is the only way the
#      model reaches the separate verifier: harbor's artifact download goes
#      through Modal's filesystem API, which caps single files at 5 GiB.
#
#   3. Agent transcript -> /logs/artifacts/agent_logs (for the judges).
#
#   2. Code + data snapshot -> /logs/artifacts/workspace. Harbor always
#      transfers the conventional /logs/artifacts dir to the host and into the
#      verifier, so the contamination judge sees the agent's code (and its
#      generated training data) there. The snapshot is size-budgeted instead
#      of relying on exclude patterns: agents leave arbitrary multi-GB dirs
#      behind (checkpoints, `final_model2`, datasets), and one such dir is
#      enough to blow harbor's 120 s tar timeout / Modal's 5 GiB download
#      cap and lose the judge's view of the code. Files are taken smallest
#      first, so source code always fits; weight formats are never taken.
set -u

# Harbor only surfaces a collect hook's output when it fails; keep our own
# copy in the agent logs dir, which harbor downloads to <trial>/agent/.
mkdir -p /logs/agent
exec > >(tee -a /logs/agent/ptb_collect.log) 2>&1

WORKSPACE="${PTB_WORKSPACE:-/home/agent/workspace}"
VOLUME_DIR="${PTB_VOLUME_DIR:-/mnt/ptb_final_model}"
SNAPSHOT_DIR="${PTB_SNAPSHOT_DIR:-/logs/artifacts/workspace}"
# Budget: harbor gzips the snapshot with a 120 s timeout and Modal caps the
# download at 5 GiB; 2 GB total is comfortably inside both. Per-file cap keeps
# a single stray blob from eating the budget.
MAX_FILE_BYTES="${PTB_SNAPSHOT_MAX_FILE_BYTES:-536870912}"    # 512 MiB
MAX_TOTAL_BYTES="${PTB_SNAPSHOT_MAX_TOTAL_BYTES:-2147483648}" # 2 GiB
MAX_FILES="${PTB_SNAPSHOT_MAX_FILES:-50000}"

echo "=== ptb_collect: $(date -u +%FT%TZ) ==="

# ---- 1. weights -> volume -------------------------------------------------
if [ -d "$WORKSPACE/final_model" ]; then
    echo "[weights] copying $WORKSPACE/final_model -> $VOLUME_DIR"
    if mkdir -p "$VOLUME_DIR" && cp -a "$WORKSPACE/final_model/." "$VOLUME_DIR/"; then
        sync
        echo "[weights] done: $(du -sh "$VOLUME_DIR" | cut -f1)"
        ls -la "$VOLUME_DIR"
    else
        echo "[weights] ERROR: copy failed (is the volume mounted at $VOLUME_DIR?)"
        ls -la "$(dirname "$VOLUME_DIR")" || true
    fi
else
    echo "[weights] no $WORKSPACE/final_model directory - nothing to hand off"
    ls -la "$WORKSPACE" || true
fi

# ---- 2. code + data snapshot -> /logs/artifacts/workspace ------------------
echo "[snapshot] staging $WORKSPACE -> $SNAPSHOT_DIR (file <= $MAX_FILE_BYTES B, total <= $MAX_TOTAL_BYTES B, max $MAX_FILES files)"
rm -rf "$SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"
python3 - "$WORKSPACE" "$SNAPSHOT_DIR" "$MAX_FILE_BYTES" "$MAX_TOTAL_BYTES" "$MAX_FILES" <<'PY' || echo "[snapshot] ERROR: snapshot failed"
import os, shutil, sys
ws, out, max_file, max_total, max_files = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
PRUNE = {".git", "__pycache__", ".cache", ".huggingface", "wandb", ".wandb", ".venv", "venv",
         "node_modules", ".ipynb_checkpoints"}
# Weight / tensor formats: never useful to the judge, always large.
SKIP_EXT = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".npy", ".npz", ".h5", ".msgpack", ".onnx"}
cands, skipped_ext, skipped_big = [], 0, 0
for root, dirs, files in os.walk(ws):
    dirs[:] = [d for d in dirs if d not in PRUNE]
    if root == ws:
        # final_model travels via the shared volume; the verifier symlinks it
        # into the judge's task dir. Keeping a partial copy (config/tokenizer
        # files pass the size filter) would shadow that symlink.
        dirs[:] = [d for d in dirs if d != "final_model"]
    for f in files:
        p = os.path.join(root, f)
        if os.path.islink(p) or not os.path.isfile(p):
            continue
        if os.path.splitext(f)[1].lower() in SKIP_EXT:
            skipped_ext += 1; continue
        sz = os.path.getsize(p)
        if sz > max_file:
            skipped_big += 1; continue
        cands.append((sz, p))
cands.sort()  # smallest first: code and configs are guaranteed to fit
total, taken, dropped = 0, 0, 0
for sz, p in cands:
    if taken >= max_files or total + sz > max_total:
        dropped += 1; continue
    rel = os.path.relpath(p, ws)
    dst = os.path.join(out, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(p, dst)
    total += sz; taken += 1
print(f"[snapshot] done: {taken} files, {total/1e6:.1f} MB; skipped {skipped_ext} weight-format, "
      f"{skipped_big} over per-file cap, {dropped} over total budget")
PY
# Record what was left behind, for postmortems.
{ echo "# workspace top-level sizes at collection time"; du -sh "$WORKSPACE"/* 2>/dev/null; } \
    > "$SNAPSHOT_DIR/.ptb_workspace_sizes.txt" || true

# ---- 3. agent trace -> /logs/artifacts/agent_logs ---------------------------
# Harbor's installed agents tee their transcript to /logs/agent/<agent>.txt
# (claude-code.txt, codex.txt, ...). Separate verifiers never receive
# /logs/agent, so ship the transcripts through the conventional artifacts dir:
# the verifier's judges read them as solve_out.txt / solve_parsed.txt.
LOGS_DST="${PTB_AGENT_LOGS_DST:-/logs/artifacts/agent_logs}"
rm -rf "$LOGS_DST"; mkdir -p "$LOGS_DST"
n=0
for f in /logs/agent/*.txt; do
    [ -s "$f" ] || continue
    cp "$f" "$LOGS_DST/" && n=$((n+1))
done
echo "[trace] staged $n non-empty agent transcript(s) -> $LOGS_DST: $(ls "$LOGS_DST" 2>/dev/null | tr '\n' ' ')"

echo "=== ptb_collect: finished $(date -u +%FT%TZ) ==="
