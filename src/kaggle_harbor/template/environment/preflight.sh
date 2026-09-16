#!/bin/bash
# Everything run_task.sh does between "job dir is ready" and "the agent's
# first token", ported to the one hook Harbor offers before the agent
# starts: [environment.healthcheck].
#
# Upstream, all three of these live in the agent phase itself
# (run_task.sh:78 and :250):
#
#   bash src/utils/create_timer.sh $NUM_HOURS $JOB_DIR/task/timer.sh
#   ...
#   bash -c "{ python /home/ben/check_cuda.py && \
#              python /home/ben/check_cuda_writing.py || exit 1; \
#              bash /home/ben/system_monitor.sh & ... }"
#
# Harbor owns the agent command, so there is nowhere else to put them.
#
# ⚠️ The healthcheck runs ONCE on success, not repeatedly: `run_healthcheck`
# (environments/base.py:1357-1381) loops only while the command keeps failing
# and `return`s on the first rc == 0. It retries `retries` times and then
# raises HealthcheckError, which is what makes the CUDA gate abort the run.
# So every step here is written to be idempotent (a retry after a CUDA failure
# re-enters it) but nothing may *depend* on a second pass to finish its work.
#
# Ordering matters and matches upstream: the CUDA gate runs first and a
# failure aborts before the timer starts or the monitor spawns, so a node
# without a usable H100 never consumes budget.

set -u

HOME_DIR=/home/ben
TASK_DIR=/home/ben/task
HF_HOME_DIR=/home/ben/hf_cache

# --- 0. HuggingFace cache ---------------------------------------------------
# run_task.sh:287 wraps the whole agent phase in `with_huggingface_overlay`,
# so upstream's cache is already in place before the CUDA gate runs — hence
# this being step 0 rather than step 4.
#
# Upstream (run_task.sh:180-195) fuse-overlayfs's the host cache as a *lower*
# layer under a scratch *upper* layer and binds the merged tree onto
# /home/ben/hf_cache (:242). Two properties follow, and both matter:
#
#   1. the agent reads the whole 160 GB cache at /home/ben/hf_cache, and
#   2. the agent can WRITE there — new datasets, .locks, metadata — with the
#      writes landing in the upper layer and being discarded afterwards
#      (:189-193). Prompt rule 6 invites exactly this.
#
# Kaggle has neither piece: dataset mounts are read-only, and there is no
# upper layer. The mount also cannot live at /home/ben/hf_cache, because the
# sandbox home is an [[artifacts]] source and harbor would try to copy the
# cache into the verifier.
#
# So the cache mounts outside the home and is linked back in one repo at a
# time. A bare `ln -s $mount /home/ben/hf_cache` would also have kept the
# artifact small, but it makes the whole cache read-only, which the overlay
# never was; per-repo links keep /home/ben/hf_cache itself a real writable
# directory, so a repo the agent downloads mid-run is created alongside the
# links exactly as it would have been created in the upper layer. The one
# thing an overlay does that this does not is copy-up: rewriting a file inside
# an already-cached repo fails here instead of shadowing it.
#
# PTB_HF_CACHE_MOUNT is a COLON-SEPARATED LIST, PATH-style. The cache is
# ~160 GB across ~396 HuggingFace repos, which unpacks to well over the 1,000
# files a single Kaggle dataset allows without ExtraDatasetsQuota
# (DatasetConstants.cs:42), so it ships as several datasets, each mounted
# separately. Merging is free here: every shard carries the same top-level
# layout ($HF_HOME's own — `hub/` for models, `datasets/` for datasets), the
# `mkdir -p` below is idempotent, and the per-repo links from different shards
# simply land side by side in one tree. Order does not matter and there is no
# dedup to do, because a repo lives in exactly one shard.
#
# Dotted entries (.locks, .no_exist) are deliberately NOT linked — they are
# HF's own bookkeeping, they must be writable, and they are rebuilt on demand.
# `for x in dir/*` skips them for us.
#
# Absent or empty mounts: nothing happens and /home/ben/hf_cache stays the
# empty directory the Dockerfile created. That is the local-run case.
# D233. Twin of tests/test.sh's `_ptb_place` -- see the long comment there for
# the two write paths (hub/datasets--* revision mkdir, datasets/ builder lock)
# that a whole-dir symlink turns into EROFS. Kept identical on both sides so
# the agent cannot hit a failure the verifier is immune to, or vice versa.
# `cp -rs` mirrors directories and symlinks files: writable at every depth,
# zero bytes copied. models--* stay whole-dir symlinks; they are the 346 GiB.
_ptb_place_cache_entry() {
    local top="$1" repo="$2" link="$3"
    case "$top/$(basename "$repo")" in
    datasets/*|modules/*|hub/datasets--*)
        if cp -rs "$repo" "$link" 2>/dev/null; then
            chmod -R u+w "$link" 2>/dev/null || true
            # Drop inherited lock symlinks. The shipped cache already contains
            # `<version>_builder.lock` files; cp -rs turns each into a symlink
            # into the read-only mount, and filelock opens it O_CREAT|O_RDWR,
            # which fails ELOOP ("too many levels of symbolic links") instead
            # of the EROFS we just removed -- a different error, same silent
            # baseline fallback. Verified against the real 372 GiB mount by
            # the zz-hfcache probe. Removing them lets huggingface create real
            # locks in the now-writable directory.
            find "$link" -type l \( -name '*.lock' -o -name '*.incomplete' \) \
                -delete 2>/dev/null || true
            return 0
        fi
        rm -rf "$link" 2>/dev/null || true
        echo "[preflight] WARN: cp -rs unavailable for $repo; linking instead," \
             "writes inside it will fail with EROFS" >&2
        ln -s "$repo" "$link"
        ;;
    *)
        ln -s "$repo" "$link"
        ;;
    esac
}

_ptb_link_cache_shard() {
    # -e OR -L on every target: -e follows the link, so a link whose source has
    # gone would look absent and `ln -s` would then fail with "File exists".
    local mount="$1" src dst repo link top
    for src in "$mount"/*; do
        [ -e "$src" ] || continue
        dst="$HF_HOME_DIR/$(basename "$src")"
        if [ -d "$src" ]; then
            mkdir -p "$dst"
            top="$(basename "$src")"
            for repo in "$src"/*; do
                [ -e "$repo" ] || continue
                link="$dst/$(basename "$repo")"
                [ -e "$link" ] || [ -L "$link" ] || \
                    _ptb_place_cache_entry "$top" "$repo" "$link"
            done
        else
            [ -e "$dst" ] || [ -L "$dst" ] || ln -s "$src" "$dst"
        fi
    done
}

if [ -n "${PTB_HF_CACHE_MOUNT:-}" ] && [ ! -f "$HOME_DIR/.preflight_hf_ok" ]; then
    mkdir -p "$HF_HOME_DIR"
    IFS=':' read -r -a _ptb_mounts <<< "${PTB_HF_CACHE_MOUNT}"
    for _ptb_mount in "${_ptb_mounts[@]}"; do
        [ -n "$_ptb_mount" ] || continue
        if [ ! -d "$_ptb_mount" ]; then
            echo "[preflight] WARN: cache shard $_ptb_mount is not mounted" >&2
            continue
        fi
        _ptb_link_cache_shard "$_ptb_mount"
    done
    # Sentinel only when something was actually linked. An empty or
    # not-yet-populated mount must not latch a "done" flag that a later retry
    # would honour.
    if [ -n "$(ls -A "$HF_HOME_DIR" 2>/dev/null)" ]; then
        echo "[preflight] linked ${#_ptb_mounts[@]} cache shard(s) into ${HF_HOME_DIR}" >&2
        touch "$HOME_DIR/.preflight_hf_ok"
    else
        echo "[preflight] WARN: no cache shard of ${PTB_HF_CACHE_MOUNT} yielded anything; nothing linked" >&2
    fi
fi

# --- 1. CUDA gate -----------------------------------------------------------
# run_task.sh:250. check_cuda.py verifies an idle H100 is present and
# check_cuda_writing.py additionally allocates a tensor on each device. Both
# touch ./cuda_not_available on failure and exit non-zero; `|| exit 1` in the
# original turns that into an aborted run.
#
# PTB_SKIP_CUDA_CHECK exists only so the pipeline can be exercised on a
# GPU-less machine. build_tasks.py emits it into [environment.env] for
# --smoke builds only, and stamps the task metadata as a smoke build, so a
# production task can never skip the gate by accident.
if [ "${PTB_SKIP_CUDA_CHECK:-}" != "1" ]; then
    if [ ! -f "$HOME_DIR/.preflight_cuda_ok" ]; then
        cd "$TASK_DIR" || exit 1
        python "$HOME_DIR/check_cuda.py" && python "$HOME_DIR/check_cuda_writing.py" || exit 1
        touch "$HOME_DIR/.preflight_cuda_ok"
    fi
else
    echo "[preflight] PTB_SKIP_CUDA_CHECK=1 — smoke build, CUDA gate skipped" >&2
fi

# --- 2. timer.sh ------------------------------------------------------------
# run_task.sh:78. create_timer.sh is copied in verbatim and invoked here
# rather than at image build time: it bakes CREATION_DATE=$(date +%s) into
# the generated script, and upstream's stamp is job-prep time — minutes
# before the agent starts, not whenever the image happened to be built.
#
# PTB_NUM_HOURS is set in [environment.env] from the generator's
# --num-hours, the same value get_prompt.py rendered into rule 2 of the
# instruction.
if [ ! -f "$TASK_DIR/timer.sh" ]; then
    bash "$HOME_DIR/create_timer.sh" "${PTB_NUM_HOURS:-10}" "$TASK_DIR/timer.sh"
fi

# --- 3. system monitor ------------------------------------------------------
# run_task.sh:250 launches it in the background next to the agent. It writes
# system_monitor.log relative to the working directory, which upstream is
# /home/ben/task — that is the file run_task.sh:339 later collects.
if ! pgrep -f "$HOME_DIR/system_monitor.sh" > /dev/null 2>&1; then
    cd "$TASK_DIR" || exit 1
    nohup bash "$HOME_DIR/system_monitor.sh" > /dev/null 2>&1 &
fi

exit 0
