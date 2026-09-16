#!/usr/bin/env python3
"""Run the full PostTrainBench sweep on Kaggle Benchmarks.

    7 benchmarks x 4 models = 28 cells
    x 3 attempts            = 84 runs
    14 concurrent           = ~6 waves x ~12 h = ~3 days

Upstream averages three trials per cell. Those three are three INDEPENDENT
sweeps -- commit.sh has no attempt loop, it is re-run under a new
EXPERIMENT_NAME and scripts/utils.py groups the resulting directories so
scripts/aggregate.py can take mean and sample stddev per cell. Kaggle's
nAttempts is serial inside one session (3 x 10 h = 30 h, past the session cap),
so the faithful mechanism is nAttempts=1 scheduled three times. That is what
this does, and it is what the team leads decided on 2026-09-01.

Design notes, all of them learned the hard way this week:

* **Resumable.** Three days is long enough that something will interrupt it.
  All state lives in a JSON file next to this script; re-running picks up where
  it stopped and never double-schedules a (cell, attempt) already sent.
* **Capacity-driven, not wave-driven.** It keeps CONCURRENCY runs in flight
  rather than waiting for a whole wave to drain, so one slow cell does not idle
  thirteen GPUs.
* **Never schedules within COLLISION_GUARD_SEC of a push.** Two runs of one
  task version 38 s apart left the second with an empty job.log and a trial
  that never started (run 868965).
* **Reads the `error` body on push.** /tasks/push answers HTTP 200 on
  validation failure.
* **Reads runSkippedReason on schedule.** The scheduler is partial-success; a
  200 does not mean anything was scheduled.

Usage:
    python3 run_posttrainbench.py --preflight     # checks only, changes nothing
    python3 run_posttrainbench.py --generate      # build + push the 28 tasks
    python3 run_posttrainbench.py --run           # start/resume the sweep
    python3 run_posttrainbench.py --status        # one-shot progress report

Run it detached so it survives this shell:
    nohup python3 run_posttrainbench.py --run >> sweep.out 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PTB = HERE / "PostTrainBench"
BUILD_TASKS = PTB / "src" / "kaggle_harbor" / "build_tasks.py"
VERIFY = PTB / "src" / "kaggle_harbor" / "verify_fidelity.py"
STATE_PATH = HERE / "sweep_state.json"
LOG_PATH = HERE / "sweep.log"
# Touch to release ONE wave past the barrier in run_sweep(). Deliberately a
# file rather than a command: it works from any shell, needs no invocation or
# env reconstructed hours later, and survives the driver being backgrounded.
WAVE_GO = HERE / ".ptb-wave-go"

OWNER = "jonathanouyang"
DEFS_DATASET = "ptb-sweep-defs"
TASK_PREFIX = "ptb"

# The CLI build comes from the slug, not from anything here: the platform
# parses `codex-0.146.1-...` and passes `--ak version=0.146.1`, confirmed in
# the run logs ("version pin '0.146.1' (--ak version=)"). The port pins no
# versions of its own.
# --------------------------------------------------------------------------
# THE CANDIDATES — one agent slug each, and that is the ONLY thing that varies
# --------------------------------------------------------------------------
#
# An Agent on Kaggle IS the combo: harness + model version + reasoning effort,
# in one slug. So a run is fully specified by its slug, and switching
# candidates means switching this list -- nothing about the task changes.
#
# That is the whole point of the dispatcher route (see HARNESS below): the 28
# tasks are candidate-agnostic. `agents.ptb_harness:PtbHarness` reads
# KAGGLE_AGENT_HARNESS at run time, becomes the scheduled harness, and applies
# the original's non-interactive clause iff that harness is Claude
# (get_prompt.py:61). One task set, every combo, correct prompt each.
CANDIDATES = [
    "claude-code-2.1.223-claude-opus-5-max-reasoning",
    # gpt-5.4 + codex was asked for but no such Agent exists -- the codex fleet
    # is 5.5 and 5.6 only, even though gpt-5.4-2026-03-05 is a published,
    # proxy-enabled MODEL version. Nobody has built an Agent pairing them, and
    # doing so needs CreateAgent (agentic-candidates-creation, now granted).
    #
    # xhigh is the TOP of the 5.5 line: the 5.5 agents offer
    # medium/high/xhigh, while the 5.6 variants offer medium/high/max. Same
    # asymmetry we hit in August when `max` was rejected on a 5.6 model.
    "codex-0.146.1-gpt-5.5-xhigh-reasoning",
    # Third candidate. opencode is the only other harness with a PtbHarness_*
    # class, and its effort kwarg (`variant`) is now taken from the platform
    # switch rather than assumed.
    "opencode-1.18.14-gemini-3.1-pro-preview-high-reasoning",
]

# Pinned into the task via envVariables. This route -- NOT
# overrideHarnessVersionCustomSlug -- is what leaves candidateType = agents so
# a full agent slug is schedulable (measured 2026-09-03 on zz-hroute-env).
# scratch.md:1786-1798 is emphatic that pinning a harness "would have destroyed
# the benchmark", because PTB rows vary both model AND harness.
HARNESS = "agents.ptb_harness:PtbHarness"

UPSTREAM_AGENT = "ptb"                # -> metadata.json. NOT the harness name:
                                      # the dispatcher supplies the real one at
                                      # run time. Must not contain 'claude', or
                                      # get_prompt.py bakes the clause in and
                                      # the dispatcher adds it a second time.
AGENT_CONFIG = "claude-opus-5"        # -> the judges' harness-identity clause

# --------------------------------------------------------------------------
# CAPACITY — the hard constraint the whole schedule is shaped around
# --------------------------------------------------------------------------
#
#   14 H100 GPUs were granted for this sweep.
#   Each run declares `gpus = 1`, so one run occupies exactly one GPU.
#   => 14 runs in parallel. Never more.
#
# This is the ONE number to change if the grant changes; everything else
# derives from it. Do not confuse it with the 28 cells or the 84 runs:
#
#   28 cells    = 7 benchmarks x 4 models        (the matrix)
#   84 runs     = 28 cells x 3 attempts          (the work)
#   14 parallel = the GPU grant                  (the pipe)
#   ~6 waves    = 84 / 14                        (consequence)
#   ~3 days     = 6 waves x ~12.2 h              (consequence)
#
# ⚠️ 20 is EXACTLY Kaggle's ScheduledTasksLimit (BenchmarkBatchScheduler.cs:91),
# not under it. At 14 the grant bound first and the cap was irrelevant; at 20
# they coincide, so whether the 20th request is accepted depends on the
# comparison being `>` or `>=`, which has not been read. Not fatal either way:
# a refusal comes back as runSkippedReason, and schedule_one() requeues rather
# than failing. Worst case is churn on the last slot of each wave. Drop to 19
# to stay strictly under, or get ops flag
# BYPASS_BATCH_SCHEDULE_BENCHMARK_TASK_RUNS_LIMIT (26767) to go above.
#
# ⚠️ Scheduling now starts a 12 h clock (MODEL_PROXY_API_KEY is minted at
# schedule time and never re-minted). NEVER set this above the number of
# clusters actually free -- the surplus does not wait harmlessly, it burns its
# token in a queue and starts dead. That is what killed the 2026-09-04 sweep:
# 14 scheduled into a pool with 0 free, queued 15 h, all 401.
#
# The scheduler below is capacity-driven rather than wave-driven: it refills a
# GPU the moment one frees, instead of waiting for a whole wave of 14 to drain.
# One slow cell therefore costs one GPU-slot, not thirteen idle ones.
GPUS_GRANTED = 25   # 2026-09-14: 25 clusters free
CONCURRENCY = min(GPUS_GRANTED, 20)

# One attempt: a COMPLETE row across every cell before any variance data.
# 2026-09-10. Three attempts is 252 runs (~12.6 waves, most of next week) and
# buys stddev on cells whose first number is not in yet. One attempt is 84 runs
# across all 28 cells -- one full leaderboard row, ~5 waves.
#
# Not a one-way door: sweep_state.json records finished runs and skips them, so
# raising this to 2 later schedules ONLY the second attempts. Nothing is redone.
ATTEMPTS = 1
NUM_HOURS = 10                        # upstream's budget (commit.sh)
cfg_AGENT_GRACE_MINUTES = 5           # run_task.sh:225 `NUM_HOURS * 60 + 5`

# H100 pool, from HarborSessionProtoBuilder.cs:91-100:
#   208 CPUs, 1872 GB RAM, 8 GPUs, 88 GB storage on the default machine.
# Upstream's single_task.sub asks for 16 CPUs / 128 GB / 400 GB / 1 H100. CPUs
# and memory now fit inside the pool, so they are set to upstream's values --
# closer than the A100 config could get. Storage is the one axis that does not:
# 400 GB exceeds the default machine's 88 GB and routes to the scarcer 1 TB
# pool. See --preflight, which prints the consequence.
# "H100", matching config.GPU_TYPES and upstream's single_task.sub:12 pin.
# This was lowercase "h100" until 2026-09-03, on a belief that H100s were only
# reachable by the lowercase name. No evidence supports that, and both
# config.py and the task build that produced run 920881 -- a real H100 run,
# reward 0.7498 -- use "H100". verify_fidelity:519 compares the built task
# against cfg.GPU_TYPES, so the lowercase override was failing all 28 tasks.
GPU_TYPES = ["H100"]
CPUS = 16
MEMORY_MB = 131072
# single_task.sub:11 `request_disk=400G`, matched exactly. Every one of the
# nine sweep .sub files asks for 400 G; none asks for less.
#
# This was 81920 (80 GB) under D224, to stay inside the default H100 machine's
# 88 GB and avoid stranding an 8-GPU host on 2 jobs. That deviation was
# retired on 2026-09-09 after it destroyed a wave: D224 justified 80 GB with an
# estimate ("a 4B checkpoint is ~16 GB, so several fit"), and opus-5 at max
# effort wrote ~6 checkpoints plus AdamW optimizer states and vLLM compile
# caches. The disk filled, ext4 went read-only, overlay2 entered a permanent
# EIO state, and 8 of 8 completed runs died -- surfacing as
# `mmap allocate error: input/output error` in the VERIFIER build rather than
# as a disk error in the agent, which is what made it hard to read.
#
# The original never constrains disk in the prompt (prompt.txt has no mention
# of disk, storage or checkpoints), so an agent here behaves exactly as one
# with 400 G would. Coaching it to save fewer checkpoints would be a genuine
# behavioural deviation and would bias our numbers against the published
# leaderboard invisibly; giving it the disk the original gives it does not.
#
# Affordable because a whole machine is dedicated per run (confirmed
# 2026-09-09), so the 8-GPU packing argument behind D224 no longer applies.
UPSTREAM_DISK_MB = 400 * 1024        # single_task.sub:11, request_disk=400G
STORAGE_MB = UPSTREAM_DISK_MB
GPUS = 1

# Sizing. A 10 h agent plus a verifier that has measured up to ~1 h 53 m needs
# more than the 12 h default session bound -- see --preflight.
# run_task.sh:225 wraps the agent in `timeout ... "$((NUM_HOURS * 60 + 5))m"`
# -- the budget PLUS a five-minute grace, so the agent's own timer (from
# create_timer.sh) fires first and it can still write final_model and clean
# up. Dropping the grace, as `NUM_HOURS * 3600` did, means harbor kills the
# agent at exactly the budget, which can cost the checkpoint. D40 records
# 36300 and build_tasks.py already defaults to it; this override was the only
# thing putting 36000 into the pushed tasks (2026-09-09).
AGENT_TIMEOUT_SEC = (NUM_HOURS * 60 + cfg_AGENT_GRACE_MINUTES) * 60
VERIFIER_TIMEOUT_SEC = 12 * 3600

# The agent image. v2 and v3 differ in exactly one file: /usr/local/bin/
# preflight.sh, which is the healthcheck the platform runs before the agent
# starts. v3 carries the D233 agent-side cache fix (copy-on-write inside
# already-cached HF repos); v2 does not, so an agent there can add new repos
# beside the cache but cannot extend an existing one -- the upstream overlay
# could. That is a fidelity gap, NOT a scoring bug: rewards come from the
# verifier, whose copy of the fix ships with the task and is already live at
# defs v5.
#
# ⚠️ preflight.sh CANNOT be delivered by a task push. It lives inside the
# image and the healthcheck runs it from /usr/local/bin (proved by probe run
# 1375308: the task bundle's copy was ignored and the image's ran). Changing
# it therefore requires a new image tag AND a task re-push pointing at it.
#
# Default stays v2 deliberately. Flipping the default would silently re-point
# any future --generate at an image the platform may not have cached yet, and
# would split a running sweep across two agent environments with nothing in
# the task version to record it. Switch only at a sweep boundary:
#
#     PTB_AGENT_IMAGE=...ptb-agent:v3 python3 run_posttrainbench.py --generate
#
# FLIPPED TO v3 on 2026-09-08, ahead of the next re-push (which is gated on two
# Kaggle features: n_attempts concurrency and multiple tasks per GPU cluster).
# Nothing switches until the next `--generate`; the sweep running now was
# pushed at defs v5 against :v2 and is unaffected. Set PTB_AGENT_IMAGE back to
# :v2 to revert.
#
# ⚠️ NOT yet booted on a real Kaggle task. v3 is verified for cache read/write
# (A/B against v2 on a read-only volume: v2 -> Errno 30, v3 -> writes succeed)
# and is a one-file layer over v2, but no run has used it. Smoke one throwaway
# task pinned to it before the real re-push.
AGENT_IMAGE = os.environ.get(
    "PTB_AGENT_IMAGE",
    "us-west1-docker.pkg.dev/kaggle-playground-170215/"
    "kaggle-benchmarks/ptb-agent:v3")
# The verifier BASE. Each task builds its own verifier as `FROM <this>` plus
# its own tests/ -- it is NOT used directly as the verifier image.
#
# ⚠️ Passing this as --verifier-docker-image instead is what produced the
# 2026-09-04 scoring failure. harbor skips the tests upload for a separate
# verifier env (trial.py:618, verifier.py:96-101) and pulls rather than builds
# when docker_image is set (definition.py:26-36), so the image's BAKED /tests
# is the only /tests any task ever sees. v5 was built from the gsm8k task, so
# all 28 tasks ran gsm8k's metadata, test.sh, score_run.py and
# repo/src/eval/tasks/gsm8k/ -- a healthbench run's own verifier printed
# `benchmark_id=gsm8k` and every cell returned gsm8k's zero-shot baseline
# 0.12661106899166036 while reporting `completed`. See D231.
VERIFIER_BASE_IMAGE = ("us-west1-docker.pkg.dev/kaggle-playground-170215/"
                       "kaggle-benchmarks/ptb-verifier:v5")  # judge codex 0.146.1, D227

# --------------------------------------------------------------------------
# HF CACHE — exactly ONE mount, because a session cannot have two
# --------------------------------------------------------------------------
#
# Measured 2026-09-02 (mount-investigation-log.md): a Harbor session mounts
# exactly one Kaggle dataset. Two or more fails with ErroredMountingDataset,
# regardless of size -- 7 mounts totalling 103.9 GB fails, 1 mount of 36.4 GB
# succeeds, and every multi-mount failure dies at the same ~17.5 min whether
# it is 89.7 GB or 394.8 GB. Constant time under a 4.4x size range means a
# fixed retry ladder expiring, not a transfer running out of budget.
#
# So the 14 cache shards were merged into one dataset holding 34 `part-NN`
# directories (630 repos, 397.3 GiB). Kaggle extracts `part-NN.zip` into
# `part-NN/` -- the same behaviour that gave the old shards their `hub/` and
# `datasets/` trees -- so under a single mount they arrive as
# MOUNT_ROOT/part-01 .. part-34, each carrying the cache's own top-level
# layout. `preflight.sh` already merges an arbitrary number of such trees, so
# it needs no change: PTB_HF_CACHE_MOUNT is just the colon-joined part list.
# ...-merged (no suffix) is the WEDGED slug from the first upload attempt: it
# never minted a version and answers 403 to its own owner. -v1 is the live one.
MERGED_DATASET = f"{OWNER}/ptb-hf-cache-merged-v1"
MERGED_MANIFEST = Path("/home/kaggle/ptb-hf-merged-manifest.json") if Path("/home/kaggle/ptb-hf-merged-manifest.json").exists() else (HERE / "ptb-hf-merged-manifest.json")
MOUNT_ROOT = "/mnt/hf-cache"

# A run whose scheduling or execution fails is retried, but only within its
# own pass and only this many times. Unbounded retries on a systematically
# broken cell would burn the GPU grant on one task forever.
#
# RAISED 2 -> 4 on 2026-09-11. The image build has a 6 min deadline and the
# harness npm install takes ~5.5 min, so a codex/opencode start is close to a
# coin flip (observed 0.55 failure on wave 1). At p=0.55 the chance a cell
# never lands in n+1 tries is p^(n+1): 2 retries leaves ~3.3 of 20 cells
# permanently empty, 4 leaves ~1.0. Empty cells are unrecoverable holes in the
# leaderboard row, and a retry that fails costs ~11 min, not 10 h -- these
# runs die during setup, before training.
#
# Safe against the runaway this bound exists to prevent, because a genuinely
# broken cell fails for a DIFFERENT reason and its errors do not match the
# setup-timeout test below, so the breaker still stops the sweep after 3.
MAX_RETRIES = 4

# Global circuit breaker. MAX_RETRIES bounds ONE unit; this bounds the sweep.
#
# The GPU canary only guards the very first run: once any run has proven an
# H100, the gate opens to all 14 and every later failure is merely requeued.
# That is the wrong behaviour for a systemic fault -- a broken image, an
# expired proxy credential, a platform regression like the 8-GPU allocation
# that killed 8/8 runs on 2026-09-04 -- because the sweep would keep feeding
# 10-hour runs into it and quietly burn the grant while nobody is watching.
#
# So: N consecutive terminal failures with no success in between and the whole
# script stops, leaving state on disk to resume from. Counted consecutively
# rather than as a rate, because a rate needs a denominator that does not
# exist early on. Reset by ANY successful run.
#
# 3 is deliberately low. Each run is a 10 h agent plus a 4 h verifier, so three
# failures is already most of a day of cluster time; there is no version of
# "wait and see" that is cheaper than stopping and reading the error.
MAX_CONSECUTIVE_FAILURES = 3

# A run that ends this fast never trained anything, so retrying it is free and
# cannot double-count partial work.
#
# This is a PROXY for "died during setup, before the agent launched". The thing
# we actually want to test is the exception's origin -- _setup_agent versus the
# agent run -- but runs/list only reports "most common error was
# NonZeroAgentExitCodeError", which both produce. Duration is the only
# discriminator available without downloading each failure's artifacts.
#
# Calibrated against the two failures we have measured, and the gap matters in
# one direction much more than the other:
#
#   run 1120769  EIO in _setup_agent, agent never launched     32 s  -> retry
#   run 1120803  agent launched, then 401 expired token       181 s  -> do NOT
#
# Misclassifying the 401 as infrastructure is the expensive mistake: its retry
# would carry the same expired token, fail identically, and chew through the
# fast-fail budget instead of tripping the breaker that should stop the sweep.
# So the threshold sits well below 181 s rather than just under it -- 180 would
# have separated these two by a single second, which is not a margin.
FAST_FAIL_SEC = 120

# ...but a fast failure is NOT free to ignore. These are infrastructure faults
# whose root cause we do NOT have -- the known one is a gcsfuse read returning
# `input/output error` on a path that should return ENOENT
# (/kaggle/input/<defs>/<task>/environment/.env, which exists in 0 of 28
# tasks). It is transient rather than a bad snapshot: 13 of 14 runs read that
# exact file successfully within the same two minutes, off the same dataset
# version, while one did not.
#
# Retrying an unexplained fault silently is how it becomes permanent. So the
# retries are BUDGETED: they do not trip the consecutive-failure breaker (they
# are not a systemic fault and should not stop a 3-day sweep), but they are
# counted, logged individually with their run ids, and once the budget is gone
# the sweep halts and hands over the evidence.
#
# 20 is ~12% of 168. The observed rate was 1/14 (~7%), so a normal sweep should
# use around a dozen. Blowing through 20 means the rate is materially worse
# than measured and the assumption behind retrying at all no longer holds.
MAX_FAST_FAILS = 80
DEFER_WAVE_FAILURES = True

# --------------------------------------------------------------------------
# GPU GATE — no GPU means STOP, never "carry on without one"
# --------------------------------------------------------------------------
#
# An H100 request without BENCHMARKS_ALLOW_USE_ACCELERATOR does NOT error. It
# silently hands back a CPU machine (HarborSessionProtoBuilder.cs:490-493).
# Fourteen concurrent 10 h runs on CPU would consume the entire grant and
# produce nothing, and the runs would look normal while doing it.
#
# So the sweep opens ONE slot until a canary has proven it got a real H100,
# and holds the other 13 shut. Upstream's own gate does the detecting for us:
# preflight.sh:130 runs check_cuda.py, whose test at line 44 is
# `if "H100" in name` -- it rejects a CPU and it rejects a non-H100 GPU. That
# gate runs during startup, so a run still alive after GPU_VERIFY_SEC is past
# it and the accelerator is real.
#
# If the canary dies instead, the sweep HALTS. It does not retry, does not
# fall back, does not widen. A missing GPU is an operator problem, and the
# only safe thing an unattended 3-day script can do with one is stop.
# Was 20 min: hold every slot but one until a canary had SURVIVED that long,
# on the theory that a silent CPU-pool fallback would otherwise waste 20 runs
# x 14 h. That theory is dead. D229 put check_cuda.py in the healthcheck, so a
# non-H100 allocation now fails in ~86 s (measured, 2026-09-04 sanity sweep) --
# and the fast-fail budget and the breaker both catch it. Paying 20 min x 19
# idle clusters to re-detect something that self-reports in 90 s is a bad
# trade.
#
# The gate is kept, cheaply, for a DIFFERENT reason than it was written for:
# scheduling mints a 12 h proxy token, so committing 20 runs to a pool that
# cannot start them is how they die. Seeing one run reach `running` proves the
# pool is actually dispatching, which is the thing worth knowing before
# minting 19 more tokens. That costs a poll cycle, not twenty minutes.
GPU_VERIFY_SEC = 0

# Scheduling mints the 12 h proxy token, so queue time is not free -- it is
# spent credential. Warn at 1 h, give up at 6 h: past that the run has burned
# half its token before starting, and its verifier would be racing expiry even
# if it did start. 2026-09-04 sat 15 h and lost every run.
# A newly scheduled run is unstarted by definition for a few seconds. Only
# past this does "unstarted" mean the pool has actually filled up.
DISPATCH_STALL_SEC = 120

QUEUE_WARN_SEC = 1 * 3600
QUEUE_ABORT_SEC = 6 * 3600

# States that mean a run is over. Used only as a backstop -- `hasEndTime` is
# the primary signal, because this list cannot be trusted to be exhaustive
# (see the note in in_flight() about state="unspecified" on LIVE runs).
_TERMINAL_STATES = frozenset({
    "completed", "failed", "cancelled", "canceled", "errored", "error",
    "timeout", "timedOut", "skipped", "aborted"})

# Poll cadence. A finished run must be noticed quickly so its GPU is refilled,
# but a run lasts ~12 h and polling every minute through the middle of it just
# burns API calls. in_flight() reports the age of the oldest live run and the
# loop switches between these two.
POLL_IDLE_SEC = 300                   # everything mid-run: nothing to learn
POLL_BUSY_SEC = 60                    # a run may finish now, or a slot is free
FAST_POLL_AFTER_SEC = 9 * 3600        # past this age a run is near its budget
POLL_GATE_SEC = 20                    # dispatch gate shut: 19 slots idle, poll hard
# Sized for gcsfuse, which mounts the GCS prefix instead of copying it onto the
# exeunit. Measured 2026-09-03, same 89.1 GB dataset, before and after the fix:
#
#   old path (full copy)   FAILED after 12.0 min
#   gcsfuse (prefix mount) COMPLETED in  2.2 min   <- incl. image pull + verifier
#
# So the mount is effectively free and there is no copy to stampede. An earlier
# revision set these to 3600/240 on the assumption of a ~45 min copy per run;
# that rationale is gone.
#
# The collision guard still earns its keep for a different reason: a push fires
# its own validation run, and two sessions on one task version 38 s apart left
# the second with an empty job.log and a trial that never started (run 868965).
COLLISION_GUARD_SEC = 300             # let the push's validation run settle
# Between consecutive schedule calls. Was 30 s, which cost 19 x 30 s = ~10 min
# on the commit phase and undid most of the probe-then-commit speedup.
#
# The 30 s came from run 868965, where two sessions claiming ONE task version
# 38 s apart left the second with an empty job.log. That is a same-task-version
# race. Consecutive schedules here go to 20 DIFFERENT slugs, so they cannot
# collide that way -- the guard was protecting against something that cannot
# happen in this loop. The push-validation collision it really guards is
# handled separately by wait_for_push_validation() at the end of generate().
#
# RAISED 5 -> 60 on 2026-09-11, for a reason 5 s did not anticipate. It is not
# about task-version races; it is about CONTAINER STARTUP CONTENTION.
#
# Wave 1 scheduled 19 runs at 5 s spacing -- the whole wave inside ~2 min --
# and 11 of them died at 10-12 min with AgentSetupTimeoutError, before any
# training began. The split was by CANDIDATE, not by task: codex 5/6 and
# opencode 5/6 dead, claude 1/7. Every task version had a healthy opus run on
# it at the same time, so the definitions were fine.
#
# The cause is agent setup, which for this port is an npm install inside each
# container: PTB_DEFAULT_VERSION = "latest" (ptb_harness.py) makes every
# adapter render `@latest`, reproducing upstream's update_agent_cli.sh. Twenty
# containers hitting the public registry at once is enough to push the slower
# CLIs past their setup deadline. claude survives because its install is
# faster / better cached, which is exactly why the failure looked like a
# candidate-specific fault rather than a capacity one.
#
# 60 s spreads a 20-run wave over ~20 min of startup instead of ~2 min. The
# cost is ~19 min added to a ~12 h wave (~2.6%), paid once per wave, against
# an observed 55% failure rate on the two affected harnesses.
#
# This is a SCHEDULING change only. It does not touch the task definitions,
# the harness, or PTB_DEFAULT_VERSION, so there is no fidelity deviation: the
# original's `@latest` behaviour is preserved exactly, it is merely no longer
# invoked by twenty containers simultaneously.
SCHEDULE_SPACING_SEC = 60

BENCHMARKS = ["aime2025", "arenahardwriting", "bfcl", "gpqamain", "gsm8k",
              "healthbench", "humaneval"]

# ---------------------------------------------------------------------------
# ⏸  TEMPORARILY DEFERRED — remove both entries once the model-proxy token TTL
#    change ships. This is a scheduling hold, NOT a scope cut.
# ---------------------------------------------------------------------------
#
# These two are the only benchmarks whose EVALUATION calls the model proxy:
#
#     arenahardwriting: ["OPENAI_API_KEY"]      src/eval/tasks/*/info.json
#     healthbench:      ["OPENAI_API_KEY"]
#
# Neither has a mechanical right answer -- you cannot unit-test "is this good
# writing" -- so evaluate.py grades with an LLM. That grading runs over the
# trained model's outputs, which only exist after the agent finishes, so its
# proxy calls are the LAST network traffic of the run.
#
# MODEL_PROXY_API_KEY is minted at SCHEDULE time with a 12 h TTL and is never
# re-minted when a queued run starts. With a 10 h agent budget the verifier
# begins around hour 10, and a measured evaluation phase is ~1 h 52 m, so the
# grading calls land at hour ~12 -- against a credential that dies at 12:00.
# Working judges (fixed this session) push evaluation later still. There is no
# ordering fix: judges already run first, in the port and in the original.
#
# The other five evaluate entirely locally with vLLM and never touch the proxy
# after the agent finishes, so a token that expires during their evaluation
# costs nothing.
#
# cells() and EXPECTED_TASK_SLUGS are deliberately NOT filtered: all 28 tasks
# still get pushed, so the slug set stays frozen and the leaderboard mappings
# stay intact. Only units() -- what gets SCHEDULED -- is held back. Emptying
# this set and re-running --run picks the remainder up; sweep_state.json means
# LLM-as-judge benchmarks (arenahardwriting and healthbench) un-deferred per user instruction.
DEFERRED_BENCHMARKS: set[str] = set()

# Benchmarks whose task definition is known-broken and being repaired. Unlike
# DEFERRED_BENCHMARKS this is NOT a scope change: held benchmarks are still
# generated, still pushed, still part of the 20, still in EXPECTED_TASK_SLUGS.
# They simply stop being SCHEDULED, so no more GPU is spent on a definition we
# already know produces a garbage score. Set at runtime, not in code, so
# lifting a hold cannot be done by accident:
#
#     PTB_HOLD=bfcl python3 run_posttrainbench.py --run
#
# Live use (2026-09-07): bfcl, pending the D233 re-push. Its 4 already-running
# runs cannot be stopped -- Kaggle exposes no cancel RPC (the benchmark service
# has 11 methods and none of them cancel) -- so the hold governs the 32 bfcl
# units NOT yet scheduled, which is where the GPU actually goes.
HOLD_BENCHMARKS = {b.strip() for b in os.environ.get("PTB_HOLD", "").split(",")
                   if b.strip()}
MODELS = ["Qwen/Qwen3-1.7B-Base", "Qwen/Qwen3-4B-Base",
          "HuggingFaceTB/SmolLM3-3B-Base", "google/gemma-3-4b-pt"]

BASE = "https://www.kaggle.com"

# --------------------------------------------------------------------------
# SMOKE MODE
# --------------------------------------------------------------------------
#
# A short, disposable dry run of the WHOLE machine before three days of GPU
# time are committed. It deliberately reuses generate() and schedule_one()
# rather than reimplementing them, because a smoke test that exercises a
# different code path than the sweep proves nothing about the sweep.
#
# What it holds constant with the real run: the images, the merged cache mount,
# storage_mb, the dispatcher env var, the candidate slugs, the push body, the
# schedule body. What it changes: the task slugs (disposable), the defs dataset
# (disposable), the state file, one cell instead of 28, one attempt instead of
# 3, and the duration.
#
# The slugs are throwaway on purpose. Pushing the real ptb-* slugs mints task
# versions that the leaderboard maps runs onto, so a smoke push would either
# pollute the official row or force a re-push that orphans it.
SMOKE = False
SMOKE_STATE = HERE / "smoke_state.json"


def enable_smoke(hours: int, agents: list[str] | None = None) -> str:
    """Rebind the module globals so --smoke drives the real path, small.

    Returns the disposable tag. The tag is persisted in the smoke state file
    and reused on a second invocation, so an interrupted smoke run resumes
    onto its own tasks instead of minting a fresh set every time.
    """
    global SMOKE, TASK_PREFIX, DEFS_DATASET, STATE_PATH, MODELS, BENCHMARKS
    global ATTEMPTS, NUM_HOURS, AGENT_TIMEOUT_SEC, VERIFIER_TIMEOUT_SEC
    global COLLISION_GUARD_SEC, CANDIDATES, SCHEDULE_SPACING_SEC

    SMOKE = True
    STATE_PATH = SMOKE_STATE
    st = load_state()
    tag = st.get("smoke_tag")
    if not tag:
        # Not random for its own sake: a fresh tag guarantees the push creates
        # a brand-new task with no prior versions, which is the only way to be
        # sure the run we watch used the definition we just built.
        tag = "zz-smk-" + uuid.uuid4().hex[:6]
        st["smoke_tag"] = tag
        save_state(st)

    TASK_PREFIX = tag
    DEFS_DATASET = f"ptb-smoke-defs-{tag.rsplit('-', 1)[-1]}"
    # Qwen3-1.7B-Base: smallest weights in the matrix (3.20 GiB), so if the
    # cache mount is wrong the failure shows up as a download attempt fastest.
    # gsm8k: the cheapest verifier of the seven.
    MODELS = MODELS[:1]
    BENCHMARKS = ["gsm8k"]
    ATTEMPTS = 1
    NUM_HOURS = hours
    AGENT_TIMEOUT_SEC = hours * 3600
    VERIFIER_TIMEOUT_SEC = 3600
    COLLISION_GUARD_SEC = 120
    if agents:
        CANDIDATES = list(agents)
    # 5 min between schedules, not the sweep's 30 s. Every candidate here runs
    # against the SAME task version, which is precisely the configuration that
    # produced run 868965: two sessions claiming one task version 38 s apart,
    # the second left with an empty job.log and a trial that never started. In
    # the real sweep the runs are spread over 28 different slugs, so 30 s is
    # fine there; here they are stacked on one.
    SCHEDULE_SPACING_SEC = 300
    return tag


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


_CACHED_TOKEN = ""
_CACHED_TOKEN_TIME = 0.0


def token() -> str:
    global _CACHED_TOKEN, _CACHED_TOKEN_TIME
    now = time.time()
    if _CACHED_TOKEN and (now - _CACHED_TOKEN_TIME) < 300:
        return _CACHED_TOKEN
    last_err = ""
    for attempt in range(5):
        out = subprocess.run(["kaggle", "auth", "print-access-token"],
                             capture_output=True, text=True)
        tok = out.stdout.strip()
        if out.returncode == 0 and tok:
            _CACHED_TOKEN = tok
            _CACHED_TOKEN_TIME = now
            return tok
        last_err = out.stderr.strip() or out.stdout.strip()
        time.sleep(2 * (attempt + 1))
    if _CACHED_TOKEN:
        # If transient refresh failed, return cached token as fallback
        return _CACHED_TOKEN
    raise SystemExit(f"no Kaggle token: {last_err}")


def api(path: str, body: dict | None = None, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json"})
    # HTTPError is a RESPONSE and is returned to the caller to interpret.
    # URLError/socket timeouts are not -- they are the transport failing, and
    # an unhandled one killed the poll loop at 17:44 on 2026-09-09, ending a
    # night's scheduling on a single blip. Retry those with a short backoff;
    # if they persist, raise, because a poller that silently reports "nothing
    # running" while the network is down is worse than one that stops.
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return {"_http_error": e.code, "_body": e.read().decode()[:500]}
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
    raise last


def verify_python() -> str:
    """An interpreter that can `import harbor`, for verify_fidelity.py.

    verify_fidelity imports agents.ptb_harness, which imports harbor, which is
    not installed in the system python. Invoking it with sys.executable made
    the gate die on ModuleNotFoundError -- and because generate() logged only
    stdout, that surfaced as "verify_fidelity FAILED" with a blank line under
    it, which reads exactly like a real fidelity failure. A crashing gate and
    a failing gate need opposite responses, so this resolves the interpreter
    explicitly and refuses to guess.
    """
    seen = []
    for cand in (sys.executable, "/tmp/hv/bin/python",
                 str(Path.home() / ".local/share/uv/tools/harbor/bin/python")):
        if not cand or cand in seen or not Path(cand).exists():
            continue
        seen.append(cand)
        probe = subprocess.run([cand, "-c", "import harbor"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return cand
    raise SystemExit(
        "\nHALTED — no interpreter can import `harbor`, so the fidelity gate "
        "cannot run.\n\n"
        f"  tried: {', '.join(seen) or '(none found)'}\n\n"
        "The gate is what stands between a bad build and 28 pushed tasks, so "
        "this is not skippable.\n"
        "Recreate the venv, e.g.:\n"
        "  python3 -m venv /tmp/hv && /tmp/hv/bin/pip install -q harbor-cli\n"
        "then re-run --generate.")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"defs_version": None, "pushed": {}, "scheduled": {}, "done": {}}


def save_state(s: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    tmp.replace(STATE_PATH)


def model_key(model: str) -> str:
    return model.split("/")[-1].lower().replace(".", "-")


def cells() -> list[tuple[str, str, str]]:
    """(benchmark, model, task_slug) for all 28 cells, BENCHMARK-MAJOR.

    Benchmark outer, model inner: all 4 models for aime2025, then all 4 for
    bfcl, and so on. Combined with the loops in units(), the full order is

        for attempt -> for benchmark -> for model -> for candidate

    so one EVAL is finished across every model and every candidate before the
    next eval starts.

    REVERSED from model-major 2026-09-10. Kaggle's aggregation is live: each
    eval parent node averages its 4 model children and repopulates as results
    land. Under model-major no eval node went fully green until all 4 models
    had run it -- the 22nd run of the pass rather than the 4th -- so every node
    sat partially filled and nothing was complete enough to read until the end.
    Benchmark-major completes one node at a time, which is the unit of review
    asked for: after each wave, one eval is done across all 4 SLMs by all 3
    candidates.

    The trade-off this accepts: no MODEL family is complete across every eval
    until the sweep is nearly over. That is the mirror of the old cost, and the
    lesser one here -- a finished eval node is directly readable, whereas a
    finished model row spans nodes that are each still partial.
    """
    out = []
    for b in BENCHMARKS:
        for m in MODELS:
            # DEFERRED benchmarks are excluded HERE, not just from units(), so
            # they are never generated and never PUSHED. Pushing a task fires an
            # auto-validation run that occupies a cluster for 10-14 h, so
            # pushing 28 to run 20 wasted 8 clusters for a day.
            if b in DEFERRED_BENCHMARKS:
                continue
            out.append((b, m, f"{TASK_PREFIX}-{b}-{model_key(m)}"))
    return out


# The 28 slugs, frozen. cells() derives them from BENCHMARKS x MODELS, which is
# convenient but means an innocent edit to either list -- a renamed benchmark, a
# model key that normalises differently -- silently pushes a NEW set of tasks
# instead of new versions of these. The runs already on the old slugs are then
# stranded: not deleted, just orphaned, and the leaderboard mappings point at
# tasks nothing will ever schedule against again.
#
# So the derivation stays (it is what keeps cells() readable) and this asserts
# the answer it must produce. Changing the matrix on purpose means updating this
# tuple in the same commit, which is exactly the moment to think about what
# happens to the existing runs.
#
# Reordered 2026-09-11 to benchmark-major, matching cells()' loop order as of
# the 2026-09-10 swap. The SET is byte-for-byte the twenty already pushed --
# verified against sweep_state.json (all 20 at task v6) and against the twenty
# leaderboard column mappings, which key on task-version id and are therefore
# indifferent to this tuple's order. Nothing is orphaned; no re-push follows.
EXPECTED_TASK_SLUGS = (
    "ptb-aime2025-qwen3-1-7b-base",
    "ptb-aime2025-qwen3-4b-base",
    "ptb-aime2025-smollm3-3b-base",
    "ptb-aime2025-gemma-3-4b-pt",
    "ptb-arenahardwriting-qwen3-1-7b-base",
    "ptb-arenahardwriting-qwen3-4b-base",
    "ptb-arenahardwriting-smollm3-3b-base",
    "ptb-arenahardwriting-gemma-3-4b-pt",
    "ptb-bfcl-qwen3-1-7b-base",
    "ptb-bfcl-qwen3-4b-base",
    "ptb-bfcl-smollm3-3b-base",
    "ptb-bfcl-gemma-3-4b-pt",
    "ptb-gpqamain-qwen3-1-7b-base",
    "ptb-gpqamain-qwen3-4b-base",
    "ptb-gpqamain-smollm3-3b-base",
    "ptb-gpqamain-gemma-3-4b-pt",
    "ptb-gsm8k-qwen3-1-7b-base",
    "ptb-gsm8k-qwen3-4b-base",
    "ptb-gsm8k-smollm3-3b-base",
    "ptb-gsm8k-gemma-3-4b-pt",
    "ptb-healthbench-qwen3-1-7b-base",
    "ptb-healthbench-qwen3-4b-base",
    "ptb-healthbench-smollm3-3b-base",
    "ptb-healthbench-gemma-3-4b-pt",
    "ptb-humaneval-qwen3-1-7b-base",
    "ptb-humaneval-qwen3-4b-base",
    "ptb-humaneval-smollm3-3b-base",
    "ptb-humaneval-gemma-3-4b-pt",
)


def assert_slugs_frozen() -> None:
    """Refuse to run if cells() no longer produces EXPECTED_TASK_SLUGS."""
    got = tuple(slug for _b, _m, slug in cells())
    if got != EXPECTED_TASK_SLUGS:
        added = sorted(set(got) - set(EXPECTED_TASK_SLUGS))
        gone = sorted(set(EXPECTED_TASK_SLUGS) - set(got))
        raise SystemExit(
            "\nHALTED — the task slug set changed.\n\n"
            f"  would push : {len(got)}\n"
            f"  new slugs  : {added or '(none)'}\n"
            f"  missing    : {gone or '(none)'}\n"
            f"  reordered  : {added == [] and gone == []}\n\n"
            "Pushing these would create tasks the existing runs and leaderboard\n"
            "mappings know nothing about. If the change is intentional, update\n"
            "EXPECTED_TASK_SLUGS in the same commit.")


def units() -> list[tuple[str, int, str]]:
    """The (task_slug, attempt, agent_slug) units of work, attempt-major.

    Attempt-major so that attempt 1 is scheduled for every cell before attempt 2
    begins: pass 1 alone fills a complete leaderboard row, and the stddev later
    passes buy is worthless until that row exists. If the sweep is cut short you
    then hold one complete pass rather than several partial ones. At ATTEMPTS=1
    this is moot, but the ordering is what makes raising ATTEMPTS later safe.

    Ordering alone is not enough to guarantee that, which is why `requeue()`
    exists -- a pass-1 cell that fails has to be retried inside pass 1, not
    appended behind the whole of pass 2.
    """
    # (task_slug, attempt, agent_slug), CELL-MAJOR.
    #
    # Cell outer, candidate inner: every wave of 20 therefore carries all three
    # candidates (~7/7/6) instead of being one candidate's row.
    #
    # REVERSED 2026-09-10, having previously been candidate-major. That order
    # finished opus's whole row first, which reads better on a leaderboard but
    # means a candidate-specific fault -- a broken harness, an auth failure, an
    # empty trace -- does not surface until that candidate's row STARTS, i.e.
    # hour ~20 for gemini. Two thirds of the budget is committed before you can
    # see it. Interleaving surfaces the same fault in wave 1, because every
    # candidate has ~7 runs in it, and 2 bad runs are enough to spot a broken
    # one. It also equalises pool conditions across candidates, so cross-model
    # comparison is not confounded by time-of-day capacity.
    #
    # The cost, accepted deliberately: no candidate's row is complete until the
    # sweep is nearly over. Wave 1 is a fault detector, NOT a scoreboard -- each
    # candidate draws DIFFERENT cells there, so a low number may be a hard cell
    # rather than a weak model. Do not rank on a partial wave.
    # HELD benchmarks are excluded HERE ONLY -- not from cells(), not from
    # EXPECTED_TASK_SLUGS, not from the push identity. A hold means "this
    # task's definition is being repaired, stop giving it new runs", which is
    # a scheduling fact, not a scope change: the sweep is still 20 tasks and
    # still pushed as 20. Contrast DEFERRED_BENCHMARKS, which is a scope
    # change and therefore lives in cells(). Lifting the hold needs no code
    # edit -- unset PTB_HOLD and restart, and the held units schedule against
    # whatever task_version state records by then (D233: bfcl at v5).
    return [(slug, a, c)
            for a in range(1, ATTEMPTS + 1)
            for _b, _m, slug in cells()
            if _b not in HOLD_BENCHMARKS
            for c in CANDIDATES]


def requeue(work: list[tuple[str, int, str]], unit: tuple[str, int, str]) -> None:
    """Put a failed unit back at the end of ITS OWN pass.

    `work.append()` would drop a failed pass-1 cell behind every unit of
    passes 2 and 3 -- roughly two days later -- leaving the leaderboard row
    this ordering exists to complete with a hole in it for the entire sweep.
    """
    _slug, attempt, _c = unit
    for i, (_s, a, _c2) in enumerate(work):
        if a > attempt:
            work.insert(i, unit)
            return
    work.append(unit)


def hf_cache_mounts() -> list[str]:
    """MOUNT_ROOT/part-NN for every part in the merged cache dataset.

    Read from the merge manifest rather than hardcoded. A stale list here
    would not error: `preflight.sh` warns about a missing mount path and skips
    it, so the run would proceed with a partial cache and quietly produce
    wrong numbers.
    """
    if not MERGED_MANIFEST.exists():
        raise SystemExit(f"missing {MERGED_MANIFEST} — run merge_cache.py first")
    man = json.loads(MERGED_MANIFEST.read_text())
    return [f"{MOUNT_ROOT}/{p}" for p in sorted(man["parts"])]


def merged_version() -> int | None:
    d = api(f"/api/v1/datasets/view/{MERGED_DATASET}")
    return d.get("currentVersionNumber")


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def preflight() -> int:
    """Everything that would waste days if it were wrong. Changes nothing."""
    print("=" * 74)
    print("PREFLIGHT — nothing is created or scheduled")
    print("=" * 74)
    problems, warnings = [], []

    total = len(cells()) * ATTEMPTS
    waves = -(-total // CONCURRENCY)
    per_run_h = NUM_HOURS + VERIFIER_TIMEOUT_SEC / 3600 * 0.5 + 0.25
    rows = [
        (f"{len(BENCHMARKS)} benchmarks x {len(MODELS)} models",
         f"{len(cells())} cells", "the matrix"),
        (f"{len(cells())} cells x {ATTEMPTS} attempts",
         f"{total} runs", "the work"),
        ("GPUs granted, 1 GPU per run",
         f"{CONCURRENCY} parallel", "THE PIPE"),
        (f"{total} runs / {CONCURRENCY} parallel", f"~{waves} waves", ""),
        (f"~{per_run_h:.1f} h per run x {waves} waves",
         f"~{waves * per_run_h / 24:.1f} days", ""),
    ]
    print()
    for lhs, rhs, note in rows:
        print(f"  {lhs:<34} = {rhs:<13} {note}")
    print(f"\ncandidates ({len(CANDIDATES)}), all reached through {HARNESS}:")
    for c in CANDIDATES:
        print(f"    {c}")

    # 1. Session bound. The killer.
    session_needed_h = (AGENT_TIMEOUT_SEC + VERIFIER_TIMEOUT_SEC) / 3600 + 0.25
    print(f"\n[1] session bound: task declares "
          f"{AGENT_TIMEOUT_SEC/3600:.0f} h agent + "
          f"{VERIFIER_TIMEOUT_SEC/3600:.0f} h verifier "
          f"(~{session_needed_h:.1f} h with setup)")
    if session_needed_h > 12:
        problems.append(
            f"Needs a ~{session_needed_h:.1f} h session but the default bound is 12 h "
            f"(BenchmarkTaskVersionLimits.MaxSessionExecutionTime).\n"
            f"       Without feature flag BENCHMARKS_ALLOW_DYNAMIC_SESSION_DURATION "
            f"(35030) the session is\n"
            f"       TRUNCATED at 12 h -- the agent's 10 h would finish and the "
            f"verifier would be cut off,\n"
            f"       losing the run after its full budget was spent. The flag is "
            f"USER_CAN_SEE, i.e. self-serve.")
    else:
        print("      fits inside the 12 h default bound")

    # 2. GPU routing. Silent downgrade if unavailable.
    print(f"\n[2] gpu_types={GPU_TYPES} gpus={GPUS} cpus={CPUS} "
          f"memory={MEMORY_MB} MB storage={STORAGE_MB} MB")
    if "h100" not in [g.lower() for g in GPU_TYPES]:
        warnings.append("not asking for h100 by name; H100s are only reachable "
                        "via gpu_types=['h100']")
    # Upstream asks for 400 G and we now match it. The old check warned when
    # storage exceeded the default machine's 88 GB; that ceiling is what
    # starved opus-5 and killed a wave, so the check is inverted: warn when we
    # are BELOW upstream, because that is the condition that bites.
    if STORAGE_MB < UPSTREAM_DISK_MB:
        warnings.append(
            f"storage {STORAGE_MB} MB is BELOW upstream's "
            f"{UPSTREAM_DISK_MB} MB (single_task.sub:11). A 4B "
            f"full fine-tune with several checkpoints and optimizer states "
            f"overruns 80 GB, which fills the disk and puts the node into a "
            f"permanent EIO state (2026-09-09, 8/8 runs lost).")
    else:
        print(f"      matches upstream request_disk "
              f"({UPSTREAM_DISK_MB} MB); needs a full machine per run")
    # The old canary preflight lived here. Removed: it gated on "has a run
    # survived N minutes", which a QUEUED run satisfies, and its purpose --
    # catching a silent CPU-pool fallback -- is covered by check_cuda.py in
    # the healthcheck since D229 (a non-H100 self-reports in ~86 s). --run now
    # gates on the thing that actually costs us: never mint a token the pool
    # cannot start. See the scheduling rule in run_sweep().

    # 3. Batch cap.
    print(f"\n[3] batch cap: scheduling one (task x model) per call, "
          f"{CONCURRENCY} in flight")
    if CONCURRENCY > 20:
        problems.append("CONCURRENCY > ScheduledTasksLimit (20).")
    else:
        print("      under ScheduledTasksLimit=20")

    # 4. Images present.
    print("\n[4] images")
    for img in (AGENT_IMAGE, VERIFIER_BASE_IMAGE):
        r = subprocess.run(["docker", "buildx", "imagetools", "inspect", img],
                           capture_output=True, text=True)
        ok = r.returncode == 0
        print(f"      {'ok  ' if ok else 'MISS'} {img.split('/')[-1]}")
        if not ok:
            problems.append(f"image not in the registry: {img}")

    # 4b. HF cache. Silent when wrong, which is the whole problem.
    print("\n[4b] hf cache")
    if not MERGED_MANIFEST.exists():
        problems.append(f"no {MERGED_MANIFEST} — run merge_cache.py")
    else:
        man = json.loads(MERGED_MANIFEST.read_text())
        mv = merged_version()
        print(f"      merged tree: {man['n_repos']} repos, "
              f"{man['total_bytes']/1024**3:.1f} GiB, {man['n_parts']} parts")
        if not mv:
            problems.append(
                f"{MERGED_DATASET} has no version — the upload has not finished. "
                f"Pushing now yields 28 tasks that mount nothing, and a missing "
                f"cache path is only a WARNING in preflight.sh, so all 84 runs "
                f"would spend their full GPU budget against an empty cache.")
        else:
            print(f"      dataset {MERGED_DATASET}/versions/{mv}")
            print(f"      1 dataset mount at {MOUNT_ROOT}  "
                  f"(> 1 mount = ErroredMountingDataset; measured)")
            if man["n_parts"] != len(hf_cache_mounts()):
                problems.append("part list disagrees with the manifest")

    # 5. Fidelity.
    print("\n[5] fidelity")
    if not VERIFY.exists():
        problems.append(f"missing {VERIFY}")
    else:
        print("      run: python3 PostTrainBench/src/kaggle_harbor/"
              "verify_fidelity.py --tasks <generated>")

    # 6. Credentials.
    print("\n[6] credentials")
    try:
        token()
        print("      kaggle token ok")
    except SystemExit as e:
        problems.append(str(e))

    print("\n" + "=" * 74)
    for w in warnings:
        print(f"WARN     {w}")
    for p in problems:
        print(f"BLOCKER  {p}")
    if not problems:
        print("no blockers")
    print("=" * 74)
    return 1 if problems else 0


# --------------------------------------------------------------------------
# generate + push
# --------------------------------------------------------------------------

def generate(out_dir: Path) -> None:
    """Build all 28 task directories and push them as one dataset."""
    if not SMOKE:          # smoke deliberately uses disposable throwaway slugs
        assert_slugs_frozen()
    state = load_state()
    out_dir.mkdir(parents=True, exist_ok=True)

    mounts = hf_cache_mounts()
    mv = merged_version()
    if not mv:
        raise SystemExit(
            f"{MERGED_DATASET} has no version yet — the merged cache upload has "
            f"not finished. Generating now would push 28 tasks that mount "
            f"nothing, and preflight.sh only WARNS on a missing cache path, so "
            f"every run would burn 10 GPU-hours against an empty cache.")
    log(f"cache: {MERGED_DATASET}/versions/{mv} -> {MOUNT_ROOT} "
        f"({len(mounts)} part dirs, 1 dataset mount)")

    log(f"generating {len(cells())} tasks into {out_dir}")
    cmd = [sys.executable, str(BUILD_TASKS), "--all",
           "--agent", UPSTREAM_AGENT, "--agent-config", AGENT_CONFIG,
           "--num-hours", str(NUM_HOURS),
           "--agent-timeout-sec", str(AGENT_TIMEOUT_SEC),
           "--verifier-timeout-sec", str(VERIFIER_TIMEOUT_SEC),
           # --agent-base-image, NOT --docker-image. A prebuilt docker_image
           # makes harbor PULL and never build, so the task's own home/ never
           # reaches /home/ben and every agent sees the base image's baked
           # gsm8k evaluate.py + 1319-item test_data.json (D237). This flag
           # instead builds a thin per-task layer on the same base.
           "--agent-base-image", AGENT_IMAGE,
           "--verifier-base-image", VERIFIER_BASE_IMAGE,
           "--gpu-types", ",".join(GPU_TYPES),
           "--cpus", str(CPUS), "--memory-mb", str(MEMORY_MB),
           "--storage-mb", str(STORAGE_MB),
           # Bakes PTB_HF_CACHE_MOUNT into every task.toml. It MUST agree with
           # the mountPath sent at push time below; a mismatch is silent --
           # preflight.sh warns, skips, and the run proceeds cacheless.
           "--hf-cache", ",".join(mounts),
           "--output", str(out_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"build_tasks failed:\n{r.stdout}\n{r.stderr}")
    log(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "generated")

    log("verifying fidelity of the generated tasks")
    # NO WAIVERS. This previously passed `--waive storage_mb` for D224 (81920
    # against the original's 409600). D224 was retired on 2026-09-09 -- the
    # 80 GB ceiling filled the disk, put the node into a permanent EIO state
    # and cost 8 of 8 runs -- so STORAGE_MB now matches upstream and the check
    # passes on its own.
    #
    # Deliberately not left as a harmless no-op waiver: a waiver that covers
    # nothing still suppresses the class, so the day it regresses the gate
    # would stay green. The whole point of this failure was that we waived the
    # one check that mattered, 20 times a build, for a week.
    r = subprocess.run([verify_python(), str(VERIFY), "--tasks", str(out_dir)],
                       capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    log(f"  {tail[0]}")
    if r.returncode != 0:
        if SMOKE:
            # Expected, and not a licence to ignore it. A smoke build sets
            # num_hours=1 against the original's 10, so the duration checks
            # (num_hours, agent timeout, verifier timeout) MUST fail -- that
            # deviation is the entire point of a short run. Gating on them
            # would make --smoke impossible.
            #
            # The gate stays hard for the real --generate, which is the push
            # that actually has to be 1:1.
            log("  smoke: fidelity is ADVISORY here (duration is deliberately "
                f"{NUM_HOURS} h, not 10 h). The real --generate still gates.")
        else:
            # stdout AND stderr. Printing only stdout hid a ModuleNotFoundError
            # behind a blank line and made a crashing gate look like a failing
            # one -- the two need completely different fixes.
            raise SystemExit("verify_fidelity FAILED -- refusing to push.\n"
                             + (r.stdout or "") + "\n--- stderr ---\n"
                             + (r.stderr or "(none)"))

    # Rename generated dirs to our slugs and upload as one dataset.
    # WIPE the bundle first. `cp -a SRC DST` nests SRC *inside* DST when DST
    # already exists, so a leftover bundle from a previous --generate does not
    # get replaced -- it gets a subdirectory added and keeps its old files at
    # the root.
    #
    # That is exactly what shipped on 2026-09-05. /tmp/ptb-sweep/_bundle
    # survived from the 09-04 run, so every task dir ended up as:
    #
    #     <slug>/task.toml                                  <- STALE v1
    #     <slug>/kaggle-posttrainbench-<cell>/task.toml      <- the new v2
    #
    # Kaggle mounts the subPath root, so it read the v1 task.toml -- which
    # still carried `docker_image = ptb-verifier:v5`. harbor therefore took the
    # prebuilt path, never built the per-task verifier, and every one of the 28
    # runs executed gsm8k's baked /tests. D231 and the judge fix were both in
    # the archive, one directory below where anything would look at them.
    #
    # Confirmed by the Benchmarks team from the uploaded zip
    # (dataset version 19429269) and by `benchmark_id=gsm8k` in the verifier
    # stdout of run 1247812, a humaneval task.
    bundle = out_dir / "_bundle"
    subprocess.run(["rm", "-rf", str(bundle)], check=True)
    bundle.mkdir(parents=True)
    (bundle / "_placeholder").mkdir(exist_ok=True)
    (bundle / "_placeholder" / "k.txt").write_text("k\n")
    subprocess.run(["cp", "-a", str(out_dir / "agents"), str(bundle / "agents")])
    subprocess.run(["rm", "-rf", str(bundle / "agents" / "__pycache__")])
    for b, m, slug in cells():
        src = out_dir / f"kaggle-posttrainbench-{b}-{model_key(m)}"
        if not src.exists():
            raise SystemExit(f"expected generated dir missing: {src}")
        subprocess.run(["cp", "-a", str(src), str(bundle / slug)])
    # Assert the layout Kaggle will actually mount: task.toml at the root of
    # each subPath, and no stray build dir nested inside it. Cheap, and it is
    # the check that would have caught the 09-05 packaging failure before it
    # cost 28 runs.
    for _b, _m, slug in cells():
        d = bundle / slug
        if not (d / "task.toml").exists():
            raise SystemExit(f"packaging: {slug}/task.toml missing at subPath root")
        nested = [x.name for x in d.iterdir()
                  if x.is_dir() and x.name.startswith("kaggle-posttrainbench-")]
        if nested:
            raise SystemExit(
                f"packaging: {slug} contains a nested build dir {nested} -- the "
                f"bundle was not wiped and Kaggle would mount the STALE root "
                f"task.toml (this is the 2026-09-05 failure)")
        txt = (d / "task.toml").read_text()
        if "docker_image" in txt.split("[verifier.environment]")[-1].split("[")[0]:
            raise SystemExit(
                f"packaging: {slug} [verifier.environment] still sets "
                f"docker_image -- harbor would PULL instead of building the "
                f"per-task verifier (D231)")
    log(f"  packaging verified: {len(cells())} task dirs, no stale roots")

    (bundle / "dataset-metadata.json").write_text(json.dumps({
        "title": "PostTrainBench sweep definitions",
        "id": f"{OWNER}/{DEFS_DATASET}",
        "licenses": [{"name": "CC0-1.0"}]}) + "\n")

    exists = api(f"/api/v1/datasets/view/{OWNER}/{DEFS_DATASET}")
    curr_v = exists.get("currentVersionNumber")
    if curr_v and curr_v > (state.get("defs_version") or 0):
        log(f"  dataset version {curr_v} already created on Kaggle (ahead of state defs_version {state.get('defs_version')})")
        v = curr_v
    else:
        log("uploading definition dataset")
        verb = ["version", "-m", "sweep"] if "currentVersionNumber" in exists else ["create"]
        r = subprocess.run(["kaggle", "datasets", *verb, "-p", str(bundle),
                            "-r", "zip", "--dir-mode", "zip"],
                           capture_output=True, text=True)
        log(f"  {(r.stdout or r.stderr).strip().splitlines()[-1]}")
        time.sleep(30)
        v = api(f"/api/v1/datasets/view/{OWNER}/{DEFS_DATASET}")["currentVersionNumber"]
    state["defs_version"] = v
    log(f"  dataset version {v}")

    # PUSH SCOPE. Uploading a new dataset version bumps `v`, which makes the
    # `rec["defs"] == v` test below false for EVERY slug -- so an unscoped
    # --generate re-pushes all 20 even when one task changed. Each push fires a
    # 10-14 h auto-validation run, so that is 16 clusters burned to fix 4, and
    # it re-versions 16 healthy tasks for no reason.
    #
    # PTB_PUSH_ONLY names the benchmarks (or exact slugs) allowed to receive a
    # new task version. Everything else keeps its existing task_version, which
    # still pins the OLD defs dataset version -- Kaggle keeps every version, so
    # those tasks go on reading the definitions they were validated against and
    # are untouched by this upload. Their state records defs=<old>, which stays
    # true; no bookkeeping is faked to make the loop skip them.
    push_only = {s.strip() for s in os.environ.get("PTB_PUSH_ONLY", "").split(",")
                 if s.strip()}
    scoped = [(b, m, s) for b, m, s in cells()
              if not push_only or b in push_only or s in push_only]
    if push_only:
        log(f"  PUSH SCOPED to {sorted(push_only)}: "
            f"{len(scoped)} of {len(cells())} tasks will be pushed; the other "
            f"{len(cells()) - len(scoped)} keep their current task version")
        for _b, _m, s in scoped:
            log(f"    will push: {s}")
    for b, m, slug in scoped:
        rec = state["pushed"].get(slug)
        if isinstance(rec, dict) and rec.get("defs") == v:
            continue
        d = api("/api/v1/benchmarks/tasks/push", {
            "slug": slug,
            "definition": {"harborKaggleDatasets": {
                "definitionSource": {
                    "datasetVersionSlug": f"{OWNER}/{DEFS_DATASET}/versions/{v}",
                    "subPath": slug},
                # EXACTLY ONE mount. Two or more is not a degraded mode, it is
                # a hard failure of the whole session (see the HF CACHE block).
                "mounts": [{
                    "datasetVersionSlug": f"{MERGED_DATASET}/versions/{mv}",
                    "mountPath": MOUNT_ROOT}],
                # Pins our harness AND makes the task accept model versions
                # (CreateBenchmarkTaskVersionFromHarborKaggleDatasetsHandler:199).
                # ⚠️ THIS DOES NOT PIN THE HARNESS -- see D228.
                #
                # It reads like a pin, and Instructions.md:427 documents the
                # field as one. It is not. The value is a DISPATCHER: it reads
                # KAGGLE_AGENT_HARNESS at run time and becomes whichever
                # harness the scheduled agent slug named. The harness is chosen
                # per run by CANDIDATES, not here.
                #
                # Why: the env-var route is what keeps candidateType = agents
                # (so a full agent slug is schedulable), and it is the only way
                # to get our code running without registering a HarnessVersion.
                # Our code has to run because the original's non-interactive
                # clause is conditional on the harness (get_prompt.py:61).
                #
                # If runs start failing with "KAGGLE_AGENT_HARNESS is not a
                # harness Harbor knows" or the dispatcher gets an empty value,
                # the platform has stopped emitting it under an override.
                # Fix = register the HarnessVersions and drop this block
                # entirely (D228 has the runbook).
                "envVariables": [{"key": "KAGGLE_HARBOR_AGENT_OVERRIDE",
                                  "value": HARNESS}],
                # Explicit: server precedence is request ?? config.yaml ?? 1,
                # so writing it down makes the decision final.
                "nAttempts": 1}}})
        err = d.get("error") or d.get("_body", "")
        if err or "_http_error" in d:
            raise SystemExit(f"push rejected for {slug}: {err}")
        # Both numbers are needed: `defs` says whether this slug is already at
        # the current definition dataset, `task_version` is what gets pinned at
        # schedule time so every run in the sweep uses one definition.
        state["pushed"][slug] = {"defs": v,
                                 "task_version": d["slug"]["versionNumber"]}
        log(f"  pushed {slug} -> task v{d['slug']['versionNumber']}")
        save_state(state)
    save_state(state)
    log(f"all {len(scoped)} tasks pushed at defs v{v}"
        + (f" (scoped to {sorted(push_only)}; "
           f"{len(cells()) - len(scoped)} left at their previous version)"
           if push_only else ""))
    # Poll the validation runs rather than sleeping a flat COLLISION_GUARD_SEC.
    # The old code logged this wait without ever performing it -- chaining
    # --generate and --run inside one --smoke process is what exposed that.
    # Polling is both safer (it confirms the runs actually settled instead of
    # assuming a duration) and usually faster, since a validation run that
    # fast-fails clears its slug immediately.
    # Only the slugs this run actually pushed have a NEW validation run to wait
    # on. Passing all 20 under a scoped push would poll 16 tasks whose latest
    # validation run finished hours ago and is not a collision risk.
    wait_for_push_validation([slug for _b, _m, slug in scoped], state,
                             min_age_sec=COLLISION_GUARD_SEC // 2,
                             max_wait_sec=COLLISION_GUARD_SEC * 6)


# --------------------------------------------------------------------------
# scheduling
# --------------------------------------------------------------------------

def run_state(slug: str) -> list[dict]:
    q = urllib.parse.urlencode({"taskSlug.ownerSlug": OWNER,
                                "taskSlug.taskSlug": slug, "pageSize": "40"})
    d = api(f"/api/v1/benchmarks/tasks/runs/list?{q}")
    return d.get("runs", []) or []


def in_flight(state: dict) -> tuple[int, list[tuple[str, int]], float]:
    """How many of OUR runs are still going, plus units that need retrying.

    Returns (live_count, failed_units, oldest_live_run_age_sec). A run that
    ends in a non-terminal-success state is a hole in the leaderboard row, so
    it is handed back for requeueing inside its own pass rather than left
    recorded as 'done'. The age drives the adaptive poll interval.
    """
    live, failed = [], []
    unstarted = 0        # ours: scheduled (token minted) but not yet executing
    oldest_unstarted = 0.0
    # One runs/list call per SLUG per cycle, not per unit. A slug carries all 3
    # of its attempts, and without this the same endpoint was hit once for
    # every scheduled-but-unfinished unit -- triple the calls for no new data.
    by_slug: dict[str, dict] = {}
    oldest = 0.0
    for key, rec in list(state["scheduled"].items()):
        if key in state["done"]:
            continue
        if rec["slug"] not in by_slug:
            by_slug[rec["slug"]] = {str(r["id"]): r
                                    for r in run_state(rec["slug"])}
        r = by_slug[rec["slug"]].get(str(rec["run_id"]))
        if r is None:
            continue
        # A run is LIVE until the API gives it an end time.
        #
        # Do NOT go back to classifying by a whitelist of state strings. The
        # runs/list endpoint reports a healthy in-progress run as
        # state="unspecified" -- observed on smoke runs 1095811 and 1095815,
        # five minutes in, hasEndTime=false, hasResult=false. A whitelist of
        # ("running","queued","pending") reads that as "not running", falls
        # through to the terminal branch below, and hands a perfectly healthy
        # run to the canary's failure path. That HALTED the sweep on its first
        # run. hasEndTime is the field that actually distinguishes the two.
        ended = bool(r.get("hasEndTime")) or r["state"] in _TERMINAL_STATES
        if not ended:
            live.append(key)
            started = r.get("startTime", "")[:19]
            try:
                age = (datetime.now(timezone.utc)
                       - datetime.fromisoformat(started).replace(
                           tzinfo=timezone.utc)).total_seconds()
            except ValueError:
                age = 0.0
            oldest = max(oldest, age)
            # A token is minted at SCHEDULE time and dies 12 h later, so a
            # scheduled-but-not-started run is spending credential on nothing.
            # Count them: run_sweep() refuses to mint another while any is
            # outstanding. Measured distinction -- an executing run reports
            # state "running"; a queued one reports "unspecified".
            if r.get("state") != "running":
                unstarted += 1
                oldest_unstarted = max(oldest_unstarted, age)
            elif not state.get("dispatch_ok"):
                # First observed dispatch. Latched: the pool has proven it can
                # start our work, so the probe phase is over for good.
                state["dispatch_ok"] = rec["run_id"]
                log(f"  DISPATCH OK: run {rec['run_id']} is running — pool "
                    f"confirmed, opening all {CONCURRENCY} slots")

            # Queue watch. Scheduling mints the proxy token, so a run that has
            # not started is burning a 12 h TTL doing nothing. Warn early and
            # give up on it well before expiry rather than discovering the 401
            # ten hours later.
            queued = r.get("state") != "running"
            if queued and age > QUEUE_WARN_SEC:
                key_q = f"_queued_warned_{rec['run_id']}"
                if not state.get(key_q):
                    state[key_q] = True
                    log(f"  ⚠️  QUEUED {rec['run_id']} has not started after "
                        f"{age/3600:.1f} h (state={r.get('state')!r}). Its proxy "
                        f"token was minted at schedule time and dies at 12 h. "
                        f"The pool is oversubscribed — this run will 401 if it "
                        f"ever starts.")
            if queued and age > QUEUE_ABORT_SEC:
                save_state(state)
                raise SystemExit(
                    f"\nHALTED — run {rec['run_id']} has sat queued for "
                    f"{age/3600:.1f} h without starting.\n\n"
                    f"Its MODEL_PROXY_API_KEY was minted when it was scheduled "
                    f"and expires 12 h later, so it is already doomed or will "
                    f"be shortly. Scheduling more runs into a pool that cannot "
                    f"start them just mints more dead tokens — which is how the "
                    f"2026-09-04 sweep lost 14 runs.\n\n"
                    f"Wait for cluster capacity, then re-run --run. Finished "
                    f"runs are not repeated.")
            continue
        res = (r.get("result") or {}).get("numericResult")
        state["done"][key] = {"state": r["state"], "reward": res,
                              "run_id": rec["run_id"], "slug": rec["slug"]}
        log(f"  DONE {key}  {r['state']}  reward={res}")
        # Same lesson as above: do not require the literal string "completed".
        # A run that produced a numeric result reached the verifier, which
        # means it got its accelerator, whatever the state string says.
        if r["state"] == "completed" or res is not None:
            # Any success clears the breaker: what it is looking for is a
            # systemic fault, and a run that got all the way to a result
            # proves the pipeline still works end to end.
            state["consec_failures"] = 0
            state["recent_failures"] = []
            continue
        # Infrastructure fast-fail: died before it could have trained anything.
        # Retryable, but budgeted and individually logged -- see MAX_FAST_FAILS.
        ran_for = 0.0
        try:
            _st = datetime.fromisoformat(r.get("startTime", "")[:19])
            _en = datetime.fromisoformat((r.get("endTime") or "")[:19])
            ran_for = (_en - _st).total_seconds()
        except ValueError:
            ran_for = 0.0
        # A setup/build timeout is the SAME class of fault as a fast-fail --
        # infrastructure, nothing trained, retryable -- but it does not look
        # like one, because it dies at 10-12 min rather than under 120 s.
        #
        # The image build has a 6 MINUTE deadline, and the harness npm install
        # takes ~5.5 min (operator, 2026-09-11). That is a coin flip, not a
        # broken task: the same slug that fails one attempt succeeds the next.
        # Wave 1 lost 11 of 20 this way at 10-12 min, and because every one of
        # them sailed past FAST_FAIL_SEC they were counted as systemic and
        # tripped the breaker after three -- exactly backwards for a fault
        # whose documented remedy is "just try uploading it again".
        #
        # Matched on the error TEXT rather than on a longer duration window,
        # so a genuine systemic failure that happens to die at 11 min still
        # trips the breaker. These share the fast-fail budget, so an infra
        # fault that stops being occasional still halts the sweep.
        err_txt = (r.get("errorMessage") or "")
        setup_timeout = ("AgentSetupTimeoutError" in err_txt
                         or "NetworkConnectionError" in err_txt
                         or "HealthcheckError" in err_txt
                         or "out of disk space" in err_txt)
        if 0 < ran_for < FAST_FAIL_SEC or setup_timeout:
            state.setdefault("fast_fails", []).append({
                "run_id": rec["run_id"], "slug": rec["slug"],
                "candidate": rec["candidate"], "seconds": round(ran_for),
                "state": r["state"],
                "error": (r.get("errorMessage") or "")[:160]})
            n_ff = len(state["fast_fails"])
            log(f"  INFRA {'SETUP-TIMEOUT' if setup_timeout else 'FAST-FAIL'} "
                f"{rec['run_id']} died in {ran_for:.0f}s "
                f"({n_ff}/{MAX_FAST_FAILS} budget) — retrying, not counted as "
                f"a systemic failure. Pull artifacts to confirm the cause: "
                f"runs/{rec['run_id']}/output")
            if n_ff >= MAX_FAST_FAILS:
                save_state(state)
                rows = "\n".join(
                    f"  {f['run_id']}  {f['seconds']:>4}s  {f['slug']}"
                    for f in state["fast_fails"])
                raise SystemExit(
                    f"\nHALTED — {n_ff} infrastructure fast-failures, budget "
                    f"exhausted.\n\n{rows}\n\n"
                    f"These are runs that died in under {FAST_FAIL_SEC}s, i.e. "
                    f"before training could start. They were retried rather "
                    f"than treated as real failures, on the measured assumption "
                    f"that the underlying mount fault is transient (~7%).\n"
                    f"At {n_ff} the assumption no longer holds.\n\n"
                    f"Confirm the cause before resuming:\n"
                    f"  curl -sSL -H \"Authorization: Bearer "
                    f"$(kaggle auth print-access-token)\" \\\n"
                    f"    https://www.kaggle.com/api/v1/benchmarks/tasks/runs/"
                    f"{state['fast_fails'][-1]['run_id']}/output -o out.zip\n"
                    f"  # then read jobs/*/*/exception.txt\n\n"
                    f"State is saved; finished runs are not repeated.")
            # Deliberately NOT counted against consec_failures: a transient
            # platform fault is not the systemic breakage that breaker exists
            # for, and letting three unlucky ones stop a 3-day sweep would be
            # the wrong trade.
            save_state(state)
            n = state.setdefault("retries", {}).get(key, 0)
            state["retries"][key] = n + 1
            del state["scheduled"][key]
            del state["done"][key]
            failed.append((rec["slug"], rec["attempt"], rec["candidate"]))
            continue

        # Circuit breaker. Record the failure BEFORE the retry logic, so a unit
        # that is about to be requeued still counts against the run of bad
        # outcomes -- otherwise a systemic fault hides behind its own retries.
        state["consec_failures"] = state.get("consec_failures", 0) + 1
        state.setdefault("recent_failures", []).append({
            "run_id": rec["run_id"], "slug": rec["slug"],
            "candidate": rec["candidate"], "state": r["state"],
            "error": (r.get("errorMessage") or "")[:200]})
        state["recent_failures"] = state["recent_failures"][-MAX_CONSECUTIVE_FAILURES:]
        if state["consec_failures"] >= MAX_CONSECUTIVE_FAILURES:
            save_state(state)
            rows = "\n".join(
                f"  run {f['run_id']:>8}  {f['state']:<10} {f['candidate'][:44]}\n"
                f"      {f['error'] or '(no error message)'}"
                for f in state["recent_failures"])
            raise SystemExit(
                f"\nHALTED — {state['consec_failures']} consecutive failures, "
                f"no success in between.\n\n{rows}\n\n"
                f"{len(state['done'])} run(s) finished before this. The sweep "
                f"stops here rather than feeding more 10 h runs into what looks "
                f"like a systemic fault.\n\n"
                f"Inspect one with:\n"
                f"  curl -H \"Authorization: Bearer $(kaggle auth print-access-token)\" \\\n"
                f"    https://www.kaggle.com/api/v1/benchmarks/tasks/runs/"
                f"{state['recent_failures'][-1]['run_id']}/output -o out.zip\n"
                f"  # then read jobs/*/*/exception.txt and trial.log\n\n"
                f"State is saved. Fix the cause, then re-run --run: finished "
                f"runs are not repeated and nothing is double-scheduled.")
        n = state.setdefault("retries", {}).get(key, 0)
        if n >= MAX_RETRIES:
            log(f"  GIVING UP on {key} after {n} retries — cell stays empty")
            continue
        state["retries"][key] = n + 1
        # Clear both records so schedule_one() will act on the key again.
        del state["scheduled"][key]
        del state["done"][key]
        failed.append((rec["slug"], rec["attempt"], rec["candidate"]))
        log(f"  RETRY {key} ({n + 1}/{MAX_RETRIES}) — requeued inside pass "
            f"{rec['attempt']}")
    return len(live), failed, oldest, unstarted, oldest_unstarted


def _run_age_sec(r: dict) -> float:
    try:
        started = datetime.fromisoformat(r.get("startTime", "")[:19])
        return (datetime.now(timezone.utc)
                - started.replace(tzinfo=timezone.utc)).total_seconds()
    except ValueError:
        return 0.0


def wait_for_push_validation(slugs: list[str], state: dict,
                             min_age_sec: int = 120,
                             max_wait_sec: int = 1800) -> None:
    """Poll until each slug's auto-created validation run is safely underway.

    Replaces a blind `time.sleep(COLLISION_GUARD_SEC)`. A push auto-creates a
    validation run against a default candidate, and scheduling on top of one
    that is still starting is what produced run 868965: two sessions on a
    single task version 38 s apart, the second left with an empty job.log and
    a trial that never ran. Silent, and indistinguishable from a scheduled run
    until you open the artifacts.

    Deliberately NOT "wait until it finishes". Historically the validation run
    errored within minutes, but it runs the real task definition -- with the
    D229 healthcheck fixed there is nothing left to stop it running the full
    10 h. Blocking on completion would stall the sweep for half a day.

    So a slug clears when its validation run has EITHER ended (the historical
    fast-fail path, and then there is nothing to collide with) OR reached
    min_age_sec (it has claimed its session; 868965 happened at 38 s, so 120 s
    is ~3x that). Whichever comes first, which makes the common case fast
    instead of a flat 5-minute wait.

    Runs already recorded in state["scheduled"] are ours and are ignored --
    only the untracked ones are validation runs.
    """
    ours = {rec["run_id"] for rec in state.get("scheduled", {}).values()}
    deadline = time.time() + max_wait_sec
    pending = list(slugs)
    log(f"waiting for push-validation runs on {len(pending)} task(s) to settle "
        f"(clear at end-of-run or {min_age_sec}s age, cap {max_wait_sec}s)")
    while pending and time.time() < deadline:
        still = []
        for slug in pending:
            others = [r for r in run_state(slug) if r.get("id") not in ours]
            if not others:
                continue                      # none appeared; nothing to wait on
            r = max(others, key=lambda x: x.get("id", 0))
            if r.get("hasEndTime") or _run_age_sec(r) >= min_age_sec:
                continue
            still.append(slug)
        # Reassign BEFORE the exit test. Breaking out while `pending` still
        # held the previous iteration's list made the success path fall into
        # the "unsettled" warning below.
        pending = still
        if not pending:
            break
        log(f"  {len(pending)} still settling; next check in 15s")
        time.sleep(15)
    if pending:
        log(f"  WARN: {len(pending)} slug(s) still unsettled after "
            f"{max_wait_sec}s; proceeding anyway: {pending[:5]}")
    else:
        log("  all validation runs settled — safe to schedule")


def verify_scheduled_agent(slug: str, run_id: int, candidate: str,
                           state: dict) -> None:
    """Read the agent slug back off the run the server actually created.

    Necessary because /benchmarks/tasks/schedule ignores unknown fields
    without complaint -- probed 2026-09-02, a request carrying a field named
    `totallyMadeUpField` returned a byte-identical response. So the request
    cannot tell us the candidate was applied.

    The agent slug IS the candidate: harness + model + reasoning effort in one
    string. Reading it back proves all three, which is why this replaced the
    older check that only looked at the reasoning config.

    Checked once per candidate. After that the request shape is proven and
    re-checking every run is just extra API calls.
    """
    seen = state.setdefault("agent_verified", {})
    if candidate in seen:
        return
    for _ in range(10):                        # the record can lag the schedule
        for r in run_state(slug):
            if str(r["id"]) != str(run_id):
                continue
            got = r.get("agentSlug") or ""
            if not got:
                break                          # not populated yet; wait
            if got == candidate:
                seen[candidate] = run_id
                log(f"  AGENT VERIFIED: run {run_id} is '{got}'")
                save_state(state)
                return
            save_state(state)
            raise SystemExit(
                f"\nHALTED — wrong candidate on the first run of this slug.\n\n"
                f"  asked for : {candidate}\n"
                f"  got       : {got or '(none)'}\n"
                f"  run       : {run_id} ({slug})\n\n"
                f"The schedule API ignores unknown fields silently, so this "
                f"means the request shape is wrong and every run would have\n"
                f"used the wrong harness/model/effort. Fix schedule_one() "
                f"before continuing; state is saved and nothing is "
                f"double-scheduled.")
        time.sleep(15)
    log(f"  WARNING: could not read agentSlug back for run {run_id} — "
        f"candidate UNVERIFIED")


def schedule_one(slug: str, attempt: int, candidate: str, state: dict) -> bool:
    key = f"{slug}#a{attempt}#{candidate}"
    if key in state["scheduled"]:
        return False
    # versionNumber is pinned deliberately. Without it the scheduler takes
    # whatever version is current, so a re-push mid-sweep would silently move
    # later runs onto a different task definition than earlier ones.
    body = {"taskSlugs": [{"taskSlug": slug,
                           "versionNumber": state["pushed"][slug]["task_version"]}],
            "agentSlugs": [candidate]}
    d = api("/api/v1/benchmarks/tasks/schedule", body)
    if "_http_error" in d:
        log(f"  SCHEDULE FAILED {key}: HTTP {d['_http_error']} {d['_body'][:200]}")
        # 401/403 mean this account may not schedule this agent slug. That is
        # a permanent fact, not a transient one, so the unit must be RETIRED
        # rather than requeued -- requeue() would put it straight back and the
        # runner would spin on the same rejection forever, burning the API
        # quota and never draining the queue. Measured: scheduling
        # claude-code-2.1.223-claude-fable-5-max-reasoning returns 403 "One or
        # more benchmark task run identifiers are not authorized."
        if d["_http_error"] in (401, 403):
            state["unauthorized"] = sorted(
                set(state.get("unauthorized", [])) | {candidate})
            log(f"  RETIRED {candidate}: not authorized for this account, "
                f"will not be retried")
            save_state(state)
        return False
    for res in d.get("results", []):
        if res.get("runScheduled"):
            state["scheduled"][key] = {"slug": slug, "attempt": attempt,
                                       "candidate": candidate,
                                       "run_id": res["runId"],
                                       "at": datetime.now(timezone.utc).isoformat()}
            log(f"  scheduled {key} -> run {res['runId']}")
            save_state(state)
            verify_scheduled_agent(slug, res["runId"], candidate, state)
            return True
        log(f"  SKIPPED {key}: {res.get('runSkippedReason')}")
    return False


_WAVE_INDEX: dict[tuple[str, int, str], int] = {}


def wave_of(unit: tuple[str, int, str]) -> tuple[int, str]:
    """(wave_number, label) -- the WAVE a unit belongs to.

    A wave is simply the next CONCURRENCY units in units() order: units 0-19
    are wave 1, 20-39 wave 2, and so on. Every wave is exactly 20 runs, so
    every GPU of the grant is busy for the whole wave.

    2026-09-11, twice. It was (attempt, benchmark) -- one eval across 4 models
    x 3 candidates. That is 12 runs, and 12 does not divide into 20, so eight
    GPUs idled for the entire sweep: 5 waves x ~12 h = ~60 h at 60% use. The
    first fix was (attempt, candidate): 20 runs exactly, but it bought the
    utilisation by making every wave one candidate's row, which delays any
    cross-candidate read until wave 2 lands.

    Chunking by POSITION instead buys the same 100% without that cost, because
    a wave no longer has to be a semantically uniform group to be worth 20 --
    it just has to be twenty of something. units() is cell-major with the
    candidate loop innermost, so consecutive units walk (cell, candidate)
    pairs and a 20-slice lands ~7/7/6 across the three candidates. Wave 1 is
    therefore a fault detector for ALL THREE harnesses at once: a broken auth
    or an empty trace shows up in hour 12, not hour 24. It also equalises pool
    conditions across candidates, so cross-model comparison is not confounded
    by time-of-day capacity.

    The barrier is unaffected and stays hard. It fires on `live == 0`, which
    still happens between waves because scheduling is restricted to the
    current wave in run_sweep() -- see the cur_wave filter there.

    Position is taken from units(), not from `work`, so a unit keeps its wave
    across restarts and across requeue(): `work` shrinks as units are
    scheduled and as failures are re-appended, and chunking a shrinking list
    would silently re-cut the boundaries mid-sweep and let two waves overlap.
    The index is rebuilt whenever an unknown unit appears, which is what makes
    this correct under diag_sweep's `R.units = diag_units` rebinding.
    """
    if unit not in _WAVE_INDEX:
        _WAVE_INDEX.clear()
        _WAVE_INDEX.update({u: i for i, u in enumerate(units())})
    i = _WAVE_INDEX[unit]
    n = i // CONCURRENCY + 1
    return (n, f"wave {n}")


def _benchmark_of_slug(slug: str) -> str:
    """The benchmark half of a task slug, e.g. ptb-bfcl-qwen3-4b-base -> bfcl.

    Derived by matching against the known benchmark names rather than by
    splitting on '-': both benchmark names and model keys contain hyphens, so
    no fixed field count parses `ptb-gpqamain-gemma-3-4b-pt` correctly.
    Matching is anchored on the TASK_PREFIX so a benchmark name that happens to
    appear inside a model key cannot match.
    """
    for b in BENCHMARKS:
        if slug.startswith(f"{TASK_PREFIX}-{b}-"):
            return b
    raise SystemExit(f"cannot determine benchmark for slug {slug!r} -- "
                     "wave grouping would be wrong, refusing to schedule")


def run_sweep(target_wave: int | None = None, makeup: bool = False) -> int:
    state = load_state()
    if not state.get("defs_version"):
        raise SystemExit("no tasks pushed yet -- run --generate first")

    if makeup:
        work = [u for u in units()
                if f"{u[0]}#a{u[1]}#{u[2]}" in state.get("deferred_units", [])
                and f"{u[0]}#a{u[1]}#{u[2]}" not in state.get("done", {})]
        log(f"MAKEUP SWEEP: {len(work)} deferred units queued")
        WAVE_GO.touch(exist_ok=True)
    elif target_wave is not None:
        work = [u for u in units()
                if wave_of(u)[0] == target_wave
                and f"{u[0]}#a{u[1]}#{u[2]}" not in state["scheduled"]
                and f"{u[0]}#a{u[1]}#{u[2]}" not in state.get("done", {})
                and f"{u[0]}#a{u[1]}#{u[2]}" not in state.get("deferred_units", [])]
        log(f"WAVE {target_wave} SWEEP: {len(work)} units queued")
        # Explicit target wave acts as wave barrier release
        WAVE_GO.touch(exist_ok=True)
    else:
        work = [u for u in units()
                if f"{u[0]}#a{u[1]}#{u[2]}" not in state["scheduled"]
                and f"{u[0]}#a{u[1]}#{u[2]}" not in state.get("deferred_units", [])
                and f"{u[0]}#a{u[1]}#{u[2]}" not in state.get("done", {})]
    log(f"sweep start: {len(work)} of {len(units())} units remaining, "
        f"concurrency {CONCURRENCY}")

    while True:
        live, failed, oldest, unstarted, oldest_unstarted = in_flight(state)
        for unit in failed:
            # A held benchmark must not come back through the retry path. Its
            # units were filtered out of `work` at startup, but the ones
            # already in flight when the hold went on are still tracked in
            # state["scheduled"], and a failure there would requeue a unit the
            # hold exists to prevent -- re-scheduling the exact definition we
            # know is broken. Retire it instead; the hold is lifted by
            # re-pushing, not by retrying.
            if any(f"-{b}-" in unit[0] for b in HOLD_BENCHMARKS):
                log(f"  HELD, not requeued: {unit[0]}#a{unit[1]} "
                    f"(benchmark on hold pending re-push)")
                continue
            key = f"{unit[0]}#a{unit[1]}#{unit[2]}"
            if DEFER_WAVE_FAILURES and not makeup:
                if key not in state.setdefault("deferred_units", []):
                    state["deferred_units"].append(key)
                log(f"  DEFERRED to makeup wave: {key}")
                continue
            requeue(work, unit)
        save_state(state)
        done = len(state["done"])
        if not work and live == 0:
            log(f"SWEEP COMPLETE: {done}/{len(units())} runs finished")
            return 0
        # THE SCHEDULING RULE, and the only one that matters:
        #
        #     a token is minted when a run is SCHEDULED and dies 12 h later,
        #     so never schedule a run the pool cannot start right now.
        #
        # There is no API for free capacity, but there is a direct observable:
        # a run we already scheduled that has not reached state "running" is
        # proof the pool is not dispatching. While any such run exists we mint
        # nothing further -- one wasted token, not twenty.
        #
        # This replaces the old GPU canary, which asked the wrong question. It
        # gated on "has a run survived 20 min", which a QUEUED run satisfies;
        # it certified run 1120464 on 2026-09-04 and opened all 14 slots into a
        # pool with zero free clusters, and every token died. Its stated
        # purpose -- catching a silent CPU-pool fallback -- is now redundant
        # anyway: D229 put check_cuda.py in the healthcheck, so a non-H100
        # allocation self-reports in ~86 s.
        #
        # It also covers the case that stopped the 06:30 launch: `--generate`
        # fires one auto-validation run per task, and since D229 those execute
        # the real 10 h task instead of dying in 90 s. 28 pushes filled a
        # 20-cluster pool, so nothing of ours could start. Under this rule the
        # sweep simply waits instead of minting 20 doomed tokens.
        # PROBE ONCE, THEN COMMIT.
        #
        # The only thing minting tokens slowly buys is finding out whether the
        # pool dispatches -- and that is answered by the FIRST run. Once one
        # run reaches `running`, runs 2..N tell us nothing new, so re-probing
        # per slot just burns wall clock (~17 min to reach 20 at one per
        # cycle, for information already in hand).
        #
        # So: mint one, wait for it to start, then open to full width. The
        # exposure stays one token, not CONCURRENCY -- which is the thing that
        # actually cost us. On 2026-09-04 fourteen tokens were minted blind
        # into a pool with zero free clusters and all fourteen expired unused;
        # on 09-05 the same would have happened behind 27 validation runs.
        #
        # After that, `unstarted` is time-boxed rather than absolute. A run
        # that was just scheduled is briefly unstarted BY DEFINITION, so
        # treating any unstarted run as "pool is full" would throttle every
        # normal refill. Only a run still not dispatched after
        # DISPATCH_STALL_SEC means the pool has genuinely filled -- other
        # teams' jobs arriving mid-sweep -- and then we stop minting.
        # ---- WAVE BARRIER -------------------------------------------------
        # Added 2026-09-10 at the operator's request. The scheduler is
        # capacity-driven: it refills a GPU the instant one frees, so runs from
        # the next wave start while the previous wave is still finishing and
        # there is no moment at which a wave is "done" to be inspected. That is
        # the right behaviour for throughput and the wrong one for a sweep
        # being reviewed between waves.
        #
        # The barrier makes the wave boundary real: when the pool has fully
        # drained and work remains, STOP and wait for an explicit release. Not
        # a timer -- nothing starts without a human acting.
        #
        #     touch /tmp/ptb-wave-go      # release exactly one wave
        #
        # The file is consumed (deleted) on release, so one touch buys one wave
        # and a forgotten file cannot silently free the rest of the sweep.
        # Checked only at live == 0, so it never interferes with refills WITHIN
        # a wave -- a wave still runs at full 20-wide concurrency.
        # `done` guards the START: at sweep start live is 0 and work is full,
        # which is not a wave boundary. Invoking --run IS the authorisation for
        # the first wave; the barrier governs every wave after it.
        if not makeup and live == 0 and work and done > 0:
            barrier_file = None
            if WAVE_GO.exists():
                barrier_file = WAVE_GO
            elif Path("/tmp/ptb-wave-go").exists():
                barrier_file = Path("/tmp/ptb-wave-go")
            if barrier_file is not None:
                try:
                    barrier_file.unlink()
                except Exception:
                    pass
                state.pop("_barrier_logged", None)
                save_state(state)
                log(f"WAVE BARRIER released ({len(work)} units remaining). "
                    f"Starting the next wave.")
            else:
                if not state.get("_barrier_logged"):
                    log(f"WAVE BARRIER: pool drained, {done} runs done, "
                        f"{len(work)} units remaining. Holding. "
                        f"Release the next wave with:  touch {WAVE_GO}")
                    state["_barrier_logged"] = True
                    save_state(state)
                time.sleep(POLL_IDLE_SEC)
                continue

        if unstarted and oldest_unstarted > DISPATCH_STALL_SEC:
            if not state.get("_wait_logged"):
                log(f"WAITING: a scheduled run has not reached 'running' after "
                    f"{oldest_unstarted/60:.1f} min. The pool is not "
                    f"dispatching; minting no further tokens until it does.")
                state["_wait_logged"] = True
            free = 0
        elif not state.get("dispatch_ok"):
            # Probe phase: exactly one token until we have seen one run start.
            state.pop("_wait_logged", None)
            free = 0 if unstarted else min(1, CONCURRENCY - live)
        else:
            state.pop("_wait_logged", None)
            free = CONCURRENCY - live
        # A wave is one model, and NOTHING from the next model starts until this
        # one is fully done. Without this the capacity-driven refill pulls the
        # next model's runs into the gap as soon as a slot frees, and the wave
        # boundary the barrier depends on never occurs -- `live` never reaches
        # 0 until the whole sweep is over. Restricting `work` to the current
        # wave is what turns the barrier from decoration into a real gate.
        # The MINIMUM pending wave, not wave_of(work[0]).
        #
        # requeue() puts a failed unit at the end of its own PASS, which at
        # ATTEMPTS=1 is the end of the whole list -- behind every later wave.
        # Reading the head would then call wave 2 current while a wave-1 unit
        # was still queued, starting wave 2 early and defeating the barrier.
        # Observed 2026-09-11 on run 1725724, which errored 6 min in and was
        # requeued. Harmless before waves became position-based: wave_of() was
        # (attempt, candidate) and requeue() preserves attempt, so the head was
        # in the right group by construction.
        if makeup:
            cur_wave = (99, "makeup wave")
            wave_work = work
        else:
            cur_wave = min((wave_of(u) for u in work), default=None)
            wave_work = [u for u in work if wave_of(u) == cur_wave]
        if free > 0 and wave_work:
            pass_no = work[0][1]
            left_in_pass = sum(1 for _s, a, _c in work if a == pass_no)
            log(f"{live} in flight, {free} free, {len(work)} queued "
                f"({len(wave_work)} left in {cur_wave[1]}, "
                f"{left_in_pass} in pass {pass_no}), {done} done")
            # `tried` bounds the cycle to one attempt per unit. requeue() puts
            # a failed unit back into `work` inside its own pass, so without
            # this a pass with fewer units left than free slots would pop the
            # same unit repeatedly and hammer the scheduler -- 14 slots and 2
            # units left meant ~7 attempts each per cycle.
            tried: set[tuple[str, int]] = set()
            for _ in range(min(free, len(wave_work))):
                nxt = next((u for u in work
                            if u not in tried and (makeup or wave_of(u) == cur_wave)), None)
                if nxt is None:
                    break                       # everything left already tried
                work.remove(nxt)
                tried.add(nxt)
                slug, attempt, candidate = nxt
                if not schedule_one(slug, attempt, candidate, state):
                    # Do not requeue a candidate the API has permanently
                    # refused -- see the 401/403 branch in schedule_one().
                    if candidate in state.get("unauthorized", []):
                        work[:] = [u for u in work if u[2] != candidate]
                    else:
                        requeue(work, nxt)
                time.sleep(SCHEDULE_SPACING_SEC)
        else:
            log(f"{live} in flight, {len(work)} queued, {done} done — waiting")

        # Adaptive poll. A run lasts ~12 h, so polling every minute for its
        # first nine hours is thousands of API calls that cannot learn
        # anything. But once a run is near its budget, a slow poll leaves a
        # GPU idle for up to POLL_IDLE_SEC -- across 84 runs that is hours of
        # wasted grant. So: poll slowly while everything is mid-run, and
        # switch to fast polling once any run is old enough to finish soon, or
        # whenever there is free capacity waiting on work.
        near_done = oldest > FAST_POLL_AFTER_SEC
        # `free` is what the gate above actually decided we may mint; the old
        # `limit` variable went away with the canary. Left behind, it crashed
        # run_sweep() on its first poll after scheduling.
        idle_capacity = free > 0 and bool(work)
        # While the dispatch gate is shut, every slot but one is idle and the
        # ONLY thing we are waiting for is the canary flipping to `running` --
        # which takes a minute or two. Polling on the idle schedule here would
        # hold 19 clusters for five minutes to learn something that is already
        # true, reintroducing exactly the waste that shrinking the gate
        # removed. `idle_capacity` does not catch this case: with limit
        # clamped to 1 and one run live there is no free slot, so it reads
        # false.
        # Waiting on the pool to dispatch: poll hard, it flips in a minute
        # or two and every idle cluster is wasted grant until it does.
        gate_shut = unstarted > 0 and bool(work)
        time.sleep(POLL_GATE_SEC if gate_shut
                   else POLL_BUSY_SEC if (near_done or idle_capacity)
                   else POLL_IDLE_SEC)


def status() -> int:
    state = load_state()
    done, sched = state["done"], state["scheduled"]
    print(f"pushed  {len(state['pushed'])}/{len(cells())} tasks "
          f"(defs v{state.get('defs_version')})")
    print(f"runs    {len(sched)} scheduled, {len(done)} finished, "
          f"{len(units()) - len(sched)} not yet started")

    # Per-pass, because pass 1 is the deliverable: it is the complete
    # leaderboard row. Passes 2 and 3 only add the stddev on top of it.
    n_cells = len(cells())
    print()
    for a in range(1, ATTEMPTS + 1):
        keys = [k for k in done if k.endswith(f"#a{a}")]
        ok = [k for k in keys if done[k]["state"] == "completed"]
        bar = "#" * (20 * len(ok) // n_cells)
        tag = "  <- LEADERBOARD ROW" if a == 1 else ""
        print(f"pass {a}  [{bar:<20}] {len(ok):>2}/{n_cells} complete"
              f"{f', {len(keys) - len(ok)} failed' if len(keys) > len(ok) else ''}"
              f"{tag}")
    if state.get("retries"):
        print(f"\nretried: {len(state['retries'])} unit(s)")
    if state.get("deferred_units"):
        print(f"\ndeferred to makeup wave: {len(state['deferred_units'])} unit(s)")
        for du in state["deferred_units"]:
            print(f"  {du}")
    if done:
        print(f"\n{'cell':<44} {'att':>3} {'state':<10} reward")
        for key, rec in sorted(done.items()):
            slug, a = key.rsplit("#a", 1)
            print(f"  {slug:<42} {a:>3} {rec['state']:<10} {rec['reward']}")
        rewards = [r["reward"] for r in done.values()
                   if isinstance(r.get("reward"), (int, float))]
        if rewards:
            print(f"\n{len(rewards)} scored, mean {sum(rewards)/len(rewards):.4f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--generate", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--makeup", action="store_true",
                   help="run deferred units from earlier waves as a makeup pass")
    p.add_argument("--wave", type=int, default=None,
                   help="target a specific wave (e.g. 2 for units 20..39)")
    g.add_argument("--smoke", action="store_true",
                   help="short disposable end-to-end check: one cell, one "
                        "attempt, throwaway slugs, both candidates. Touches "
                        "no ptb-* task and no sweep state.")
    p.add_argument("--hours", type=int, default=1,
                   help="--smoke duration (default 1)")
    p.add_argument("--agents", type=str, default="",
                   help="--smoke only: comma/newline-separated agent slugs to "
                        "run instead of CANDIDATES. Leaves the sweep's own "
                        "candidate list untouched.")
    p.add_argument("--out", type=Path, default=Path("/tmp/ptb-sweep"))
    a = p.parse_args()
    if a.smoke:
        picked = [s.strip() for s in a.agents.replace("\n", ",").split(",")
                  if s.strip()]
        tag = enable_smoke(a.hours, picked)
        log(f"SMOKE {tag}: {a.hours} h, 1 cell x {len(CANDIDATES)} candidates, "
            f"disposable slugs, state -> {STATE_PATH.name}")
        # Resume rather than re-push. A second --smoke after a crash should
        # rejoin the runs already in flight; calling generate() again would
        # mint a new defs version and a new task version underneath them.
        st = load_state()
        if st.get("defs_version") and len(st.get("pushed", {})) >= len(cells()):
            log(f"smoke: {len(st['pushed'])} task(s) already pushed at defs "
                f"v{st['defs_version']} — resuming the run phase")
        else:
            generate(a.out / "smoke")
        return run_sweep()
    if a.preflight:
        return preflight()
    if a.generate:
        generate(a.out)
        return 0
    if a.makeup:
        return run_sweep(makeup=True)
    if a.run:
        return run_sweep(target_wave=a.wave)
    return status()


if __name__ == "__main__":
    sys.exit(main())
