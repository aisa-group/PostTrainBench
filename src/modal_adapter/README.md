# Running PostTrainBench on Modal

This adapter runs the full benchmark pipeline on [Modal](https://modal.com)
(serverless cloud GPUs) instead of an HTCondor cluster. It is purely
additive: the cluster path (`src/run_task.sh`, `src/commit_utils/`) is
untouched, and each pipeline phase here is a direct port of the
corresponding section of `run_task.sh`.

Each run (agent x model x task) is a chain of three Modal Functions:

```
submit.py (your laptop; exits immediately)
  └─ run_agent   standard image,   1..8x H100   the agent post-trains for num_hours
       └─ run_judge   standard image,  CPU only    codex contamination judge, fresh container
            └─ run_eval    vllm_debug image, 1x H100   evaluate.py max-tokens retry ladder
```

All phases write into the `ptb-results` volume using the exact `EVAL_DIR`
layout of the cluster, so downloaded results feed `scripts/collect.py` and
friends unchanged.

## One-time setup

1. Install and authenticate the Modal CLI:

   ```bash
   pip install modal        # or: uv tool install modal
   modal setup
   ```

2. Create the secret with your API keys (unused ones can be empty; the last
   two are only for subscription-based agents and replace the `auth.json` /
   `oauth_token` files that `run_task.sh` copies from `agents/<agent>/`):

   ```bash
   modal secret create posttrainbench-keys \
       OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... \
       OPENCODE_API_KEY="" DASHSCOPE_API_KEY="" ZAI_API_KEY="" \
       CODEX_AUTH_JSON="" CLAUDE_OAUTH_TOKEN=""
   ```

3. Deploy the app (builds both container images on first run; expect a
   while for the vLLM/flash-attn layers):

   ```bash
   modal deploy src/modal_adapter/app.py
   ```

4. Sanity-check without benchmark spend:

   ```bash
   modal run src/modal_adapter/app.py::smoke_check       # CPU: images, CLIs, volumes, secret
   modal run src/modal_adapter/app.py::gpu_smoke_check   # ~1 min on an H100 (~$0.10)
   ```

5. Seed the HuggingFace cache volume. The download runs *inside* Modal
   (nothing is uploaded from your machine). For a first end-to-end test,
   seed the minimal subset; the full cache covers all of
   `containers/download_hf_cache/resources.json` and takes hours:

   ```bash
   modal run src/modal_adapter/app.py::seed_smoke            # 1 model + 1 dataset, minutes
   modal run --detach src/modal_adapter/app.py::seed_full    # everything (run detached)
   ```

## Submitting runs

```bash
# Cheap end-to-end test (~$10-15): 1 hour of claude on gsm8k
python src/modal_adapter/submit.py \
    --agent claude --agent-config claude-opus-4-5 \
    --eval gsm8k --model Qwen/Qwen3-1.7B-Base --num-hours 1

# A real cell (~$55-60): 10 hours, then judge + eval
python src/modal_adapter/submit.py \
    --agent claude --agent-config claude-opus-4-5 \
    --eval healthbench --model Qwen/Qwen3-4B-Base --num-hours 10

# Sweeps: --eval and --model are repeatable
python src/modal_adapter/submit.py \
    --agent codex --agent-config gpt-5.3-codex \
    --eval gsm8k --eval humaneval --eval gpqamain \
    --model Qwen/Qwen3-4B-Base --model HuggingFaceTB/SmolLM3-3B-Base \
    --num-hours 10
```

Spawned runs keep going after your laptop disconnects. Each submission is
recorded in `modal_runs.jsonl` at the repo root.

## Monitoring and collecting results

```bash
python src/modal_adapter/status.py       # per-run phase progress
modal app logs posttrainbench            # live logs (also on the web dashboard)

modal volume get ptb-results / ./results_modal
python scripts/collect.py --data-dir ./results_modal   # existing tooling, unchanged
```

## Costs (Modal list prices, mid-2026)

| Item | Cost |
|---|---|
| H100 | ~$3.95/h (billed per second) |
| Standard run (10h agent + judge + eval) | ~$55–60 |
| Full sweep, 7 tasks x 4 models, one agent config | ~$1.6k |
| 1h smoke run | ~$10–15 |
| Storage (cache + results volumes) | per byte-day; small next to compute |

Concurrency: Modal's Starter plan allows 10 concurrent GPUs (excess runs
queue automatically); the Team plan allows 50.

## Differences from the cluster setup

1. **HF cache overlay**: fuse-overlayfs cannot mount inside Modal's gVisor
   runtime, so the read-only cache volume is exposed through a symlink farm
   on container SSD (`hf_cache_lib.sh`). Same throwaway-write semantics; the
   one edge case is that overwriting an already-cached file in place fails
   instead of copying up.
2. **Judge runs CPU-only.** It is code inspection via the codex CLI; on the
   cluster it merely happened to run on a GPU node.
3. **Eval always uses 1x H100**, regardless of the agent phase's
   `--num-gpus` (final models are <=4B).
4. **Preemption**: Modal GPU containers can be preempted (rare, but
   possible on long runs, and it cannot be disabled). A preempted phase
   restarts from scratch automatically (up to 2 retries for the agent
   phase); every attempt is recorded in the run's `attempts.log`, and a
   finished eval is never re-run (`metrics.json` short-circuit).
5. **Isolation**: the agent runs as a subprocess inside the same container
   as the phase harness (on the cluster, the harness runs on the host
   outside apptainer). The judge and eval still run in fresh containers, so
   nothing the agent does to its environment can leak into them.
6. **Run IDs** are submission epoch-seconds instead of condor cluster IDs
   (still integers, still ordered, as `scripts/utils.py` expects).
7. **`--num-hours` is capped at 22** (Modal Functions max out at 24h). The
   8-GPU / 50-100h experiments from `commit.sh` are out of scope here;
   `--num-gpus` up to 8 works for runs within the cap.
