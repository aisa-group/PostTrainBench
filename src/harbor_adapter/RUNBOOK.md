# PostTrainBench × Harbor on Modal — Reproducibility Runbook

**Purpose:** exact, verified versions + commands for running the PostTrainBench Harbor adapter on Modal GPUs. Captured because the pipeline has several non-obvious requirements that cause silent failures if missed.

**Last verified:** 2026-06-15, on a clean run that built the image, trained a Qwen3-1.7B model, and produced a valid `final_model`. All container versions below are from `pip freeze` inside the **live running sandbox** (ground truth, not inferred).

> ⚠️ Read the **[Gotchas](#gotchas--why-others-fail)** section first — every item there has caused a real failure.

---

## TL;DR — the recipe that works

```bash
# 0. One-time host setup
uv tool install 'harbor[modal]' --force      # harbor 0.13.2 WITH the modal extra (see Gotcha #1/#2)
python -m modal setup                          # authenticate Modal (interactive)
export ANTHROPIC_API_KEY="sk-ant-..."          # terminus-2 agent (LLM loop runs host-side)
export OPENAI_API_KEY="sk-..."                 # contamination judge + some evals' judge

# 1. Repo env (for run_adapter.py + the `modal` CLI helpers)
cd src/harbor_adapter
uv sync

# 2. Generate a task (keeps the fastapi==0.136.0 pin from containers/requirements-direct.txt)
uv run python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --num-hours 2 --output ./tasks

# 3. Run it (VERIFIED-WORKING flag set)
uv run harbor run \
    --path tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent terminus-2 \
    --model anthropic/claude-opus-4-8 \
    --env modal \
    --yes \
    --max-retries 3
```

Key choices and **why** (details in Gotchas):
- `harbor[modal]`, not `harbor` — cloud backends are optional extras in harbor 0.13.x.
- `--agent terminus-2`, **not `claude-code`** — claude-code self-kills (its prompt is in argv; the agent's own `pkill -f evaluate.py` matches and kills it).
- `--yes` — harbor 0.13.x prompts to confirm env vars; without it, headless runs abort.
- `--max-retries 3` — rides through transient Modal/network flakiness.
- **No `--ak temperature=...`** — harbor 0.13.2 handles `claude-opus-4-8`'s deprecated `temperature` param correctly (needed only on harbor ≤0.1.x).

---

## Host tooling versions (verified)

| Tool | Version | Notes |
|------|---------|-------|
| OS (host) | macOS / darwin | orchestrator runs here |
| `uv` | 0.10.6 | |
| host Python | 3.14.4 | |
| **harbor** | **0.13.2** | installed as a **`uv tool`** (`~/.local/bin/harbor`), **UNPINNED** — drifts; runs on Python 3.14. Install with the `[modal]` extra. |
| `modal` (harbor tool env) | 1.5.0 | pulled by `harbor[modal]` |
| `modal` (adapter `.venv`) | 1.3.0.post1 | used by `uv run modal ...` helper commands |

> **harbor is not pinned anywhere** (not in `pyproject.toml`/`uv.lock`) — it is a standalone uv tool. Pin your team to a known-good version: **0.13.2**. Versions **< 0.13** break the run timer (the `task.toml [environment.healthcheck]` that writes `/timer_start` is unsupported on old harbor → agents have no clock).

Reproduce host versions:
```bash
uv --version
uv tool list | grep harbor          # shows harbor + version
cd src/harbor_adapter && uv run harbor --version
uv run modal --version
```

---

## Container image versions (verified from live sandbox)

Built by harbor from `src/harbor_adapter/template/environment/Dockerfile`.

| Layer | Value |
|-------|-------|
| Base image | `nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04` |
| OS | Ubuntu 22.04.5 LTS |
| Container Python | 3.10.12 |
| CUDA (nvcc) | 12.9 |
| GPU (runtime) | NVIDIA H100 80GB HBM3, driver 580.95.05 (Modal may serve H200 too) |
| Node.js | v22.22.3 |

### Python packages in the container (key — full `pip freeze` is 228 pkgs)

| Package | Version | Source / note |
|---------|---------|---------------|
| torch | **2.8.0+cu128** | `vllm==0.11.0 --torch-backend=cu128` |
| vllm | 0.11.0 | pinned in Dockerfile |
| flash_attn | 2.8.3 | `--no-build-isolation` |
| transformers | 4.57.3 | requirements-direct.txt |
| trl | 0.27.2 | |
| peft | 0.18.1 | |
| accelerate | 1.12.0 | |
| datasets | 4.5.0 | |
| tokenizers | 0.22.2 | |
| huggingface-hub | 0.36.0 | |
| bitsandbytes | 0.49.1 | |
| safetensors | 0.8.0 | |
| numpy | 2.2.6 | |
| **fastapi** | **0.136.0** | **PINNED BY US** (see Gotcha #4) — NOT the latest 0.137.x |
| starlette | 1.3.1 | |
| prometheus-fastapi-instrumentator | 8.0.0 | (the bug is fastapi, not this) |
| uvicorn | 0.49.0 | |
| pydantic | 2.13.4 | |
| openai | 2.17.0 | |
| tiktoken | 0.12.0 | |
| lm-eval | 0.4.10 | requirements-direct.txt |
| ninja | 1.13.0 | |
| packaging | 26.0 | |
| inspect_ai | fork: `rank-and-file/inspect_ai_vllm_stdout` (installed as `inspect_ai`, overrides `inspect-ai==0.3.150`) |
| inspect_evals | `UKGovernmentBEIS/inspect_evals` @ commit `06001a83e6d7c709c2ede0570dce7f1031a0bad8` |

### Agent CLIs baked into the image (npm, from Dockerfile)
```
@anthropic-ai/claude-code@2.1.76
@openai/codex@0.98.0
@google/gemini-cli@0.18.4
opencode-ai@1.1.59
```
(Note: terminus-2 does **not** use these — it drives the sandbox over tmux with a host-side LLM loop. These matter only for the `--agent claude-code|codex|...` "installed" agents.)

### Pinned ML deps file
The Dockerfile installs `containers/requirements-direct.txt` **after** vllm. Full pinned list:
```
accelerate==1.12.0   bitsandbytes==0.49.1   boto3==1.40.61   certifi==2026.1.4
datasets==4.5.0      evaluate==0.4.6        inspect-ai==0.3.150   lm-eval==0.4.10
matplotlib==3.10.8   openai==2.17.0         pandas==2.2.3    peft==0.18.1
scikit-learn==1.7.2  shortuuid==1.0.13      tiktoken==0.12.0  tokenizers==0.22.2
transformers==4.57.3 trl==0.27.2            ninja==1.13.0    packaging==26.0
huggingface-hub==0.36.0
fastapi==0.136.0     # <-- added to fix vLLM /v1/models 500 (see Gotcha #4)
```

To regenerate the exact container freeze yourself (while a sandbox is live):
```bash
CID=$(uv run modal container list --json | python3 -c "import sys,json;d=json.load(sys.stdin);print([ (c.get('Container ID') or c.get('container_id')) for c in d if 'harbor' in str(c).lower()][0])")
uv run modal container exec "$CID" -- bash -lc 'pip freeze' > container_freeze.txt
```

---

## Exact commands used (full sequence)

```bash
# --- one-time, host ---
uv tool install 'harbor[modal]' --force
python -m modal setup
export ANTHROPIC_API_KEY=...   export OPENAI_API_KEY=...

# --- repo env ---
cd src/harbor_adapter && uv sync

# --- generate tasks ---
uv run python run_adapter.py --benchmark gsm8k    --model qwen3-1.7b --num-hours 2 --output ./tasks
uv run python run_adapter.py --benchmark preflight --model qwen3-1.7b --num-hours 3 --output ./tasks   # PR-only benchmark, see note

# --- run (gsm8k, 2h agent cap, verification on) ---
uv run harbor run --path tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent terminus-2 --model anthropic/claude-opus-4-8 \
    --env modal --yes --max-retries 3

# --- run (preflight, 3h cap) ---
uv run harbor run --path tasks/posttrainbench-preflight-qwen3-1.7b \
    --agent terminus-2 --model anthropic/claude-opus-4-8 \
    --env modal --yes --max-retries 3

# --- add --disable-verification for a fast smoke test (no eval/score) ---
```

**Wall clocks used:** gsm8k `--num-hours 2`, pre-flight `--num-hours 3` (pre-flight's eval is heavier: `--max-tokens 16000`). Minimum useful for a base benchmark ≈ 1h; don't go below ~1h (sub-hour only tests provisioning). Full default is 10h.

---

## Gotchas — why others fail

1. **Install `harbor[modal]`, not `harbor`.** Plain harbor 0.13.x fails `--env modal` with `MissingExtraError: The 'modal' package is required`. Fix: `uv tool install 'harbor[modal]' --force`.

2. **harbor version matters — use ≥ 0.13.2.** harbor is an unpinned uv tool. On stale **0.1.44** the `[environment.healthcheck]` that writes `/timer_start` is unsupported, so `timer.sh` always says *"Timer not initialized"* and agents run with no clock (they get disoriented and fail to save a model in time). 0.13.2 runs the healthcheck → timer works.

3. **Use `--yes`.** harbor 0.13.x adds an interactive env-var confirmation prompt; headless/background runs abort without `--yes`.

4. **Pin `fastapi==0.136.0` (already in `containers/requirements-direct.txt`).** vLLM 0.11.0 declares `fastapi[standard]>=0.115.0` (no upper bound), so it floats to **0.137.x**, whose router refactor adds `_IncludedRouter` (a `BaseRoute` with no `.path`). vLLM's prometheus middleware then crashes (`AttributeError: '_IncludedRouter' object has no attribute 'path'`) → **`/v1/models` returns HTTP 500 forever** → `inspect_ai`'s health-check poll hangs → eval never starts (GPU loads the model and sits idle — looks like a silent hang). Pinning fastapi below 0.137.0 fixes it. **This affects the verifier too.**

5. **Prefer `--agent terminus-2` over `claude-code`.** harbor passes the claude-code prompt in **argv** (`claude --print -- '<instruction>'`), and the instruction text contains "evaluate.py". When the agent runs `pkill -f evaluate.py` (a natural step to free the GPU before training), it matches and **kills its own process** (exit 137/143, no result). terminus-2's LLM loop runs **host-side** and drives the sandbox over tmux, so it's immune. (This is **not** fixed in harbor 0.13.2.)

6. **Do NOT pass `--ak temperature=...` on harbor 0.13.2.** `claude-opus-4-8` rejects non-default temperature (`'temperature' is deprecated for this model`). On 0.13.2 terminus-2 defaults temperature to `None` and only sends it if set, so omit it. (On legacy harbor ≤0.1.x you must pass `--ak temperature=1`, because it sent `0.7` unconditionally.)

7. **`OPENAI_API_KEY` is required even when the agent is Claude/terminus.** The verifier's contamination judge uses the OpenAI Codex CLI, and `arenahardwriting`/`healthbench` evals use an OpenAI judge.

8. **The run is bound to the launching machine.** The harbor orchestrator (and terminus-2's LLM loop) run locally; Modal only hosts the GPU sandbox. **WiFi loss or laptop sleep kills the run.** For unattended long runs, launch from an always-on host (tmux on a VM) or wrap `harbor run` in a detached Modal Function.

9. **`pre-flight` (and ds1000/frontierscience/dabstep/kernelbench) are not in the base repo** — they live in the `mercor-code-envs/posttrainbench-samples` PR. To run pre-flight you need its `src/eval/tasks/preflight/` files + the `adapter.py` `BENCHMARKS` entry from that PR. `inspect_evals.pre_flight` imports cleanly at the pinned commit.

---

## Observability / verifying a run

- **Modal dashboard:** `https://modal.com/apps/<workspace>/main` (sandboxes appear under the `__harbor__` app).
- **Local job artifacts:** `src/harbor_adapter/jobs/<job-name>/` → `result.json`, per-trial `trial.log`, `agent/` (trajectory, system_monitor.log), `verifier/` (`metrics.json`, `reward.txt`).
- **Exec into a live sandbox** (read-only checks):
  ```bash
  CID=$(uv run modal container list --json | python3 -c "import sys,json;d=json.load(sys.stdin);print([ (c.get('Container ID') or c.get('container_id')) for c in d if 'harbor' in str(c).lower()][0])")
  uv run modal container exec "$CID" -- bash -lc 'cd /home/agent/workspace && bash timer.sh; nvidia-smi; ls final_model'
  ```
- **A healthy run shows:** `timer.sh` reports real remaining time (not "not initialized"); GPU utilized during training; `final_model/` is a complete HF dir (`model.safetensors` ~3.4 GB for 1.7B, `config.json` with `Qwen3ForCausalLM`); verifier `reward.txt` holds the accuracy.
