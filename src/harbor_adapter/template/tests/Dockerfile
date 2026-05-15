FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Verifier image for harbor's separate-verifier mode
# (https://www.harborframework.com/docs/tasks#verifier-environment-shared-vs-separate).
#
# Built from the task's tests/ directory. Harbor does NOT upload tests/
# at runtime in separate mode, so test.sh + every eval helper the
# verifier needs (evaluate.py, templates/, evaluation_code/,
# contamination_judge.py, metadata.json) must be baked in. They get
# copied into /tests/ via `COPY . /tests/` near the bottom.
#
# Mirrors environment/Dockerfile (containers/opus_4_6_1m.def) so any
# Python the verifier imports resolves identically — same vllm,
# transformers, flash-attn, inspect_evals, inspect_ai_vllm_stdout
# pins. Also keeps the agent CLIs (codex, claude-code, gemini-cli,
# opencode) because test.sh shells out to `codex exec ...` for the
# contamination judge.
#
# /home/agent/workspace is created empty here; harbor's verifier-input
# transfer (driven by the task.toml [[artifacts]] block) populates it
# with final_model/ + the rest of the agent's workspace contents
# before test.sh runs.

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Node.js 22.x (for codex CLI used by the contamination judge)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# vllm. Pin torch backend to CUDA 12.8 wheels explicitly (Modal build VM
# has no GPU/nvidia-smi, so --torch-backend=auto falls back to CPU and
# breaks the CUDA-only xformers requirement).
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu128

# Pinned ML deps (synced from containers/requirements-direct.txt;
# adapter copies that file into the build context).
COPY requirements-direct.txt /opt/requirements-direct.txt
RUN uv pip install --system --no-cache -r /opt/requirements-direct.txt

RUN uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

# Pinned agent CLIs. Codex is required for the contamination judge in
# test.sh; the others are kept for parity so the verifier image is a
# drop-in match for the agent image.
RUN npm install -g \
    @anthropic-ai/claude-code@2.1.76 \
    @openai/codex@0.98.0 \
    @google/gemini-cli@0.18.4 \
    opencode-ai@1.1.59

RUN mkdir -p /opt && \
    cd /opt && \
    git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd /opt/inspect_evals && \
    git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 && \
    uv pip install --system --no-cache .

RUN cd /opt && \
    git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git && \
    cd inspect_ai_vllm_stdout && \
    uv pip install --system --no-cache .

# Entrypoint streams /logs/{agent,verifier}/*.txt to PID 1 stdout so
# Modal's sandbox dashboard shows live verifier output. Also kicks off
# a system monitor daemon (parity with condor's
# src/utils/system_monitor.sh). Healthcheck (in task.toml) writes
# /timer_start once the streaming daemon is up.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh

# Workspace placeholder. Populated at runtime by harbor's
# verifier-input transfer from the agent env.
RUN mkdir -p /home/agent/workspace

# Bake the verifier package into /tests/. The Dockerfile itself is
# excluded via .dockerignore. entrypoint.sh, system_monitor.sh, and
# requirements-direct.txt also land here (build-context bystanders) and
# are stripped to keep /tests/ clean.
RUN mkdir -p /tests
COPY . /tests/
RUN rm -f /tests/entrypoint.sh \
          /tests/system_monitor.sh \
          /tests/requirements-direct.txt && \
    chmod +x /tests/test.sh

# WORKDIR matches the agent env so test.sh's `cd $WORKSPACE && codex
# exec ...` resolves the same way; harbor's transfer puts the agent's
# files at /home/agent/workspace.
WORKDIR /home/agent/workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
