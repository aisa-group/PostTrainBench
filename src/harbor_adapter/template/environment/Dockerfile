FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# This Dockerfile mirrors containers/opus_5.def (the PostTrainBench v1.1
# apptainer image used by the HTCondor pipeline) for cross-environment parity.
# Differences vs. apptainer build:
#   - --torch-backend=cu128 is explicit. Modal's build VM has no
#     nvidia-smi/CUDA driver, so --torch-backend=auto would resolve to CPU
#     torch and fail to satisfy vllm's CUDA-only xformers dependency.
#   - ENTRYPOINT streams /logs/agent/*.txt to PID 1 stdout so Modal's
#     dashboard shows live agent output (apptainer doesn't need this).

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

# Node.js 22.x (for the agent CLIs)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# vllm. Pin torch backend to CUDA 12.8 wheels explicitly (Modal build VM
# has no GPU/nvidia-smi, so --torch-backend=auto falls back to CPU and
# breaks the CUDA-only xformers requirement).
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu128

# Pinned ML deps (synced from containers/requirements-direct.txt; adapter
# copies that file into the build context).
COPY requirements-direct.txt /opt/requirements-direct.txt
RUN uv pip install --system --no-cache -r /opt/requirements-direct.txt

RUN uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

# Agent CLIs, pinned to containers/opus_5.def (the PostTrainBench v1.1
# container era this image mirrors). opus_5.def additionally installs the
# xAI Grok and Cursor CLIs for agents this adapter does not run; they are
# omitted here. Override the Claude Code version per run without rebuilding
# via `harbor run --ak version=<x.y.z>`.
RUN npm install -g \
    @anthropic-ai/claude-code@2.1.219 \
    @openai/codex@0.144.0 \
    @google/gemini-cli@0.18.4 \
    opencode-ai@1.17.18

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

# Entrypoint streams /logs/{agent,verifier}/*.txt to PID 1 stdout so Modal's
# sandbox dashboard shows live agent + verifier output. Also kicks off a
# system monitor daemon (parity with condor's src/utils/system_monitor.sh).
# Healthcheck (in task.toml) writes /timer_start once the streaming daemon
# is up.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
# Post-agent collection (weights -> shared volume, code snapshot -> /logs/artifacts);
# run by the [[verifier.collect]] hook in task.toml.
COPY ptb_collect.sh /usr/local/bin/ptb_collect.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh /usr/local/bin/ptb_collect.sh

# Self-decontamination tooling (parity with run_task.sh, which copies both
# into the sandbox HOME): the judges' n-gram checker and the benchmark test
# set live one level above the workspace, exactly where the prompt's
# "Decontamination Tool" section points (`../contamination_check.py`,
# `../test_data.json`).
COPY contamination_check.py /home/agent/contamination_check.py
COPY test_data.json /home/agent/test_data.json

# Workspace
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Task files. The Dockerfile is excluded via .dockerignore. entrypoint.sh,
# system_monitor.sh, ptb_collect.sh, requirements-direct.txt, contamination_check.py
# and test_data.json land here too via `COPY .`,
# but they're container-internal — strip them from the workspace.
COPY . /home/agent/workspace/
RUN rm -f /home/agent/workspace/entrypoint.sh \
          /home/agent/workspace/system_monitor.sh \
          /home/agent/workspace/ptb_collect.sh \
          /home/agent/workspace/requirements-direct.txt \
          /home/agent/workspace/contamination_check.py \
          /home/agent/workspace/test_data.json

RUN chmod -R a+rw /home/agent/workspace/ && \
    chmod +x /home/agent/workspace/timer.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
