FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# This Dockerfile mirrors containers/opus_4_6_1m.def (the apptainer
# image used by the HTCondor pipeline) for cross-environment parity.
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

# Pinned agent CLIs (matches opus_4_6_1m.def).
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

# Entrypoint streams /logs/{agent,verifier}/*.txt to PID 1 stdout so Modal's
# sandbox dashboard shows live agent + verifier output. Also kicks off a
# system monitor daemon (parity with condor's src/utils/system_monitor.sh).
# Healthcheck (in task.toml) writes /timer_start once the streaming daemon
# is up.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh

# Workspace
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Task files. The Dockerfile is excluded via .dockerignore. entrypoint.sh,
# system_monitor.sh, and requirements-direct.txt land here too via `COPY .`,
# but they're container-internal — strip them from the workspace.
COPY . /home/agent/workspace/
RUN rm -f /home/agent/workspace/entrypoint.sh \
          /home/agent/workspace/system_monitor.sh \
          /home/agent/workspace/requirements-direct.txt

RUN chmod -R a+rw /home/agent/workspace/ && \
    chmod +x /home/agent/workspace/timer.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
