FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"
# Task budget in seconds (matches agent.timeout_sec in task.toml).
# Injected at task-generation time by adapter.py.
ENV TASK_BUDGET_SECS={task_budget_secs}

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create python3 symlink
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# Install Node.js (LTS version 22.x) for npm
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install vllm. Repinned from 0.11.0 → 0.19.1: 0.11.0 required
# xformers==0.0.32.post1 which is no longer on PyPI for manylinux_x86_64,
# and 0.19.1 is what we successfully built and end-to-end tested on Modal.
RUN uv pip install --system --no-cache vllm==0.19.1 --torch-backend=auto

# Install AI CLI tools via npm (agents can use these)
RUN npm install -g \
    @anthropic-ai/claude-code \
    @openai/codex \
    @google/gemini-cli

RUN uv pip install --system --no-cache ninja packaging

# Install the required ML packages
RUN uv pip install --system --no-cache \
    accelerate \
    boto3 \
    bitsandbytes \
    datasets \
    evaluate \
    lm-eval \
    openai \
    pandas \
    scikit-learn \
    shortuuid \
    tokenizers \
    transformers \
    trl \
    peft \
    tiktoken \
    inspect-ai \
    matplotlib \
    certifi

# Note: flash_attn requires GPU to compile - install at runtime if needed:
# pip install flash_attn --no-build-isolation

# Install inspect_evals pinned to commit 03cb4bc2 (2026-03-15), the last
# commit on `main` that still declares `requires-python = ">=3.10"`. The
# next commit (b4d1356a, "Remove Python 3.10 from CI and bump minimum to
# 3.11") drops 3.10 support, which clashes with the python3.10 we install
# above. To bump past this pin, either upgrade the base image's Python or
# pick the latest inspect_evals commit that still supports 3.10.
RUN uv pip install --system --no-cache \
    "inspect_evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals.git@03cb4bc212890a828d9cc73c17e6c491ec24fbee"

# Setup workspace
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Install timer daemon + entrypoint at /home/agent/ (outside the agent workspace).
# Authoritative copies live at /home/agent/; the agent-writable task folder
# does not have them.
COPY timer.sh /home/agent/timer.sh
COPY entrypoint.sh /home/agent/entrypoint.sh
RUN chmod +x /home/agent/timer.sh /home/agent/entrypoint.sh

# Timer daemon — three start mechanisms for Harbor/Modal reliability:
#   1. ENTRYPOINT       — fires on container start under plain Docker.
#   2. /etc/bash.bashrc — fires on interactive bash exec (Modal's usual path).
#   3. BASH_ENV profile — fires on non-interactive bash exec.
# The timer is idempotent (PID file), so multiple triggers are safe.
RUN printf '# Start PostTrainBench task timer if not already running\n/home/agent/timer.sh >/dev/null 2>&1 &\n' \
        >> /etc/bash.bashrc && \
    printf '#!/usr/bin/env bash\n[ -x /home/agent/timer.sh ] && /home/agent/timer.sh >/dev/null 2>&1 &\n' \
        > /etc/profile.d/ptb-timer.sh && \
    chmod +x /etc/profile.d/ptb-timer.sh
ENV BASH_ENV=/etc/profile.d/ptb-timer.sh

# Copy all task files (populated by adapter; .dockerignore excludes Dockerfile)
COPY . /home/agent/workspace/

# Set permissions on agent workspace. Remove any duplicate copies of timer/entrypoint
# that the wildcard COPY pulled in — the authoritative copies live at /home/agent/.
RUN chmod -R a+rw /home/agent/workspace/ && \
    rm -f /home/agent/workspace/timer.sh /home/agent/workspace/entrypoint.sh

ENTRYPOINT ["/home/agent/entrypoint.sh"]
# Keep the container alive when Harbor/Modal creates the sandbox without a command.
CMD ["tail", "-f", "/dev/null"]
