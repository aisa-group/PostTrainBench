FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

# Verifier image for harbor's separate-verifier mode (PR #1655). Built
# from `tests/` as context — Harbor builds this image whenever the
# task's [verifier].environment_mode = "separate", and uploads the
# agent's [[artifacts]] into the resulting container at the same paths.
#
# Mirrors environment/Dockerfile (the agent image) so any Python the
# verifier imports — vllm, transformers, inspect_evals — resolves
# identically across both containers. Differences vs. agent Dockerfile:
#   - No `COPY . /home/agent/workspace/`: agent's workspace contents
#     arrive at runtime via the artifact transfer, NOT baked in.
#   - `COPY . /tests/` instead: test.sh, evaluate.py, templates/,
#     contamination_judge.py, etc. are baked in (Harbor does not
#     upload `tests/` at runtime for separate verifier envs).

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

# Node.js 22.x — only needed for the codex contamination judge invoked
# from test.sh. Drop later if we move the judge to a Python SDK call.
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# vllm with cu128 explicit (Modal build VM has no GPU).
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=cu128

# Pinned ML deps (adapter copies containers/requirements-direct.txt
# into the tests/ build context).
COPY requirements-direct.txt /opt/requirements-direct.txt
RUN uv pip install --system --no-cache -r /opt/requirements-direct.txt

RUN uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

# Codex CLI (contamination judge). Other agent CLIs deliberately omitted —
# the verifier doesn't run an agent.
RUN npm install -g @openai/codex@0.98.0

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

# Entrypoint streams /logs/{agent,verifier}/*.txt to PID 1 stdout (Modal
# dashboard) and kicks off the system monitor. Healthcheck (in task.toml)
# writes /timer_start once the streaming daemon is up.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY system_monitor.sh /usr/local/bin/system_monitor.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/system_monitor.sh

# Workspace exists but stays empty until the artifact transfer fills it.
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Bake the entire tests/ build context (test.sh, evaluate.py, templates/,
# evaluation_code/, contamination_judge.py, metadata.json, and any
# task_context/* files the adapter dropped in) into /tests/.
# entrypoint.sh / system_monitor.sh / requirements-direct.txt land here
# too via `COPY .`; they're already installed elsewhere, so strip them
# from /tests/ to keep test.sh's listing clean.
COPY . /tests/
RUN rm -f /tests/Dockerfile \
          /tests/entrypoint.sh \
          /tests/system_monitor.sh \
          /tests/requirements-direct.txt && \
    chmod +x /tests/test.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
