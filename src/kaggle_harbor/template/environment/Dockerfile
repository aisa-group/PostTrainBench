# Agent environment — a Docker port of containers/standard.def.
#
# Package set, versions and install order are transcribed step for step from
# that file. Read the two side by side; every difference is called out below.
#
# Deliberate differences, and why:
#
#   1. TORCH_BACKEND is an ARG defaulting to "auto", which is what
#      standard.def passes. `auto` probes the NVIDIA driver, so a builder
#      with no GPU resolves to CPU torch and then fails to satisfy vllm's
#      CUDA-only xformers dependency. Build with
#      `--build-arg TORCH_BACKEND=cu129` on a GPU-less builder — cu129 is the
#      wheel `auto` actually resolves to on Kaggle's A100, measured by probe
#      run 767447, which returned 129. (An earlier revision of this comment
#      said cu128; that was a guess, and the probe replaced it.) The ARG
#      default stays `auto` so a substitution is always a visible, deliberate
#      act rather than a silent divergence.
#
#   2. standard.def has no ENTRYPOINT (apptainer exec supplies the command).
#      Harbor also supplies the command, so none is set here either.
#
#   3. The agent CLIs are not installed. standard.def:43 installs codex only,
#      with the comment "only for the judge, the other ones are installed in
#      solve.sh". Harbor owns agent installation and version pinning (via
#      `--ak version=`), which replaces solve.sh entirely; codex is kept
#      because the image is otherwise identical upstream.
#
#   4. The sandbox home is baked in rather than bind-mounted. Upstream builds
#      JOB_DIR on the host and mounts it with
#      `--home "${JOB_DIR}:/home/ben" --pwd "/home/ben/task"`
#      (run_task.sh:246-247). The resulting in-container layout is identical.

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ARG TORCH_BACKEND=auto

# standard.def %post: `export DEBIAN_FRONTEND=noninteractive`
ENV DEBIAN_FRONTEND=noninteractive

# standard.def %environment
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# HOME is deliberately NOT set here — see the ENV at the end of this file.
#
# apptainer splits build from runtime: standard.def's %post runs as root with
# HOME=/root (which is why its `curl … astral.sh/uv/install.sh | sh` lands in
# /root/.local/bin and its `export PATH="/root/.local/bin:$PATH"` at :31
# matches), while %environment sets HOME=/home/ben only for the running
# container. Docker has one ENV namespace for both, so setting HOME=/home/ben
# up here sent the uv installer to /home/ben/.local/bin while PATH still
# pointed at /root/.local/bin — uv installed, unreachable, and every
# `uv pip install` below died with exit 127 (command not found). That is what
# broke the first real Kaggle run.

RUN chmod 1777 /tmp

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-dev \
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

# 1) vllm first (controls torch/CUDA versions)
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=${TORCH_BACKEND}

# 2) Pinned direct dependencies
COPY requirements-direct.txt /opt/requirements-direct.txt
RUN uv pip install --system --no-cache -r /opt/requirements-direct.txt

# 3) flash-attn (needs no-build-isolation)
RUN uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

# 4) AI CLI tools via npm (only for the judge, the other ones are installed in solve.sh)
RUN npm install -g @openai/codex@0.137.0

# 5) inspect_evals from source (pinned commit)
RUN mkdir -p /opt && \
    cd /opt && \
    git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd /opt/inspect_evals && \
    git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 && \
    uv pip install --system --no-cache .

# ---------------------------------------------------------------------------
# Sandbox home — the JOB_DIR that run_task.sh assembles on the host.
#
# home/ is staged by build_tasks.py and mirrors run_task.sh:55-155 exactly:
#
#   /home/ben/task/{evaluate.py,templates/,evaluation_code/,<task_context/*>}
#   /home/ben/.codex/                  containers/other_home_data/.codex
#   /home/ben/check_cuda.py            src/utils/
#   /home/ben/check_cuda_writing.py    src/utils/
#   /home/ben/system_monitor.sh        src/utils/
#   /home/ben/timestamp_lines.py       src/utils/
#   /home/ben/update_agent_cli.sh      src/utils/
#   /home/ben/create_timer.sh          src/utils/
#   /home/ben/contamination_check.py   src/judges/judge_tools/
#   /home/ben/test_data.json           src/eval/tasks/<T>/
#
# timer.sh is NOT staged. Upstream generates it during job prep
# (run_task.sh:78), stamping CREATION_DATE at a moment minutes before the
# agent starts. Image build time is hours-to-days before, so preflight.sh
# runs the same create_timer.sh at container start instead.
# ---------------------------------------------------------------------------
COPY home/ /home/ben/

COPY preflight.sh /usr/local/bin/preflight.sh
RUN chmod +x /usr/local/bin/preflight.sh && \
    chmod -R a+rw /home/ben && \
    mkdir -p /home/ben/hf_cache

# standard.def's %environment equivalent — runtime only, after every build
# step, so the uv installer above still resolves against root's HOME.
# run_task.sh:246 makes /home/ben the sandbox home, and
# containers/other_home_data/.codex/config.toml hardcodes HOME=/home/ben, so
# the codex CLI resolves its config the same way.
ENV HOME=/home/ben

WORKDIR /home/ben/task

# standard.def %labels, verbatim. LABEL is Docker's exact analogue, so there is no
# reason not to carry them; an earlier revision dropped them as "purely
# documentary", and documentary is not a reason to diverge.
#
# Placed LAST, which is both where standard.def puts %labels (after %post and
# %environment) and the only position that does not cost anything: a LABEL
# high up invalidates every cached layer beneath it, which on this image
# means a ~25 GB rebuild to add two strings of metadata.
#
# apptainer's %help has no Docker equivalent — there is no `docker run-help` —
# so it is the one part of the .def with genuinely nowhere to go.
LABEL Version="v1.1"
LABEL Description="Python ML container with CUDA support for transformers and LLM training (using uv) + AI CLI tools - pinned dependencies"
