# Verifier environment.
#
# Upstream runs the post-agent phases in two images, neither of them the
# agent's:
#
#   judges      containers/gpt_5_5.def     (judge_lib.sh:37, JUDGE_CONTAINER)
#   evaluation  containers/vllm_debug.def  (run_task.sh:419)
#
# Harbor gives a task one verifier environment, so this image is the union.
# The two defs are byte-identical apart from their npm CLI pins, so the union
# is exact rather than approximate:
#
#   vllm_debug.def   claude-code@2.1.34   codex@0.98.0   gemini@0.18.4   opencode@1.1.59
#   gpt_5_5.def      claude-code@2.1.116  codex@0.124.0  gemini@0.39.1   opencode@1.14.20
#
# The merge forces one choice: whose pins? This image takes gpt_5_5.def's,
# because that is the image that actually runs the judges upstream. All four
# CLIs are installed, not just codex — an earlier revision kept codex alone
# because it is the only one any verifier code path invokes, but that was an
# efficiency argument and efficiency is not grounds to diverge. general_judge
# additionally npm-installs its own pinned codex at judge time, exactly as
# judge_lib.sh does upstream.
#
# Everything else — base image, apt set, python symlinks, node, uv, vllm,
# requirements-direct.txt, flash-attn, inspect_evals at its pinned commit,
# inspect_ai_vllm_stdout, and the %environment exports — is transcribed from
# vllm_debug.def step for step.
#
# See environment/Dockerfile for the TORCH_BACKEND note; it applies here too.
#
# Harbor builds this from `tests/` as the build context and does NOT upload
# tests/ at runtime for a separate verifier environment
# (trial.py:_verifier_env_build_context, skip_tests_upload=True), so
# everything test.sh reads has to be baked in below.

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ARG TORCH_BACKEND=auto

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

# HOME is set at the END of this file, not here. Same reason as
# environment/Dockerfile: apptainer's %post runs as root (HOME=/root), so
# gpt_5_5.def/vllm_debug.def's uv installer lands in /root/.local/bin and
# matches the PATH above. Setting HOME=/home/ben at build time instead sends
# uv to /home/ben/.local/bin, leaving every `uv pip install` below to fail
# with exit 127.

RUN chmod 1777 /tmp

RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-dev \
        git \
        wget \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=${TORCH_BACKEND}

COPY requirements-direct.txt /opt/requirements-direct.txt
RUN uv pip install --system --no-cache -r /opt/requirements-direct.txt

RUN uv pip install --system --no-cache flash-attn==2.8.3 --no-build-isolation

# gpt_5_5.def:44-48, all four pins, verbatim.
#
# Only codex is ever invoked in here (the judges use it; evaluate.py uses no
# CLI), and an earlier revision of this file installed codex alone to keep the
# image smaller. That was an efficiency argument, and efficiency is not a
# reason to diverge: the project rule is no deviation unless it is
# physically impossible, inefficiency notwithstanding (2026-09-01).
# Installing all four is possible, so all four are installed.
#
# ⚠️ codex is 0.146.1, NOT gpt_5_5.def's 0.124.0 -- see D227. 0.124.0 has no
# HTTPS fallback, so the model proxy's 307 on `wss://$BASE/responses` is fatal
# to it; 0.146.1 is the version Kaggle registers as its canonical codex harness
# and falls back cleanly. Bumping it removes the custom model_provider block
# test.sh had to inject.
#
# The pins are gpt_5_5.def's, not vllm_debug.def's. The two upstream images
# differ ONLY in these four versions, so merging them (D90) forces a choice,
# and gpt_5_5 is the image that actually runs the judges upstream.
RUN npm install -g \
        @anthropic-ai/claude-code@2.1.116 \
        @openai/codex@0.146.1 \
        @google/gemini-cli@0.39.1 \
        opencode-ai@1.14.20

# install inspect evals
RUN mkdir -p /opt && \
    cd /opt && \
    git clone https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd /opt/inspect_evals && \
    git checkout 06001a83e6d7c709c2ede0570dce7f1031a0bad8 && \
    uv pip install --system --no-cache .

# install inspect ai with debug
RUN cd /opt && \
    git clone https://github.com/rank-and-file/inspect_ai_vllm_stdout.git && \
    cd inspect_ai_vllm_stdout && \
    uv pip install --system --no-cache .

# ---------------------------------------------------------------------------
# /tests — the untampered copies the verifier runs against.
#
# Staged by build_tasks.py, mirroring what run_task.sh re-copies for the
# judge and evaluation phases so neither ever executes an agent-modified
# file (run_task.sh:358 prepare_judge_sandbox, and the evaluation's
# `--bind "${REPO_ROOT}:${REPO_ROOT}"` which serves evaluate.py and
# templates/ from the repo, not the sandbox):
#
#   /tests/test.sh                    the port of run_task.sh:290-518
#   /tests/score_run.py               the port of scripts/collect.py's per-cell rules
#   /tests/strip_web_search.py        diagnostic shim, inert unless PTB_DISABLE_WEB_SEARCH=1
#   /tests/metadata.json              generated: benchmark, model, weights, baseline
#   /tests/judge_model_map.json       generated: upstream model name -> proxy slug
#   /tests/docker-compose.yaml        generated: HF-cache bind, when there is one
#
# Everything copied from upstream lives under /tests/repo/, at its ORIGINAL
# upstream path, so relative paths inside those files resolve unchanged:
#
#   /tests/repo/src/eval/tasks/<T>/   evaluate.py, templates/, test_data.json,
#                                     benchmark.txt, info.json, evaluation_code/
#   /tests/repo/src/eval/templates/   the chat templates evaluate.py resolves via
#                                     --templates-dir ../../../../src/eval/templates
#   /tests/repo/src/judges/           get_judge_prompt.py, judge_tools/,
#                                     <judge>/judge.conf + prompt.md
#   /tests/repo/src/trace_parsing/    parse_trace.py and the per-agent parsers
#   /tests/repo/containers/           delete_hf_models.py, other_home_data/.codex
#
# An earlier revision of this comment described a flattened layout
# (/tests/eval/, /tests/judge_tools/, /tests/dot_codex/, ...) that this port
# has never produced. Corrected 2026-09-01; the mirrored-tree layout above is
# what `find /tests` actually shows and is the whole reason upstream's
# relative paths work untouched (D98).
# ---------------------------------------------------------------------------
COPY . /tests/
RUN rm -f /tests/Dockerfile /tests/requirements-direct.txt && \
    chmod +x /tests/test.sh

# %environment equivalent — runtime only, after every build step.
ENV HOME=/home/ben

WORKDIR /tests

# gpt_5_5.def %labels, verbatim. LABEL is Docker's exact analogue, so there is no
# reason not to carry them; an earlier revision dropped them as "purely
# documentary", and documentary is not a reason to diverge.
#
# Placed LAST, which is both where gpt_5_5.def puts %labels (after %post and
# %environment) and the only position that does not cost anything: a LABEL
# high up invalidates every cached layer beneath it, which on this image
# means a ~25 GB rebuild to add two strings of metadata.
#
# apptainer's %help has no Docker equivalent — there is no `docker run-help` —
# so it is the one part of the .def with genuinely nowhere to go.
LABEL Version="v1.0"
LABEL Description="Python ML container with CUDA support for transformers and LLM training (using uv) + AI CLI tools"
