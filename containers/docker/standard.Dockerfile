FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/opt/posttrainbench-venv
ENV PATH="${VIRTUAL_ENV}/bin:/root/.local/bin:${PATH}"
ENV PYTHONNOUSERSITE=1
ENV NO_PROXY=localhost,127.0.0.1
ENV no_proxy=localhost,127.0.0.1

RUN chmod 1777 /tmp \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.11 \
    && uv venv "${VIRTUAL_ENV}" --python 3.11

RUN uv pip install --no-cache vllm==0.11.0

RUN npm install -g \
        @anthropic-ai/claude-code@2.0.55 \
        @openai/codex@0.79.0 \
        @google/gemini-cli@0.18.4 \
        opencode-ai@1.1.59

COPY containers/requirements-direct.txt /tmp/posttrainbench-requirements.txt
RUN uv pip install --no-cache -r /tmp/posttrainbench-requirements.txt \
    && uv pip install --no-cache flash_attn --no-build-isolation \
    && rm /tmp/posttrainbench-requirements.txt

RUN mkdir -p /opt \
    && cd /opt \
    && git clone --depth=1 https://github.com/UKGovernmentBEIS/inspect_evals.git \
    && cd /opt/inspect_evals \
    && uv pip install --no-cache .

WORKDIR /workspace
CMD ["bash"]
