FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:$PATH"
ENV NO_PROXY="localhost,127.0.0.1"
ENV no_proxy="localhost,127.0.0.1"

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

# Install vllm
RUN uv pip install --system --no-cache vllm==0.11.0 --torch-backend=auto

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

# Install inspect evals
RUN mkdir -p /opt && \
    cd /opt && \
    git clone --depth=1 https://github.com/UKGovernmentBEIS/inspect_evals.git && \
    cd /opt/inspect_evals && \
    uv pip install --system --no-cache .

# Setup workspace
RUN mkdir -p /home/agent/workspace
WORKDIR /home/agent/workspace

# Copy task files (will be populated by adapter)
COPY ./evaluate.py /home/agent/workspace/evaluate.py
COPY ./templates/ /home/agent/workspace/templates/
COPY ./timer.sh /home/agent/workspace/timer.sh
COPY ./metadata.json /home/agent/workspace/metadata.json
COPY ./contamination_judge.py /home/agent/workspace/contamination_judge.py

# Set permissions
RUN chmod -R a+rw /home/agent/workspace/ && \
    chmod +x /home/agent/workspace/timer.sh
