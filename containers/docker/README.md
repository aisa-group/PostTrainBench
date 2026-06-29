# Docker-native container image

This directory contains a Docker-native build of the standard PostTrainBench
environment. It is useful for cloud GPU environments that already run workloads
inside Docker containers and do not expose Apptainer/Singularity to the user.
Runpod Pods are one example of this deployment style.

This image is only the execution environment. It does not replace the current
HTCondor submission scripts, and the existing Apptainer flow remains the
canonical path for cluster runs.

## Build

Build from the repository root so the Dockerfile can read
`containers/requirements-direct.txt`:

```bash
docker build \
  -f containers/docker/standard.Dockerfile \
  -t posttrainbench-standard:docker .
```

## Smoke test

On a machine with an NVIDIA GPU and the NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all posttrainbench-standard:docker \
  python - <<'PY'
from importlib.metadata import version

import torch
import vllm
import inspect_ai

print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
print("vllm=", version("vllm"))
print("inspect_ai=", version("inspect-ai"))
PY
```

To work with a local PostTrainBench checkout:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace/PostTrainBench" \
  -w /workspace/PostTrainBench \
  posttrainbench-standard:docker \
  bash
```

## Notes

- The image uses Python 3.11 in `/opt/posttrainbench-venv` because current
  `inspect_evals` releases require Python 3.11 or newer.
- The image installs the same CLI agent tools as `containers/standard.def`:
  Claude Code, Codex CLI, Gemini CLI, and OpenCode.
- If a cloud provider requires SSH access, add that provider-specific SSH setup
  in a downstream image or through the provider template. SSH is intentionally
  not part of this base Docker image.
