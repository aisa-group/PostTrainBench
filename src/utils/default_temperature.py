#!/usr/bin/env python3
"""Print whether vLLM decodes a model greedily when a request sets no temperature.

The final evaluation's inspect tasks set no temperature, so the vLLM server takes it from the model's generation
config (generation_config.json, else config.json), as ModelConfig.get_diff_sampling_param does with the default
--generation-config auto. Without one, it uses 1.0. Writes "greedy <temperature>" or "sampling <temperature>" to
--output-file (not stdout, which also gets vLLM's log lines).

Run it in the eval container (src/eval/run_final_eval.sh does), so it uses the evaluation's vLLM and transformers.
"""
from __future__ import annotations

import argparse

from vllm.sampling_params import _SAMPLING_EPS
from vllm.transformers_utils.config import try_get_generation_config

# The temperature of a request that sets none, without one in the generation config
# (vllm.entrypoints.openai.protocol.ChatCompletionRequest._DEFAULT_SAMPLING_PARAMS, a pydantic private attribute).
VLLM_DEFAULT_TEMPERATURE = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Directory of the model that vLLM serves.")
    parser.add_argument("--output-file", required=True, help="File to write the result to.")
    args = parser.parse_args()

    config = try_get_generation_config(args.model_dir, trust_remote_code=False)
    # vLLM keeps only the values that differ from the HF defaults.
    diff = {} if config is None else config.to_diff_dict()
    temperature = diff.get("temperature")
    if temperature is None:
        temperature = VLLM_DEFAULT_TEMPERATURE
    # vLLM decodes greedily below _SAMPLING_EPS (SamplingParams).
    mode = "greedy" if temperature < _SAMPLING_EPS else "sampling"
    with open(args.output_file, "w") as f:
        f.write(f"{mode} {temperature}\n")


if __name__ == "__main__":
    main()
