#!/usr/bin/env python3
"""Print whether vLLM decodes a model greedily when a request sets no temperature.

The final evaluation's inspect tasks set no temperature, so the vLLM server takes it from the model's generation
config (generation_config.json, else config.json), as ModelConfig.get_diff_sampling_param does with the default
--generation-config auto. Without one, it uses 1.0. Writes "greedy <temperature>" or "sampling <temperature>" to
--output-file (not stdout, which also gets vLLM's log lines).

With --batch-output, it checks many model dirs in one process (vLLM's import is slow) and writes
{model_dir: [mode, temperature]} as JSON, mode "unknown" (with the error) for a model it cannot read
(scripts/rerun_final_eval_parallel.py plans its seeds with it).

Run it in the eval container (src/eval/run_final_eval.sh does), so it uses the evaluation's vLLM and transformers.
"""
from __future__ import annotations

import argparse
import json

from vllm.sampling_params import _SAMPLING_EPS
from vllm.transformers_utils.config import try_get_generation_config

# The temperature of a request that sets none, without one in the generation config
# (vllm.entrypoints.openai.protocol.ChatCompletionRequest._DEFAULT_SAMPLING_PARAMS, a pydantic private attribute).
VLLM_DEFAULT_TEMPERATURE = 1.0


def decoding(model_dir: str) -> tuple[str, float]:
    """("greedy" or "sampling", temperature) of a request without a temperature to vLLM serving model_dir."""
    config = try_get_generation_config(model_dir, trust_remote_code=False)
    # vLLM keeps only the values that differ from the HF defaults.
    diff = {} if config is None else config.to_diff_dict()
    temperature = diff.get("temperature")
    if temperature is None:
        temperature = VLLM_DEFAULT_TEMPERATURE
    # vLLM decodes greedily below _SAMPLING_EPS (SamplingParams).
    mode = "greedy" if temperature < _SAMPLING_EPS else "sampling"
    return mode, temperature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+",
                        help="Directory of the model that vLLM serves (several with --batch-output).")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-file", help="File to write the result of the one model dir to.")
    output.add_argument("--batch-output", help="JSON file to write the results of all model dirs to.")
    args = parser.parse_args()

    if args.output_file is not None:
        if len(args.model_dirs) != 1:
            parser.error("--output-file takes exactly one model dir")
        mode, temperature = decoding(args.model_dirs[0])
        with open(args.output_file, "w") as f:
            f.write(f"{mode} {temperature}\n")
        return
    # As in run_final_eval.sh, a model whose default temperature cannot be read may still load in vLLM, so it is
    # recorded as "unknown" (with the error) for the caller to evaluate with all seeds, not a reason to stop.
    results = {}
    for model_dir in args.model_dirs:
        try:
            results[model_dir] = list(decoding(model_dir))
        except Exception as err:  # noqa: BLE001  (reported per model, see above)
            results[model_dir] = ["unknown", f"{type(err).__name__}: {err}"]
    with open(args.batch_output, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
