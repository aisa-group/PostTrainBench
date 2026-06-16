#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import modal

# --- Modal Environment Setup ---
# Define the container image with system tools, Python dependencies, and local file mounts.
image = (
    modal.Image.debian_slim()
    .apt_install("procps")  # Required by inspect_ai for process management (provides 'pkill')
    .pip_install(
        "inspect_ai",
        "inspect_evals",
        "openai",
        "vllm" 
    )
    # Upload the local templates directory to the remote container, preserving the structure
    .add_local_dir("./src/eval/templates", remote_path="/root/src/eval/templates")
)

# Initialize the Modal application
app = modal.App("modal-eval-preflight", image=image)


# --- Remote Execution ---
@app.function(gpu="A100", timeout=10000)
def run():
    """Executes the Inspect AI evaluation on a remote A100 container."""
    # Guarded imports: Loaded only within the remote container
    from inspect_ai import eval as inspect_eval  
    from inspect_ai.util._display import init_display_type 
    import inspect_evals.pre_flight

    # Suppress heavy terminal UI banners during remote execution
    init_display_type("plain")

    # Configure vLLM arguments, pointing to the exact remote path of the Jinja template
    model_args = {
        'gpu_memory_utilization': 0.8,
        'chat_template': "/root/src/eval/templates/qwen3.jinja"
    }
    
    # Execute the evaluation task
    eval_out = inspect_eval(
        "inspect_evals/pre_flight",
        model="vllm/Qwen/Qwen3-4B", 
        model_args=model_args,
        score_display=False,
        timeout=10000,
        attempt_timeout=300,
        log_realtime=False,
        log_format='json',
        max_tokens=8000,
        max_connections=1,
    )
    
    # Catch and report internal evaluation failures (e.g., vLLM startup crashes)
    if eval_out[0].status == "error":
        print("\n[!] The evaluation crashed internally:")
        print(eval_out[0].error)
        return None
    
    # Parse and validate results safely
    assert len(eval_out) == 1, eval_out
    assert eval_out[0].results is not None, "Results are None despite no error status."
    assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
    
    # Extract evaluation metrics into a clean dictionary
    metrics = {k: v.value for k, v in eval_out[0].results.scores[0].metrics.items()}
    
    print("\nEvaluation Metrics for vllm/Qwen/Qwen3-4B:")
    print(json.dumps(metrics, indent=2))
    
    return metrics


@app.local_entrypoint()
def main():
    print("Submitting Qwen evaluation to Modal...")
    run.remote()
