#!/usr/bin/env python3
"""HealthBench evaluation for PostTrainBench.

Evaluates a model on physician-curated medical conversations using
LLM-as-judge grading against rubric criteria.

The Easy dataset contains 450 examples filtered for meaningful base→instruct
separation to demonstrate post-training progress.

Expected performance:
- Base models: 5-17% overall
- Instruct models: 26-40% overall
- Gap: ~20-23 percentage points

Usage:
    python evaluate.py --model-path Qwen/Qwen3-1.7B-Base --limit 5
    python evaluate.py --model-path final_model/ --json-output-file results.json
"""

import os
import argparse
import atexit
import json
import random
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

from evaluation_code.data_loader import load_healthbench_easy, HealthBenchExample
from evaluation_code.grader import grade_examples_parallel, ExampleResult
from evaluation_code.scoring import aggregate_scores, BenchmarkResult


# Constants
API_MAX_RETRY = 3
API_RETRY_SLEEP = 5
DEFAULT_JUDGE_WORKERS = 64
VLLM_HEALTH_TIMEOUT = 600
VLLM_REQUEST_TIMEOUT = 300
VLLM_GENERATION_RETRY = 3

JUDGE_MODEL = "gpt-5-mini"
BENCHMARK_NAME = "HealthBench"


def _model_alias(model_path: str) -> str:
    """Extract model alias from path."""
    if os.path.isdir(model_path):
        return Path(model_path).name
    return model_path.split("/")[-1]


def _find_available_port() -> int:
    """Find an available port for vLLM server."""
    for _ in range(100):
        port = random.randint(20000, 65000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Unable to find an available port for vLLM server.")


def _wait_for_vllm_server(port: int, process: subprocess.Popen) -> None:
    """Wait for vLLM server to become ready."""
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + VLLM_HEALTH_TIMEOUT

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM server exited unexpectedly while starting.")
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)

    raise TimeoutError("Timed out waiting for vLLM server to become ready.")


class VLLMServer:
    """Manage vLLM server lifecycle."""
    
    def __init__(self, args, model_path: str):
        self.args = args
        self.model_path = model_path
        self.port: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> int:
        """Start vLLM server and return port."""
        if self.process is not None:
            raise RuntimeError("vLLM server already started.")

        port = _find_available_port()
        command = [
            "vllm",
            "serve",
            self.model_path,
            "--port",
            str(port),
            "--trust-remote-code",
        ]
        command.extend(template_args(self.args))

        self.process = subprocess.Popen(command)
        self.port = port

        try:
            _wait_for_vllm_server(port, self.process)
        except Exception:
            self.stop(force=True)
            raise

        atexit.register(self.stop)
        return port

    def stop(self, force: bool = False) -> None:
        """Stop vLLM server."""
        if self.process is None:
            return
        if self.process.poll() is None:
            if force:
                self.process.kill()
            else:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        self.process = None
        self.port = None


def model_type(args) -> str:
    """Detect model type from path."""
    if 'qwen' in args.model_path.lower():
        return 'qwen'
    if 'llama' in args.model_path.lower():
        return 'llama'
    if 'gemma' in args.model_path.lower():
        return 'gemma'
    if 'smollm' in args.model_path.lower():
        return 'smollm'

    # Try to read from config.json
    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        architecture = config.get('architectures', [''])[0].lower()
        if 'gemma' in architecture:
            return 'gemma'
        if 'llama' in architecture:
            return 'llama'
        if 'qwen' in architecture:
            return 'qwen'
        if 'smollm' in architecture:
            return 'smollm'
    
    # Default to qwen (no template override)
    return 'qwen'


def template_args(args) -> list:
    """Get vLLM template arguments based on model type."""
    model_type_str = model_type(args)
    if model_type_str == 'qwen':
        return []
    elif model_type_str == 'llama':
        template = 'llama3.jinja'
    elif model_type_str == 'gemma':
        template = 'gemma3.jinja'
    elif model_type_str == 'smollm':
        template = 'smollm.jinja'
    else:
        return []
    
    return ['--chat-template', os.path.join(args.templates_dir, template)]


def generate_answers(
    args,
    examples: List[HealthBenchExample]
) -> List[str]:
    """Generate model responses for all examples."""
    server = VLLMServer(args, args.model_path)
    print(f"[generate] Starting vLLM server for model {args.model_path}")

    try:
        port = server.start()
        endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        session = requests.Session()

        responses = []
        print(f"[generate] Generating answers for {len(examples)} examples")
        
        for example in tqdm(examples, desc="Generating answers"):
            # Build messages from conversation
            messages = example.conversation.copy()
            
            payload = {
                "model": args.model_path,
                "messages": messages,
                "max_tokens": args.max_new_tokens,
            }

            answer_text: Optional[str] = None
            for attempt in range(1, VLLM_GENERATION_RETRY + 1):
                try:
                    response = session.post(
                        endpoint,
                        json=payload,
                        timeout=VLLM_REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    completion = response.json()
                    choices = completion.get("choices", [])
                    if not choices:
                        raise ValueError("vLLM response missing 'choices'.")
                    message = choices[0].get("message")
                    if not message or "content" not in message:
                        raise ValueError("vLLM response missing message content.")
                    answer_text = message["content"].strip()
                    break
                except (requests.RequestException, ValueError) as err:
                    if attempt == VLLM_GENERATION_RETRY:
                        raise RuntimeError(
                            f"Failed to generate answer for {example.example_id} after {VLLM_GENERATION_RETRY} attempts"
                        ) from err
                    backoff = 2 ** attempt
                    print(f"[generate] Error (attempt {attempt}): {err}. Retrying in {backoff}s.")
                    time.sleep(backoff)

            if answer_text is None:
                raise RuntimeError(f"No answer generated for {example.example_id}")

            # Strip thinking tags if present (for reasoning models)
            if answer_text.startswith("<think>"):
                answer_text = answer_text.split("</think>", maxsplit=1)[-1].strip()

            responses.append(answer_text)

        return responses
    finally:
        server.stop()


def _compute_metrics(results: List[ExampleResult], examples: List[HealthBenchExample]) -> Dict:
    """Compute final metrics."""
    benchmark_result = aggregate_scores(results, examples)
    
    return {
        "accuracy": benchmark_result.accuracy,
        "stderr": benchmark_result.stderr,
        "n_examples": benchmark_result.n_examples,
        "total_grader_calls": benchmark_result.total_grader_calls,
        "by_theme": benchmark_result.by_theme,
        "by_axis": benchmark_result.by_axis,
    }


def main():
    parser = argparse.ArgumentParser(description="Run HealthBench evaluation.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Hugging Face model ID or local path."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per response."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples for quicker runs."
    )
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=DEFAULT_JUDGE_WORKERS,
        help="Number of concurrent judge jobs to run in parallel."
    )
    parser.add_argument(
        '--templates-dir',
        type=str,
        default='templates/',
        help="Directory containing chat templates."
    )
    parser.add_argument(
        '--json-output-file',
        type=str,
        default=None,
        help="Optional path to output the metrics as a JSON file."
    )
    parser.add_argument(
        '--grader-model',
        type=str,
        default=JUDGE_MODEL,
        help=f"Model for grading (default: {JUDGE_MODEL})."
    )
    args = parser.parse_args()

    model_alias = _model_alias(args.model_path)

    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please export your OpenAI API key before running."
        )

    # Load data
    print(f"[data] Loading {BENCHMARK_NAME} dataset...")
    examples = load_healthbench_easy(limit=args.limit)

    # Generate answers
    responses = generate_answers(args, examples)
    print(f"[generate] Generated {len(responses)} responses")

    # Grade responses
    print(f"[judge] Grading responses with {args.grader_model}...")
    pbar = tqdm(total=len(examples), desc="Judging answers")
    
    def update_progress(completed, total):
        pbar.n = completed
        pbar.refresh()
    
    results = grade_examples_parallel(
        examples=examples,
        responses=responses,
        grader_model=args.grader_model,
        example_workers=min(4, len(examples)),
        criteria_workers=8,
        max_concurrent_requests=args.judge_workers,
        progress_callback=update_progress
    )
    pbar.close()

    # Compute metrics
    metrics = _compute_metrics(results, examples)
    metrics["benchmark"] = BENCHMARK_NAME

    # Print summary
    print(f"\n[done] {BENCHMARK_NAME} Evaluation Complete")
    print(f"  Model: {model_alias}")
    print(f"  Examples: {metrics['n_examples']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} (±{metrics['stderr']:.4f})")
    print(f"  Grader calls: {metrics['total_grader_calls']}")
    
    if metrics['by_theme']:
        print(f"\n  By Theme:")
        for theme, score in sorted(metrics['by_theme'].items()):
            print(f"    {theme}: {score:.4f}")
    
    if metrics['by_axis']:
        print(f"\n  By Axis:")
        for axis, score in sorted(metrics['by_axis'].items()):
            print(f"    {axis}: {score:.4f}")

    # Save metrics
    if args.json_output_file is not None:
        with open(args.json_output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[done] Metrics saved to {args.json_output_file}")


if __name__ == "__main__":
    main()
