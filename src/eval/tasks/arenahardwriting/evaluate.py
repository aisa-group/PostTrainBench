import os

import argparse
import atexit
import json
import math
import random
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import shortuuid
import tiktoken
from tqdm import tqdm

from evaluation_code.utils.add_markdown_info import count_markdown_elements, remove_pattern
from evaluation_code.utils.completion import (
    load_model_answers,
    load_questions,
    make_config,
)
from evaluation_code.utils.judge_utils import JUDGE_SETTINGS
from evaluation_code.show_result import load_judgments, print_leaderboard


API_MAX_RETRY = 3
API_RETRY_SLEEP = 5
DEFAULT_JUDGE_WORKERS = 64
VLLM_HEALTH_TIMEOUT = 600
VLLM_REQUEST_TIMEOUT = 300
VLLM_GENERATION_RETRY = 3

BENCHMARK = "arena-hard-v2.0"
JUDGE_MODEL = "gpt-5-mini"
REASONING_EFFORT = "medium"
JUDGE_CONFIG = "evaluation_code/config/arena-hard-v2.0.yaml"
JUDGE_MAX_COMPLETION = 49152
DATA_PATH = Path("evaluation_code/data/" + BENCHMARK)

def get_questions(args):
    data_dir = DATA_PATH 
    questions = load_questions(str(data_dir / "question.jsonl"))
    if args.limit is not None and args.limit != -1:
        random.Random(42).shuffle(questions)
        questions = questions[: args.limit]

    if args.limit == -1:
        random.Random(42).shuffle(questions)

    return questions


def _model_alias(model_path: str) -> str:
    if os.path.isdir(model_path):
        return Path(model_path).name
    return model_path.split("/")[-1]


def _find_available_port() -> int:
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
    def __init__(self, args, model_path: str):
        self.args = args
        self.model_path = model_path
        self.port: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None

    def start(self) -> int:
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
            "--api-key",
            os.environ.get("VLLM_API_KEY", ""),
        ]
        command.extend(template_args(self.args))

        self.process = subprocess.Popen(command)  # noqa: S603,S607
        self.port = port

        try:
            _wait_for_vllm_server(port, self.process)
        except Exception:
            self.stop(force=True)
            raise

        atexit.register(self.stop)
        return port

    def stop(self, force: bool = False) -> None:
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


def _make_metadata(answer: str) -> Dict:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    token_len = len(encoding.encode(answer, disallowed_special=()))
    metadata = {"token_len": token_len}

    markdown_info = count_markdown_elements(
        remove_pattern(answer, re.compile("```([^`]*)```")),
        suffix="",
    )
    metadata.update(markdown_info)
    return metadata


def generate_answers(args) -> tuple:
    """Generate answers and optionally save to disk.

    Returns:
        Tuple of (output_path or None, dict mapping uid to answer record)
    """
    data_dir = DATA_PATH
    output_dir = data_dir / "model_answer"
    output_path = output_dir / f"{args.model_alias}.jsonl"

    questions = get_questions(args)
    server = VLLMServer(args, args.model_path)
    print(f"[generate] Starting vLLM server for model {args.model_path}.")

    answers_dict: Dict[str, Dict] = {}

    try:
        port = server.start()
        endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
        session = requests.Session()
        vllm_api_key = os.environ.get("VLLM_API_KEY")
        if vllm_api_key:
            session.headers["Authorization"] = f"Bearer {vllm_api_key}"

        for question in tqdm(questions, desc="Generating answers"):
            payload = {
                "model": args.model_path,
                "messages": [
                    {"role": "user", "content": question["prompt"]},
                ],
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
                            f"Failed to generate answer for uid {question['uid']} after {VLLM_GENERATION_RETRY} attempts"
                        ) from err
                    backoff = 2 ** attempt
                    print(
                        f"[generate] Error from vLLM (attempt {attempt}/{VLLM_GENERATION_RETRY}): {err}. Retrying in {backoff}s."
                    )
                    time.sleep(backoff)

            if answer_text is None:
                raise RuntimeError(f"No answer generated for uid {question['uid']}.")

            if answer_text.startswith("<think>") and ("</think>" in answer_text):
                answer_text = answer_text.split("</think>", maxsplit=1)[1]
                answer_text = answer_text.strip()

            messages = [
                {"role": "user", "content": question["prompt"]},
                {"role": "assistant", "content": {"answer": answer_text}},
            ]

            record = {
                "uid": question["uid"],
                "ans_id": shortuuid.uuid(),
                "model": args.model_alias,
                "messages": messages,
                "tstamp": time.time(),
                "metadata": _make_metadata(answer_text),
            }
            answers_dict[question["uid"]] = record

        if args.store_outputs:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[generate] Writing answers to {output_path}")
            with open(output_path, "w", encoding="utf-8") as fout:
                for record in answers_dict.values():
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            return output_path, answers_dict

        return None, answers_dict
    finally:
        server.stop()


def call_openai(messages: List[Dict]):
    import openai

    client = openai.OpenAI()
    request_kwargs = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "max_completion_tokens": JUDGE_MAX_COMPLETION,
    }
    if REASONING_EFFORT is not None:
        request_kwargs["reasoning_effort"] = REASONING_EFFORT

    for attempt in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(**request_kwargs)
            return {
                "answer": completion.choices[0].message.content,
            }
        except openai.BadRequestError as err:
            if "reasoning" in str(err).lower() and "reasoning_effort" in request_kwargs:
                print("[judge] reasoning_effort not supported; retrying without it.")
                request_kwargs.pop("reasoning_effort", None)
                continue
            wait_time = API_RETRY_SLEEP * (2**attempt)
            print(f"[judge] OpenAI API error ({type(err).__name__}): {err}. Retry in {wait_time}s.")
            time.sleep(wait_time)
        except Exception as err:  # pylint: disable=broad-except
            wait_time = API_RETRY_SLEEP * (2**attempt)
            print(f"[judge] OpenAI API error ({type(err).__name__}): {err}. Retry in {wait_time}s.")
            time.sleep(wait_time)
    print("[judge] Exhausted retries; returning None.")
    return None


def get_score(judgment: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        compiled = re.compile(pattern)
        matches = [
            m for m in compiled.findall(judgment.upper()) if isinstance(m, str) and m
        ]
        if matches:
            return matches[-1].strip("\n")
        if matches and isinstance(matches[-1], tuple):
            for item in matches[-1]:
                if item:
                    return item.strip("\n")
    return None


def judge_answers(args, candidate_answers: Optional[Dict[str, Dict]] = None) -> tuple:
    """Judge model answers and optionally save to disk.

    Args:
        args: Command-line arguments
        candidate_answers: Optional dict mapping uid to answer record (for in-memory answers)

    Returns:
        Tuple of (output_path or None, list of judgment records)
    """
    judge_config = make_config(JUDGE_CONFIG)
    prompt_template = judge_config["prompt_template"]
    regex_patterns = judge_config["regex_patterns"]

    data_dir = DATA_PATH
    answer_dir = data_dir / "model_answer"
    judgment_dir = data_dir / "model_judgment" / JUDGE_MODEL
    output_path = judgment_dir / f"{args.model_alias}.jsonl"

    if "OPENAI_API_KEY" not in os.environ:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please export your OpenAI API key before judging."
        )

    questions = get_questions(args)

    model_answers = load_model_answers(str(answer_dir))

    if candidate_answers is not None:
        model_answers[args.model_alias] = candidate_answers

    if args.model_alias not in model_answers:
        raise FileNotFoundError(
            f"Cannot find answers for model '{args.model_alias}' in {answer_dir}."
        )

    results: List[Optional[Dict]] = [None] * len(questions)

    with ThreadPoolExecutor(max_workers=args.judge_workers) as executor:
        futures = {
            executor.submit(
                _judge_single_question,
                question,
                args,
                model_answers,
                prompt_template,
                regex_patterns,
            ): idx
            for idx, question in enumerate(questions)
        }

        with tqdm(total=len(futures), desc="Judging answers") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    pbar.update(1)
                    raise
                pbar.update(1)

    if args.store_outputs:
        judgment_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            for record in results:
                if record is None:
                    continue
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output_path, results

    return None, results


def _judge_single_question(
    question: Dict,
    args,
    model_answers: Dict[str, Dict],
    prompt_template: str,
    regex_patterns: Iterable[str],
):
    uid = question["uid"]
    category = question["category"]

    baseline_model = JUDGE_SETTINGS[category]["baseline"]
    if baseline_model not in model_answers:
        raise FileNotFoundError(
            f"Baseline model '{baseline_model}' answers not found in data/model_answer"
        )

    candidate_answer = model_answers[args.model_alias].get(uid)
    baseline_answer = model_answers[baseline_model].get(uid)

    if candidate_answer is None:
        print(f"[judge] Candidate missing answer for UID {uid}. Skipping.")
        return None
    if baseline_answer is None:
        print(f"[judge] Baseline missing answer for UID {uid}. Skipping.")
        return None

    prompt_args = {
        "QUESTION": question["prompt"],
        "ANSWER_A": baseline_answer["messages"][-1]["content"]["answer"],
        "ANSWER_B": candidate_answer["messages"][-1]["content"]["answer"],
    }
    user_prompt = prompt_template.format(**prompt_args)
    messages = [
        {
            "role": "system",
            "content": JUDGE_SETTINGS[category]["system_prompt"],
        },
        {"role": "user", "content": user_prompt},
    ]

    result_ab = call_openai(
        messages=messages,
    )
    score_ab = get_score(result_ab["answer"], regex_patterns) if result_ab else None

    prompt_args_swap = {
        "QUESTION": question["prompt"],
        "ANSWER_A": candidate_answer["messages"][-1]["content"]["answer"],
        "ANSWER_B": baseline_answer["messages"][-1]["content"]["answer"],
    }
    user_prompt_swap = prompt_template.format(**prompt_args_swap)
    messages_swap = [
        {
            "role": "system",
            "content": JUDGE_SETTINGS[category]["system_prompt"],
        },
        {"role": "user", "content": user_prompt_swap},
    ]

    result_ba = call_openai(
        messages=messages_swap,
    )
    score_ba = get_score(result_ba["answer"], regex_patterns) if result_ba else None

    return {
        "uid": uid,
        "category": category,
        "judge": JUDGE_MODEL,
        "model": candidate_answer["model"],
        "baseline": baseline_answer["model"],
        "games": [
            {"score": score_ab, "judgment": result_ab, "prompt": messages},
            {"score": score_ba, "judgment": result_ba, "prompt": messages_swap},
        ],
    }


def _compute_metrics(battles) -> Dict[str, float]:
    scores = battles["scores"].astype(float)
    num_samples = len(scores)

    if num_samples == 0:
        return {"accuracy": 0.0, "stderr": 0.0}

    accuracy = float(scores.mean())
    if num_samples == 1:
        stderr = 0.0
    else:
        std_dev = float(scores.std(ddof=1))
        stderr = std_dev / math.sqrt(num_samples) if not math.isnan(std_dev) else 0.0

    return {"accuracy": accuracy, "stderr": stderr}


def _judgments_to_battles(judgments: List[Optional[Dict]]):
    """Convert in-memory judgment records to battles DataFrame."""
    import pandas as pd

    battles_data = []
    for record in judgments:
        if record is None:
            continue

        games = record.get("games", [])
        if len(games) < 2:
            continue

        score_ab = games[0].get("score")
        score_ba = games[1].get("score")

        if score_ab is None or score_ba is None:
            continue

        score_map = {"A": 1, "B": 0, "TIE": 0.5, "A>>B": 1, "A>B": 1, "B>A": 0, "B>>A": 0}
        score_ab_val = score_map.get(score_ab.upper(), 0.5)
        score_ba_val = score_map.get(score_ba.upper(), 0.5)
        # Game 1: A=baseline, B=candidate. Flip to get candidate's win probability.
        # Game 2: A=candidate, B=baseline. score_ba_val is already candidate's win probability.
        score_ab_flipped = 1 - score_ab_val

        avg_score = (score_ab_flipped + score_ba_val) / 2

        battles_data.append({
            "uid": record["uid"],
            "category": record["category"],
            "model": record["model"],
            "baseline": record["baseline"],
            "scores": avg_score,
        })

    return pd.DataFrame(battles_data)


def summarize_results(model_alias: str, judgments: Optional[List[Optional[Dict]]] = None) -> Optional[Dict[str, float]]:
    if judgments is not None:
        battles = _judgments_to_battles(judgments)
    else:
        try:
            battles = load_judgments([JUDGE_MODEL], BENCHMARK)
        except FileNotFoundError:
            print("[summary] No judgments found for summary.")
            return {"accuracy": 0.0, "stderr": 0.0}

    battles = battles[battles.model == model_alias].reset_index(drop=True)

    if battles.empty:
        print(f"[summary] No battles recorded for model '{model_alias}'.")
        return {"accuracy": 0.0, "stderr": 0.0}

    categories = battles.category.unique().tolist()
    for category in categories:
        print_leaderboard(battles[battles.category == category].reset_index(drop=True), category)

    return _compute_metrics(battles)

def main():
    parser = argparse.ArgumentParser(description="Run Arena-Hard evaluation for local or Hugging Face models.")
    parser.add_argument("--model-path", required=True, help="Hugging Face model ID or local path.")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    # this is a good limit for this task, just keep it like that (or use less in case you want faster tests)
    parser.add_argument("--limit", type=int, default=32, help="Limit number of questions for quicker runs.")
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=DEFAULT_JUDGE_WORKERS,
        help="Number of concurrent judge jobs to run in parallel.",
    )
    parser.add_argument(
        '--templates-dir',
        type=str,
        default='templates/',
    )
    parser.add_argument(
        '--json-output-file',
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help="Skip answer generation and use existing answers from model_answer/.",
    )
    parser.add_argument(
        '--store-outputs',
        action='store_true',
        help="Store model answers and judgments to disk (default: off).",
    )
    args = parser.parse_args()

    model_alias = _model_alias(args.model_path)
    args.model_alias = model_alias

    candidate_answers = None


    if args.skip_generation:
        ans_path = DATA_PATH / "model_answer" / f"{model_alias}.jsonl"
        if ans_path.exists():
            print(f"[skip] Skipping answer generation, loading existing answers from {ans_path}")
            candidate_answers = {}
            with open(ans_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    candidate_answers[record["uid"]] = record
        else:
            print(f"[skip] File {ans_path} not found, generating answers instead")
            ans_path, candidate_answers = generate_answers(args)
            if ans_path:
                print(f"[done] Answers saved to {ans_path}")
            else:
                print("[done] Answers generated (not saved to disk)")
    else:
        ans_path, candidate_answers = generate_answers(args)
        if ans_path:
            print(f"[done] Answers saved to {ans_path}")
        else:
            print("[done] Answers generated (not saved to disk)")

    judge_path, judgments = judge_answers(args, candidate_answers)
    if judge_path:
        print(f"[done] Judgments saved to {judge_path}")
    else:
        print("[done] Judgments generated (not saved to disk)")

    metrics = summarize_results(model_alias, judgments if not args.store_outputs else None)

    if args.json_output_file is not None and metrics is not None:
        with open(args.json_output_file, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)
        print(f"[done] Metrics saved to {args.json_output_file}")
    if metrics is None:
        print("Failed to compute metrics.")
    
    print("Score (winrate) is:", metrics['accuracy'])

def model_type(args) -> str:
    if 'qwen' in args.model_path.lower():
        return 'qwen'
    if 'llama' in args.model_path.lower():
        return 'llama'
    if 'gemma' in args.model_path.lower():
        return 'gemma'
    if 'smollm' in args.model_path.lower():
        return 'smollm'

    with open(os.path.join(args.model_path, "config.json"), 'r') as f:
        config = json.load(f)
    architecture = config['architectures'][0].lower()
    if 'gemma' in architecture:
        return 'gemma'
    if 'llama' in architecture:
        return 'llama'
    if 'qwen' in architecture:
        return 'qwen'
    if 'smollm' in architecture:
        return 'smollm'
    raise ValueError(architecture)

def template_args(args) -> list:
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
        raise ValueError(model_type_str)
    return [
        '--chat-template', os.path.join(args.templates_dir, template)
    ]
    

if __name__ == "__main__":
    main()
