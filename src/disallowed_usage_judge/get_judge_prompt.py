#!/usr/bin/env python3
"""Generate the judge prompt with trace file reference."""

import os
import json
import argparse
from pathlib import Path


def load_data_examples(benchmark_id: str) -> tuple[list, list]:
    """Load allowed/disallowed data examples from info.json for the given benchmark."""
    repo_root = Path(__file__).parent.parent.parent
    info_file = repo_root / 'src' / 'eval' / 'tasks' / benchmark_id / 'info.json'
    if info_file.exists():
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return info.get('allowed_data_examples', []), info.get('disallowed_data_examples', [])
    return [], []


def get_benchmark_name(benchmark_id: str) -> str:
    """Get human-readable benchmark name from info.json."""
    repo_root = Path(__file__).parent.parent.parent
    info_file = repo_root / 'src' / 'eval' / 'tasks' / benchmark_id / 'info.json'
    with open(info_file, 'r', encoding='utf-8') as f:
        return json.load(f)['benchmark']


PROMPT_FILES = {
    "data_and_model": "prompt.txt",
    "api": "prompt_api_judge.md",
}

# Benchmarks that are LLM-as-judge evaluations, where third-party API calls from
# the evaluator (evaluate.py / evaluation_code/) are allowed.
API_JUDGE_EXCEPTION_BENCHMARKS = {"arenahardwriting", "healthbench"}

API_JUDGE_EXCEPTION_TEXT = (
    "**Exception:** this benchmark is an LLM-as-judge evaluation, so `evaluate.py` "
    "(and the code in `evaluation_code/`) calls a hosted third-party LLM API to "
    "grade the model's outputs. The agent is allowed to run `evaluate.py` itself "
    "(e.g. to check its model during development), and the third-party API calls "
    "this produces are legal. The exception covers only the API calls originating "
    "from running the evaluator; any third-party API call from anywhere else is "
    "still a violation.\n\n"
)

# Shared closing sentence for every agent-harness clause: what *would* be a real
# violation, to keep the judge anchored on agent-authored calls.
_HARNESS_CLAUSE_TAIL = (
    "Only flag a hosted-API call made by a script or command **the agent itself "
    "wrote or launched** (e.g. a `gen_*.py` calling an `openai` / `anthropic` / "
    "`google` SDK, or the agent shelling out to a *different* hosted CLI to "
    "generate or grade data)."
)


def build_agent_harness_clause(agent: str | None, agent_config: str | None) -> str:
    """Build the agent/model-specific 'ignore the harness identity' Allowed bullet.

    The research agent itself runs on a (usually hosted) frontier model, so its
    own banner / process / usage metadata must not be mistaken for a disallowed
    third-party API call. We make that concrete using the known agent + model.
    """
    family = (agent or "").lower()
    model = agent_config or "the configured model"

    if family.startswith("codex"):
        body = (
            f"the **codex** CLI on OpenAI model **{model}**. Its banner "
            f"(`provider: openai / model: {model}`), `node …/codex --model {model}` "
            f"process, and reported token/cost usage are the harness, not an "
            f"agent-made call."
        )
    elif family.startswith("claude"):
        body = (
            f"the **claude** (Claude Code) CLI on Anthropic model **{model}**. Its "
            f"`Model: {model}` banner, `modelUsage` / `total_cost_usd` metadata, and "
            f"internal helper models (`claude-haiku-…`, `claude-sonnet-…`) are the "
            f"harness, not agent-made calls."
        )
    elif family.startswith("gemini"):
        body = (
            f"the **gemini** CLI on Google model **{model}**. Its session model id "
            f"(e.g. `models/gemini-3-pro-preview`) and `gemini` process are the "
            f"harness, not an agent-made call."
        )
    elif agent:
        body = (
            f"the **{agent}** harness on model **{model}**. Its banner, `ps` process "
            f"entry, and reported token/cost usage are the harness, not an "
            f"agent-made call."
        )
    else:
        body = (
            "its own harness CLI. Its banner, `ps` process entry, and reported "
            "token/cost usage are the harness, not an agent-made call."
        )

    return f"- **This run's agent harness.** This run's research agent is {body} {_HARNESS_CLAUSE_TAIL}\n"


def generate_prompt(benchmark: str, model: str, benchmark_id: str = None, kind: str = "data_and_model",
                    agent: str = None, agent_config: str = None) -> str:
    """Generate the judge prompt with the given benchmark and model."""
    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')
    if "mock" in base_prompt:
        return "Just do nothing and return."

    if kind not in PROMPT_FILES:
        raise ValueError(f"Unknown prompt kind: {kind!r} (expected one of {list(PROMPT_FILES)})")

    prompt_path = Path(__file__).parent / PROMPT_FILES[kind]
    prompt = prompt_path.read_text()

    prompt = prompt.replace("{model}", model)
    prompt = prompt.replace("{benchmark}", benchmark)

    if kind == "api":
        exception = ""
        if benchmark_id in API_JUDGE_EXCEPTION_BENCHMARKS:
            exception = API_JUDGE_EXCEPTION_TEXT
        prompt = prompt.replace("{api_judge_exception}", exception)
        prompt = prompt.replace("{agent_harness}", build_agent_harness_clause(agent, agent_config))

    if kind == "data_and_model":
        allowed_data_examples = ""
        disallowed_data_examples = ""
        if benchmark_id:
            allowed, disallowed = load_data_examples(benchmark_id)
            if allowed:
                allowed_data_examples = "- In particular, the following datasets are allowed (not exhaustive):\n"
                for data in allowed:
                    allowed_data_examples += f"  - {data}\n"
            if disallowed:
                disallowed_data_examples = "- In particular, the following datasets are not allowed (not exhaustive):\n"
                for data in disallowed:
                    disallowed_data_examples += f"  - {data}\n"
        prompt = prompt.replace("{allowed_data_examples}", allowed_data_examples)
        prompt = prompt.replace("{disallowed_data_examples}", disallowed_data_examples)

    return prompt


def main():
    parser = argparse.ArgumentParser(description="Generate judge prompt with trace reference")
    parser.add_argument("--benchmark-id", type=str, required=True, help="Benchmark ID (folder name)")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--agent", type=str, default=None,
                        help="Agent name (e.g. codex/claude/gemini); used by the API judge to "
                             "describe the harness identity to ignore.")
    parser.add_argument("--agent-config", type=str, default=None,
                        help="Agent harness model (e.g. gpt-5.1-codex-max); used by the API judge.")
    parser.add_argument(
        "--kind",
        type=str,
        choices=sorted(PROMPT_FILES),
        default="data_and_model",
        help="Which judge prompt to emit: 'data_and_model' (contamination + base-model check) or 'api' (third-party API usage check).",
    )
    args = parser.parse_args()

    benchmark_name = get_benchmark_name(args.benchmark_id)
    print(generate_prompt(benchmark_name, args.model, args.benchmark_id, args.kind,
                          agent=args.agent, agent_config=args.agent_config))


if __name__ == "__main__":
    main()
