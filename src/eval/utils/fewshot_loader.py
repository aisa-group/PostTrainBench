"""
1. Load few-shot examples from JSON files
2. Detect if a model is a base model (pre-trained) vs instruction-tuned
3. Format few-shot examples into prompts for different benchmark types
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Directory containing few-shot example JSON files
FEWSHOT_DIR = Path(__file__).parent.parent / "fewshot_examples"


def is_base_model(model_path: str) -> bool:
    """
    Detect if a model is a base (pre-trained) model vs instruction-tuned.

    Base models benefit from few-shot examples since they haven't been
    trained to follow instructions in a 0-shot format.

    Args:
        model_path: Path or HuggingFace model identifier

    Returns:
        True if the model appears to be a base model, False if instruction-tuned
    """
    model_path_lower = model_path.lower()

    # Instruction-tuned model indicators (return False for these)
    instruct_indicators = [
        '-it', '-instruct', '-chat', '-sft',
        '/instruct', '/chat', '-rlhf', '-dpo'
    ]

    for indicator in instruct_indicators:
        if indicator in model_path_lower:
            return False

    # Base model indicators (return True for these)
    base_indicators = ['-pt', '-base', '_base', 'base-', '/base']

    for indicator in base_indicators:
        if indicator in model_path_lower:
            return True

    # If no clear indicator, assume it's an instruct model
    return False


def should_use_fewshot(fewshot_mode: str, model_path: str) -> bool:
    """
    Determine if few-shot examples should be used based on mode and model type.

    Args:
        fewshot_mode: 'auto', 'always', or 'never'
        model_path: Path or HuggingFace model identifier

    Returns:
        True if few-shot should be used, False otherwise
    """
    if fewshot_mode == 'always':
        return True
    elif fewshot_mode == 'never':
        return False
    else:  # auto
        return is_base_model(model_path)


def load_fewshot_examples(benchmark: str) -> Dict[str, Any]:
    """
    Load few-shot examples for a specific benchmark.

    Args:
        benchmark: Name of the benchmark (e.g., 'aime2025', 'gsm8k')

    Returns:
        Dictionary containing benchmark info and examples list
    """
    filepath = FEWSHOT_DIR / f"{benchmark}.json"

    if not filepath.exists():
        return {"benchmark": benchmark, "examples": []}

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_math_fewshot_prompt(
    examples: List[Dict[str, str]],
    num_examples: Optional[int] = None
) -> str:
    """
    Format few-shot examples for math reasoning tasks (AIME, GSM8K).

    Args:
        examples: List of example dicts with 'problem', 'reasoning', 'answer' keys
        num_examples: Number of examples to include (None = all)

    Returns:
        Formatted string with few-shot examples
    """
    if num_examples is not None:
        examples = examples[:num_examples]

    formatted_parts = []

    for i, ex in enumerate(examples, 1):
        part = f"Example {i}:\n"
        part += f"Problem: {ex['problem']}\n\n"
        part += f"Solution:\n{ex['reasoning']}\n\n"
        part += f"ANSWER: {ex['answer']}"
        formatted_parts.append(part)

    return "\n\n" + "=" * 50 + "\n\n".join(formatted_parts) + "\n\n" + "=" * 50 + "\n\n"


def format_mcq_fewshot_prompt(
    examples: List[Dict[str, Any]],
    num_examples: Optional[int] = None
) -> str:
    """
    Format few-shot examples for multiple choice tasks (GPQA).

    Args:
        examples: List of example dicts with 'question', 'choices', 'reasoning', 'answer' keys
        num_examples: Number of examples to include (None = all)

    Returns:
        Formatted string with few-shot examples
    """
    if num_examples is not None:
        examples = examples[:num_examples]

    formatted_parts = []

    for i, ex in enumerate(examples, 1):
        part = f"Example {i}:\n"
        part += f"Question: {ex['question']}\n"

        # Format choices
        for j, choice in enumerate(ex['choices']):
            letter = chr(ord('A') + j)
            part += f"({letter}) {choice}\n"

        part += f"\nReasoning: {ex['reasoning']}\n"
        part += f"Answer: {ex['answer']}"
        formatted_parts.append(part)

    return "\n\n" + "-" * 50 + "\n\n".join(formatted_parts) + "\n\n" + "-" * 50 + "\n\n"


def get_fewshot_prompt(
    benchmark: str,
    num_examples: Optional[int] = None
) -> str:
    """
    Get formatted few-shot prompt for a benchmark.

    Args:
        benchmark: Name of the benchmark
        num_examples: Number of examples to include (None = all available)

    Returns:
        Formatted few-shot prompt string, or empty string if no examples
    """
    data = load_fewshot_examples(benchmark)
    examples = data.get("examples", [])

    if not examples:
        return ""

    format_type = data.get("format", "math_reasoning")

    if format_type == "math_reasoning":
        return format_math_fewshot_prompt(examples, num_examples)
    elif format_type == "multiple_choice":
        return format_mcq_fewshot_prompt(examples, num_examples)
    else:
        # Default to math reasoning format
        return format_math_fewshot_prompt(examples, num_examples)
