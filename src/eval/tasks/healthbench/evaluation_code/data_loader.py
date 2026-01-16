"""Load and parse HealthBench Easy dataset for PostTrainBench evaluation."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


@dataclass
class RubricCriterion:
    """Single rubric criterion for grading."""
    criterion: str          # The criterion text
    points: int             # Weight/points for this criterion
    tags: List[str]         # Tags like ["level:example", "axis:accuracy"]
    
    @property
    def axis(self) -> str:
        """Extract axis from tags (e.g., 'axis:accuracy' -> 'accuracy')."""
        for tag in self.tags:
            if tag.startswith("axis:"):
                return tag.split(":")[1]
        return "unknown"
    
    @property
    def criterion_id(self) -> str:
        """Generate a criterion ID from the text."""
        return self.criterion[:50].replace(" ", "_").lower()


@dataclass
class HealthBenchExample:
    """Single HealthBench conversation with rubric."""
    prompt_id: str                       # Unique identifier
    prompt: List[dict]                   # [{"role": "user/assistant", "content": "..."}]
    rubrics: List[RubricCriterion]       # List of rubric criteria
    example_tags: List[str]              # Tags like ["theme:emergency_referrals"]
    
    @property
    def example_id(self) -> str:
        """Alias for prompt_id."""
        return self.prompt_id
    
    @property
    def conversation(self) -> List[dict]:
        """Alias for prompt."""
        return self.prompt
    
    @property
    def rubric_criteria(self) -> List[RubricCriterion]:
        """Alias for rubrics."""
        return self.rubrics
    
    @property
    def theme(self) -> str:
        """Extract theme from example_tags (e.g., 'theme:communication' -> 'communication')."""
        for tag in self.example_tags:
            if tag.startswith("theme:"):
                return tag.split(":")[1]
        return "unknown"
    
    @property
    def n_criteria(self) -> int:
        return len(self.rubrics)
    
    @property
    def max_possible_score(self) -> float:
        """Sum of positive point values."""
        return sum(c.points for c in self.rubrics if c.points > 0)


def parse_rubric(raw: dict) -> RubricCriterion:
    """Parse a raw rubric JSON object into RubricCriterion."""
    return RubricCriterion(
        criterion=raw["criterion"],
        points=raw["points"],
        tags=raw.get("tags", [])
    )


def parse_example(raw: dict) -> HealthBenchExample:
    """Parse a raw JSON object into HealthBenchExample."""
    return HealthBenchExample(
        prompt_id=raw["prompt_id"],
        prompt=raw["prompt"],
        rubrics=[parse_rubric(r) for r in raw["rubrics"]],
        example_tags=raw.get("example_tags", [])
    )


def load_healthbench_easy(
    limit: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> List[HealthBenchExample]:
    """Load HealthBench Easy dataset for PostTrainBench evaluation.
    
    The Easy dataset contains 245 examples designed for maximum base→instruct
    separation to demonstrate post-training progress.
    
    Filtering criteria (Easy V3):
    - Multi-turn conversations (≥5 turns) - forces context tracking
    - Completeness axis required - where base models score ~0%
    - ≤2 negative criteria - limits penalty exposure
    
    Expected performance:
    - Base models: 4.7-13.7% overall
    - Instruct models: 30.6-47.9% overall
    - Gap: 25-43 percentage points
    
    Args:
        limit: Maximum number of examples to load (for fast iteration)
        cache_dir: Directory containing data (defaults to ./data/)
    
    Returns:
        List of HealthBenchExample objects (245 total, or limited)
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data"
    
    cache_path = cache_dir / "healthbench_easy.jsonl"
    
    if not cache_path.exists():
        raise FileNotFoundError(
            f"HealthBench Easy dataset not found at {cache_path}. "
            "Please ensure healthbench_easy.jsonl is in the data/ directory."
        )
    
    print(f"[data] Loading HealthBench Easy from: {cache_path}")
    data = cache_path.read_text()
    
    examples = []
    for line in data.strip().split("\n"):
        if not line:
            continue
        raw = json.loads(line)
        example = parse_example(raw)
        examples.append(example)
        if limit and len(examples) >= limit:
            break
    
    print(f"[data] Loaded {len(examples)} examples")
    return examples


def get_theme_distribution(examples: List[HealthBenchExample]) -> dict:
    """Get distribution of examples by theme."""
    distribution = {}
    for ex in examples:
        distribution[ex.theme] = distribution.get(ex.theme, 0) + 1
    return distribution


def get_axis_distribution(examples: List[HealthBenchExample]) -> dict:
    """Get distribution of rubric criteria by axis."""
    distribution = {}
    for ex in examples:
        for rubric in ex.rubrics:
            axis = rubric.axis
            distribution[axis] = distribution.get(axis, 0) + 1
    return distribution
