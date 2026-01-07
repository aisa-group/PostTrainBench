"""Load and parse HealthBench Hard dataset."""

import json
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

# HealthBench Hard URL (1,000 examples)
HEALTHBENCH_HARD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"


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


def load_healthbench_hard(
    limit: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> List[HealthBenchExample]:
    """Load HealthBench Hard dataset.
    
    Downloads from OpenAI blob storage and caches locally.
    
    Args:
        limit: Maximum number of examples to load (for fast iteration)
        use_cache: Whether to use cached data if available
        cache_dir: Directory to cache data (defaults to ./data/)
    
    Returns:
        List of HealthBenchExample objects
    """
    if cache_dir is None:
        # Default to data/ subdirectory relative to this file
        cache_dir = Path(__file__).parent.parent / "data"
    
    cache_path = cache_dir / "healthbench_hard.jsonl"
    
    # Check cache first
    if use_cache and cache_path.exists():
        print(f"[data] Loading from cache: {cache_path}")
        data = cache_path.read_text()
    else:
        # Download from blob storage
        print(f"[data] Downloading HealthBench Hard from {HEALTHBENCH_HARD_URL}")
        response = requests.get(HEALTHBENCH_HARD_URL, timeout=60)
        response.raise_for_status()
        data = response.text
        
        # Cache locally
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(data)
        print(f"[data] Cached to {cache_path}")
    
    # Parse JSONL
    examples = []
    for line in data.strip().split("\n"):
        if not line:
            continue
        raw = json.loads(line)
        example = parse_example(raw)
        examples.append(example)
        
        if limit and len(examples) >= limit:
            break
    
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

