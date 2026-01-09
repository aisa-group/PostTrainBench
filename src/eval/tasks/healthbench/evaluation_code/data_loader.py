"""Load and parse HealthBench datasets (Hard and Easy subsets)."""

import json
import random
import requests
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

# HealthBench URLs
HEALTHBENCH_HARD_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl"
HEALTHBENCH_FULL_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"

# Easy subset parameters
EASY_MAX_NEGATIVE_CRITERIA = 2  # Maximum negative criteria per example
EASY_TARGET_SIZE = 1000         # Target number of examples after subsampling


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


def load_healthbench_easy(
    limit: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None,
    seed: int = 42
) -> List[HealthBenchExample]:
    """Load HealthBench Easy dataset.
    
    Easy subset is created by:
    1. Starting from full HealthBench (5,000 examples)
    2. Excluding examples in the Hard subset
    3. Filtering to examples with ≤2 negative criteria
    4. Stratified subsampling to 1,000 examples (preserving theme distribution)
    
    This yields examples that are inherently easier for models to score well on,
    targeting ~40-50% base model performance vs ~0% on Hard.
    
    Args:
        limit: Maximum number of examples to load (for fast iteration)
        use_cache: Whether to use cached data if available
        cache_dir: Directory to cache data (defaults to ./data/)
        seed: Random seed for reproducible subsampling
    
    Returns:
        List of HealthBenchExample objects
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data"
    
    cache_path = cache_dir / "healthbench_easy.jsonl"
    
    # Check cache first
    if use_cache and cache_path.exists():
        print(f"[data] Loading Easy subset from cache: {cache_path}")
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
        return examples
    
    # Need to create Easy subset from Full dataset
    print(f"[data] Creating Easy subset from full HealthBench...")
    
    # Download full dataset
    full_cache = cache_dir / "healthbench_full.jsonl"
    if full_cache.exists():
        print(f"[data] Loading full dataset from cache: {full_cache}")
        full_data = full_cache.read_text()
    else:
        print(f"[data] Downloading full HealthBench from {HEALTHBENCH_FULL_URL}")
        response = requests.get(HEALTHBENCH_FULL_URL, timeout=120)
        response.raise_for_status()
        full_data = response.text
        full_cache.parent.mkdir(parents=True, exist_ok=True)
        full_cache.write_text(full_data)
        print(f"[data] Cached full dataset to {full_cache}")
    
    # Parse full dataset
    full_examples_raw = []
    for line in full_data.strip().split("\n"):
        if not line:
            continue
        full_examples_raw.append(json.loads(line))
    print(f"[data] Full dataset: {len(full_examples_raw)} examples")
    
    # Get Hard subset IDs to exclude
    hard_cache = cache_dir / "healthbench_hard.jsonl"
    if hard_cache.exists():
        hard_data = hard_cache.read_text()
    else:
        print(f"[data] Downloading Hard subset to get exclusion IDs...")
        response = requests.get(HEALTHBENCH_HARD_URL, timeout=60)
        response.raise_for_status()
        hard_data = response.text
        hard_cache.parent.mkdir(parents=True, exist_ok=True)
        hard_cache.write_text(hard_data)
    
    hard_ids = set()
    for line in hard_data.strip().split("\n"):
        if line:
            hard_ids.add(json.loads(line)["prompt_id"])
    print(f"[data] Excluding {len(hard_ids)} Hard examples")
    
    # Filter: non-hard + max 2 negative criteria
    candidates = []
    for raw in full_examples_raw:
        if raw["prompt_id"] in hard_ids:
            continue
        n_negative = sum(1 for r in raw["rubrics"] if r["points"] < 0)
        if n_negative <= EASY_MAX_NEGATIVE_CRITERIA:
            candidates.append(raw)
    
    print(f"[data] After filtering (≤{EASY_MAX_NEGATIVE_CRITERIA} negative criteria): {len(candidates)} examples")
    
    # Stratified subsample to preserve theme distribution
    rng = random.Random(seed)
    
    # Group by theme
    by_theme = defaultdict(list)
    for raw in candidates:
        theme = "unknown"
        for tag in raw.get("example_tags", []):
            if tag.startswith("theme:"):
                theme = tag.split(":")[1]
                break
        by_theme[theme].append(raw)
    
    print(f"[data] Theme distribution before sampling: {dict((k, len(v)) for k, v in by_theme.items())}")
    
    # Calculate proportional sample sizes
    total_candidates = len(candidates)
    target_size = min(EASY_TARGET_SIZE, total_candidates)
    
    sampled = []
    for theme, theme_examples in by_theme.items():
        # Proportional allocation
        n_sample = int(len(theme_examples) / total_candidates * target_size)
        n_sample = max(1, n_sample)  # At least 1 per theme
        n_sample = min(n_sample, len(theme_examples))  # Don't oversample
        
        rng.shuffle(theme_examples)
        sampled.extend(theme_examples[:n_sample])
    
    # If we're short of target, add more randomly
    if len(sampled) < target_size:
        remaining = [ex for ex in candidates if ex not in sampled]
        rng.shuffle(remaining)
        sampled.extend(remaining[:target_size - len(sampled)])
    
    # If we're over target (due to rounding), trim
    if len(sampled) > target_size:
        rng.shuffle(sampled)
        sampled = sampled[:target_size]
    
    print(f"[data] Easy subset: {len(sampled)} examples")
    
    # Cache the Easy subset
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        for raw in sampled:
            f.write(json.dumps(raw) + "\n")
    print(f"[data] Cached Easy subset to {cache_path}")
    
    # Parse and return
    examples = [parse_example(raw) for raw in sampled]
    
    if limit:
        examples = examples[:limit]
    
    return examples


def load_healthbench(
    subset: str = "hard",
    limit: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: Optional[Path] = None
) -> List[HealthBenchExample]:
    """Load HealthBench dataset (unified interface).
    
    Args:
        subset: Which subset to load ("hard" or "easy")
        limit: Maximum number of examples
        use_cache: Whether to use cached data
        cache_dir: Cache directory
    
    Returns:
        List of HealthBenchExample objects
    """
    if subset == "hard":
        return load_healthbench_hard(limit=limit, use_cache=use_cache, cache_dir=cache_dir)
    elif subset == "easy":
        return load_healthbench_easy(limit=limit, use_cache=use_cache, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown subset: {subset}. Must be 'hard' or 'easy'.")


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

