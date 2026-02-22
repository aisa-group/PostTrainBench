"""Score aggregation for HealthBench evaluation."""

import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Dict, TYPE_CHECKING

from .grader import ExampleResult, GradingResult

if TYPE_CHECKING:
    from .data_loader import HealthBenchExample


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result."""
    accuracy: float           # Primary metric (0-1 normalized score, clipped)
    stderr: float             # Bootstrap standard error
    n_examples: int
    by_theme: Dict[str, float]  # theme -> avg score
    by_axis: Dict[str, float]   # axis -> avg score
    total_grader_calls: int


def aggregate_scores(
    results: List[ExampleResult],
    examples: Optional[List["HealthBenchExample"]] = None
) -> BenchmarkResult:
    """Aggregate individual example results into benchmark score.
    
    Follows HealthBench methodology:
    - Overall score: mean of per-example normalized scores, clipped to [0, 1]
    - By-theme: average normalized_score for examples with each theme
    - By-axis: score computed per-axis using only criteria with that axis
    
    Args:
        results: List of ExampleResult from grading
        examples: Original HealthBenchExample objects (for theme/axis metadata)
    
    Returns:
        BenchmarkResult with overall and stratified scores
    """
    if not results:
        return BenchmarkResult(
            accuracy=0.0,
            stderr=0.0,
            n_examples=0,
            by_theme={},
            by_axis={},
            total_grader_calls=0
        )
    
    # 1. Compute overall score (mean of normalized scores, clipped)
    normalized_scores = [r.normalized_score for r in results]
    accuracy = float(np.clip(np.mean(normalized_scores), 0, 1))
    
    # Compute bootstrap standard error
    stderr = compute_bootstrap_std(normalized_scores)
    
    # 2. Compute by-theme breakdown
    by_theme = {}
    if examples:
        by_theme = compute_scores_by_theme(results, examples)
    
    # 3. Compute by-axis breakdown  
    by_axis = {}
    if examples:
        by_axis = compute_scores_by_axis(results, examples)
    
    # 4. Count grader calls
    total_grader_calls = sum(len(r.grading_results) for r in results)
    
    return BenchmarkResult(
        accuracy=accuracy,
        stderr=stderr,
        n_examples=len(results),
        by_theme=by_theme,
        by_axis=by_axis,
        total_grader_calls=total_grader_calls
    )


def compute_bootstrap_std(scores: List[float], n_bootstrap: int = 1000) -> float:
    """Compute bootstrap standard error for scores."""
    if len(scores) < 2:
        return 0.0
    
    scores_arr = np.array(scores)
    bootstrap_means = []
    
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
        bootstrap_means.append(np.clip(np.mean(sample), 0, 1))
    
    return float(np.std(bootstrap_means))


def compute_scores_by_theme(
    results: List[ExampleResult],
    examples: List["HealthBenchExample"]
) -> Dict[str, float]:
    """Compute average score for each theme."""
    if len(results) != len(examples):
        raise ValueError(f"results ({len(results)}) and examples ({len(examples)}) must have same length")
    
    # Group scores by theme
    theme_scores: Dict[str, List[float]] = defaultdict(list)
    
    for result, example in zip(results, examples):
        theme = example.theme
        theme_scores[theme].append(result.normalized_score)
    
    # Compute mean for each theme, clipped to [0, 1]
    by_theme = {}
    for theme, scores in theme_scores.items():
        mean_score = float(np.mean(scores))
        by_theme[theme] = float(np.clip(mean_score, 0, 1))
    
    return by_theme


def compute_scores_by_axis(
    results: List[ExampleResult],
    examples: List["HealthBenchExample"]
) -> Dict[str, float]:
    """Compute average score for each behavioral axis."""
    if len(results) != len(examples):
        raise ValueError(f"results ({len(results)}) and examples ({len(examples)}) must have same length")
    
    # Collect (rubric_item, grading_result) pairs by axis
    axis_items_grades: Dict[str, List[tuple]] = defaultdict(list)
    
    for result, example in zip(results, examples):
        for rubric, grading in zip(example.rubrics, result.grading_results):
            axis = rubric.axis
            axis_items_grades[axis].append((rubric, grading))
    
    # Compute score for each axis
    by_axis = {}
    for axis, items_grades in axis_items_grades.items():
        # Total possible = sum of positive point values
        total_possible = sum(
            rubric.points for rubric, _ in items_grades if rubric.points > 0
        )
        
        if total_possible == 0:
            continue
        
        # Achieved = sum of points where criteria_met
        achieved = sum(
            rubric.points for rubric, grading in items_grades if grading.criteria_met
        )
        
        score = achieved / total_possible
        by_axis[axis] = float(np.clip(score, 0, 1))
    
    return by_axis

