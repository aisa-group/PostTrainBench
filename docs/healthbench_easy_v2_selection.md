# HealthBench Easy V2: Filtering Criteria Selection

**Author:** Shira Eisenberg  
**Date:** January 15, 2026  
**Status:** Validated and ready for integration

---

## Summary

We developed a new "Easy V2" subset of HealthBench that provides meaningful base→instruct performance separation for PostTrainBench. The key insight: **multi-turn conversations with completeness requirements** are the discriminating factor.

| Subset | Base Model | Instruct Model | Gap |
|--------|------------|----------------|-----|
| Easy V1 (original) | ~27% | ~50% | 23pp |
| **Easy V2 (new)** | **5-17%** | **26-40%** | **20-23pp** |

---

## Problem with Easy V1

The original Easy filter (≤2 negative criteria) was **too easy for base models**:

- SmolLM3-3B-Base scored **27% overall** and **40% on accuracy**
- This left insufficient headroom for post-training improvement
- Base models were "lucking into" correct answers on simple factual questions

### Root Cause Analysis

| Factor | Easy V1 | Impact |
|--------|---------|--------|
| Single-turn examples | 58% | Base models can answer simple Qs without instruction-following |
| Context awareness | 65% single-turn | No conversation history to track |
| Completeness | Low bar | Partial answers counted as success |

---

## Easy V2 Filtering Criteria

After analyzing the full HealthBench dataset (5,000 examples), we identified filtering criteria that target axes where base models consistently fail:

```python
def easy_v2_filter(example):
    return (
        len(example["prompt"]) >= 3 and              # Multi-turn required
        has_axis(example, "completeness") and        # Must have completeness criteria
        count_negative_criteria(example) <= 2        # Limit penalty exposure
    )
```

### Why These Criteria?

| Criterion | Rationale |
|-----------|-----------|
| **Multi-turn (≥3 turns)** | Forces context tracking across conversation history. Base models can't maintain coherence over multiple exchanges. |
| **Completeness axis** | Base models score **0%** on completeness in Hard subset. This axis requires thorough, well-structured responses that only instruction-tuned models can provide. |
| **≤2 negative criteria** | Avoids penalty-heavy examples that create noisy signal. |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total examples | 450 |
| Avg criteria/example | 7.6 |
| Completeness coverage | 100% |
| Accuracy coverage | 90% |
| Context awareness coverage | 51% |

Theme distribution:
- communication: 135 (30%)
- hedging: 118 (26%)
- global_health: 84 (19%)
- context_seeking: 42 (9%)
- complex_responses: 31 (7%)
- emergency_referrals: 22 (5%)
- health_data_tasks: 18 (4%)

---

## Validation Results

We validated Easy V2 on two model pairs with 50-sample runs using GPT-5-mini as the grader:

### SmolLM3-3B (3B parameters)

| Axis | Base | Instruct | Gap |
|------|------|----------|-----|
| **Overall** | **5.2%** | **26.1%** | **+20.9pp** |
| accuracy | 20.0% | 26.8% | +6.8pp |
| completeness | 0.0% | 24.9% | +24.9pp |
| context_awareness | 3.8% | 21.7% | +17.9pp |
| instruction_following | 7.4% | 31.9% | +24.5pp |
| communication_quality | 21.9% | 61.5% | +39.6pp |

### Qwen3-4B (4B parameters)

| Axis | Base | Instruct | Gap |
|------|------|----------|-----|
| **Overall** | **17.1%** | **39.7%** | **+22.6pp** |
| accuracy | 23.0% | 45.0% | +22.0pp |
| completeness | 12.4% | 39.7% | +27.3pp |
| context_awareness | 9.2% | 31.3% | +22.1pp |
| instruction_following | 42.6% | 22.3% | -20.3pp ⚠️ |
| communication_quality | 24.0% | 54.2% | +30.2pp |

---

## Known Quirk: instruction_following Anomaly

The instruction_following axis shows an unexpected reversal on Qwen (base > instruct). Investigation revealed this is a **rubric design issue**, not an evaluation bug:

### Root Cause

4 negative instruction_following criteria penalize behaviors that instruct models are trained to do:

| Criterion | Points | Why Instruct Fails |
|-----------|--------|-------------------|
| "Includes additional statements unrelated to prompt" | -4 | Instruct adds helpful context |
| "Does not provide examples or seek unnecessary context" | -4 | Instruct explains thoroughly |
| "Model should refuse unsafe requests" | -6 | Instruct may comply with safety guidelines |
| "Wrong language" | -9 | Training data dependent |

**Impact:** 23 potential penalty points (24.5% of positive points)

### Recommendation

This quirk affects only the instruction_following axis. The **completeness**, **accuracy**, and **context_awareness** axes show clean, expected base→instruct separation. We recommend:

1. Interpreting instruction_following with caution
2. Focusing on completeness and accuracy as primary metrics
3. Potentially excluding negative instruction_following criteria in future versions

---

## Files & Usage

### Dataset Location
```
data/healthbench_easy_v2.jsonl  (450 examples)
```

### Loading the Dataset
```python
from src.eval.tasks.healthbench.data_loader import load_healthbench_easy_v2

# Load all examples
examples = load_healthbench_easy_v2()

# Load with limit for testing
examples = load_healthbench_easy_v2(limit=50)
```

### Running Evaluation
```bash
python src/eval/tasks/healthbench/evaluate.py \
  --model-path Qwen/Qwen3-4B-Base \
  --subset easy_v2 \
  --limit 50 \
  --grader-model gpt-5-mini
```

---

## Next Steps

1. ✅ Easy V2 dataset generated and validated
2. ⏳ Port to PTB repo (`ptb_port/`)
3. ⏳ Run full 450-sample validation on all PTB target models
4. ⏳ Document in PTB PR description

---

## Appendix: Alternative Filters Considered

| Filter | Count | Reason Not Selected |
|--------|-------|---------------------|
| V7: completeness OR context | 1,247 | Too many single-turn examples |
| V9: multi-turn + context + 3+ pos | 250 | Slightly smaller pool |
| V10: multi-turn + complete + context | 231 | Both zero-axes; may be too strict |
| V11: multi-turn + 5+ positive | 324 | Lower axis coverage |

V8 (multi-turn + completeness) was selected as the best balance of:
- Sample size (450 examples)
- Discriminating power (completeness axis)
- Multi-turn requirement (forces instruction-following)

---

*Analysis scripts: `scripts/analyze_easy_filtering.py`, `scripts/analyze_criteria_complexity.py`*  
*Detailed analysis: `docs/healthbench_easy_filtering_analysis.md`*

