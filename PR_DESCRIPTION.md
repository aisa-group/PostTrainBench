# Add HealthBench Evaluation Task

## Summary

Adds **HealthBench** as a new evaluation task — the first rubric-based, LLM-as-judge eval in PostTrainBench.

The dataset contains **450 examples** designed for meaningful base→instruct separation:
- **Base models:** 5-17% overall
- **Instruct models:** 26-40% overall  
- **Gap:** ~20-23 percentage points

Uses physician-curated medical conversations with ~7 rubric criteria per example, graded by GPT-5-mini against physician-written standards.

## Why HealthBench?

Current PTB tasks have **binary pass/fail signals**:
- GSM8K: Math is right or wrong
- HumanEval: Code passes tests or doesn't
- BFCL: Function call format matches or doesn't

HealthBench introduces **multi-dimensional rubric scoring** (-10 to +10 per criterion), testing whether agents can post-train for fuzzy, open-ended objectives where success is measured by nuanced evaluation rather than exact matches.

## Baseline Results (50 samples, GPT-5-mini grader)

### SmolLM3-3B

| Axis | Base | Instruct | Gap |
|------|------|----------|-----|
| **Overall** | **5.2%** | **26.1%** | **+20.9pp** |
| Accuracy | 20.0% | 26.8% | +6.8pp |
| Completeness | 0.0% | 24.9% | +24.9pp |
| Context Awareness | 3.8% | 21.7% | +17.9pp |
| Communication | 21.9% | 61.5% | +39.6pp |

### Qwen3-4B

| Axis | Base | Instruct | Gap |
|------|------|----------|-----|
| **Overall** | **17.1%** | **39.7%** | **+22.6pp** |
| Accuracy | 23.0% | 45.0% | +22.0pp |
| Completeness | 12.4% | 39.7% | +27.3pp |
| Context Awareness | 9.2% | 31.3% | +22.1pp |
| Communication | 24.0% | 54.2% | +30.2pp |

**Key finding:** ~20pp gap provides clear headroom for post-training agents to improve. Completeness axis is the key discriminator (base models score ~0-12%, instruct ~25-40%).

## Dataset Filtering

The Easy dataset was carefully filtered for meaningful base→instruct separation:

**Filtering criteria:**
- Multi-turn conversations (≥3 turns) — forces context tracking
- Completeness axis required — where base models score ~0%
- ≤2 negative criteria — limits penalty exposure

**Result:** 450 examples with good theme diversity.

See `docs/healthbench_easy_v2_selection.md` for detailed analysis.

## Files Added

```
src/eval/tasks/healthbench/
├── benchmark.txt                    # "HealthBench"
├── evaluate.py                      # Main entry point (vLLM + LLM judge)
├── evaluation_code/
│   ├── __init__.py
│   ├── data_loader.py               # Loads HealthBench Easy data
│   ├── grader.py                    # LLM-as-judge grading (GPT-5-mini)
│   └── scoring.py                   # Score aggregation with bootstrap stderr
├── data/
│   └── healthbench_easy.jsonl       # 450 examples
└── task_context/
    └── README.md                    # Agent instructions for post-training
```

## Usage

```bash
# Quick test (5 examples)
python src/eval/tasks/healthbench/evaluate.py \
  --model-path Qwen/Qwen3-4B-Base \
  --limit 5

# Full evaluation with output
python src/eval/tasks/healthbench/evaluate.py \
  --model-path final_model/ \
  --json-output-file results.json
```

## Output Format

```json
{
  "accuracy": 0.171,
  "stderr": 0.025,
  "n_examples": 50,
  "total_grader_calls": 365,
  "by_theme": {
    "communication": 0.06,
    "hedging": 0.32,
    ...
  },
  "by_axis": {
    "accuracy": 0.23,
    "completeness": 0.12,
    "context_awareness": 0.09,
    ...
  }
}
```

Primary metric is `accuracy` (overall normalized score, 0-1). Additional breakdowns by theme (7 categories) and axis (5 behavioral dimensions) provided for analysis.

## Runtime & Cost

| Configuration | Examples | Runtime | Grader Cost |
|--------------|----------|---------|-------------|
| Quick test | 5 | ~1 min | ~$0.15 |
| Dev iteration | 50 | ~5 min | ~$1.00 |
| Full eval | 450 | ~25 min | ~$12 |

*Runtimes on H100. Grader costs using GPT-5-mini.*

## Requirements

- `OPENAI_API_KEY` environment variable (for grader)
- vLLM (for model inference)
- Python packages: `openai`, `tiktoken` (explicit in `containers/standard.def`), plus `numpy`, `requests`, `tqdm` (transitive dependencies of vLLM/pandas/transformers)

**Note:** All required packages are already in the standard PTB container. No new dependencies needed.

## Known Quirks

**instruction_following axis anomaly:** On some models, base scores higher than instruct on this axis due to negative criteria penalizing "over-helpful" behaviors (adding context, explaining thoroughly). Focus on completeness and accuracy axes for reliable base→instruct comparison.

## References

- [HealthBench Paper](https://arxiv.org/abs/2505.08775) — OpenAI, May 2025
- [HealthBench Data](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/)
- [Filtering Analysis](docs/healthbench_easy_v2_selection.md)

## Authors

- Shira Eisenberg
- Karina Nguyen
