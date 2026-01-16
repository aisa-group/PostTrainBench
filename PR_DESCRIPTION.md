# Add HealthBench Evaluation Task

## Summary

Adds **HealthBench** as a new evaluation task — the first rubric-based, LLM-as-judge eval in PostTrainBench.

The dataset contains **245 examples** designed for maximum base→instruct separation:
- **Base models:** 4.7-13.7% overall
- **Instruct models:** 30.6-47.9% overall  
- **Gap:** 25.5-43.2 percentage points

Uses physician-curated medical conversations with ~7 rubric criteria per example, graded by GPT-5-mini against physician-written standards.

## Why HealthBench?

Current PTB tasks have **binary pass/fail signals**:
- GSM8K: Math is right or wrong
- HumanEval: Code passes tests or doesn't
- BFCL: Function call format matches or doesn't

HealthBench introduces **multi-dimensional rubric scoring** (-10 to +10 per criterion), testing whether agents can post-train for fuzzy, open-ended objectives where success is measured by nuanced evaluation rather than exact matches.

## Baseline Results (50 samples, GPT-5-mini grader)

| Model | Base | Instruct | Gap |
|-------|------|----------|-----|
| **Gemma-3-4B** | 4.7% | 47.9% | **+43.2pp** |
| **Qwen3-4B** | 13.7% | 41.7% | **+27.9pp** |
| **Qwen3-1.7B** | 10.7% | 37.3% | **+26.7pp** |
| **SmolLM3-3B** | 5.0% | 30.6% | **+25.5pp** |

### Key Axes (SmolLM3-3B example)

| Axis | Base | Instruct | Gap |
|------|------|----------|-----|
| Accuracy | 26.7% | 27.2% | +0.6pp |
| **Completeness** | **0.0%** | **19.3%** | **+19.3pp** |
| Context Awareness | 16.9% | 33.8% | +16.9pp |
| Communication | 40.5% | 29.8% | -10.7pp |

**Key finding:** Completeness is the primary discriminator — base models score ~0%, instruct ~20-40%. All models show >25pp overall gap.

## Dataset Filtering (V3)

The Easy dataset was carefully filtered for maximum base→instruct separation:

**Filtering criteria:**
- Multi-turn conversations (≥5 turns) — forces context tracking
- Completeness axis required — where base models score ~0%
- ≤2 negative criteria — limits penalty exposure

**Result:** 245 examples with excellent separation across all 4 target models.

See `docs/healthbench_easy_v3_selection.md` for detailed analysis.

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
│   └── healthbench_easy.jsonl       # 245 examples
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
  "accuracy": 0.137,
  "stderr": 0.025,
  "n_examples": 50,
  "total_grader_calls": 350,
  "by_theme": {
    "communication": 0.06,
    "hedging": 0.32,
    ...
  },
  "by_axis": {
    "accuracy": 0.175,
    "completeness": 0.044,
    "context_awareness": 0.224,
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
| Full eval | 245 | ~15 min | ~$8 |

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
- [Filtering Analysis](docs/healthbench_easy_v3_selection.md)

## Authors

- Shira Eisenberg
- Karina Nguyen
