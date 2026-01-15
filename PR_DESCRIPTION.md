# Add HealthBench Evaluation Task (Hard + Easy Subsets)

## Summary

Adds **HealthBench** as a new evaluation task — the first rubric-based, LLM-as-judge eval in PostTrainBench.

Includes **two subsets**:
- **HealthBench Hard** (default): 1,000 difficult examples where base models score ~0%
- **HealthBench Easy** (new): 1,000 filtered examples targeting ~40% base model accuracy

Both subsets use physician-curated medical conversations with ~6-12 rubric criteria per example, graded by GPT-5-mini against physician-written standards.

## Why HealthBench?

Current PTB tasks have **binary pass/fail signals**:
- GSM8K: Math is right or wrong
- HumanEval: Code passes tests or doesn't
- BFCL: Function call format matches or doesn't

HealthBench introduces **multi-dimensional rubric scoring** (-10 to +10 per criterion), testing whether agents can post-train for fuzzy, open-ended objectives where success is measured by nuanced evaluation rather than exact matches.

## Baseline Results (200 samples)

| Model | Overall | Accuracy Axis | Communication Axis |
|-------|---------|---------------|-------------------|
| Qwen3-1.7B-Base | 0% | 0% | 0% |
| Qwen3-4B-Base | 0% | 1% | 3.4% |
| SmolLM3-3B-Base | 0% | 2% | **18%** |
| DeepSeek-V2-Lite | 0% | 0.1% | **17%** |
| Gemma-3-4B-pt | 0% | 2% | **14%** |
| Qwen3-1.7B (instruct) | 0% | **7.8%** | 2% |

**Post-training headroom:** ~8% accuracy improvement from base → instruct demonstrates the potential gains agents can achieve.

**Key finding:** Different base models show very different communication scores (0-18%), suggesting pre-training data/approach significantly affects this axis.

## HealthBench Easy (NEW)

For PTB 1.0, includes a filtered "Easy" subset that allows agents to show progress during post-training:

| Metric | Easy | Hard |
|--------|------|------|
| Examples | 1,000 | 1,000 |
| Avg criteria/example | 6.3 | 11.8 |
| Negative criteria/example | 1.3 | 4.1 |
| **SmolLM3-3B-Base overall** | **31.6%** | ~0% |
| **SmolLM3-3B-Base accuracy** | **43.2%** | 2% |

**Filtering criteria:**
- Non-hard examples (excluded from Hard subset)
- ≤2 negative criteria per example
- Stratified sampled to preserve theme distribution

```bash
# Easy subset (for PTB 1.0)
python evaluate.py --model-path Qwen/Qwen3-1.7B-Base --subset easy --limit 200

# Hard subset (default)
python evaluate.py --model-path Qwen/Qwen3-1.7B-Base --subset hard --limit 200
```

## Files Added

```
src/eval/tasks/healthbench/
├── benchmark.txt                    # "HealthBench Hard"
├── benchmark_easy.txt               # "HealthBench Easy"
├── evaluate.py                      # Main entry point (vLLM + LLM judge)
├── evaluation_code/
│   ├── __init__.py
│   ├── data_loader.py               # Downloads/caches HealthBench data
│   ├── grader.py                    # LLM-as-judge grading (GPT-5-mini)
│   └── scoring.py                   # Score aggregation with bootstrap stderr
└── task_context/
    └── README.md                    # Agent instructions for post-training
```

## Usage

```bash
# Quick test - Easy subset (5 examples)
python src/eval/tasks/healthbench/evaluate.py \
  --model-path Qwen/Qwen3-1.7B-Base \
  --subset easy \
  --limit 5

# Quick test - Hard subset (5 examples)
python src/eval/tasks/healthbench/evaluate.py \
  --model-path Qwen/Qwen3-1.7B-Base \
  --subset hard \
  --limit 5

# Full evaluation with output
python src/eval/tasks/healthbench/evaluate.py \
  --model-path final_model/ \
  --subset easy \
  --json-output-file results.json

# Use GPT-4.1 for paper-comparable results
python src/eval/tasks/healthbench/evaluate.py \
  --model-path final_model/ \
  --subset hard \
  --grader-model gpt-4.1 \
  --json-output-file results.json
```

## Output Format

```json
{
  "accuracy": 0.078,
  "stderr": 0.012,
  "n_examples": 200,
  "total_grader_calls": 2367,
  "by_theme": {
    "communication": 0.05,
    "emergency_referrals": 0.12,
    ...
  },
  "by_axis": {
    "accuracy": 0.078,
    "communication": 0.02,
    "context_seeking": 0.03,
    ...
  }
}
```

Primary metric is `accuracy` (overall normalized score, 0-1). Additional breakdowns by theme (7 categories) and axis (5 behavioral dimensions) provided for analysis.

## Runtime & Cost

| Configuration | Examples | Runtime | Grader Cost |
|--------------|----------|---------|-------------|
| Quick test | 5 | ~1 min | ~$0.15 |
| Dev iteration | 50 | ~3 min | ~$1.50 |
| Full eval | 200 | ~6 min | ~$6.50 |
| Complete | 1000 | ~25 min | ~$33 |

*Runtimes on H100. Grader costs using GPT-5-mini.*

## Requirements

- `OPENAI_API_KEY` environment variable (for grader)
- vLLM (for model inference)
- Python packages: `openai`, `tiktoken` (explicit in `containers/standard.def`), plus `numpy`, `requests`, `tqdm` (transitive dependencies of vLLM/pandas/transformers)

**Note:** All required packages are already in the standard PTB container. No new dependencies needed.

Data downloads automatically from OpenAI blob storage on first run and caches locally to `data/` (gitignored).

## Open Questions

1. **Templates path:** Code expects `--templates-dir` relative to repo root (default: `templates/`). Is this the correct convention?

2. **Data cache location:** Currently caches to `data/healthbench_hard.jsonl` within the task directory. Should this go elsewhere?

3. **Default limit:** Currently `None` (full 1000 examples). Should we default to a smaller number for faster iteration?

## Future Work

- **Judge comparison:** GPT-4.1 vs GPT-4o-mini vs GPT-5-mini correlation analysis (to validate cheaper graders)

## References

- [HealthBench Paper](https://arxiv.org/abs/2505.08775) — OpenAI, May 2025
- [HealthBench Data](https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl)

## Authors

- Shira Eisenberg
- Karina Nguyen

