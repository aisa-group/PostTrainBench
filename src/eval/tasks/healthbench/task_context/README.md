# HealthBench Easy - Task Context for Agents

## Overview

HealthBench Easy evaluates models on physician-curated medical conversations from OpenAI's HealthBench benchmark. Your goal is to post-train the base model to generate better responses to health-related queries.

## Dataset

The benchmark contains **1,000 examples** filtered for moderate difficulty:
- Base models achieve **~27% overall** (with high accuracy but poor communication)
- Instruction-tuned models achieve **~50% overall** (balanced across all axes)
- This **~23% gap** demonstrates meaningful improvement from post-training

**Filtering criteria:** Non-hard examples with ≤2 negative criteria, stratified sampled to preserve theme distribution.

## Evaluation

Responses are graded by GPT-5-mini against **physician-written rubric criteria**. Each example has ~12 criteria on average, covering:

- **Accuracy**: Is the medical information correct?
- **Communication**: Is the response clear and appropriate?
- **Context-seeking**: Does the model ask for necessary clarification?
- **Instruction-following**: Does the model follow the user's request?
- **Completeness**: Is the response thorough?

## Scoring

- Each criterion has a point value (-10 to +10)
- Positive criteria: gain points if met
- Negative criteria: lose points if met (penalizes bad behavior)
- Final score = total points earned / maximum possible points
- **Note:** Each example has ~12 criteria on average, so evaluation involves ~12,000 grader calls for 1,000 examples

## Tips for Post-Training

1. **Focus on completeness** - HealthBench rewards thorough responses
2. **Emergency referrals matter** - Always advise seeking emergency care for serious symptoms
3. **Context-seeking is valued** - Ask clarifying questions when appropriate
4. **Avoid harmful advice** - Negative criteria penalize dangerous recommendations
5. **Communication quality counts** - Clear, empathetic responses score better

## Themes

The benchmark covers 7 themes:
- Emergency referrals
- Global health
- Health data tasks
- Context-seeking
- Expertise-tailored communication
- Response depth
- Responding under uncertainty

## Data Sources for Training

**Do NOT use HealthBench test data for training.**

Suggested alternative datasets:
- MedQA / MedMCQA (medical Q&A)
- PubMedQA (biomedical questions)
- Instruction-following datasets (Alpaca, Dolly)
- Medical dialogue datasets (filtered for quality)

## Evaluation Command

```bash
# Quick check
python evaluate.py --model-path final_model/ --limit 50

# Full evaluation
python evaluate.py --model-path final_model/
```

## Expected Baseline Scores (SmolLM3-3B, 50 samples, PTB templates)

| Metric | Base | Instruct | Gap |
|--------|------|----------|-----|
| **Overall** | 27.2% | **50.3%** | +23.1% |
| Accuracy | 63.9% | 50.8% | -13.1% |
| Context Awareness | 25.9% | 50.3% | +24.4% |
| Completeness | 5.1% | 37.5% | +32.4% |
| Communication | 0% | **100%** | +100% |

### Key Insights

1. **Overall gap**: 27% → 50% shows meaningful improvement from post-training
2. **Communication/Completeness**: Instruct model is dramatically better (0% → 100% communication)
3. **Accuracy paradox**: Base model scores higher on raw accuracy but gives incomplete/poorly formatted responses

The gap between base and instruction-tuned models demonstrates the potential improvement available through post-training. Target: improve overall score by enhancing communication and completeness while maintaining accuracy.
