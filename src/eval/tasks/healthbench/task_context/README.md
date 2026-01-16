# HealthBench Easy - Task Context for Agents

## Overview

HealthBench Easy evaluates models on physician-curated medical conversations from OpenAI's HealthBench benchmark. Your goal is to post-train the base model to generate better responses to health-related queries.

## Dataset

The benchmark contains **245 examples** designed for maximum base→instruct separation:
- Base models achieve **4.7-13.7% overall**
- Instruction-tuned models achieve **30.6-47.9% overall**
- This **25-43% gap** demonstrates significant improvement from post-training

**Filtering criteria:** Multi-turn conversations (≥5 turns) with completeness criteria and ≤2 negative criteria.

## Evaluation

Responses are graded by GPT-5-mini against **physician-written rubric criteria**. Each example has ~7 criteria on average, covering:

- **Accuracy**: Is the medical information correct?
- **Completeness**: Is the response thorough? (key discriminator)
- **Context-awareness**: Does the model track conversation history?
- **Communication**: Is the response clear and appropriate?
- **Instruction-following**: Does the model follow the user's request?

## Scoring

- Each criterion has a point value (-10 to +10)
- Positive criteria: gain points if met
- Negative criteria: lose points if met (penalizes bad behavior)
- Final score = total points earned / maximum possible points
- **Note:** Each example has ~7 criteria on average, so evaluation involves ~1,700 grader calls for 245 examples

## Tips for Post-Training

1. **Focus on completeness** - Base models score ~0%, instruct models ~20-40%
2. **Multi-turn context matters** - All examples require tracking conversation history (≥5 turns)
3. **Emergency referrals matter** - Always advise seeking emergency care for serious symptoms
4. **Context-seeking is valued** - Ask clarifying questions when appropriate
5. **Avoid harmful advice** - Negative criteria penalize dangerous recommendations
6. **Communication quality counts** - Clear, empathetic responses score better

## Themes

The benchmark covers 7 themes:
- Communication (31%)
- Hedging/uncertainty (29%)
- Global health (15%)
- Complex responses (9%)
- Context-seeking (9%)
- Emergency referrals (5%)
- Health data tasks (2%)

## Data Sources for Training

**Do NOT use HealthBench test data for training.**

Suggested alternative datasets:
- MedQA / MedMCQA (medical Q&A)
- PubMedQA (biomedical questions)
- ChatDoctor / HealthCareMagic (medical conversations)
- Instruction-following datasets (Alpaca, Dolly)

## Evaluation Command

```bash
# Quick check
python evaluate.py --model-path final_model/ --limit 50

# Full evaluation  
python evaluate.py --model-path final_model/
```

## Expected Baseline Scores (50 samples)

| Model | Base | Instruct | Gap |
|-------|------|----------|-----|
| **Gemma-3-4B** | 4.7% | 47.9% | **+43.2pp** |
| **Qwen3-4B** | 13.7% | 41.7% | **+27.9pp** |
| **Qwen3-1.7B** | 10.7% | 37.3% | **+26.7pp** |
| **SmolLM3-3B** | 5.0% | 30.6% | **+25.5pp** |

### Axis Breakdown (SmolLM3-3B)

| Axis | Base | Instruct | Gap |
|------|------|----------|-----|
| Accuracy | 26.7% | 27.2% | +0.6pp |
| **Completeness** | **0.0%** | **19.3%** | **+19.3pp** |
| Context Awareness | 16.9% | 33.8% | +16.9pp |
| Communication | 40.5% | 29.8% | -10.7pp |
| Instruction Following | 4.4% | 54.9% | +50.5pp |

## Key Insights

1. **25-43pp gap**: All 4 target models show significant base→instruct improvement
2. **Completeness is the key discriminator**: Base models score ~0%, instruct ~20-40%
3. **Multi-turn requirement (≥5 turns)**: Forces context tracking that base models struggle with
4. **Gemma shows largest gap**: +43pp indicates high post-training potential

The gap between base and instruction-tuned models demonstrates the potential improvement available through post-training. Target: improve overall score by enhancing completeness and context tracking while maintaining accuracy.
