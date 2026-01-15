# HealthBench Easy - Task Context for Agents

## Overview

HealthBench Easy evaluates models on physician-curated medical conversations from OpenAI's HealthBench benchmark. Your goal is to post-train the base model to generate better responses to health-related queries.

## Dataset

The benchmark contains **450 examples** designed for meaningful base→instruct separation:
- Base models achieve **5-17% overall**
- Instruction-tuned models achieve **26-40% overall**
- This **~20-23% gap** demonstrates meaningful improvement from post-training

**Filtering criteria:** Multi-turn conversations (≥3 turns) with completeness criteria and ≤2 negative criteria.

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
- **Note:** Each example has ~7 criteria on average, so evaluation involves ~3,150 grader calls for 450 examples

## Tips for Post-Training

1. **Focus on completeness** - Base models score ~0% on completeness, instruct models ~25-40%
2. **Multi-turn context matters** - All examples require tracking conversation history
3. **Emergency referrals matter** - Always advise seeking emergency care for serious symptoms
4. **Context-seeking is valued** - Ask clarifying questions when appropriate
5. **Avoid harmful advice** - Negative criteria penalize dangerous recommendations
6. **Communication quality counts** - Clear, empathetic responses score better

## Themes

The benchmark covers 7 themes:
- Communication (30%)
- Hedging/uncertainty (26%)
- Global health (19%)
- Context-seeking (9%)
- Complex responses (7%)
- Emergency referrals (5%)
- Health data tasks (4%)

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

## Expected Baseline Scores

### SmolLM3-3B (50 samples)

| Metric | Base | Instruct | Gap |
|--------|------|----------|-----|
| **Overall** | **5.2%** | **26.1%** | **+20.9pp** |
| Accuracy | 20.0% | 26.8% | +6.8pp |
| Completeness | 0.0% | 24.9% | +24.9pp |
| Context Awareness | 3.8% | 21.7% | +17.9pp |
| Communication | 21.9% | 61.5% | +39.6pp |

### Qwen3-4B (50 samples)

| Metric | Base | Instruct | Gap |
|--------|------|----------|-----|
| **Overall** | **17.1%** | **39.7%** | **+22.6pp** |
| Accuracy | 23.0% | 45.0% | +22.0pp |
| Completeness | 12.4% | 39.7% | +27.3pp |
| Context Awareness | 9.2% | 31.3% | +22.1pp |
| Communication | 24.0% | 54.2% | +30.2pp |

## Key Insights

1. **Overall gap**: 5-17% → 26-40% shows meaningful improvement from post-training
2. **Completeness is the key discriminator**: Base models score ~0-12%, instruct ~25-40%
3. **Multi-turn requirement**: Forces context tracking that base models struggle with
4. **~20pp gap** provides clear headroom for post-training agents to improve

The gap between base and instruction-tuned models demonstrates the potential improvement available through post-training. Target: improve overall score by enhancing completeness and context tracking while maintaining accuracy.
