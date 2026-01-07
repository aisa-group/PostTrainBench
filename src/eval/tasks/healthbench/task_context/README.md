# HealthBench Hard - Task Context for Agents

## Overview

HealthBench Hard is a subset of 1,000 challenging medical conversations from OpenAI's HealthBench benchmark. Your goal is to post-train the base model to generate better responses to health-related queries.

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
python evaluate.py --model-path final_model/ --limit 50  # Quick check
python evaluate.py --model-path final_model/              # Full evaluation
```

## Expected Baseline Scores

**Overall scores (HealthBench Hard is extremely difficult):**
- Base models: 0% overall (expected — base models can't follow instructions)
- Instruction-tuned: 0% overall, but ~7-8% accuracy

**Sub-axis scores (more informative):**
| Model Type | Accuracy | Communication | Instr-Following |
|------------|----------|---------------|-----------------|
| Qwen Base | 0-1% | 0-4% | 0% |
| SmolLM3/Gemma/DeepSeek Base | 0-2% | 14-18% | 0-5% |
| Qwen Instruct | **7.8%** | 2% | 0% |

**Key Insight:** Instruction-tuning improves accuracy from ~0% to ~8%. This is the headroom available for post-training agents to target.

The gap between base and instruction-tuned shows the potential improvement available through post-training. Target improving accuracy while maintaining or improving communication quality.

