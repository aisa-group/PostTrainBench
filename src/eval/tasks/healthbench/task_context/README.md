# HealthBench - Task Context for Agents

## Overview

HealthBench evaluates models on physician-curated medical conversations from OpenAI's HealthBench benchmark. Your goal is to post-train the base model to generate better responses to health-related queries.

## Subsets

Two subsets are available:

### HealthBench Hard (default)
- **1,000 examples** selected for maximum difficulty
- Base models score **~0%** (essentially random)
- Tests if agents can post-train for challenging medical tasks
- Use `--subset hard` (or omit, as this is the default)

### HealthBench Easy
- **1,000 examples** filtered for moderate difficulty
- Base models target **40-50%** performance
- Allows agents to demonstrate incremental progress
- Use `--subset easy`

**Filtering criteria for Easy:** Non-hard examples with ≤2 negative criteria, stratified sampled to preserve theme distribution.

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
# Hard subset (default)
python evaluate.py --model-path final_model/ --subset hard --limit 50  # Quick check
python evaluate.py --model-path final_model/ --subset hard              # Full evaluation

# Easy subset
python evaluate.py --model-path final_model/ --subset easy --limit 50  # Quick check
python evaluate.py --model-path final_model/ --subset easy              # Full evaluation
```

## Expected Baseline Scores

### HealthBench Hard

**Overall scores (extremely difficult):**
- Base models: 0% overall (expected — base models can't follow instructions)
- Instruction-tuned: 0% overall, but ~7-8% accuracy

**Sub-axis scores (more informative):**
| Model Type | Accuracy | Communication | Instr-Following |
|------------|----------|---------------|-----------------|
| Qwen Base | 0-1% | 0-4% | 0% |
| SmolLM3/Gemma/DeepSeek Base | 0-2% | 14-18% | 0-5% |
| Qwen Instruct | **7.8%** | 2% | 0% |

### HealthBench Easy

**Target performance:**
- Base models: 40-50% overall (allows demonstrating progress)
- Instruction-tuned: Higher (establishes ceiling)

### Key Insights

1. **Hard subset:** The gap between base (~0%) and instruction-tuned (~8% accuracy) shows the potential improvement available through post-training.
2. **Easy subset:** Allows agents to show meaningful progress during training, not just at the end.
3. **Target:** Improve accuracy while maintaining or improving communication quality.

