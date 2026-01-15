# HuggingFace Datasets for HealthBench Training

This document explains the dataset recommendations for agents training models to improve on HealthBench.

## HealthBench Evaluation Axes

HealthBench evaluates medical conversations on several axes:
- **Accuracy**: Medical correctness, appropriate recommendations
- **Communication**: Clear explanations, appropriate tone, patient-friendly language
- **Context-Seeking**: Asking clarifying questions, gathering relevant information
- **Completeness**: Thorough responses covering relevant aspects
- **Instruction Following**: Adhering to the conversation's implicit requirements

## Dataset Categories

### 1. Medical Q&A (for Accuracy)

| Dataset | Size | Description |
|---------|------|-------------|
| `openlifescienceai/medmcqa` | ~200MB | 194K medical MCQs from Indian medical exams. Broad clinical coverage. |
| `qiaojin/PubMedQA` | ~50MB | Biomedical QA from PubMed abstracts. Tests medical reasoning. |
| `medalpaca/medical_meadow_medqa` | ~50MB | USMLE-style questions formatted for instruction tuning. |
| `medalpaca/medical_meadow_medical_flashcards` | ~30MB | Medical flashcard Q&A pairs for factual accuracy. |
| `keivalya/MedQuad-MedicalQnADataset` | ~20MB | NIH-sourced medical Q&A. Authoritative information. |

### 2. Medical Conversations (for Communication & Context-Seeking)

| Dataset | Size | Description |
|---------|------|-------------|
| `lavita/ChatDoctor-HealthCareMagic-100k` | ~150MB | **Key dataset**: 100K real doctor-patient conversations. Models communication style and follow-up questions. |
| `ruslanmv/ai-medical-chatbot` | ~100MB | Medical chatbot conversations with dialogue context. |
| `medalpaca/medical_meadow_wikidoc_patient_information` | ~50MB | Patient-friendly medical explanations. Teaches lay communication. |

### 3. Medical Instructions (for Instruction Following)

| Dataset | Size | Description |
|---------|------|-------------|
| `Mohammed-Altaf/medical-instruction-100k` | ~200MB | 100K medical instruction-following examples. |
| `nlpie/Llama2-MedTuned-Instructions` | ~100MB | Medical instructions formatted for fine-tuning. |
| `axiong/pmc_llama_instructions` | ~150MB | PubMed Central derived instructions. |

### 4. Medical Knowledge (for Completeness)

| Dataset | Size | Description |
|---------|------|-------------|
| `medalpaca/medical_meadow_wikidoc` | ~100MB | Broad medical knowledge from WikiDoc. |
| `gamino/wiki_medical_terms` | ~50MB | Medical terminology definitions. |
| `BI55/MedText` | ~500MB | Large medical text corpus. |

## Priority Recommendations

For agents with limited training time, prioritize:

1. **`lavita/ChatDoctor-HealthCareMagic-100k`** - Real doctor-patient conversations are most aligned with HealthBench's conversational evaluation format.

2. **`openlifescienceai/medmcqa`** - Large dataset covering clinical medicine, good for accuracy.

3. **`medalpaca/medical_meadow_wikidoc_patient_information`** - Teaches patient-friendly communication style.

4. **`Mohammed-Altaf/medical-instruction-100k`** - General medical instruction following.

## Note on Dataset Sizes

All recommended datasets are under 15GB (most are under 500MB). Total recommended: ~2GB.

## Datasets Already in PTB Cache

The following medical-relevant datasets are already in `resources.json`:
- `GBaker/MedQA-USMLE-4-options` - USMLE questions
- `cais/mmlu` configs: `anatomy`, `clinical_knowledge`, `college_medicine`, `medical_genetics`, `nutrition`, `professional_medicine`, `virology`
- `camel-ai/biology`, `camel-ai/chemistry`

