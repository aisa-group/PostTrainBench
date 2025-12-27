"""Download every cached Hugging Face model and dataset if missing."""

import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

MODELS: List[str] = [
    'HuggingFaceTB/SmolLM3-3B-Base',
    'Qwen/Qwen2.5-3B',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-Math-7B-Instruct',
    'Qwen/Qwen3-1.7B-Base',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-4B-Base',
    'Qwen/Qwen3-4B-Instruct-2507',
    'deepseek-ai/DeepSeek-Math-7B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'deepseek-ai/deepseek-math-7b-instruct',
    'google/gemma-3-4b-it',
    'google/gemma-3-4b-pt'
]

DATASETS: List[Dict[str, Any]] = [
    {'dataset': 'AI-MO/aimo-validation-aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'AI-MO___aimo-validation-aime'},
    {'dataset': 'AI-MO/aimo-validation-amc', 'config': 'default', 'splits': ['train'], 'cache_key': 'AI-MO___aimo-validation-amc'},
    {'dataset': 'AI-MO/aimo-validation-math-level-5', 'config': 'default', 'splits': ['train'], 'cache_key': 'AI-MO___aimo-validation-math-level-5'},
    {'dataset': 'AI-MO/NuminaMath-CoT', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'AI-MO___numina_math-co_t'},
    {'dataset': 'AI-MO/NuminaMath-TIR', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'AI-MO___numina_math-tir'},
    {'dataset': 'AIMERgzr/PapperScore', 'config': 'default', 'splits': ['train'], 'cache_key': 'AIMERgzr___papper_score'},
    {'dataset': 'AMead10/Universal-glaive-function-calling-v2', 'config': 'default', 'splits': ['train'], 'cache_key': 'AMead10___universal-glaive-function-calling-v2'},
    {'dataset': 'Asap7772/aime_gpt-4o-mini_responses_evaluated_flatturn', 'config': 'default', 'splits': ['train'], 'cache_key': 'Asap7772___aime_gpt-4o-mini_responses_evaluated_flatturn'},
    {'dataset': 'AymanTarig/function-calling-v0.2-with-r1-cot', 'config': 'default', 'splits': ['train'], 'cache_key': 'AymanTarig___function-calling-v0.2-with-r1-cot'},
    {'dataset': 'BitAgent/tool_calling', 'config': 'default', 'splits': ['train'], 'cache_key': 'BitAgent___tool_calling'},
    {'dataset': 'BitStarWalkin/AIME_1983_2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'BitStarWalkin___aime_1983_2024'},
    {'dataset': 'chilleD/SVAMP', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'ChilleD___svamp'},
    {'dataset': 'CohereLabs/m-ArenaHard-v2.0', 'config': 'en', 'splits': ['test'], 'cache_key': 'CohereLabs___m-arena_hard-v2.0'},
    {'dataset': 'Deepexi/function-calling-small', 'config': 'default', 'splits': ['train'], 'cache_key': 'Deepexi___function-calling-small'},
    {'dataset': 'Digital-nimbus/llama-2-oai-function-calling', 'config': 'default', 'splits': ['train'], 'cache_key': 'Digital-nimbus___llama-2-oai-function-calling'},
    {'dataset': 'EleutherAI/hendrycks_math', 'config': 'algebra', 'splits': ['test', 'train'], 'cache_key': 'EleutherAI___hendrycks_math'},
    {'dataset': 'EleutherAI/hendrycks_math', 'config': 'counting_and_probability', 'splits': ['test', 'train'], 'cache_key': 'EleutherAI___hendrycks_math'},
    {'dataset': 'EleutherAI/hendrycks_math', 'config': 'geometry', 'splits': ['test', 'train'], 'cache_key': 'EleutherAI___hendrycks_math'},
    {'dataset': 'EleutherAI/hendrycks_math', 'config': 'intermediate_algebra', 'splits': ['test', 'train'], 'cache_key': 'EleutherAI___hendrycks_math'},
    {'dataset': 'EleutherAI/hendrycks_math', 'config': 'number_theory', 'splits': ['test', 'train'], 'cache_key': 'EleutherAI___hendrycks_math'},
    {'dataset': 'GBaker/MedQA-USMLE-4-options', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'GBaker___med_qa-usmle-4-options'},
    {'dataset': 'Goekdeniz-Guelmez/Function_Calling_Unfiltered', 'config': 'default', 'splits': ['train'], 'cache_key': 'Goekdeniz-Guelmez___function_calling_unfiltered'},
    {'dataset': 'HuggingFaceH4/aime_2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'HuggingFaceH4___aime_2024'},
    {'dataset': 'HuggingFaceH4/Bespoke-Stratos-17k', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'HuggingFaceH4___bespoke-stratos-17k'},
    {'dataset': 'HuggingFaceH4/CodeAlpaca_20K', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'HuggingFaceH4___code_alpaca_20_k'},
    {'dataset': 'HuggingFaceH4/CodeAlpaca_20K', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'HuggingFaceH4___code_alpaca_20k'},
    {'dataset': 'HuggingFaceH4/ultrachat_200k', 'config': 'default', 'splits': ['test_gen', 'test_sft', 'train_gen', 'train_sft'], 'cache_key': 'HuggingFaceH4___ultrachat_200k'},
    {'dataset': 'HuggingFaceH4/ultrafeedback_binarized', 'config': 'default', 'splits': ['test_gen', 'test_prefs', 'test_sft', 'train_gen', 'train_prefs', 'train_sft'], 'cache_key': 'HuggingFaceH4___ultrafeedback_binarized'},
    {'dataset': 'HydraLM/glaive_function_calling_v1_standardized', 'config': 'default', 'splits': ['train'], 'cache_key': 'HydraLM___glaive_function_calling_v1_standardized'},
    {'dataset': 'Idavidrein/gpqa', 'config': 'gpqa_diamond', 'splits': ['train'], 'cache_key': 'Idavidrein___gpqa'},
    {'dataset': 'Idavidrein/gpqa', 'config': 'gpqa_experts', 'splits': ['train'], 'cache_key': 'Idavidrein___gpqa'},
    {'dataset': 'Idavidrein/gpqa', 'config': 'gpqa_extended', 'splits': ['train'], 'cache_key': 'Idavidrein___gpqa'},
    {'dataset': 'Idavidrein/gpqa', 'config': 'gpqa_main', 'splits': ['train'], 'cache_key': 'Idavidrein___gpqa'},
    {'dataset': 'Locutusque/hercules-v2.0', 'config': 'default', 'splits': ['train'], 'cache_key': 'Locutusque___hercules-v2.0'},
    {'dataset': 'LongQ/leetcode_python', 'config': 'default', 'splits': ['dev', 'test', 'train'], 'cache_key': 'LongQ___leetcode_python'},
    {'dataset': 'MCES10-Software/Python-Code-Solutions', 'config': 'default', 'splits': ['train'], 'cache_key': 'MCES10-Software___python-code-solutions'},
    {'dataset': 'magpie-align/Magpie-Pro-300K-Filtered', 'config': 'default', 'splits': ['train'], 'cache_key': 'Magpie-Align___magpie-pro-300_k-filtered'},
    {'dataset': 'Magpie-Align/Magpie-Qwen2.5-Math-Pro-300K-v0.1', 'config': 'default', 'splits': ['train'], 'cache_key': 'Magpie-Align___magpie-qwen2.5-math-pro-300_k-v0.1'},
    {'dataset': 'Magpie-Align/Magpie-Qwen2.5-Pro-300K-Filtered', 'config': 'default', 'splits': ['train'], 'cache_key': 'Magpie-Align___magpie-qwen2.5-pro-300_k-filtered'},
    {'dataset': 'Maxwell-Jia/AIME_2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'Maxwell-Jia___aime_2024'},
    {'dataset': 'Na0s/sft-ready-hendrycks-competition_math', 'config': 'default', 'splits': ['train'], 'cache_key': 'Na0s___sft-ready-hendrycks-competition_math'},
    {'dataset': 'Nan-Do/code-search-net-python', 'config': 'default', 'splits': ['train'], 'cache_key': 'Nan-Do___code-search-net-python'},
    {'dataset': 'Nan-Do/instructional_code-search-net-python', 'config': 'default', 'splits': ['train'], 'cache_key': 'Nan-Do___instructional_code-search-net-python'},
    {'dataset': 'NewEden/xlam-function-calling-60k-shareGPT', 'config': 'default', 'splits': ['train'], 'cache_key': 'NewEden___xlam-function-calling-60k-share_gpt'},
    {'dataset': 'NousResearch/hermes-function-calling-v1', 'config': 'func_calling_singleturn', 'splits': ['train'], 'cache_key': 'NousResearch___hermes-function-calling-v1'},
    {'dataset': 'Open-Orca/SlimOrca', 'config': 'default', 'splits': ['train'], 'cache_key': 'Open-Orca___slim_orca'},
    {'dataset': 'Pandores/aime-1983-2025', 'config': 'default', 'splits': ['train'], 'cache_key': 'Pandores___aime-1983-2025'},
    {'dataset': 'Post-training-Data-Flywheel/gorilla-openfunctions-v1', 'config': 'default', 'splits': ['train'], 'cache_key': 'Post-training-Data-Flywheel___gorilla-openfunctions-v1'},
    {'dataset': 'Prompt48/AIME_Problem_Set_1983-2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'Prompt48___aime_problem_set_1983-2024'},
    {'dataset': 'RayBernard/leetcode', 'config': 'default', 'splits': ['train'], 'cache_key': 'RayBernard___leetcode'},
    {'dataset': 'RayBernard/leetcode1000', 'config': 'default', 'splits': ['train'], 'cache_key': 'RayBernard___leetcode1000'},
    {'dataset': 'Rock23210/AIME_Deepseek_Clean', 'config': 'default', 'splits': ['train'], 'cache_key': 'Rock23210___aime_deepseek_clean'},
    {'dataset': 'Saxo/alpaca_function_calling_dataset', 'config': 'default', 'splits': ['train'], 'cache_key': 'Saxo___alpaca_function_calling_dataset'},
    {'dataset': 'TIGER-Lab/MathInstruct', 'config': 'default', 'splits': ['train'], 'cache_key': 'TIGER-Lab___math_instruct'},
    {'dataset': 'TIGER-Lab/MMLU-Pro', 'config': 'default', 'splits': ['test', 'validation'], 'cache_key': 'TIGER-Lab___mmlu-pro'},
    {'dataset': 'TIGER-Lab/TheoremQA', 'config': 'default', 'splits': ['test'], 'cache_key': 'TIGER-Lab___theorem_qa'},
    {'dataset': 'TigerResearch/tigerbot-kaggle-leetcodesolutions-en-2k', 'config': 'default', 'splits': ['train'], 'cache_key': 'TigerResearch___tigerbot-kaggle-leetcodesolutions-en-2k'},
    {'dataset': 'Vishal24/function_calling', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'Vishal24___function_calling'},
    {'dataset': 'XxCotHGxX/242K_Python_Docstring_Pairs', 'config': 'default', 'splits': ['test', 'train', 'validation'], 'cache_key': 'XxCotHGxX___242_k_python_docstring_pairs'},
    {'dataset': 'Zaynes/multiple_samples_sympy_numina_aime_with_sol_trace', 'config': 'default', 'splits': ['train'], 'cache_key': 'Zaynes___multiple_samples_sympy_numina_aime_with_sol_trace'},
    {'dataset': 'ZeroAgency/gemma3-pythonic-function-tool-calling-v1', 'config': 'default', 'splits': ['test', 'train', 'valid'], 'cache_key': 'ZeroAgency___gemma3-pythonic-function-tool-calling-v1'},
    {'dataset': 'aadajinkya/python_codes_sample', 'config': 'default', 'splits': ['train'], 'cache_key': 'aadajinkya___python_codes_sample'},
    {'dataset': 'agentlans/train-of-thought', 'config': 'train', 'splits': ['train'], 'cache_key': 'agentlans___train-of-thought'},
    {'dataset': 'allenai/ai2_arc', 'config': 'ARC-Challenge', 'splits': ['test', 'train', 'validation'], 'cache_key': 'allenai___ai2_arc'},
    {'dataset': 'allenai/ai2_arc', 'config': 'ARC-Easy', 'splits': ['test', 'train', 'validation'], 'cache_key': 'allenai___ai2_arc'},
    {'dataset': 'allenai/openbookqa', 'config': 'main', 'splits': ['test', 'train', 'validation'], 'cache_key': 'allenai___openbookqa'},
    {'dataset': 'allenai/qasc', 'config': 'default', 'splits': ['test', 'train', 'validation'], 'cache_key': 'allenai___qasc'},
    {'dataset': 'allenai/sciq', 'config': 'default', 'splits': ['test', 'train', 'validation'], 'cache_key': 'allenai___sciq'},
    {'dataset': 'allenporter/assist-llm-function-calling', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'allenporter___assist-llm-function-calling'},
    {'dataset': 'cais/mmlu', 'config': 'abstract_algebra', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'all', 'splits': ['auxiliary_train', 'dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'anatomy', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'astronomy', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'auxiliary_train', 'splits': ['train'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'clinical_knowledge', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'college_biology', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'college_chemistry', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'college_computer_science', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'college_mathematics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'college_medicine', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'college_physics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'computer_security', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'conceptual_physics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'electrical_engineering', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'elementary_mathematics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'high_school_biology', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'high_school_chemistry', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'high_school_computer_science', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'high_school_mathematics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'high_school_physics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'high_school_statistics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'human_aging', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'machine_learning', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'medical_genetics', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'nutrition', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'professional_medicine', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'cais/mmlu', 'config': 'virology', 'splits': ['dev', 'test', 'validation'], 'cache_key': 'cais___mmlu'},
    {'dataset': 'camel-ai/biology', 'config': 'default', 'splits': ['train'], 'cache_key': 'camel-ai___biology'},
    {'dataset': 'camel-ai/chemistry', 'config': 'default', 'splits': ['train'], 'cache_key': 'camel-ai___chemistry'},
    {'dataset': 'camel-ai/math', 'config': 'default', 'splits': ['train'], 'cache_key': 'camel-ai___math'},
    {'dataset': 'camel-ai/physics', 'config': 'default', 'splits': ['train'], 'cache_key': 'camel-ai___physics'},
    {'dataset': 'chenggong1995/MATH-lighteval-olympiads_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'chenggong1995___math-lighteval-olympiads_aime'},
    {'dataset': 'codeparrot/codecomplex', 'config': 'default', 'splits': ['train'], 'cache_key': 'codeparrot___codecomplex'},
    {'dataset': 'dakopi/aime_23', 'config': 'default', 'splits': ['train'], 'cache_key': 'dakopi___aime_23'},
    {'dataset': 'daman1209arora/jeebench', 'config': 'default', 'splits': ['test'], 'cache_key': 'daman1209arora___jeebench'},
    {'dataset': 'deepmind/aqua_rat', 'config': 'raw', 'splits': ['test', 'train', 'validation'], 'cache_key': 'deepmind___aqua_rat'},
    {'dataset': 'deepmind/code_contests', 'config': 'default', 'splits': ['test', 'train', 'valid'], 'cache_key': 'deepmind___code_contests'},
    {'dataset': 'derek-thomas/ScienceQA', 'config': 'default', 'splits': ['test', 'train', 'validation'], 'cache_key': 'derek-thomas___science_qa'},
    {'dataset': 'di-zhang-fdu/AIME_1983_2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'di-zhang-fdu___aime_1983_2024'},
    {'dataset': 'dim/competition_math', 'config': 'default', 'splits': ['train'], 'cache_key': 'dim___competition_math'},
    {'dataset': 'dim/competition_math_selected', 'config': 'default', 'splits': ['train'], 'cache_key': 'dim___competition_math_selected'},
    {'dataset': 'dim/leetcodesolutions_en_2k', 'config': 'default', 'splits': ['train'], 'cache_key': 'dim___leetcodesolutions_en_2k'},
    {'dataset': 'dinushiTJ/gemma-function-calling', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'dinushiTJ___gemma-function-calling'},
    {'dataset': 'dmayhem93/agieval-gaokao-biology', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-gaokao-biology'},
    {'dataset': 'dmayhem93/agieval-gaokao-chemistry', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-gaokao-chemistry'},
    {'dataset': 'dmayhem93/agieval-gaokao-mathqa', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-gaokao-mathqa'},
    {'dataset': 'dmayhem93/agieval-gaokao-physics', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-gaokao-physics'},
    {'dataset': 'dmayhem93/agieval-logiqa-en', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-logiqa-en'},
    {'dataset': 'dmayhem93/agieval-lsat-ar', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-lsat-ar'},
    {'dataset': 'dmayhem93/agieval-lsat-lr', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-lsat-lr'},
    {'dataset': 'dmayhem93/agieval-sat-en', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-sat-en'},
    {'dataset': 'dmayhem93/agieval-sat-math', 'config': 'default', 'splits': ['test'], 'cache_key': 'dmayhem93___agieval-sat-math'},
    {'dataset': 'evalplus/mbppplus', 'config': 'default', 'splits': ['test'], 'cache_key': 'evalplus___mbppplus'},
    {'dataset': 'flytech/python-codes-25k', 'config': 'default', 'splits': ['train'], 'cache_key': 'flytech___python-codes-25k'},
    {'dataset': 'garage-bAInd/Open-Platypus', 'config': 'default', 'splits': ['train'], 'cache_key': 'garage-bAInd___open-platypus'},
    {'dataset': 'glaiveai/glaive-function-calling', 'config': 'default', 'splits': ['train'], 'cache_key': 'glaiveai___glaive-function-calling'},
    {'dataset': 'glaiveai/Glaive-function-calling-v2', 'config': 'default', 'splits': ['train'], 'cache_key': 'glaiveai___glaive-function-calling-v2'},
    {'dataset': 'gneubig/aime-1983-2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'gneubig___aime-1983-2024'},
    {'dataset': 'google/IFEval', 'config': 'default', 'splits': ['train'], 'cache_key': 'google___ifeval'},
    {'dataset': 'google-research-datasets/mbpp', 'config': 'full', 'splits': ['prompt', 'test', 'train', 'validation'], 'cache_key': 'google-research-datasets___mbpp'},
    {'dataset': 'google-research-datasets/mbpp', 'config': 'sanitized', 'splits': ['prompt', 'test', 'train', 'validation'], 'cache_key': 'google-research-datasets___mbpp'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'chatable', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'exec_multiple', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'exec_parallel_multiple', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'exec_simple', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'java', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'javascript', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'parallel', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'rest', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'gorilla-llm/Berkeley-Function-Calling-Leaderboard', 'config': 'sql', 'splits': ['train'], 'cache_key': 'gorilla-llm___berkeley-function-calling-leaderboard'},
    {'dataset': 'greengerong/leetcode', 'config': 'default', 'splits': ['train'], 'cache_key': 'greengerong___leetcode'},
    {'dataset': 'hbXNov/numina_amc_aime_deepseek_r1_responses', 'config': 'default', 'splits': ['train'], 'cache_key': 'hbXNov___numina_amc_aime_deepseek_r1_responses'},
    {'dataset': 'hbXNov/numina_amc_aime_in_depth_deepseek_r1_questions', 'config': 'default', 'splits': ['train'], 'cache_key': 'hbXNov___numina_amc_aime_in_depth_deepseek_r1_questions'},
    {'dataset': 'hendrydong/aime24', 'config': 'default', 'splits': ['test'], 'cache_key': 'hendrydong___aime24'},
    {'dataset': 'hiyouga/glaive-function-calling-v2-sharegpt', 'config': 'default', 'splits': ['train'], 'cache_key': 'hiyouga___glaive-function-calling-v2-sharegpt'},
    {'dataset': 'huntz47/aimee', 'config': 'default', 'splits': ['train'], 'cache_key': 'huntz47___aimee'},
    {'dataset': 'hypervariance/function-calling-sharegpt', 'config': 'default', 'splits': ['train'], 'cache_key': 'hypervariance___function-calling-sharegpt'},
    {'dataset': 'iamtarun/python_code_instructions_18k_alpaca', 'config': 'default', 'splits': ['train'], 'cache_key': 'iamtarun___python_code_instructions_18k_alpaca'},
    {'dataset': 'interstellarninja/tool-calls-singleturn', 'config': 'default', 'splits': ['train'], 'cache_key': 'interstellarninja___tool-calls-singleturn'},
    {'dataset': 'ise-uiuc/Magicoder-Evol-Instruct-110K', 'config': 'default', 'splits': ['train'], 'cache_key': 'ise-uiuc___magicoder-evol-instruct-110_k'},
    {'dataset': 'ise-uiuc/Magicoder-OSS-Instruct-75K', 'config': 'default', 'splits': ['train'], 'cache_key': 'ise-uiuc___magicoder-oss-instruct-75_k'},
    {'dataset': 'jinaai/code_search_net_clean', 'config': 'default', 'splits': ['test.go', 'test.java', 'test.javascript', 'test.php', 'test.python', 'test.ruby', 'train.go', 'train.java', 'train.javascript', 'train.php', 'train.python', 'train.ruby', 'validation.go', 'validation.java', 'validation.javascript', 'validation.php', 'validation.python', 'validation.ruby'], 'cache_key': 'jinaai___code_search_net_clean'},
    {'dataset': 'jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_deepseek-r1-distill-qwen-1.5b_traces_32768'},
    {'dataset': 'jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_deepseek-r1-distill-qwen-14b_traces_32768'},
    {'dataset': 'jonathanyin/aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_deepseek-r1-distill-qwen-7b_traces_32768'},
    {'dataset': 'jonathanyin/aime_1983_2023_deepseek-r1_traces_16384', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_deepseek-r1_traces_16384'},
    {'dataset': 'jonathanyin/aime_1983_2023_deepseek-r1_traces_32768', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_deepseek-r1_traces_32768'},
    {'dataset': 'jonathanyin/aime_1983_2023_grok-3-mini-high_traces_32768', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_grok-3-mini-high_traces_32768'},
    {'dataset': 'jonathanyin/aime_1983_2023_qwq-32b_traces_16384', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_qwq-32b_traces_16384'},
    {'dataset': 'jonathanyin/aime_1983_2023_qwq-32b_traces_32768', 'config': 'default', 'splits': ['train'], 'cache_key': 'jonathanyin___aime_1983_2023_qwq-32b_traces_32768'},
    {'dataset': 'khaimaitien/multi-hop-qa-function-calling-format-V1.0', 'config': 'default', 'splits': ['train', 'validation'], 'cache_key': 'khaimaitien___multi-hop-qa-function-calling-format-v1.0'},
    {'dataset': 'lighteval/mmlu', 'config': 'abstract_algebra', 'splits': ['auxiliary_train', 'dev', 'test', 'validation'], 'cache_key': 'lighteval___mmlu'},
    {'dataset': 'lighteval/mmlu', 'config': 'all', 'splits': ['auxiliary_train', 'dev', 'test', 'validation'], 'cache_key': 'lighteval___mmlu'},
    {'dataset': 'lockon/glaive_toolcall_en', 'config': 'default', 'splits': ['train'], 'cache_key': 'lockon___glaive_toolcall_en'},
    {'dataset': 'm-a-p/CodeFeedback-Filtered-Instruction', 'config': 'default', 'splits': ['train'], 'cache_key': 'm-a-p___code_feedback-filtered-instruction'},
    {'dataset': 'magpie-align/Magpie-Pro-300K-Filtered', 'config': 'default', 'splits': ['train'], 'cache_key': 'magpie-align___magpie-pro-300_k-filtered'},
    {'dataset': 'martim00/math_aime_2023', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'martim00___math_aime_2023'},
    {'dataset': 'math-ai/aime24', 'config': 'default', 'splits': ['test'], 'cache_key': 'math-ai___aime24'},
    {'dataset': 'math-ai/aime25', 'config': 'default', 'splits': ['test'], 'cache_key': 'math-ai___aime25'},
    {'dataset': 'meta-math/MetaMathQA', 'config': 'default', 'splits': ['train'], 'cache_key': 'meta-math___meta_math_qa'},
    {'dataset': 'microsoft/orca-math-word-problems-200k', 'config': 'default', 'splits': ['train'], 'cache_key': 'microsoft___orca-math-word-problems-200k'},
    {'dataset': 'milsunone/cural-functionary-small-5000', 'config': 'default', 'splits': ['train'], 'cache_key': 'milsunone___cural-functionary-small-5000'},
    {'dataset': 'mjalg/function-code', 'config': 'default', 'splits': ['train'], 'cache_key': 'mjalg___function-code'},
    {'dataset': 'mlfoundations-dev/4o_annotated_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___4o_annotated_aime'},
    {'dataset': 'mlfoundations-dev/a1_math_numina_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___a1_math_numina_aime'},
    {'dataset': 'mlfoundations-dev/a1_math_openmathinstruct_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___a1_math_openmathinstruct_aime'},
    {'dataset': 'mlfoundations-dev/bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___bespokelabs-sky-t1-numina-amc-aime-subset-unfiltered'},
    {'dataset': 'mlfoundations-dev/multiple_samples_all_numina_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___multiple_samples_all_numina_aime'},
    {'dataset': 'mlfoundations-dev/multiple_samples_ground_truth_numina_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___multiple_samples_ground_truth_numina_aime'},
    {'dataset': 'mlfoundations-dev/multiple_samples_majority_consensus_numina_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___multiple_samples_majority_consensus_numina_aime'},
    {'dataset': 'mlfoundations-dev/multiple_samples_majority_consensus_numina_aime_math_verify', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___multiple_samples_majority_consensus_numina_aime_math_verify'},
    {'dataset': 'mlfoundations-dev/r1_annotated_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'mlfoundations-dev___r1_annotated_aime'},
    {'dataset': 'narrative-io/narrative-function-calling-v1', 'config': 'default', 'splits': ['train'], 'cache_key': 'narrative-io___narrative-function-calling-v1'},
    {'dataset': 'nickrosh/Evol-Instruct-Code-80k-v1', 'config': 'default', 'splits': ['train'], 'cache_key': 'nickrosh___evol-instruct-code-80k-v1'},
    {'dataset': 'notbadai/python_functions_reasoning', 'config': 'default', 'splits': ['train'], 'cache_key': 'notbadai___python_functions_reasoning'},
    {'dataset': 'nvidia/OpenCodeInstruct', 'config': 'train', 'splits': ['train'], 'cache_key': 'nvidia___open_code_instruct'},
    {'dataset': 'nvidia/OpenMathInstruct-1', 'config': 'default', 'splits': ['train', 'validation'], 'cache_key': 'nvidia___open_math_instruct-1'},
    {'dataset': 'open-r1/codeforces', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'open-r1___codeforces'},
    {'dataset': 'open-r1/OpenR1-Math-220k', 'config': 'default', 'splits': ['train'], 'cache_key': 'open-r1___open_r1-math-220k'},
    {'dataset': 'openai/gsm8k', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'openai___gsm8k'},
    {'dataset': 'openai/gsm8k', 'config': 'main', 'splits': ['test', 'train'], 'cache_key': 'openai___gsm8k'},
    {'dataset': 'openai/openai_humaneval', 'config': 'openai_humaneval', 'splits': ['test'], 'cache_key': 'openai___openai_humaneval'},
    {'dataset': 'philschmid/AIME_1983_2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'philschmid___aime_1983_2024'},
    {'dataset': 'prem-research/Funcdex-MT-Function-Calling', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'prem-research___funcdex-mt-function-calling'},
    {'dataset': 'qfq/genminiall_onlyqwenwrong_aimegpqatrain_domain_powerlaw_steps', 'config': 'default', 'splits': ['train'], 'cache_key': 'qfq___genminiall_onlyqwenwrong_aimegpqatrain_domain_powerlaw_steps'},
    {'dataset': 'qwedsacf/competition_math', 'config': 'default', 'splits': ['train'], 'cache_key': 'qwedsacf___competition_math'},
    {'dataset': 'redwoodresearch/mbpp_extended', 'config': 'default', 'splits': ['train'], 'cache_key': 'redwoodresearch___mbpp_extended'},
    {'dataset': 'roborovski/synthetic-tool-calls', 'config': 'default', 'splits': ['train'], 'cache_key': 'roborovski___synthetic-tool-calls'},
    {'dataset': 'roborovski/synthetic-tool-calls-v2', 'config': 'default', 'splits': ['train'], 'cache_key': 'roborovski___synthetic-tool-calls-v2'},
    {'dataset': 'rulins/DeepSeek-R1-Distill-Qwen-32B_NUMINA_train_amc_aime', 'config': 'default', 'splits': ['train'], 'cache_key': 'rulins___deep_seek-r1-distill-qwen-32_b_numina_train_amc_aime'},
    {'dataset': 'sahil2801/CodeAlpaca-20k', 'config': 'default', 'splits': ['train'], 'cache_key': 'sahil2801___code_alpaca-20k'},
    {'dataset': 'semeru/text-code-galeras-code-generation-from-docstring-3k-deduped', 'config': 'default', 'splits': ['train'], 'cache_key': 'semeru___text-code-galeras-code-generation-from-docstring-3k-deduped'},
    {'dataset': 'sharkchill-xy/HumanEval_mbpp_format', 'config': 'default', 'splits': ['train'], 'cache_key': 'sharkchill-xy___human_eval_mbpp_format'},
    {'dataset': 'simplescaling/aime_nofigures', 'config': 'default', 'splits': ['train'], 'cache_key': 'simplescaling___aime_nofigures'},
    {'dataset': 'smolagents/toolcalling', 'config': 'default', 'splits': ['train'], 'cache_key': 'smolagents___toolcalling'},
    {'dataset': 'starfishdata/AIME_MATH_1000_LONG_COT', 'config': 'default', 'splits': ['train'], 'cache_key': 'starfishdata___aime_math_1000_long_cot'},
    {'dataset': 'starfishdata/AIME_MATH_100_LONG_COT', 'config': 'default', 'splits': ['train'], 'cache_key': 'starfishdata___aime_math_100_long_cot'},
    {'dataset': 'tau/commonsense_qa', 'config': 'default', 'splits': ['test', 'train', 'validation'], 'cache_key': 'tau___commonsense_qa'},
    {'dataset': 'teknium/OpenHermes-2.5', 'config': 'default', 'splits': ['train'], 'cache_key': 'teknium___open_hermes-2.5'},
    {'dataset': 'theblackcat102/evol-codealpaca-v1', 'config': 'default', 'splits': ['train'], 'cache_key': 'theblackcat102___evol-codealpaca-v1'},
    {'dataset': 'think-a-tron/aime-math-train', 'config': 'default', 'splits': ['train'], 'cache_key': 'think-a-tron___aime-math-train'},
    {'dataset': 'togethercomputer/glaive-function-calling-v2-formatted', 'config': 'default', 'splits': ['test', 'train'], 'cache_key': 'togethercomputer___glaive-function-calling-v2-formatted'},
    {'dataset': 'TokenBender/code_instructions_122k_alpaca_style', 'config': 'default', 'splits': ['train'], 'cache_key': 'tokenbender___code_instructions_122k_alpaca_style'},
    {'dataset': 'vikp/python_code_instructions_filtered', 'config': 'default', 'splits': ['train'], 'cache_key': 'vikp___python_code_instructions_filtered'},
    {'dataset': 'weijiezz/math-datasets-100k', 'config': 'default', 'splits': ['test_aime24', 'test_aime25', 'test_amc23', 'test_gsm8k', 'test_math500', 'test_minervamath', 'test_olympiadbench', 'train'], 'cache_key': 'weijiezz___math-datasets-100k'},
    {'dataset': 'weijiezz/math-datasets-20k', 'config': 'default', 'splits': ['test_aime24', 'test_aime25', 'test_amc23', 'test_gsm8k', 'test_math500', 'test_minervamath', 'test_olympiadbench', 'train'], 'cache_key': 'weijiezz___math-datasets-20k'},
    {'dataset': 'weijiezz/NuminaMath-20k', 'config': 'default', 'splits': ['test_aime24', 'test_aime25', 'test_amc23', 'test_gsm8k', 'test_math500', 'test_minervamath', 'test_numina', 'test_olympiadbench', 'train'], 'cache_key': 'weijiezz___numina_math-20k'},
    {'dataset': 'xw27/scibench', 'config': 'default', 'splits': ['train'], 'cache_key': 'xw27___scibench'},
    {'dataset': 'yahma/alpaca-cleaned', 'config': 'default', 'splits': ['train'], 'cache_key': 'yahma___alpaca-cleaned'},
    {'dataset': 'zhiyuanhucs/AIME_1983_2024', 'config': 'default', 'splits': ['aime'], 'cache_key': 'zhiyuanhucs___aime_1983_2024'},
    {'dataset': 'zhuzilin/aime-2024', 'config': 'default', 'splits': ['train'], 'cache_key': 'zhuzilin___aime-2024'},
    {'dataset': 'zuom/AIME-solutions', 'config': 'default', 'splits': ['train'], 'cache_key': 'zuom___aime-solutions'},
    {'dataset': 'ByteDance-Seed/Code-Contests-Plus', 'config': None, 'splits': [], 'cache_key': 'ByteDance-Seed___code-contests-plus'},
    {'dataset': 'WNJXYK/AIME_1983_2024-Reasoning-Paths', 'config': None, 'splits': [], 'cache_key': 'WNJXYK___aime_1983_2024-reasoning-paths'},
    {'dataset': 'internlm/Agent-FLAN', 'config': None, 'splits': [], 'cache_key': 'internlm___agent-flan'},
    {'dataset': 'nvidia/OpenMathInstruct-2', 'config': None, 'splits': [], 'cache_key': 'nvidia___open_math_instruct-2'},
]

CACHE_ROOT = Path(os.environ.get('HF_HOME') or Path.home() / '.cache' / 'huggingface')
HUB_ROOT = CACHE_ROOT / 'hub'
MODEL_CACHE_DIRS = tuple(dict.fromkeys([HUB_ROOT, CACHE_ROOT, Path(os.environ.get('TRANSFORMERS_CACHE') or (CACHE_ROOT / 'models'))]))
DATASET_CACHE_DIR = Path(os.environ.get('HF_DATASETS_CACHE') or (CACHE_ROOT / 'datasets'))


def _repo_folder(prefix: str, repo_id: str) -> str:
    owner, name = repo_id.split('/', 1)
    return f"{prefix}--{owner}--{name}"


def _any_exists(paths):
    for path in paths:
        if path and path.exists():
            return True
    return False


for model_name in MODELS:
    repo_folder = _repo_folder('models', model_name)
    candidates = [base / repo_folder for base in MODEL_CACHE_DIRS]
    if _any_exists(candidates):
        print(f"Skipping model: {model_name} (already cached)")
        continue
    print(f"Downloading model: {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    print(f"Model {model_name} downloaded successfully")

for entry in DATASETS:
    dataset_name = entry['dataset']
    config_name = entry['config']
    splits = entry['splits']
    cache_key = entry['cache_key']
    label = f"{dataset_name} ({config_name})" if config_name else dataset_name
    repo_folder = _repo_folder('datasets', dataset_name)
    dataset_cached = _any_exists([HUB_ROOT / repo_folder, CACHE_ROOT / repo_folder, DATASET_CACHE_DIR / cache_key])
    if dataset_cached:
        print(f"Skipping dataset: {label} (already cached)")
        continue
    if splits:
        for split in splits:
            print(f"Downloading dataset: {label} [split={split}]...")
            kwargs = dict(split=split)
            if config_name:
                kwargs['name'] = config_name
            load_dataset(dataset_name, **kwargs)
    else:
        print(f"Downloading dataset: {label}...")
        kwargs = {}
        if config_name:
            kwargs['name'] = config_name
        load_dataset(dataset_name, **kwargs)

print(f"Cache location: {CACHE_ROOT}")
