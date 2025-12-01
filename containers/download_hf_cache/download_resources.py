from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import os

MODELS = [
    'Qwen/Qwen3-4B-Base',
    'Qwen/Qwen3-1.7B-Base',
    'google/gemma-3-4b-pt',
    'HuggingFaceTB/SmolLM3-3B-Base',
]

for model_name in MODELS:
    print(f"Downloading model: {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    AutoModel.from_pretrained(model_name)
    print(f"Model {model_name} downloaded successfully")

load_dataset('openai/openai_humaneval', split='test')
load_dataset('openai/gsm8k', 'main')
load_dataset('opencompass/AIME2025', 'AIME2025-I')
load_dataset('Idavidrein/gpqa', 'gpqa_main')

print(f"Cache location: {os.environ.get('HF_HOME')}")