import os

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')
    if "mock" in base_prompt:
        print("Just do nothing and return.")
        return

    with open('src/disallowed_usage_judge/prompt.txt', 'r') as f:
        prompt = f.read()

    prompt = prompt.replace("{model}", args.model)
    prompt = prompt.replace("{benchmark}", args.benchmark)

    other_allowed_data = ""
    if 'gsm8k' in args.benchmark.lower():
        other_allowed_data = "- Usage of the training subset of GSM8K for training.\n"

    prompt = prompt.replace("{other_allowed_data}", other_allowed_data)
    
    print(prompt)

if __name__ == "__main__":
    main()
