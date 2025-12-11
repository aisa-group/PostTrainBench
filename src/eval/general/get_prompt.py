#!/usr/bin/env python3
import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--model-to-train', type=str, required=True)
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--num-hours', type=str, required=True)

    args = parser.parse_args()

    base_prompt = os.environ.get('POST_TRAIN_BENCH_PROMPT', 'prompt')

    template_path = f'src/eval/general/{base_prompt}.txt'

    with open(template_path, 'r') as f:
        template = f.read()

    datetime = subprocess.run(['date', '-u'], capture_output=True, text=True).stdout.strip()

    result = template.replace('{model}', args.model_to_train)
    result = result.replace('{benchmark}', args.benchmark)
    result = result.replace('{num_hours}', args.num_hours)
    result = result.replace('{datetime}', datetime)

    if args.agent == 'claude':
        result += """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""
    print(result)

if __name__ == '__main__':
    main()
