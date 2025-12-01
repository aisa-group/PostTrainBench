#!/usr/bin/env python3
import argparse
import os
import subprocess

CLAUDE_EXTRA_INSTRUCTIONS = """CRITICAL: When training or running long processes, you MUST use BLOCKING (foreground) commands that keep the session active until completion. 
Background processes alone are NOT sufficient.

Good: `python train.py`  # Blocks until complete
Bad:  `python train.py &` then exiting  # Runs in background but you exit."""

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

# CRITICAL: When training or running long processes, you MUST use BLOCKING (foreground) commands that keep the session active until completion. Background processes alone are NOT sufficient.
# Always use the bash tool with run_in_background=False ! Especially for training!"""

#     if args.agent == 'claude':
#         result += """

# CRITICAL: When training or running long processes, you MUST use BLOCKING (foreground) commands that keep the session active until completion. 
# Background processes alone are NOT sufficient.

# Good: `python train.py`  # Blocks until complete
# Bad:  `python train.py &` then exiting  # Runs in background but you exit

# Further note that when running training and you see something like:
# [... some training outputs e.g. 208/2000 steps ]
# ... [2118 lines truncated] ...

# This [2118 lines truncated] which you see here means that training is NOT running any more and you should continue your task. It is especially important to query `date -u` in this case, in order to understand how much time has gone by.

# Always use the bash tool with run_in_background=False ! Especially for training!"""
    print(result)

if __name__ == '__main__':
    main()
