"""Text processing utilities for HealthBench evaluation."""

import re

MAX_REPETITIONS = 5  # Maximum allowed repetitions of any pattern


def limit_repetitions(text: str, max_reps: int = MAX_REPETITIONS) -> str:
    """
    Limit repetitive patterns in generated text to at most max_reps repetitions.

    Handles multiple types of repetition:
    1. Consecutive identical lines
    2. Repeating blocks of lines (e.g., groups separated by blank lines)
    3. Repeated string patterns (via regex)

    Args:
        text: The input text to process
        max_reps: Maximum allowed repetitions of any pattern (default: 5)

    Returns:
        Text with repetitions limited to max_reps
    """

    def _limit_consecutive_lines(txt: str) -> tuple:
        """Find consecutive identical lines and limit them."""
        lines = txt.split('\n')
        result = []
        i = 0
        modified = False

        while i < len(lines):
            line = lines[i]
            count = 1
            j = i + 1
            while j < len(lines) and lines[j] == line:
                count += 1
                j += 1

            if count > max_reps:
                result.extend([line] * max_reps)
                modified = True
            else:
                result.extend([line] * count)
            i = j

        return '\n'.join(result), modified

    def _limit_block_patterns(txt: str) -> tuple:
        """Find repeating blocks of lines and limit them."""
        lines = txt.split('\n')
        modified = False

        for block_size in range(2, min(30, len(lines) // 5 + 1)):
            i = 0
            new_lines = []
            last_end = 0

            while i <= len(lines) - block_size:
                block = lines[i:i + block_size]

                count = 1
                j = i + block_size
                while j + block_size <= len(lines):
                    next_block = lines[j:j + block_size]
                    if next_block == block:
                        count += 1
                        j += block_size
                    else:
                        break

                if count > max_reps:
                    new_lines.extend(lines[last_end:i])
                    for _ in range(max_reps):
                        new_lines.extend(block)

                    # Check if remaining lines are partial repetition of the block
                    remaining_lines = lines[j:]
                    if remaining_lines:
                        is_partial_repeat = True
                        for k, rem_line in enumerate(remaining_lines):
                            if k >= len(block):
                                is_partial_repeat = False
                                break
                            block_line = block[k % len(block)]
                            if rem_line != block_line:
                                if not (rem_line.strip() and block_line.startswith(rem_line.strip())):
                                    if not (block_line.strip() and rem_line.startswith(block_line.strip())):
                                        is_partial_repeat = False
                                        break
                        if is_partial_repeat:
                            last_end = len(lines)
                        else:
                            last_end = j
                    else:
                        last_end = j

                    i = len(lines)
                    modified = True
                else:
                    i += 1

            if modified:
                new_lines.extend(lines[last_end:])
                return '\n'.join(new_lines), True

        return txt, False

    def _limit_regex_patterns(txt: str) -> tuple:
        """Use regex to find and limit repeated string patterns."""
        modified = False

        while True:
            changed_this_round = False

            for min_len, max_len in [(3, 30), (30, 60), (60, 120), (120, 250), (250, 500)]:
                pattern = rf'(.{{{min_len},{max_len}}}?)\1{{{max_reps},}}'

                def replace_func(match):
                    nonlocal changed_this_round, modified
                    unit = match.group(1)
                    full_match = match.group(0)
                    count = len(full_match) // len(unit)
                    if count > max_reps:
                        modified = True
                        changed_this_round = True
                        return unit * max_reps
                    return full_match

                txt = re.sub(pattern, replace_func, txt, flags=re.DOTALL)

            if not changed_this_round:
                break

        return txt, modified

    # Apply strategies iteratively until no more changes
    for _ in range(10):
        m_any = False

        text, m1 = _limit_consecutive_lines(text)
        m_any = m_any or m1

        text, m2 = _limit_block_patterns(text)
        m_any = m_any or m2

        text, m3 = _limit_regex_patterns(text)
        m_any = m_any or m3

        if not m_any:
            break

    return text
