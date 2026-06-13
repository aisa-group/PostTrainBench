from score import grade, strip_boxed


def test_suffix_false_positive_rejected():
    _, ok = grade("ANSWER: 711", "11")
    assert not ok


def test_dataset_suffix_collision_rejected():
    _, ok = grade("ANSWER: 149", "49")
    assert not ok


def test_correct_answer_line():
    _, ok = grade("Step by step...\nANSWER: 127", "127")
    assert ok


def test_answer_line_beats_trailing_wrong_number():
    _, ok = grade("work shows 149\nANSWER: 49", "49")
    assert ok


def test_strip_boxed_multiple():
    assert strip_boxed(r"\boxed{1} and \boxed{2}") == "1 and 2"
