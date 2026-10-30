"""Torch-free stopping helpers for open-ended generation."""


def is_looping(ids, min_repeats=6, max_period=64, min_len=48):
    """True if the tail of `ids` is one short block repeated many times (a degenerate loop).
    Needs at least `min_len` repeating tokens, so a short legitimate run like '=====' is not flagged."""
    n = len(ids)
    for p in range(1, max_period + 1):
        need = max(p * min_repeats, min_len)
        if n < need:
            continue
        tail = ids[-need:]
        if all(tail[i] == tail[i - p] for i in range(p, need)):
            return True
    return False
