"""Torch-free automatic tokenizer growth.

The tokenizer has no unknown-word limit (any text encodes through byte fallback) and no vocabulary
ceiling unless one is configured. It grows only by APPENDING merges, so every old id keeps its meaning.
Growth is judged on held-out text: a round of merges is kept only if it lowers tokens-per-byte there."""
import time
from dataclasses import dataclass
from .tokenizer import BPETokenizer, BASE_VOCAB


@dataclass
class TokenizerPolicy:
    round_size: int = 128            # merges tried per round
    min_gain: float = 0.005          # keep a round only if held-out tokens/byte drops by >= 0.5%
    min_freq: int = 4                # a pair must occur this often to become a token
    max_rounds: int = 16             # bounds work per cycle, NOT the vocabulary
    sample_chars: int = 300_000      # text sampled per cycle (pure-Python BPE is slow on big samples)
    line_chars: int = 400            # sample is cut into pieces of at most this many chars
    max_vocab: int = 0               # 0 = no vocabulary ceiling
    time_budget_s: float = 120.0


def _pieces(texts, line_chars, limit):
    out, used = [], 0
    for t in texts:
        for i in range(0, len(t), line_chars):
            piece = t[i:i + line_chars]
            if piece.strip():
                out.append(piece)
                used += len(piece)
                if used >= limit:
                    return out
    return out


def tokens_per_byte(tok, pieces):
    n_tok = n_byte = 0
    for p in pieces:
        n_tok += len(tok.encode(p))
        n_byte += len(p.encode("utf-8", errors="replace"))
    return n_tok / n_byte if n_byte else 1.0


def grow_tokenizer(tok, texts, policy=None):
    """Return (new_tokenizer, report). `tok` is never modified. Old merges stay a strict prefix."""
    policy = policy or TokenizerPolicy()
    base = BPETokenizer(getattr(tok, "merges", []))
    pieces = _pieces(texts, policy.line_chars, policy.sample_chars)
    report = {"vocab_before": base.vocab_size, "vocab_after": base.vocab_size, "added": 0, "rounds": 0,
              "tpb_before": None, "tpb_after": None, "stopped": ""}
    if len(pieces) < 10:
        report["stopped"] = "not enough text sampled"
        return base, report
    learn = [p for i, p in enumerate(pieces) if i % 5 != 0]       # 80% to learn merges
    probe = [p for i, p in enumerate(pieces) if i % 5 == 0]       # 20% held out to judge them
    report["tpb_before"] = cur_tpb = tokens_per_byte(base, probe)
    cand = BPETokenizer(base.merges)
    keep = len(base.merges)
    t0 = time.time()
    for r in range(policy.max_rounds):
        if policy.max_vocab and cand.vocab_size >= policy.max_vocab:
            report["stopped"] = "max_vocab reached"
            break
        n = policy.round_size
        if policy.max_vocab:
            n = min(n, policy.max_vocab - cand.vocab_size)
        added = cand.learn_merges(learn, n, policy.min_freq)
        report["rounds"] = r + 1
        if added == 0:
            report["stopped"] = "no frequent pair left"
            break
        new_tpb = tokens_per_byte(cand, probe)
        gain = (cur_tpb - new_tpb) / cur_tpb if cur_tpb else 0.0
        if gain < policy.min_gain:
            report["stopped"] = f"round gain {gain:.2%} < {policy.min_gain:.2%}"
            break                                                  # this round is not kept
        keep, cur_tpb = len(cand.merges), new_tpb
        if time.time() - t0 > policy.time_budget_s:
            report["stopped"] = "time budget"
            break
    else:
        report["stopped"] = "max_rounds"
    final = BPETokenizer(cand.merges[:keep])                       # truncating = dropping only the unkept tail
    report.update(vocab_after=final.vocab_size, added=final.vocab_size - base.vocab_size, tpb_after=cur_tpb)
    return final, report
