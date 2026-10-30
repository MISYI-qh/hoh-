from pathlib import Path
import numpy as np
from .sft import (DatasetError, IGNORE_INDEX, read_jsonl, record_to_text, load_sft_examples,
                  example_to_xy, split_items)
from .tokenizer import ByteTokenizer, PAD

TEXT_SUFFIXES = {".txt", ".md", ".jsonl"}


def read_text_file(path):
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset path not found: {p}")
    if p.is_dir():
        chunks = [read_text_file(f) for f in sorted(p.rglob("*"))
                  if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES]
        chunks = [c for c in chunks if c.strip()]
        if not chunks:
            raise DatasetError(f"No non-empty .txt/.md/.jsonl files found in {p}")
        return "\n\n".join(chunks)
    if p.suffix.lower() == ".jsonl":
        return "\n\n".join(record_to_text(obj, f"{p}:{n}") for n, obj in read_jsonl(p))
    try:
        return p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise DatasetError(f"{p}: not valid UTF-8 ({e})") from e


def load_tokens(path, tokenizer=None):
    """Pretraining stream: one flat token list (BOS ... EOS)."""
    tokenizer = tokenizer or ByteTokenizer()
    text = read_text_file(path)
    n = len(text.encode("utf-8"))
    if n < 32:
        raise DatasetError(f"Dataset too small ({n} bytes). Provide a real text corpus.")
    return tokenizer.encode(text, add_bos=True, add_eos=True)


def split_tokens(tokens, block_size, val_fraction=0.02):
    """Contiguous tail split (no overlap). Needs >= block_size+2 tokens on each side."""
    need = block_size + 2
    n_val = max(int(len(tokens) * val_fraction), need)
    if len(tokens) - n_val < need:
        raise DatasetError(
            f"Dataset too small for block_size={block_size}: {len(tokens)} tokens, need at least {2 * need}.")
    return tokens[:-n_val], tokens[-n_val:]


class TokenBatcher:
    """`device` is accepted for call-site compatibility with the PyTorch build and ignored (NumPy = CPU)."""

    def __init__(self, tokens, block_size, device, batch_size, rng=None):
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.block_size, self.batch_size = block_size, batch_size
        self.rng = rng if rng is not None else np.random.default_rng()
        if len(self.tokens) < block_size + 2:
            raise DatasetError(f"Need at least block_size+2 tokens; got {len(self.tokens)} for block_size={block_size}.")

    def get(self):
        ix = self.rng.integers(0, len(self.tokens) - self.block_size - 1, self.batch_size)
        off = np.arange(self.block_size)
        return self.tokens[ix[:, None] + off], self.tokens[ix[:, None] + off + 1]


class SFTBatcher:
    """Right-padded batches; prompt tokens and padding get IGNORE_INDEX targets."""

    def __init__(self, examples, device, batch_size, rng=None):
        if not examples:
            raise DatasetError("SFTBatcher needs at least one example")
        self.examples, self.batch_size = examples, batch_size
        self.rng = rng if rng is not None else np.random.default_rng()

    def get(self):
        ix = self.rng.integers(0, len(self.examples), self.batch_size).tolist()
        pairs = [example_to_xy(*self.examples[i]) for i in ix]
        width = max(len(x) for x, _ in pairs)
        X = np.full((len(pairs), width), PAD, dtype=np.int64)
        Y = np.full((len(pairs), width), IGNORE_INDEX, dtype=np.int64)
        for r, (x, y) in enumerate(pairs):
            X[r, :len(x)] = x
            Y[r, :len(y)] = y
        return X, Y
