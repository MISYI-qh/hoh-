"""Byte tokenizer + append-only BPE growth.

IDs 0-255 are bytes, 256-259 are PAD/BOS/EOS/UNK, and learned merge k always gets id 260+k.
Merges are only ever appended, so an old id never changes meaning (old checkpoints stay valid
as a prefix of a grown vocabulary)."""
import json
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = 256, 257, 258, 259
BASE_VOCAB = 260
VOCAB_SIZE = 260
_SPECIAL_BYTES = {PAD: b"<PAD>", BOS: b"<BOS>", EOS: b"<EOS>", UNK: b"<UNK>"}
_SPECIALS = {"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK}


class ByteTokenizer:
    vocab_size = VOCAB_SIZE
    pad_token_id, bos_token_id, eos_token_id = PAD, BOS, EOS

    def _bytes_to_ids(self, text):
        return list(text.encode("utf-8", errors="replace"))

    def encode(self, text, add_bos=False, add_eos=False):
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        ids = self._bytes_to_ids(text)
        if add_bos:
            ids.insert(0, BOS)
        if add_eos:
            ids.append(EOS)
        return ids

    def _token_bytes(self, i):
        return bytes([i]) if 0 <= i < 256 else None

    def decode(self, ids, skip_special=True):
        out = bytearray()
        for i in ids:
            i = int(i)
            b = self._token_bytes(i)
            if b is not None:
                out.extend(b)
            elif not skip_special and i in _SPECIAL_BYTES:
                out.extend(_SPECIAL_BYTES[i])
        return out.decode("utf-8", errors="replace")

    def to_dict(self):
        return {"type": "utf8-byte", "vocab_size": 260, "special_tokens": dict(_SPECIALS)}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path=None):
        return load_tokenizer(path) if path else cls()


class BPETokenizer(ByteTokenizer):
    def __init__(self, merges=None):
        self.merges = []
        self._rank = {}
        self._bytes = {}
        for pair in (merges or []):
            self._append(tuple(pair))

    @property
    def vocab_size(self):
        return BASE_VOCAB + len(self.merges)

    def _append(self, pair):
        a, b = int(pair[0]), int(pair[1])
        new_id = BASE_VOCAB + len(self.merges)
        for t in (a, b):
            if not (0 <= t < 256 or BASE_VOCAB <= t < new_id):
                raise ValueError(f"merge {pair} references invalid token id {t}")
        if (a, b) in self._rank:
            raise ValueError(f"duplicate merge {(a, b)}")
        self._rank[(a, b)] = len(self.merges)
        self.merges.append((a, b))
        self._bytes[new_id] = self._token_bytes(a) + self._token_bytes(b)

    def _token_bytes(self, i):
        if 0 <= i < 256:
            return bytes([i])
        return self._bytes.get(i)

    @staticmethod
    def _merge(ids, a, b, new):
        out, i = [], 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _bytes_to_ids(self, text):
        ids = list(text.encode("utf-8", errors="replace"))
        while len(ids) > 1 and self._rank:
            best = None
            for pair in zip(ids, ids[1:]):
                r = self._rank.get(pair)
                if r is not None and (best is None or r < best):
                    best = r
            if best is None:
                break
            a, b = self.merges[best]
            ids = self._merge(ids, a, b, BASE_VOCAB + best)
        return ids

    def learn_merges(self, texts, n_new, min_freq=2):
        """Append up to n_new merges learned from `texts` (iterable of str). Returns #added.
        Policy: most frequent adjacent pair under the CURRENT tokenization; stop when
        frequency < min_freq. Existing ids are never altered."""
        seqs = [self._bytes_to_ids(t) for t in texts if t]
        added = 0
        while added < n_new:
            counts = Counter()
            for s in seqs:
                counts.update(zip(s, s[1:]))
            if not counts:
                break
            (a, b), freq = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if freq < min_freq:
                break
            new_id = BASE_VOCAB + len(self.merges)
            self._append((a, b))
            seqs = [self._merge(s, a, b, new_id) for s in seqs]
            added += 1
        return added

    def to_dict(self):
        return {"type": "bpe", "version": 1, "vocab_size": self.vocab_size,
                "special_tokens": dict(_SPECIALS), "merges": [list(m) for m in self.merges]}


def tokenizer_from_dict(d):
    t = d.get("type")
    if t == "utf8-byte":
        return ByteTokenizer()
    if t == "bpe":
        tok = BPETokenizer(d.get("merges", []))
        if d.get("vocab_size", tok.vocab_size) != tok.vocab_size:
            raise ValueError("tokenizer vocab_size does not match its merge list (corrupt file?)")
        return tok
    raise ValueError(f"unknown tokenizer type {t!r}")


def load_tokenizer(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Tokenizer file not found: {p}")
    return tokenizer_from_dict(json.loads(p.read_text(encoding="utf-8")))
