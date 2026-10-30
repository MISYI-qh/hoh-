"""Semantic (meaning-based) retrieval for HOH memory.

The default HashingEmbedder only sees character n-grams ("mobil" ~ "mobilitas", but NOT "mobil" ~ "kendaraan").
This module adds three embedders with the same interface (`text -> unit-length list[float]`, attribute `.id`):

* RandomIndexEmbedder  - distributional semantics learned from YOUR OWN text (memory, inbox, chats).
                          Words used in similar contexts get similar vectors. Quality grows with the amount of text;
                          with very little text it degrades to exact-word matching (never worse than keywords).
* SentenceEmbedder     - optional wrapper over `sentence-transformers` (use a multilingual model); strongest, needs the
                          package + downloaded weights. Not available in every environment.
* ModelEmbedder        - mean-pooled hidden states of a HOH checkpoint . Weak for a small model; mostly useful
                          because it costs nothing extra.

`best_available_embedder()` picks SentenceEmbedder if importable, else RandomIndexEmbedder (trained/loaded from `path`).
When the embedder changes call `MemoryStore.reindex()`: text is untouched, only vectors are recomputed.
HONESTY: none of these "understands" like a person; RandomIndex is a classic, cheap approximation (Sahlgren 2005 style)."""
import array, json, math, os, re, zlib
from collections import Counter, defaultdict
from pathlib import Path

_WORD = re.compile(r"[a-zA-Z\u00C0-\u024F0-9]+")
_STOP = set("yang dan di ke dari untuk dengan pada ini itu adalah ada atau juga akan sudah tidak the of a an to in and is are was for on".split())
_PREF = ("meng", "meny", "mem", "men", "me", "peng", "peny", "pem", "pen", "per", "pe", "ber", "ter", "di")
_SUFF = ("kan", "nya", "lah", "kah", "an", "i")


def stem_id(w):
    """Very light Indonesian affix stripping ('mempelajari' -> 'pelajar'-ish). Conservative: never below 4 letters."""
    for _ in range(2):
        for sfx in _SUFF:
            if w.endswith(sfx) and len(w) - len(sfx) >= 4:
                w = w[: -len(sfx)]
                break
    for pf in _PREF:
        if w.startswith(pf) and len(w) - len(pf) >= 4:
            w = w[len(pf):]
            break
    return w


def tokenize(text):
    return [stem_id(w) for w in (m.group(0).lower() for m in _WORD.finditer(text)) if w not in _STOP and len(w) > 1]


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else list(v)


class RandomIndexEmbedder:
    """Sparse ternary index vectors + context accumulation. Pure Python, deterministic, persistent."""

    def __init__(self, dim=256, window=3, nnz=8, min_count=2, max_vocab=20000, generation=0):
        self.dim, self.window, self.nnz, self.min_count, self.max_vocab = dim, window, nnz, min_count, max_vocab
        self.generation = generation
        self.ctx = {}          # word -> array('f') accumulated context (unnormalised)
        self.count = Counter() # word -> occurrences
        self.docs = 0
        self.df = Counter()    # document frequency, for idf weighting
        self._frozen = {}      # word -> normalised context vector (built by freeze())
        self._ic = {}          # cache of sparse index vectors

    # ---- identity: changes whenever the learned space changes, which tells MemoryStore to reindex ----
    @property
    def id(self):
        return f"ri-d{self.dim}-g{self.generation}"

    def _index(self, word):
        """Deterministic sparse random vector: `nnz` positions with +-1 (seeded by the word itself)."""
        c = self._ic.get(word)
        if c is not None:
            return c
        pos, h = {}, zlib.crc32(word.encode("utf-8"))
        k = 0
        while len(pos) < self.nnz:
            h = zlib.crc32(f"{word}#{k}".encode("utf-8"), h)
            pos.setdefault(h % self.dim, 1.0 if (h >> 20) & 1 else -1.0)
            k += 1
        if len(self._ic) < 200_000:
            self._ic[word] = pos
        return pos

    # ---- learning ----
    def learn(self, texts):
        for t in texts:
            toks = tokenize(t)
            if not toks:
                continue
            self.docs += 1
            self.count.update(toks)
            self.df.update(set(toks))
            for i, w in enumerate(toks):
                c = self.ctx.get(w)
                if c is None:
                    c = self.ctx[w] = array.array("f", [0.0]) * self.dim
                for j in range(max(0, i - self.window), min(len(toks), i + self.window + 1)):
                    if j == i or toks[j] == w:
                        continue
                    wt = 1.0 / abs(j - i)
                    for p, sgn in self._index(toks[j]).items():
                        c[p] += sgn * wt
        return self

    def freeze(self):
        """Finish a training round: normalise context vectors for words seen >= min_count; bump generation."""
        keep = [w for w, n in self.count.most_common(self.max_vocab) if n >= self.min_count and w in self.ctx]
        self._frozen = {w: _unit(list(self.ctx[w])) for w in keep}
        self.generation += 1
        return len(self._frozen)

    def _idf(self, w):
        return math.log((1 + self.docs) / (1 + self.df.get(w, 0))) + 1.0

    # ---- embedding ----
    def __call__(self, text):
        v = [0.0] * self.dim
        toks = tokenize(text)
        for w in toks:
            wt = self._idf(w)
            for p, sgn in self._index(w).items():          # exact-word channel: always present, so keywords still match
                v[p] += 0.6 * sgn * wt
            f = self._frozen.get(w)
            if f is not None:                                # meaning channel: what the word's neighbours say about it
                for k, x in enumerate(f):
                    v[k] += 1.0 * x * wt
        return _unit(v) if any(v) else v

    def similar_words(self, word, k=5):
        q = self._frozen.get(stem_id(word.lower()))
        if q is None:
            return []
        sc = sorted(((sum(a * b for a, b in zip(q, vec)), w) for w, vec in self._frozen.items()), reverse=True)
        return [(w, round(s, 3)) for s, w in sc if w != stem_id(word.lower())][:k]

    # ---- persistence (atomic, binary vectors + json header) ----
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        hdr = {"dim": self.dim, "window": self.window, "nnz": self.nnz, "min_count": self.min_count,
               "max_vocab": self.max_vocab, "generation": self.generation, "docs": self.docs,
               "words": list(self.ctx), "count": [self.count[w] for w in self.ctx], "df": [self.df[w] for w in self.ctx],
               "frozen": list(self._frozen)}
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            h = json.dumps(hdr, ensure_ascii=False).encode("utf-8")
            f.write(len(h).to_bytes(8, "little") + h)
            for w in self.ctx:
                self.ctx[w].tofile(f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n).decode("utf-8"))
            e = cls(hdr["dim"], hdr["window"], hdr["nnz"], hdr["min_count"], hdr["max_vocab"], hdr["generation"])
            e.docs = hdr["docs"]
            for w, c, d in zip(hdr["words"], hdr["count"], hdr["df"]):
                a = array.array("f")
                a.fromfile(f, hdr["dim"])
                e.ctx[w], e.count[w], e.df[w] = a, c, d
        e._frozen = {w: _unit(list(e.ctx[w])) for w in hdr["frozen"] if w in e.ctx}
        return e


class SentenceEmbedder:
    """Optional: sentence-transformers. Raises ImportError with a clear message when unavailable."""

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers (and download a multilingual model) for strong semantic search") from e
        self.model, self.id = SentenceTransformer(model_name), f"st-{model_name}"

    def __call__(self, text):
        return _unit([float(x) for x in self.model.encode(text, normalize_embeddings=True)])


class ModelEmbedder:
    """Mean-pooled final hidden states of a HOH checkpoint. Weak for small models; see module docstring."""

    def __init__(self, checkpoint, device="cpu"):
        from .generate import load_model
        self.model, self.dev, self.tok = load_model(checkpoint, device, return_tokenizer=True)
        self.id = f"hoh-{Path(checkpoint).name}-{os.path.getmtime(checkpoint):.0f}"

    def __call__(self, text):
        import numpy as np
        ids = self.tok.encode(text, add_bos=True, add_eos=False)[: self.model.config.block_size]
        h = self.model.hidden_states(np.asarray([ids], dtype=np.int64))
        return _unit([float(v) for v in h.mean(axis=1)[0]])


def best_available_embedder(path=None, prefer_sentence=True):
    if prefer_sentence:
        try:
            return SentenceEmbedder()
        except Exception:
            pass
    if path and Path(path).exists():
        return RandomIndexEmbedder.load(path)
    return RandomIndexEmbedder()


def train_from_memory(memory, workspace, path, layers=("knowledge", "persistent", "conversation"), extra_texts=()):
    """Learn word meaning from the memory itself (+ optional extra texts), freeze, save, and reindex the store.
    Returns {'vocab', 'generation', 'reindexed'}. Cheap enough to run during `sleep`."""
    emb = RandomIndexEmbedder.load(path) if Path(path).exists() else RandomIndexEmbedder()
    texts = [r["content"] for r in memory.rows_before(workspace, layers, float("inf"), limit=200_000)]
    emb.learn(list(texts) + list(extra_texts))
    vocab = emb.freeze()
    emb.save(path)
    memory.embed = emb
    return {"vocab": vocab, "generation": emb.generation, "reindexed": memory.reindex()}
