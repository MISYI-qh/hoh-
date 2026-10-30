"""Memory consolidation ("sleep" for the notebook): summarise old, related notes into one compact note.

Rules that keep it safe:
* Originals are ARCHIVED, never deleted: they stay searchable and `unconsolidate()` brings them back.
* Only rows older than `older_than_days`, with importance < `keep_importance`, not pinned, not already summaries,
  and (by default) not in the 'persistent' layer are touched.
* A group is replaced only when the summary is clearly shorter than the originals (`min_gain`), otherwise nothing changes.
* The default summariser is EXTRACTIVE (it picks the most central original sentences, so it cannot invent facts).
  Pass `summarizer(texts) -> str` to use the model instead; its output is length-checked the same way.
"""
import math, re, time
from collections import Counter

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_W = re.compile(r"\w+", re.U)


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) if len(a) == len(b) else 0.0


def _split_sentences(text):
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) > 12]


def extractive_summary(texts, max_chars=700, idf=None):
    """Pick the sentences most similar to the group's word centroid, drop near-duplicates, keep original order."""
    sents = []
    for ti, t in enumerate(texts):
        for si, s in enumerate(_split_sentences(t) or [t.strip()]):
            sents.append((ti, si, s))
    if not sents:
        return ""
    docs = [Counter(w.lower() for w in _W.findall(s)) for _, _, s in sents]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    weight = lambda w: math.log((1 + n) / (1 + df[w])) + 1.0
    cent = Counter()
    for d in docs:
        for w, c in d.items():
            cent[w] += c * weight(w)

    def score(d):
        num = sum(cent[w] * c * weight(w) for w, c in d.items())
        den = math.sqrt(sum(c * c for c in d.values())) or 1.0
        return num / den

    ranked = sorted(range(n), key=lambda i: -score(docs[i]))
    chosen, used, words_seen = [], 0, []
    for i in ranked:
        ws = set(docs[i])
        if any(len(ws & o) / max(1, len(ws | o)) > 0.6 for o in words_seen):       # near-duplicate of a chosen sentence
            continue
        if used + len(sents[i][2]) > max_chars and chosen:
            continue
        chosen.append(i)
        words_seen.append(ws)
        used += len(sents[i][2])
        if used >= max_chars:
            break
    chosen.sort(key=lambda i: (sents[i][0], sents[i][1]))
    return " ".join(sents[i][2] for i in chosen)


def _group(rows, embed, threshold, max_group):
    """Greedy single-pass clustering on embeddings (centroid is the running mean, re-normalised)."""
    groups = []
    for r in rows:
        v = embed(r["content"])
        best, bs = None, threshold
        for g in groups:
            if len(g["rows"]) >= max_group:
                continue
            s = _cos(v, g["c"])
            if s >= bs:
                best, bs = g, s
        if best is None:
            groups.append({"c": v, "rows": [r], "sum": list(v)})
        else:
            best["rows"].append(r)
            best["sum"] = [a + b for a, b in zip(best["sum"], v)]
            n = math.sqrt(sum(x * x for x in best["sum"])) or 1.0
            best["c"] = [x / n for x in best["sum"]]
    return groups


def consolidate(memory, workspace, older_than_days=14.0, layers=("conversation", "knowledge"), summarizer=None,
                min_group=4, max_group=12, sim_threshold=0.35, keep_importance=0.8, min_gain=0.3,
                max_rows=2000, now=None, dry_run=False, summary_layer="knowledge"):
    now = time.time() if now is None else now
    cutoff = now - older_than_days * 86400
    rows = [r for r in memory.rows_before(workspace, layers, cutoff, limit=max_rows)
            if r["importance"] < keep_importance and not r["meta"].get("pinned") and r["meta"].get("kind") != "summary"]
    rep = {"candidates": len(rows), "groups": 0, "summarised": 0, "archived": 0, "chars_before": 0, "chars_after": 0,
           "skipped_no_gain": 0, "summary_ids": []}
    for g in _group(rows, memory.embed, sim_threshold, max_group):
        if len(g["rows"]) < min_group:
            continue
        rep["groups"] += 1
        texts = [r["content"] for r in g["rows"]]
        before = sum(len(t) for t in texts)
        try:
            summ = (summarizer(texts) if summarizer else extractive_summary(texts)).strip()
        except Exception:
            summ = extractive_summary(texts)            # a failing model summariser must not block consolidation
        if not summ or len(summ) > before * (1 - min_gain):
            rep["skipped_no_gain"] += 1
            continue
        if dry_run:
            rep["summarised"] += 1
            rep["chars_before"] += before
            rep["chars_after"] += len(summ)
            continue
        t0, t1 = min(r["created"] for r in g["rows"]), max(r["created"] for r in g["rows"])
        day = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t))
        text = f"Ringkasan {len(texts)} catatan ({day(t0)} s/d {day(t1)}): {summ}"
        ids = [r["id"] for r in g["rows"]]
        sid, _ = memory.add(workspace, summary_layer, text,
                            {"kind": "summary", "consolidated_from": ids, "span": [t0, t1]},
                            importance=min(1.0, max(r["importance"] for r in g["rows"]) + 0.05))
        rep["archived"] += memory.archive_ids(workspace, ids)
        rep["summarised"] += 1
        rep["chars_before"] += before
        rep["chars_after"] += len(text)
        rep["summary_ids"].append(sid)
    return rep


def unconsolidate(memory, workspace, summary_id):
    """Undo one consolidation: restore the originals from the archive and remove the summary row."""
    s = memory.get(workspace, summary_id)
    if s is None or s["meta"].get("kind") != "summary":
        return {"restored": 0, "error": "not a summary row"}
    ids = set(s["meta"].get("consolidated_from", []))
    restored = 0
    for r in memory.db.execute("SELECT id, orig_id FROM memories_archive WHERE workspace=?", (workspace,)).fetchall():
        if r["orig_id"] in ids:
            if memory.restore(workspace, r["id"]) is not None:
                restored += 1
    memory.delete(workspace, summary_id)
    return {"restored": restored}
