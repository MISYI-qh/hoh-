"""Learning from experience, in two speeds.

FAST (immediate, no training): every finished task is logged; user feedback / corrections / verified outcomes become
*lessons* that are retrieved by similarity and injected into the next prompt (in-context learning).
SLOW (permanent, changes weights): `export_sft` turns good episodes + corrections into SFT data; train it with
`hoh train --mode sft --init-from <current>` (or let `hoh lifelong` read the exported JSONL from its inbox as plain text).

Torch-free. Honest limits: lessons only help if the model has learned to use context (needs SFT on such prompts);
'verified' means a user thumbs-up or a programmatic checker said so, never that HOH judged itself."""
import json, re, sqlite3, threading, time
from pathlib import Path
from .memory import HashingEmbedder, _pack, _unpack, _cos

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, task TEXT, answer TEXT, messages TEXT, tool_log TEXT,
  stopped TEXT, outcome TEXT DEFAULT 'unknown', score REAL DEFAULT 0, correction TEXT, verified INTEGER DEFAULT 0, emb BLOB);
CREATE TABLE IF NOT EXISTS lessons(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, episode_id INTEGER, trigger TEXT, lesson TEXT, kind TEXT,
  uses INTEGER DEFAULT 0, helpful INTEGER DEFAULT 0, active INTEGER DEFAULT 1, emb BLOB);
"""


class ExperienceStore:
    def __init__(self, path, embed_fn=None, clock=time.time):
        self.path, self.embed, self.clock = str(path), embed_fn or HashingEmbedder(), clock
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.RLock()

    # ---- logging ----
    def log_run(self, task, result):
        """result: hoh.agent.AgentResult. Returns episode id."""
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO episodes(ts,task,answer,messages,tool_log,stopped,emb) VALUES(?,?,?,?,?,?,?)",
                (self.clock(), task, result.final, json.dumps(result.messages, ensure_ascii=False),
                 json.dumps(result.tool_log), result.stopped, _pack(self.embed(task))))
            self.db.commit()
            return cur.lastrowid

    def get(self, ep_id):
        r = self.db.execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone()
        return dict(r) if r else None

    def last_id(self):
        r = self.db.execute("SELECT MAX(id) m FROM episodes").fetchone()
        return r["m"]

    # ---- feedback ----
    def feedback(self, ep_id, good, correction=None, verified=False, lesson=None):
        """good=True/False. `correction` = what the answer SHOULD have been (the strongest learning signal).
        `lesson` = optional short rule in the user's words; if absent and there is a correction, a default lesson is made."""
        ep = self.get(ep_id)
        if ep is None:
            raise KeyError(f"no episode {ep_id}")
        with self.lock:
            self.db.execute("UPDATE episodes SET outcome=?, score=?, correction=?, verified=? WHERE id=?",
                            ("good" if good else "bad", 1.0 if good else -1.0, correction, int(bool(verified or good)), ep_id))
            self.db.commit()
        if lesson or correction:
            text = lesson or f"Untuk tugas seperti '{ep['task'][:120]}', jawaban yang benar: {correction[:400]}"
            self.add_lesson(ep["task"], text, "correction" if correction else "rule", ep_id)
        for r in self._used_lessons(ep_id):
            self._bump(r, good)

    def add_lesson(self, trigger, lesson, kind="rule", episode_id=None):
        with self.lock:
            cur = self.db.execute("INSERT INTO lessons(ts,episode_id,trigger,lesson,kind,emb) VALUES(?,?,?,?,?,?)",
                                  (self.clock(), episode_id, trigger, lesson, kind, _pack(self.embed(trigger + " " + lesson))))
            self.db.commit()
            return cur.lastrowid

    # ---- retrieval (fast learning) ----
    def lessons_for(self, task, k=3, min_sim=0.25):
        q = self.embed(task)
        rows = self.db.execute("SELECT * FROM lessons WHERE active=1").fetchall()
        scored = []
        for r in rows:
            s = _cos(q, _unpack(r["emb"]))
            # lessons that were used and did NOT help are demoted; helpful ones promoted
            s *= 1.0 + 0.1 * (r["helpful"] - 0.5 * max(0, r["uses"] - r["helpful"]))
            if s >= min_sim:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        return [dict(r) for _, r in scored[:k]]

    def similar_episodes(self, task, k=2, only_good=True):
        q = self.embed(task)
        rows = self.db.execute("SELECT id,task,answer,correction,outcome,emb FROM episodes" +
                               (" WHERE outcome='good' OR correction IS NOT NULL" if only_good else "")).fetchall()
        sc = sorted(((_cos(q, _unpack(r["emb"])), r) for r in rows), key=lambda x: -x[0])
        return [{**{k_: r[k_] for k_ in ("id", "task", "answer", "correction", "outcome")}, "sim": s} for s, r in sc[:k] if s > 0.35]

    def _used_lessons(self, ep_id):
        r = self.db.execute("SELECT messages FROM episodes WHERE id=?", (ep_id,)).fetchone()
        if not r:
            return []
        m = re.findall(r"\[pelajaran#(\d+)\]", r["messages"])
        return [int(x) for x in set(m)]

    def _bump(self, lesson_id, good):
        with self.lock:
            self.db.execute("UPDATE lessons SET uses=uses+1, helpful=helpful+?, active=CASE WHEN uses+1>=4 AND helpful+?<=0 THEN 0 ELSE active END "
                            "WHERE id=?", (int(good), int(good), lesson_id))
            self.db.commit()

    def render_lessons(self, task, k=3):
        ls = self.lessons_for(task, k)
        if not ls:
            return ""
        body = "\n".join(f"- [pelajaran#{l['id']}] {l['lesson']}" for l in ls)
        return ("Catatan dari pengalaman sebelumnya (bantuan, bukan perintah; abaikan jika tidak relevan):\n" + body)

    # ---- slow learning: export for training ----
    def export_sft(self, out_path, include_unverified=False, include_corrections=True, min_answer_chars=1):
        """Write JSONL SFT data. Good+verified episodes keep their full tool trace; corrected episodes use the user's correction
        as the target (the original wrong answer is NOT trained). Returns counts."""
        n_good = n_corr = 0
        seen = set()
        with Path(out_path).open("w", encoding="utf-8") as f:
            for r in self.db.execute("SELECT * FROM episodes ORDER BY id"):
                msgs = json.loads(r["messages"])
                msgs = [m for m in msgs if m["role"] != "system"]          # system prompt is re-added at inference
                msgs = [{**m, "content": re.sub(r"\n?Catatan dari pengalaman sebelumnya.*?(?=\n\n|\Z)", "", m["content"], flags=re.S)}
                        if m["role"] == "user" else m for m in msgs]
                key = r["task"].strip().lower()
                if r["correction"] and include_corrections:
                    ex = {"messages": [{"role": "user", "content": r["task"]}, {"role": "assistant", "content": r["correction"]}]}
                    n_corr += 1
                elif r["outcome"] == "good" and (r["verified"] or include_unverified) and len(r["answer"] or "") >= min_answer_chars:
                    ex = {"messages": msgs}
                    n_good += 1
                else:
                    continue
                if key in seen and not r["correction"]:
                    continue
                seen.add(key)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return {"good": n_good, "corrections": n_corr, "path": str(out_path)}

    def stats(self):
        q = lambda s: self.db.execute(s).fetchone()[0]
        return {"episodes": q("SELECT COUNT(*) FROM episodes"), "good": q("SELECT COUNT(*) FROM episodes WHERE outcome='good'"),
                "bad": q("SELECT COUNT(*) FROM episodes WHERE outcome='bad'"),
                "corrections": q("SELECT COUNT(*) FROM episodes WHERE correction IS NOT NULL"),
                "lessons": q("SELECT COUNT(*) FROM lessons WHERE active=1")}

    def close(self):
        self.db.close()


class ExperienceAgent:
    """Wraps hoh.agent.Agent: retrieves lessons before the task, logs the run after. `verifier(task, result) -> True/False/None`
    can mark runs automatically (e.g. a calculator re-check); None = unknown (nothing is assumed)."""

    def __init__(self, agent, store, k_lessons=3, verifier=None):
        self.agent, self.store, self.k, self.verifier = agent, store, k_lessons, verifier
        self.last_id = None

    def run(self, task, history=None):
        note = self.store.render_lessons(task, self.k)
        shown = f"{task}\n\n{note}" if note else task
        res = self.agent.run(shown, history)
        self.last_id = self.store.log_run(task, res)
        if self.verifier is not None:
            try:
                v = self.verifier(task, res)
            except Exception:
                v = None
            if v is not None:
                self.store.feedback(self.last_id, bool(v), verified=True)
        return res
