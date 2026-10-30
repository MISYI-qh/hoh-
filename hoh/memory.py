"""External persistent memory (SQLite). Completely separate from model parameters: storing
something here never changes the model; it only becomes visible to the model if an agent retrieves
it into the prompt.

Retrieval modes: 'keyword' (FTS5 if available, else LIKE) and 'vector' (cosine over stored embeddings).
The default HashingEmbedder is character-n-gram hashing = SURFACE similarity, NOT semantic
understanding. Plug a real embedding model through `embed_fn` for semantic retrieval.

Elastic mode (default): nothing is discarded to make room. When the "hot" table passes `max_rows`,
the least important/least recently used CONVERSATION rows move to a searchable archive table; if only
persistent/knowledge rows remain the limit itself grows. The byte limit also grows (while the disk has
room) instead of refusing writes. `elastic=False` restores the old evict-and-raise behaviour."""
import functools, json, math, os, re, shutil, sqlite3, struct, threading, time, zlib
from hashlib import sha256
from pathlib import Path

SCHEMA_VERSION = 2
LAYERS = ("conversation", "persistent", "knowledge")


class MemoryError_(Exception):
    pass


class MemoryCorruptError(MemoryError_):
    pass


class MemoryLimitError(MemoryError_):
    pass


class HashingEmbedder:
    def __init__(self, dim=256, ngrams=(3, 4)):
        self.dim, self.ngrams = dim, ngrams

    def __call__(self, text):
        t = " " + re.sub(r"\s+", " ", text.lower()) + " "
        v = [0.0] * self.dim
        for n in self.ngrams:
            for i in range(max(0, len(t) - n + 1)):
                h = zlib.crc32(t[i:i + n].encode("utf-8"))
                v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v


def _pack(v):
    return struct.pack(f"{len(v)}f", *v)


def _unpack(b):
    return list(struct.unpack(f"{len(b) // 4}f", b))


def _cos(a, b):
    if len(a) != len(b):          # vectors from different embedders are NOT comparable: never score them
        return 0.0
    return sum(x * y for x, y in zip(a, b))  # vectors are unit length


def _locked(fn):
    @functools.wraps(fn)
    def w(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return w


class WorkingMemory:
    """Per-task scratch space; in-process only, bounded, never persisted."""

    def __init__(self, max_items=50):
        self.max_items, self.items = max_items, []

    def add(self, text):
        self.items.append(text)
        del self.items[:-self.max_items]

    def clear(self):
        self.items.clear()


class MemoryStore:
    def __init__(self, path, embed_fn=None, max_rows_per_workspace=10000, max_db_bytes=256 * 1024 * 1024,
                 elastic=True, grow_factor=2.0, min_free_bytes=512 * 1024 * 1024):
        self.path = str(path)
        self.embed = embed_fn or HashingEmbedder()
        self.max_rows, self.max_bytes = max_rows_per_workspace, max_db_bytes
        self.elastic, self.grow_factor, self.min_free_bytes = elastic, max(1.1, float(grow_factor)), min_free_bytes
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock = threading.RLock()  # tools run in worker threads
            self.db = sqlite3.connect(self.path, check_same_thread=False)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self._migrate()
            self.fts = self._init_fts()
            self._init_kv()
            if not self._kv_get("embedder_id") and not self.db.execute("SELECT 1 FROM memories LIMIT 1").fetchone() \
                    and not self.db.execute("SELECT 1 FROM memories_archive LIMIT 1").fetchone():
                self._kv_set("embedder_id", self.embedder_id)       # brand-new store: remember which embedder made its vectors
        except sqlite3.DatabaseError as e:
            raise MemoryCorruptError(f"{self.path}: database unreadable/corrupt ({e}); use recover_from_backup()") from e

    # ---- schema / migrations -------------------------------------------------
    def _migrate(self):
        v = self.db.execute("PRAGMA user_version").fetchone()[0]
        if v > SCHEMA_VERSION:
            raise MemoryError_(f"database schema {v} is newer than supported {SCHEMA_VERSION}")
        if v < 1:
            with self.db:
                self.db.executescript("""
                CREATE TABLE IF NOT EXISTS memories(
                  id INTEGER PRIMARY KEY, workspace TEXT NOT NULL, layer TEXT NOT NULL,
                  content TEXT NOT NULL, hash TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}',
                  importance REAL NOT NULL DEFAULT 0.5, created REAL NOT NULL, updated REAL NOT NULL,
                  accessed REAL NOT NULL, embedding BLOB,
                  UNIQUE(workspace, layer, hash));
                CREATE INDEX IF NOT EXISTS idx_ws_layer ON memories(workspace, layer);
                CREATE INDEX IF NOT EXISTS idx_created ON memories(created);
                """)
                self.db.execute("PRAGMA user_version=1")
            v = 1
        if v < 2:
            with self.db:
                self.db.executescript("""
                CREATE TABLE IF NOT EXISTS memories_archive(
                  id INTEGER PRIMARY KEY, orig_id INTEGER, workspace TEXT NOT NULL, layer TEXT NOT NULL,
                  content TEXT NOT NULL, hash TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}',
                  importance REAL NOT NULL DEFAULT 0.5, created REAL NOT NULL, updated REAL NOT NULL,
                  accessed REAL NOT NULL, archived REAL NOT NULL, embedding BLOB,
                  UNIQUE(workspace, layer, hash));
                CREATE INDEX IF NOT EXISTS idx_arch_ws ON memories_archive(workspace, layer);
                CREATE INDEX IF NOT EXISTS idx_arch_created ON memories_archive(created);
                """)
                self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _init_kv(self):
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT NOT NULL)")

    def _kv_get(self, k, default=""):
        r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def _kv_set(self, k, v):
        with self.db:
            self.db.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))

    @property
    def embedder_id(self):
        """Identity of the embedder in use (attribute `.id` on embed_fn, else its class name)."""
        return getattr(self.embed, "id", type(self.embed).__name__)

    def needs_reindex(self):
        """True when stored vectors were made by a different embedder than the current one."""
        stored = self._kv_get("embedder_id", "")
        if not stored:
            n = self.db.execute("SELECT (SELECT count(*) FROM memories)+(SELECT count(*) FROM memories_archive)").fetchone()[0]
            return n > 0 and self.embedder_id != "HashingEmbedder"     # legacy DBs were always hashing vectors
        return stored != self.embedder_id

    @_locked
    def reindex(self, batch=200, progress=None):
        """Recompute every stored vector (active + archive) with the current embedder. Text is untouched; safe to
        interrupt and re-run (rows are committed per batch and the embedder id is recorded only at the end)."""
        done = 0
        for tbl in ("memories", "memories_archive"):
            ids = [r[0] for r in self.db.execute(f"SELECT id FROM {tbl} ORDER BY id")]
            for i in range(0, len(ids), batch):
                with self.db:
                    for rid in ids[i:i + batch]:
                        row = self.db.execute(f"SELECT content FROM {tbl} WHERE id=?", (rid,)).fetchone()
                        self.db.execute(f"UPDATE {tbl} SET embedding=? WHERE id=?", (_pack(self.embed(row[0])), rid))
                        done += 1
                if progress:
                    progress(done)
        self._kv_set("embedder_id", self.embedder_id)
        return done

    @_locked
    def rows_before(self, workspace, layers, before_ts, limit=1000):
        """Active (not archived) rows older than `before_ts`, oldest first, WITH ids: input for consolidation."""
        for l in layers:
            self._check(workspace, l)
        ph = ",".join("?" * len(layers))
        return [self._row(r) for r in self.db.execute(
            f"SELECT * FROM memories WHERE workspace=? AND layer IN ({ph}) AND created<? ORDER BY created LIMIT ?",
            (workspace, *layers, before_ts, limit))]

    @_locked
    def archive_ids(self, workspace, ids):
        """Move rows to the searchable archive (reversible with restore()). Returns how many moved."""
        n = 0
        for i in ids:
            r = self.db.execute("SELECT 1 FROM memories WHERE id=? AND workspace=?", (i, workspace)).fetchone()
            if r:
                self._archive_row(i)
                n += 1
        return n

    def _init_fts(self):
        try:
            with self.db:
                self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content)")
            return True
        except sqlite3.OperationalError:
            return False

    # ---- CRUD ----------------------------------------------------------------
    @staticmethod
    def _check(workspace, layer):
        if not workspace or not isinstance(workspace, str):
            raise ValueError("workspace must be a non-empty string")
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}")

    @_locked
    def add(self, workspace, layer, content, meta=None, importance=0.5):
        """Returns (id, created_new). Exact duplicates (same workspace/layer/text) are merged."""
        self._check(workspace, layer)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        h = sha256(" ".join(content.split()).lower().encode()).hexdigest()
        now = time.time()
        row = self.db.execute("SELECT id FROM memories WHERE workspace=? AND layer=? AND hash=?",
                              (workspace, layer, h)).fetchone()
        if row:
            with self.db:
                self.db.execute("UPDATE memories SET updated=?, importance=max(importance,?) WHERE id=?",
                                (now, importance, row["id"]))
            return row["id"], False
        arch = self.db.execute("SELECT id FROM memories_archive WHERE workspace=? AND layer=? AND hash=?",
                               (workspace, layer, h)).fetchone()
        if arch:  # same text was archived earlier: bring it back instead of storing a second copy
            new_id = self._restore_row(arch["id"])
            with self.db:
                self.db.execute("UPDATE memories SET updated=?, importance=max(importance,?) WHERE id=?",
                                (now, importance, new_id))
            self.enforce_limits(workspace)
            return new_id, False
        emb = _pack(self.embed(content))
        with self.db:
            cur = self.db.execute(
                "INSERT INTO memories(workspace,layer,content,hash,meta,importance,created,updated,accessed,embedding)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (workspace, layer, content, h, json.dumps(meta or {}), importance, now, now, now, emb))
            if self.fts:
                self.db.execute("INSERT INTO memories_fts(rowid, content) VALUES(?,?)", (cur.lastrowid, content))
        self.enforce_limits(workspace)
        return cur.lastrowid, True

    @_locked
    def get(self, workspace, mem_id):
        r = self.db.execute("SELECT * FROM memories WHERE id=? AND workspace=?", (mem_id, workspace)).fetchone()
        return self._row(r) if r else None

    @_locked
    def update(self, workspace, mem_id, content=None, meta=None, importance=None):
        r = self.get(workspace, mem_id)
        if r is None:
            return False
        content = r["content"] if content is None else content
        if not content.strip():
            raise ValueError("content must be non-empty")
        h = sha256(" ".join(content.split()).lower().encode()).hexdigest()
        with self.db:
            self.db.execute("UPDATE memories SET content=?,hash=?,meta=?,importance=?,updated=?,embedding=? WHERE id=?",
                            (content, h, json.dumps(r["meta"] if meta is None else meta),
                             r["importance"] if importance is None else importance, time.time(),
                             _pack(self.embed(content)), mem_id))
            if self.fts:
                self.db.execute("DELETE FROM memories_fts WHERE rowid=?", (mem_id,))
                self.db.execute("INSERT INTO memories_fts(rowid, content) VALUES(?,?)", (mem_id, content))
        return True

    @_locked
    def delete(self, workspace, mem_id):
        with self.db:
            n = self.db.execute("DELETE FROM memories WHERE id=? AND workspace=?", (mem_id, workspace)).rowcount
            if n and self.fts:
                self.db.execute("DELETE FROM memories_fts WHERE rowid=?", (mem_id,))
        return bool(n)

    @_locked
    def delete_workspace(self, workspace):
        ids = [r[0] for r in self.db.execute("SELECT id FROM memories WHERE workspace=?", (workspace,))]
        for i in ids:
            self.delete(workspace, i)
        with self.db:
            n_arch = self.db.execute("DELETE FROM memories_archive WHERE workspace=?", (workspace,)).rowcount
        return len(ids) + n_arch

    @staticmethod
    def _row(r):
        d = dict(r)
        d["meta"] = json.loads(d["meta"])
        d.pop("embedding", None)
        return d

    @_locked
    def count_archived(self, workspace=None):
        if workspace:
            return self.db.execute("SELECT count(*) FROM memories_archive WHERE workspace=?", (workspace,)).fetchone()[0]
        return self.db.execute("SELECT count(*) FROM memories_archive").fetchone()[0]

    @_locked
    def count(self, workspace=None):
        if workspace:
            return self.db.execute("SELECT count(*) FROM memories WHERE workspace=?", (workspace,)).fetchone()[0]
        return self.db.execute("SELECT count(*) FROM memories").fetchone()[0]

    # ---- search --------------------------------------------------------------
    @_locked
    def search(self, workspace, query, k=5, layers=None, mode="hybrid", include_archive=True):
        if mode not in ("keyword", "vector", "hybrid"):
            raise ValueError("mode must be keyword, vector or hybrid")
        if not query.strip():
            raise ValueError("query must be non-empty")
        layers = tuple(layers or LAYERS)
        for l in layers:
            self._check(workspace, l)
        ranks = {}
        if mode in ("keyword", "hybrid"):
            for r, i in enumerate(self._keyword_ids(workspace, query, layers, 50)):
                ranks[i] = ranks.get(i, 0.0) + 1.0 / (60 + r)
        if mode in ("vector", "hybrid"):
            q = self.embed(query)
            ph = ",".join("?" * len(layers))
            scored = [(_cos(q, _unpack(r["embedding"])), r["id"]) for r in self.db.execute(
                f"SELECT id, embedding FROM memories WHERE workspace=? AND layer IN ({ph})", (workspace, *layers))]
            scored.sort(reverse=True)
            for r, (s, i) in enumerate(scored[:50]):
                if s > 0.05:
                    ranks[i] = ranks.get(i, 0.0) + 1.0 / (60 + r)
        top = sorted(ranks.items(), key=lambda kv: -kv[1])[:k]
        out = []
        now = time.time()
        for i, score in top:
            rec = self.get(workspace, i)
            rec["score"] = score
            out.append(rec)
            self.db.execute("UPDATE memories SET accessed=? WHERE id=?", (now, i))
        self.db.commit()
        if include_archive and len(out) < k:  # archived memories are still found, after the active ones
            out.extend(self._search_archive(workspace, query, layers, k - len(out), mode))
        return out

    def _search_archive(self, workspace, query, layers, need, mode):
        ph = ",".join("?" * len(layers))
        q = self.embed(query) if mode in ("vector", "hybrid") else None
        words = re.findall(r"\w+", query.lower()) if mode in ("keyword", "hybrid") else []
        scored = []
        for r in self.db.execute(f"SELECT * FROM memories_archive WHERE workspace=? AND layer IN ({ph})",
                                 (workspace, *layers)):
            cos = _cos(q, _unpack(r["embedding"])) if (q is not None and r["embedding"]) else 0.0
            low = r["content"].lower()
            kw = sum(w in low for w in words) / len(words) if words else 0.0
            if kw > 0 or cos > 0.2:
                scored.append((max(cos, 0.0) + kw, r))
        scored.sort(key=lambda x: -x[0])
        out = []
        for sc, r in scored[:need]:
            d = dict(r)
            d["meta"] = json.loads(d["meta"])
            d.pop("embedding", None)
            d["archived"], d["score"] = True, 0.0
            out.append(d)
        return out

    def _keyword_ids(self, workspace, query, layers, limit):
        ph = ",".join("?" * len(layers))
        words = re.findall(r"\w+", query.lower())
        if not words:
            return []
        if self.fts:
            match = " OR ".join('"' + w.replace('"', '') + '"' for w in words)
            rows = self.db.execute(
                f"SELECT m.id FROM memories_fts f JOIN memories m ON m.id=f.rowid "
                f"WHERE memories_fts MATCH ? AND m.workspace=? AND m.layer IN ({ph}) ORDER BY bm25(memories_fts) LIMIT ?",
                (match, workspace, *layers, limit)).fetchall()
            return [r[0] for r in rows]
        like = " OR ".join("lower(content) LIKE ?" for _ in words)
        rows = self.db.execute(
            f"SELECT id FROM memories WHERE workspace=? AND layer IN ({ph}) AND ({like}) LIMIT ?",
            (workspace, *layers, *[f"%{w}%" for w in words], limit)).fetchall()
        return [r[0] for r in rows]

    # ---- knowledge documents -------------------------------------------------
    @_locked
    def add_document(self, workspace, title, text, chunk_chars=800):
        chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars) if text[i:i + chunk_chars].strip()]
        ids = [self.add(workspace, "knowledge", c, {"title": title, "chunk": n})[0] for n, c in enumerate(chunks)]
        return ids

    # ---- limits / retention --------------------------------------------------
    @_locked
    def enforce_limits(self, workspace):
        if not self.elastic:  # legacy: evict, then refuse writes past the byte limit
            excess = self.count(workspace) - self.max_rows
            if excess > 0:
                ids = [r[0] for r in self.db.execute(
                    "SELECT id FROM memories WHERE workspace=? ORDER BY (layer='conversation') DESC, importance ASC, accessed ASC LIMIT ?",
                    (workspace, excess))]
                for i in ids:
                    self.delete(workspace, i)
            if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_bytes:
                raise MemoryLimitError(f"database exceeds {self.max_bytes} bytes; purge or raise the limit")
            return
        excess = self.count(workspace) - self.max_rows
        if excess > 0:  # keep the active set small by ARCHIVING (not deleting) the least useful conversation rows
            ids = [r[0] for r in self.db.execute(
                "SELECT id FROM memories WHERE workspace=? AND layer='conversation' ORDER BY importance ASC, accessed ASC LIMIT ?",
                (workspace, excess))]
            for i in ids:
                self._archive_row(i)
        while self.count(workspace) > self.max_rows:  # only persistent/knowledge left: the limit grows, rows stay
            self.max_rows = int(self.max_rows * self.grow_factor) + 1
        self._ensure_disk_room()

    def _ensure_disk_room(self):
        if not os.path.exists(self.path):
            return
        size = os.path.getsize(self.path)
        if size <= self.max_bytes:
            return
        new_max = self.max_bytes
        while size > new_max:
            new_max = int(new_max * self.grow_factor)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(self.path))).free
        if free - (new_max - size) < self.min_free_bytes:
            raise MemoryLimitError(f"disk almost full: cannot grow memory past {self.max_bytes} bytes "
                                   f"(free {free}, reserve {self.min_free_bytes}); free disk space")
        self.max_bytes = new_max

    def _archive_row(self, mem_id):
        r = self.db.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
        if r is None:
            return
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO memories_archive(orig_id,workspace,layer,content,hash,meta,importance,created,updated,"
                "accessed,archived,embedding) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (r["id"], r["workspace"], r["layer"], r["content"], r["hash"], r["meta"], r["importance"],
                 r["created"], r["updated"], r["accessed"], time.time(), r["embedding"]))
            self.db.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            if self.fts:
                self.db.execute("DELETE FROM memories_fts WHERE rowid=?", (mem_id,))

    def _restore_row(self, arch_id):
        r = self.db.execute("SELECT * FROM memories_archive WHERE id=?", (arch_id,)).fetchone()
        with self.db:
            cur = self.db.execute(
                "INSERT INTO memories(workspace,layer,content,hash,meta,importance,created,updated,accessed,embedding)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (r["workspace"], r["layer"], r["content"], r["hash"], r["meta"], r["importance"],
                 r["created"], time.time(), time.time(), r["embedding"]))
            if self.fts:
                self.db.execute("INSERT INTO memories_fts(rowid, content) VALUES(?,?)", (cur.lastrowid, r["content"]))
            self.db.execute("DELETE FROM memories_archive WHERE id=?", (arch_id,))
        return cur.lastrowid

    @_locked
    def restore(self, workspace, arch_id):
        """Move one archived memory back into the active set. Returns the new id or None."""
        r = self.db.execute("SELECT workspace FROM memories_archive WHERE id=?", (arch_id,)).fetchone()
        if r is None or r["workspace"] != workspace:
            return None
        new_id = self._restore_row(arch_id)
        self.enforce_limits(workspace)
        return new_id

    @_locked
    def iter_since(self, workspace, layers, since=0.0, limit=5000):
        """Memories (active AND archived) created after `since`, oldest first: the learning feed."""
        for l in layers:
            self._check(workspace, l)
        ph = ",".join("?" * len(layers))
        rows = []
        for tbl in ("memories", "memories_archive"):
            rows += [dict(r) for r in self.db.execute(
                f"SELECT layer, content, created FROM {tbl} WHERE workspace=? AND layer IN ({ph}) AND created>? "
                f"ORDER BY created LIMIT ?", (workspace, *layers, since, limit))]
        rows.sort(key=lambda r: r["created"])
        return rows[:limit]

    @_locked
    def purge_older_than(self, workspace, days, layers=("conversation",)):
        """Explicit retention (with delete/delete_workspace, the only paths that really remove rows)."""
        cutoff = time.time() - days * 86400
        ph = ",".join("?" * len(layers))
        ids = [r[0] for r in self.db.execute(
            f"SELECT id FROM memories WHERE workspace=? AND created<? AND layer IN ({ph})", (workspace, cutoff, *layers))]
        for i in ids:
            self.delete(workspace, i)
        with self.db:
            n_arch = self.db.execute(
                f"DELETE FROM memories_archive WHERE workspace=? AND created<? AND layer IN ({ph})",
                (workspace, cutoff, *layers)).rowcount
        return len(ids) + n_arch

    # ---- backup / export / integrity ----------------------------------------
    @_locked
    def integrity_ok(self):
        return self.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    @_locked
    def backup(self, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        tmp = str(dest) + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        out = sqlite3.connect(tmp)
        try:
            self.db.backup(out)
        finally:
            out.close()
        os.replace(tmp, dest)

    @_locked
    def export_json(self, workspace, path):
        rows = [self._row(r) for r in self.db.execute("SELECT * FROM memories WHERE workspace=?", (workspace,))]
        for r in self.db.execute("SELECT * FROM memories_archive WHERE workspace=?", (workspace,)):
            d = self._row(r)
            d["archived"] = True
            rows.append(d)
        Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        return len(rows)

    @_locked
    def import_json(self, workspace, path):
        n = 0
        for r in json.loads(Path(path).read_text(encoding="utf-8")):
            n += self.add(workspace, r["layer"], r["content"], r.get("meta"), r.get("importance", 0.5))[1]
        return n

    def close(self):
        self.db.close()

    @staticmethod
    def recover_from_backup(path, backup):
        """Move a damaged database aside (kept as .corrupt) and restore the backup after checking it."""
        probe = sqlite3.connect(backup)
        try:
            ok = probe.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        except sqlite3.DatabaseError:
            ok = False
        finally:
            probe.close()
        if not ok:
            raise MemoryCorruptError(f"backup {backup} is not a valid database")
        if os.path.exists(path):
            shutil.move(path, str(path) + ".corrupt")
        for ext in ("-wal", "-shm"):
            if os.path.exists(str(path) + ext):
                os.remove(str(path) + ext)
        shutil.copy2(backup, path)
