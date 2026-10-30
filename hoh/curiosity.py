"""Self-directed learning: what to study next, and how to go get it politely. Torch-free.

Curriculum = a persistent queue of GOALS (topics) + a link FRONTIER. Goals come from: you (`hoh goals add`), failures
(bad/corrected episodes, uncertain answers), and curiosity (frequent terms in what was just read). A goal that keeps
returning only duplicates/rejects is retired as saturated. Explorer = budgeted, robots.txt-respecting crawler on top of
Ingestor. Needs network: NOT VERIFIED offline (tests use injected fetch/search functions)."""
import re, sqlite3, time, urllib.parse, urllib.robotparser
from collections import Counter
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS goals(id INTEGER PRIMARY KEY AUTOINCREMENT, topic TEXT UNIQUE, priority REAL, source TEXT,
  status TEXT DEFAULT 'active', attempts INTEGER DEFAULT 0, gained INTEGER DEFAULT 0, barren INTEGER DEFAULT 0, last_ts REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS frontier(url TEXT PRIMARY KEY, goal_id INTEGER, score REAL, depth INTEGER, status TEXT DEFAULT 'new');
CREATE TABLE IF NOT EXISTS daily(day TEXT PRIMARY KEY, pages INTEGER DEFAULT 0, chars INTEGER DEFAULT 0);
"""
_STOP = set("yang dan di ke dari untuk dengan ini itu pada adalah tidak akan juga atau sebagai oleh dalam saya kami kita "
            "the and of to in is that for with as on are this be by it from or an not was were has have had which their".split())
TRUSTED_SUFFIX = (".edu", ".gov", ".ac.id", ".go.id", ".org", "wikipedia.org", "arxiv.org")


def _tokens(text):
    return [w for w in re.findall(r"[a-zA-ZÀ-ÿ]{3,}", text.lower()) if w not in _STOP]


class Curriculum:
    def __init__(self, path, clock=time.time):
        self.clock = clock
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---- goals ----
    def add_goal(self, topic, priority=1.0, source="user"):
        topic = " ".join(topic.split())[:200]
        if len(topic) < 3:
            return None
        cur = self.db.execute("INSERT OR IGNORE INTO goals(topic,priority,source) VALUES(?,?,?)", (topic, priority, source))
        if cur.rowcount == 0:                     # known topic: a repeated request makes it active and more important again
            self.db.execute("UPDATE goals SET priority=MAX(priority,?), status='active', barren=0 WHERE topic=?", (priority, topic))
        self.db.commit()
        return self.db.execute("SELECT id FROM goals WHERE topic=?", (topic,)).fetchone()["id"]

    def goals(self, status=None):
        q = "SELECT * FROM goals" + (" WHERE status=?" if status else "") + " ORDER BY priority DESC, id"
        return [dict(r) for r in self.db.execute(q, (status,) if status else ())]

    def next_goal(self, cooldown=3600.0):
        now = self.clock()
        rows = self.db.execute("SELECT * FROM goals WHERE status='active' AND (attempts=0 OR ?-last_ts>=?)", (now, cooldown)).fetchall()
        if not rows:
            return None
        return dict(max(rows, key=lambda r: r["priority"] / (1 + r["attempts"]) ))

    def record(self, goal_id, accepted, chars, max_barren=3):
        """Outcome of one study session on a goal. Saturated after `max_barren` sessions in a row with nothing new."""
        g = self.db.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()
        barren = 0 if accepted > 0 else g["barren"] + 1
        status = "saturated" if barren >= max_barren else g["status"]
        self.db.execute("UPDATE goals SET attempts=attempts+1, gained=gained+?, barren=?, status=?, last_ts=? WHERE id=?",
                        (chars, barren, status, self.clock(), goal_id))
        self.db.commit()
        return status

    # ---- where goals come from besides you ----
    def seed_from_experience(self, store, limit=10):
        n = 0
        for r in store.db.execute("SELECT task FROM episodes WHERE outcome='bad' OR correction IS NOT NULL ORDER BY id DESC LIMIT ?", (limit,)):
            n += bool(self.add_goal(r["task"][:120], 1.5, "failure"))
        return n

    def seed_from_text(self, text, k=3, min_count=3):
        """Curiosity: frequent content words of what was just read that are not already goals -> low-priority follow-ups."""
        known = {t for g in self.goals() for t in _tokens(g["topic"])}
        cnt = Counter(w for w in _tokens(text) if len(w) >= 5 and w not in known)
        n = 0
        for w, c in cnt.most_common(k * 3):
            if c >= min_count and n < k:
                n += bool(self.add_goal(w, 0.4, "curiosity"))
        return n

    # ---- frontier + budget ----
    def add_frontier(self, urls, goal_id, topic, depth, trusted_boost=0.3):
        tw = set(_tokens(topic))
        for u in urls:
            host = (urllib.parse.urlparse(u).hostname or "").lower()
            ut = set(_tokens(urllib.parse.unquote(u)))
            score = len(tw & ut) / max(1, len(tw)) + (trusted_boost if host.endswith(TRUSTED_SUFFIX) else 0.0) - 0.1 * depth
            self.db.execute("INSERT OR IGNORE INTO frontier(url,goal_id,score,depth) VALUES(?,?,?,?)", (u, goal_id, score, depth))
        self.db.commit()

    def pop_frontier(self, min_score=0.15):
        r = self.db.execute("SELECT * FROM frontier WHERE status='new' AND score>=? ORDER BY score DESC LIMIT 1", (min_score,)).fetchone()
        if r:
            self.db.execute("UPDATE frontier SET status='done' WHERE url=?", (r["url"],))
            self.db.commit()
        return dict(r) if r else None

    def day_usage(self):
        day = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        r = self.db.execute("SELECT pages, chars FROM daily WHERE day=?", (day,)).fetchone()
        return day, (r["pages"] if r else 0), (r["chars"] if r else 0)

    def add_usage(self, pages, chars):
        day, _, _ = self.day_usage()
        self.db.execute("INSERT INTO daily(day,pages,chars) VALUES(?,?,?) ON CONFLICT(day) DO UPDATE SET pages=pages+?, chars=chars+?",
                        (day, pages, chars, pages, chars))
        self.db.commit()


class Robots:
    """robots.txt per host (cached). If robots.txt cannot be fetched the host is treated as allowed (documented limitation)."""
    def __init__(self, fetcher, agent="HOH-learner"):
        self.fetcher, self.agent, self.cache = fetcher, agent, {}

    def allowed(self, url):
        pu = urllib.parse.urlparse(url)
        base = f"{pu.scheme}://{pu.netloc}"
        if base not in self.cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                body = self.fetcher(base + "/robots.txt")[0].decode("utf-8", errors="replace")
                rp.parse(body.splitlines())
            except Exception:
                rp = None
            self.cache[base] = rp
        rp = self.cache[base]
        return True if rp is None else rp.can_fetch(self.agent, url)


class Explorer:
    def __init__(self, ingestor, curriculum, robots=None, max_pages=15, max_chars=1_500_000, max_seconds=600.0,
                 per_host_delay=2.0, daily_pages=150, daily_chars=20_000_000, deny_hosts=(), max_depth=1,
                 results_per_goal=4, sleep=time.sleep, clock=time.time):
        self.ing, self.cur, self.robots = ingestor, curriculum, robots
        self.max_pages, self.max_chars, self.max_seconds, self.delay = max_pages, max_chars, max_seconds, per_host_delay
        self.daily_pages, self.daily_chars, self.deny, self.max_depth = daily_pages, daily_chars, set(deny_hosts), max_depth
        self.k, self.sleep, self.clock, self.last_hit = results_per_goal, sleep, clock, {}

    def _ok(self, url):
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if not host or host in self.deny or any(host.endswith("." + d) for d in self.deny):
            return False, "denied host"
        if self.robots and not self.robots.allowed(url):
            return False, "robots.txt disallows"
        return True, ""

    def _fetch(self, url, goal_id, topic, depth, stats):
        ok, why = self._ok(url)
        if not ok:
            stats["skipped"].append((url, why))
            return
        host = urllib.parse.urlparse(url).netloc
        wait = self.delay - (self.clock() - self.last_hit.get(host, -1e9))
        if wait > 0:
            self.sleep(wait)
        self.last_hit[host] = self.clock()
        rep = self.ing.learn(url)
        stats["pages"] += 1
        if rep.get("status") == "error":
            stats["skipped"].append((url, rep.get("error", "error")))
        stats["accepted"] += rep.get("accepted", 0)
        stats["chars"] += rep.get("chars", 0)
        stats["files"] += [d["file"] for d in rep.get("documents", []) if d.get("file")]
        if depth < self.max_depth:
            links = [l for d in rep.get("documents", []) if d["status"] == "accepted" for l in d.get("links", [])]
            self.cur.add_frontier(links, goal_id, topic, depth + 1)

    def explore(self):
        t0 = self.clock()
        stats = {"pages": 0, "accepted": 0, "chars": 0, "files": [], "skipped": [], "goals": [], "stopped": ""}
        while True:
            _, dp, dc = self.cur.day_usage()
            if stats["pages"] >= self.max_pages or dp + stats["pages"] >= self.daily_pages:
                stats["stopped"] = "page budget"; break
            if stats["chars"] >= self.max_chars or dc + stats["chars"] >= self.daily_chars:
                stats["stopped"] = "size budget"; break
            if self.clock() - t0 > self.max_seconds:
                stats["stopped"] = "time budget"; break
            item = self.cur.pop_frontier()
            if item:
                g = self.cur.db.execute("SELECT topic FROM goals WHERE id=?", (item["goal_id"],)).fetchone()
                self._fetch(item["url"], item["goal_id"], g["topic"] if g else "", item["depth"], stats)
                continue
            goal = self.cur.next_goal()
            if goal is None:
                stats["stopped"] = "no active goals"; break
            before = (stats["accepted"], stats["chars"])
            try:
                urls = self.ing.searcher(goal["topic"], self.k)
            except Exception as e:                                   # search outage must not kill the session
                stats["skipped"].append((goal["topic"], f"search failed: {e}"))
                self.cur.record(goal["id"], 0, 0, max_barren=10 ** 6)
                continue
            for u in urls:
                self._fetch(u, goal["id"], goal["topic"], 0, stats)
            # follow-up links found on those pages are consumed through the frontier in later loop turns
            stats["goals"].append(goal["topic"])
            self.cur.record(goal["id"], stats["accepted"] - before[0], stats["chars"] - before[1])
        self.cur.add_usage(stats["pages"], stats["chars"])
        return stats
