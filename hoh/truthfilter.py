"""Truth filter (backend-independent): a CORROBORATION filter, not an oracle.

ingest.quality() only asks "is this readable text?". This module asks "is there reason to believe it?", using signals a
program can actually check:

 1. source tier        - trusted / unknown / low-trust suffixes (editable policy); text YOU hand over is 'user-supplied'
 2. corroboration      - the same factual sentence (same subject words + same numbers) seen in an INDEPENDENT document
                         (different host, and not a copy: near-duplicate documents count as ONE origin)
 3. conflict           - same subject, different numbers in an independent document -> flagged, and the lone sentence is
                         dropped when >=2 independent origins agree on the other value
 4. internal checks    - written arithmetic that is wrong ("12*3=37" is verified with the calculator), impossible dates
 5. style signals      - miracle-cure / clickbait phrasing, shouting, absolute claims with no source
 6. citations          - references/links/'menurut'/doi raise confidence slightly

Verdicts: corroborated | trusted_source | user_supplied | unverified | low_trust | contradicted.
Only the first three (and 'unverified' if policy.allow_unverified) go to training; the rest are QUARANTINED (kept on disk and
re-checked whenever new documents arrive: a second independent source can release them).

LIMITS (read these): agreement between sources is not proof - widely repeated falsehoods pass; true-but-novel facts from a
single source wait in quarantine; only sentences with numbers/definitions are cross-checked; no world knowledge is used.
It lowers the chance of learning junk; it cannot make HOH infallible."""
import hashlib, json, math, re, sqlite3, time, zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from .semantic import tokenize

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUM = re.compile(r"(?<![\w.])-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|(?<![\w.])-?\d+(?:[.,]\d+)?")
_ARITH = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?(?:\s*[-+*/x×:]\s*\d+(?:[.,]\d+)?)+)\s*=\s*(-?\d+(?:[.,]\d+)?)")
_DEF = re.compile(r"\b(adalah|merupakan|disebut|didefinisikan|is a|is an|refers to)\b", re.I)
_CITE = re.compile(r"(https?://|\bdoi\b|et al\.|\bmenurut\b|\bsumber\s*:|\breferensi\b|\breferences\b|\[\d+\]|\baccording to\b)", re.I)
_SHOUT = re.compile(r"(!{2,}|\b[A-Z]{5,}\b.*\b[A-Z]{5,}\b)")
_HYPE = re.compile(r"(100\s*%\s*(sembuh|ampuh|aman|terbukti)|dijamin|obat ajaib|rahasia (yang )?(disembunyikan|dibongkar)|"
                   r"dokter (benci|terkejut)|miracle cure|doctors hate|you won'?t believe|tanpa efek samping sama sekali|"
                   r"sembuh total dalam \d+ (hari|jam)|viral!|wajib (share|bagikan)|sebarkan)", re.I)
_MONTH_DAYS = {"januari": 31, "februari": 29, "maret": 31, "april": 30, "mei": 31, "juni": 30, "juli": 31, "agustus": 31,
               "september": 30, "oktober": 31, "november": 30, "desember": 31}
_DATE = re.compile(r"\b(\d{1,2})\s+(" + "|".join(_MONTH_DAYS) + r")\b", re.I)

DEFAULT_POLICY = {
    "trusted_suffixes": [".go.id", ".gov", ".gov.uk", ".edu", ".ac.id", ".ac.uk", "who.int", "un.org", "nih.gov", "arxiv.org",
                         "nature.com", "science.org", "bps.go.id", "wikipedia.org", "britannica.com"],
    "low_suffixes": [],            # add known junk/clickbait hosts here (or via the policy file)
    "allow_unverified": False,     # True = unknown single-source text is learned too (less safe)
    "drop_conflicting_sentences": True,
    "min_support_for_drop": 2,
}


def _host(source):
    m = re.match(r"^https?://([^/:]+)", source or "")
    return (m.group(1).lower() if m else "")


def _norm_num(tok):
    t = tok.strip()
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+(?:,\d+)?", t):        # id: 12.000 / 1.234,5
        t = t.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", t):      # en: 12,000 / 1,234.5
        t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    try:
        return round(float(t), 6)
    except ValueError:
        return None


def numbers_in(s):
    out = {_norm_num(m.group(0)) for m in _NUM.finditer(s)}
    return {n for n in out if n is not None}


def _words(s):
    return {w for w in tokenize(_NUM.sub(" ", s)) if len(w) > 2}


def _relation(n, cn):
    """'agree' | 'conflict' | 'unrelated' for the number sets of two claims about the same subject."""
    if not n or not cn:
        return "agree"
    if n <= cn or cn <= n:
        return "agree"
    for x in n - cn:                       # a differing pair of similar magnitude = the same quantity with another value
        for y in cn - n:
            if x != 0 and y != 0 and 0.1 < abs(x / y) < 10 and (x > 0) == (y > 0):
                return "conflict"
    return "unrelated"


def _is_claim(s):
    if len(s) < 25 or len(s) > 400:
        return False
    return bool(_NUM.search(s)) or bool(_DEF.search(s))


def split_claims(text, limit=60):
    out = []
    for s in _SENT.split(text):
        s = s.strip()
        if _is_claim(s) and len(_words(s)) >= 3:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _minhash(text, k=16):
    ws = re.findall(r"\w+", text.lower())
    grams = {" ".join(ws[i:i + 5]) for i in range(max(1, len(ws) - 4))}
    return [min((zlib.crc32(f"{j}|{g}".encode()) for g in grams), default=0) for j in range(k)]


def _mh_sim(a, b):
    return sum(x == y for x, y in zip(a, b)) / max(1, len(a))


def check_arithmetic(text):
    """-> list of wrong equations found in the text, verified with the calculator (never guessed)."""
    from .tools import safe_eval
    bad = []
    for m in _ARITH.finditer(text):
        expr = re.sub(r"[x×]", "*", m.group(1)).replace(":", "/")
        expr = re.sub(r"(\d),(\d)", r"\1.\2", expr)
        try:
            want = safe_eval(expr)
        except Exception:
            continue
        got = _norm_num(m.group(2))
        if got is not None and abs(float(want) - got) > max(1e-6, abs(got) * 1e-6):
            bad.append(f"{m.group(0)} (hasil benar {round(float(want), 6)})")
    return bad


def check_dates(text):
    bad = []
    for m in _DATE.finditer(text):
        if int(m.group(1)) > _MONTH_DAYS[m.group(2).lower()] or int(m.group(1)) == 0:
            bad.append(m.group(0))
    return bad


@dataclass
class TruthReport:
    verdict: str = "unverified"
    score: float = 0.5
    host: str = ""
    tier: str = "unknown"
    reasons: list = field(default_factory=list)
    corroborated_by: list = field(default_factory=list)      # independent hosts that back at least one claim
    conflicts: list = field(default_factory=list)            # {"claim","other_host","other_claim"}
    dropped_sentences: int = 0
    claims_checked: int = 0
    learn: bool = False
    text: str = ""

    def meta(self):
        d = asdict(self)
        d.pop("text")
        d["conflicts"] = d["conflicts"][:5]
        return d


class TruthFilter:
    def __init__(self, db_path, policy=None, clock=time.time):
        self.policy = {**DEFAULT_POLICY, **(policy or {})}
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.clock = clock
        with self.db:
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY, sig TEXT UNIQUE, host TEXT, minhash TEXT, ts REAL, accepted INTEGER);
            CREATE TABLE IF NOT EXISTS claims(id INTEGER PRIMARY KEY, doc INTEGER, host TEXT, sentence TEXT, words TEXT, nums TEXT);
            CREATE TABLE IF NOT EXISTS cw(word TEXT, claim INTEGER);
            CREATE INDEX IF NOT EXISTS idx_cw ON cw(word);
            CREATE INDEX IF NOT EXISTS idx_claim_doc ON claims(doc);""")

    # ---- source tier ----
    def tier(self, host, user_supplied=False):
        if user_supplied:
            return "user_supplied"
        h = host.lower()
        if any(h == s.lstrip(".") or h.endswith(s if s.startswith(".") else "." + s) for s in self.policy["low_suffixes"]):
            return "low"
        if any(h == s.lstrip(".") or h.endswith(s if s.startswith(".") else "." + s) for s in self.policy["trusted_suffixes"]):
            return "trusted"
        return "unknown"

    # ---- cross-document lookup ----
    def _candidates(self, words, exclude_host, limit=40):
        rare = sorted(words, key=lambda w: -len(w))[:6]
        if not rare:
            return []
        ph = ",".join("?" * len(rare))
        rows = self.db.execute(
            f"SELECT claim, count(*) c FROM cw WHERE word IN ({ph}) GROUP BY claim HAVING c>=3 ORDER BY c DESC LIMIT ?",
            (*rare, limit)).fetchall()
        out = []
        for r in rows:
            c = self.db.execute("SELECT c.*, d.minhash, d.accepted FROM claims c JOIN docs d ON d.id=c.doc WHERE c.id=?",
                                (r["claim"],)).fetchone()
            if c and c["host"] != exclude_host:
                out.append(c)
        return out

    def assess(self, text, source="", user_supplied=False):
        host = _host(source)
        tier = self.tier(host, user_supplied)
        rep = TruthReport(host=host, tier=tier, text=text)
        base = {"trusted": 0.8, "user_supplied": 0.7, "unknown": 0.5, "low": 0.2}[tier]
        score = base
        mh = _minhash(text)
        # internal checks
        bad_math, bad_dates = check_arithmetic(text), check_dates(text)
        if bad_math:
            score -= 0.5
            rep.reasons.append("perhitungan salah: " + "; ".join(bad_math[:3]))
        if bad_dates:
            score -= 0.3
            rep.reasons.append("tanggal mustahil: " + ", ".join(bad_dates[:3]))
        if _HYPE.search(text):
            score -= 0.25
            rep.reasons.append("bahasa klik-umpan/janji berlebihan")
        if _SHOUT.search(text):
            score -= 0.1
            rep.reasons.append("banyak seruan/huruf kapital")
        if _CITE.search(text):
            score += 0.1
            rep.reasons.append("ada rujukan/sumber")
        # cross-document
        support_hosts, conflict_hosts, drop = set(), set(), []
        claims = split_claims(text)
        rep.claims_checked = len(claims)
        keep = text
        for s in claims:
            w, n = _words(s), numbers_in(s)
            sup, con = set(), []
            for c in self._candidates(w, host):
                if _mh_sim(mh, json.loads(c["minhash"])) >= 0.6:       # a copy of this very document is not independent
                    continue
                cw, cn = set(json.loads(c["words"])), set(json.loads(c["nums"]))
                if len(w & cw) < 3 or len(w & cw) / max(1, len(w | cw)) < 0.5:
                    continue
                rel = _relation(n, cn)
                if rel == "agree":
                    sup.add(c["host"])
                elif rel == "conflict":
                    con.append(c)
            support_hosts |= sup
            if con:
                conflict_hosts |= {c["host"] for c in con}
                rep.conflicts.append({"claim": s[:200], "other_host": con[0]["host"], "other_claim": con[0]["sentence"][:200]})
                others = {c["host"] for c in con}
                if self.policy["drop_conflicting_sentences"] and len(others) >= self.policy["min_support_for_drop"] and not sup:
                    drop.append(s)
        for s in drop:
            keep = keep.replace(s, "")
        rep.dropped_sentences = len(drop)
        rep.corroborated_by = sorted(support_hosts)
        score += min(0.3, 0.15 * len(support_hosts))
        score -= min(0.4, 0.2 * len(rep.conflicts) - 0.2 * len(drop)) if rep.conflicts else 0
        if rep.conflicts:
            rep.reasons.append(f"{len(rep.conflicts)} klaim bertentangan dengan sumber lain ({len(drop)} dibuang)")
        if support_hosts:
            rep.reasons.append(f"dikuatkan {len(support_hosts)} sumber independen")
        rep.text = re.sub(r"\n{3,}", "\n\n", keep).strip()
        rep.score = max(0.0, min(1.0, round(score, 3)))
        # verdict
        if bad_math or (rep.conflicts and len(conflict_hosts) > len(support_hosts) and rep.dropped_sentences == 0):
            rep.verdict = "contradicted"
        elif tier == "low":
            rep.verdict = "corroborated" if len(support_hosts) >= 2 else "low_trust"
        elif support_hosts:                      # this document + at least one independent origin say the same thing
            rep.verdict = "corroborated"
        elif tier == "trusted":
            rep.verdict = "trusted_source"
        elif tier == "user_supplied":
            rep.verdict = "user_supplied"
        else:
            rep.verdict = "unverified"
        rep.learn = rep.verdict in ("corroborated", "trusted_source", "user_supplied") and rep.score >= 0.4 or \
            (rep.verdict == "unverified" and self.policy["allow_unverified"] and rep.score >= 0.5)
        return rep

    # ---- index maintenance ----
    def register(self, text, source="", accepted=True):
        """Add this document's claims to the index (also when quarantined: a later independent source can corroborate it)."""
        host = _host(source) or hashlib.sha1(source.encode()).hexdigest()[:8]
        sig = hashlib.sha1(" ".join(text.split()).lower().encode()).hexdigest()
        with self.db:
            if self.db.execute("SELECT 1 FROM docs WHERE sig=?", (sig,)).fetchone():
                self.db.execute("UPDATE docs SET accepted=? WHERE sig=?", (int(accepted), sig))
                return 0
            cur = self.db.execute("INSERT INTO docs(sig,host,minhash,ts,accepted) VALUES(?,?,?,?,?)",
                                  (sig, host, json.dumps(_minhash(text)), self.clock(), int(accepted)))
            n = 0
            for s in split_claims(text):
                w = _words(s)
                cc = self.db.execute("INSERT INTO claims(doc,host,sentence,words,nums) VALUES(?,?,?,?,?)",
                                     (cur.lastrowid, host, s, json.dumps(sorted(w)), json.dumps(sorted(numbers_in(s))))).lastrowid
                self.db.executemany("INSERT INTO cw(word,claim) VALUES(?,?)", [(x, cc) for x in w])
                n += 1
        return n

    def close(self):
        self.db.close()
