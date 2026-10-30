"""Learn from anywhere: URL / file / folder / zip / search -> cleaned, filtered, de-duplicated text.

Torch-free. Output goes to the lifelong *inbox* (picked up while idle) and/or MemoryStore 'knowledge'.
Readers: txt md json jsonl csv tsv html xml docx pptx xlsx epub pdf zip. PDF uses pypdf/pdfminer if installed,
otherwise a minimal built-in extractor (text-based PDFs only; scanned PDFs need OCR = NOT supported).
Nothing here touches model weights: it only prepares text. Network paths are NOT VERIFIED in the build sandbox."""
import csv, hashlib, io, json, os, re, time, urllib.parse, urllib.request, zipfile, zlib
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


from . import ocr as _ocr


class IngestError(Exception):
    pass


@dataclass
class Document:
    source: str
    title: str
    text: str
    kind: str
    notes: list = field(default_factory=list)
    links: list = field(default_factory=list)


# ---------------------------------------------------------------- HTML
class _Html(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "pre", "blockquote",
             "table", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def extract_links(html, base=""):
    """Absolute http(s) links of an HTML page (for the explorer's frontier)."""
    out = []
    for m in re.finditer(r'(?is)<a\s[^>]*?href\s*=\s*["\']([^"\'#]+)', html):
        u = urllib.parse.urljoin(base, m.group(1).strip())
        if u.startswith(("http://", "https://")) and u not in out:
            out.append(u)
    return out


def html_to_text(html):
    p = _Html()
    p.feed(html)
    p.close()
    return p.title.strip(), "".join(p.out)


# ---------------------------------------------------------------- Office / epub / pdf
def _xml_text(data, tags):
    root = ET.fromstring(data)
    out = []
    for el in root.iter():
        t = el.tag.rsplit("}", 1)[-1]
        if t in tags and el.text:
            out.append(el.text)
        elif t in ("p", "br", "tr") and out:
            out.append("\n")
    return "".join(out)


def read_docx(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        return _xml_text(z.read("word/document.xml"), {"t"})


def read_pptx(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                        key=lambda n: int(re.findall(r"\d+", n)[-1]))
        return "\n\n".join(_xml_text(z.read(n), {"t"}) for n in slides)


def read_xlsx(b):
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root:
                shared.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
        rows = []
        for n in sorted(x for x in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", x)):
            for row in ET.fromstring(z.read(n)).iter():
                if not row.tag.endswith("}row"):
                    continue
                cells = []
                for c in row:
                    v = next((x.text for x in c if x.tag.endswith("}v")), None)
                    if v is None:
                        continue
                    cells.append(shared[int(v)] if c.get("t") == "s" and v.isdigit() and int(v) < len(shared) else v)
                if cells:
                    rows.append(" | ".join(cells))
        return "\n".join(rows)


def read_epub(b):
    out = []
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        for n in sorted(z.namelist()):
            if n.lower().endswith((".xhtml", ".html", ".htm")):
                out.append(html_to_text(z.read(n).decode("utf-8", errors="replace"))[1])
    return "\n\n".join(out)


_PDF_STREAM = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)


def _pdf_unescape(s):
    s = s[1:-1]
    s = re.sub(rb"\\([nrtbf()\\])", lambda m: {b"n": b"\n", b"r": b"\n", b"t": b" ", b"b": b"", b"f": b""}.get(m.group(1), m.group(1)), s)
    s = re.sub(rb"\\([0-7]{1,3})", lambda m: bytes([int(m.group(1), 8) & 255]), s)
    return s.decode("latin-1")


def read_pdf(b):
    for mod, fn in (("pypdf", lambda m: "\n\n".join((p.extract_text() or "") for p in m.PdfReader(io.BytesIO(b)).pages)),
                    ("pdfminer.high_level", lambda m: m.extract_text(io.BytesIO(b)))):
        try:
            m = __import__(mod, fromlist=["x"])
        except ImportError:
            continue
        try:
            t = fn(m)
            if t.strip():
                return t
        except Exception:
            pass
    # minimal fallback: Flate streams, literal-string text operators only
    out = []
    for m in _PDF_STREAM.finditer(b):
        raw = m.group(1)
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        if b"BT" not in raw:
            continue
        for tok in re.finditer(rb"\((?:\\.|[^\\()])*\)|\bT[dD*]\b|\bET\b|'", raw):
            t = tok.group(0)
            out.append(_pdf_unescape(t) if t.startswith(b"(") else "\n")
    text = "".join(out)
    if len(re.sub(r"\s", "", text)) < 20:
        raise IngestError("PDF has no extractable text (scanned/encrypted/unsupported fonts). "
                          "Install `pypdf` for better coverage; scans are read with OCR when tesseract is installed.")
    return text


def _csv_text(s):
    rows = list(csv.reader(io.StringIO(s), dialect=csv.Sniffer().sniff(s[:2000]) if s.strip() else csv.excel))
    return "\n".join(" | ".join(r) for r in rows)


def _json_text(s):
    obj = json.loads(s)
    out = []

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}{k}: " if not isinstance(v, (dict, list)) else "")
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        elif o is not None:
            out.append(f"{path}{o}")
    walk(obj)
    return "\n".join(out)


def _decode(b):
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("utf-8", errors="replace")


def parse_bytes(data, name="", ctype=""):
    """bytes + name/content-type -> list[Document] (a zip yields many). Raises IngestError."""
    ext = Path(urllib.parse.urlparse(name).path).suffix.lower() if name else ""
    ct = (ctype or "").split(";")[0].strip().lower()
    title = Path(name).name if name else "document"
    try:
        if ext == ".zip" or ct == "application/zip":
            docs = []
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                total = 0
                for info in z.infolist()[:500]:
                    if info.is_dir() or info.file_size > 50_000_000:
                        continue
                    total += info.file_size
                    if total > 300_000_000:
                        raise IngestError("zip expands to more than 300 MB")
                    try:
                        docs += parse_bytes(z.read(info), f"{name}!{info.filename}")
                    except IngestError:
                        continue
            return docs
        if ext in _ocr.IMAGE_EXT or ct.startswith("image/") and ct != "image/svg+xml":
            try:
                text = _ocr.ocr_image_bytes(data)
            except _ocr.OcrUnavailable as e:
                raise IngestError(f"image {title}: {e}")
            return [Document(name, title, text, "image", notes=["ocr"])]
        if ext in _ocr.AUDIO_EXT or ct.startswith("audio/"):
            try:
                text = _ocr.transcribe_audio(data, name or "audio")
            except _ocr.OcrUnavailable as e:
                raise IngestError(f"audio {title}: {e}")
            return [Document(name, title, text, "audio", notes=["speech-to-text"])]
        if ext == ".pdf" or ct == "application/pdf" or data[:5] == b"%PDF-":
            try:
                text, notes = read_pdf(data), []
            except IngestError as e:                      # no text layer -> scanned PDF -> OCR
                try:
                    text, notes = _ocr.ocr_pdf(data), ["ocr"]
                except _ocr.OcrUnavailable as e2:
                    raise IngestError(f"{e} OCR unavailable: {e2}")
                except Exception as e2:                   # damaged/odd PDF: report, never crash the learner
                    raise IngestError(f"{e} OCR failed ({type(e2).__name__}: {e2})")
            else:
                if _ocr.looks_scanned(text, data):        # a few stray characters but really a scan
                    try:
                        o = _ocr.ocr_pdf(data)
                        if len(o) > len(text):
                            text, notes = o, ["ocr"]
                    except Exception:                     # OCR is a bonus here: keep the text layer we already have
                        pass
            return [Document(name, title, text, "pdf", notes=notes)]
        if ext == ".docx":
            return [Document(name, title, read_docx(data), "docx")]
        if ext == ".pptx":
            return [Document(name, title, read_pptx(data), "pptx")]
        if ext == ".xlsx":
            return [Document(name, title, read_xlsx(data), "xlsx")]
        if ext == ".epub":
            return [Document(name, title, read_epub(data), "epub")]
        s = _decode(data)
        if ext in (".html", ".htm", ".xhtml") or ct in ("text/html", "application/xhtml+xml") or \
                (not ext and s.lstrip()[:15].lower().startswith(("<!doctype html", "<html"))):
            t, text = html_to_text(s)
            return [Document(name, t or title, text, "html", links=extract_links(s, name))]
        if ext in (".csv", ".tsv"):
            return [Document(name, title, _csv_text(s), "csv")]
        if ext == ".json":
            return [Document(name, title, _json_text(s), "json")]
        if ext == ".jsonl":
            from .sft import record_to_text
            lines = []
            for n, l in enumerate(s.splitlines(), 1):
                if l.strip():
                    try:
                        lines.append(record_to_text(json.loads(l), f"{name}:{n}"))
                    except Exception:
                        lines.append(l)
            return [Document(name, title, "\n\n".join(lines), "jsonl")]
        if "\x00" in s[:4000]:
            raise IngestError(f"binary file not supported: {name or ct}")
        return [Document(name, title, s, "text")]
    except IngestError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, KeyError, json.JSONDecodeError, csv.Error, ValueError) as e:
        raise IngestError(f"cannot read {name or 'data'} as {ext or ct or 'text'}: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- network
def fetch_url(url, timeout=20.0, max_bytes=20_000_000, user_agent="HOH-learner/1.4 (+personal research)"):
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise IngestError("only http(s) URLs")
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "text/html,application/pdf,*/*;q=0.5"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise IngestError(f"response larger than {max_bytes} bytes")
            return data, r.headers.get("Content-Type", ""), r.geturl()
    except IngestError:
        raise
    except Exception as e:
        raise IngestError(f"fetch failed ({type(e).__name__}): {e}")


def search_urls(query, n=5, env=None, fetcher=fetch_url):
    """URLs for a query. Uses HOH_WEB_SEARCH_URL (same contract as the web_search tool) if set, else DuckDuckGo's
    HTML endpoint (best effort; the markup can change). NOT VERIFIED offline."""
    env = env if env is not None else os.environ
    base = env.get("HOH_WEB_SEARCH_URL")
    if base:
        url = f"{base}{'&' if '?' in base else '?'}{urllib.parse.urlencode({'q': query, 'n': n})}"
        body, _, _ = fetcher(url)
        body = body.decode("utf-8", errors="replace")
        urls = re.findall(r"https?://[^\s\"'<>\\)]+", body)
    else:
        body, _, _ = fetcher("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}))
        urls = []
        for m in re.findall(r'href="([^"]*uddg=[^"]+)"', body.decode("utf-8", errors="replace")):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(m.replace("&amp;", "&")).query).get("uddg")
            if q:
                urls.append(q[0])
    out, seen = [], set()
    for u in urls:
        u = u.rstrip(".,;")
        host = urllib.parse.urlparse(u).netloc
        if host and u not in seen and "duckduckgo.com" not in host:
            seen.add(u)
            out.append(u)
    return out[:n]


# ---------------------------------------------------------------- cleaning / quality
_SECRET = [re.compile(p) for p in (
    r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\s*[:=]\s*\S{8,}",
    r"\b(?:sk|pk|ghp|gho|xox[bp])[-_][A-Za-z0-9_\-]{16,}",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----")]
_INJECT = re.compile(r"(?i)(ignore (all )?(previous|prior) instructions|abaikan (semua )?instruksi|you are now|system prompt|"
                     r"<tool_call>|</?tool_result|disregard the above)")
_BOILER = re.compile(r"(?i)^(share|bagikan|tweet|cookie|accept all|terima semua|subscribe|berlangganan|login|masuk|daftar|"
                     r"sign up|read more|baca juga|baca selengkapnya|copyright|©|all rights reserved|privacy policy|"
                     r"kebijakan privasi|advertisement|iklan|skip to)\b")
_ID = set("yang dan di ke dari untuk dengan ini itu pada adalah tidak akan juga atau sebagai oleh dalam saya kami kita".split())
_EN = set("the and of to in is that for with as on are this be by it from or an not".split())


def clean_text(text, drop_boilerplate=True):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    lines, out, seen = [l.strip() for l in text.split("\n")], [], set()
    for l in lines:
        if not l:
            if out and out[-1] != "":
                out.append("")
            continue
        if drop_boilerplate and len(l) < 80 and _BOILER.match(l):
            continue
        if len(l) < 40 and l in seen:           # repeated menu/footer line
            continue
        seen.add(l)
        out.append(l)
    # reflow hard-wrapped lines inside a paragraph is NOT attempted (would corrupt code/tables)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def scrub_secrets(text):
    n = 0
    for rx in _SECRET:
        text, k = rx.subn("[REDACTED]", text)
        n += k
    return text, n


def quality(text):
    """-> (score 0..1, reasons). Heuristic gate; not a truth check."""
    reasons, n = [], len(text)
    if n < 200:
        return 0.0, ["too short"]
    letters = sum(c.isalpha() for c in text)
    if letters / n < 0.45:
        reasons.append("low letter ratio")
    words = re.findall(r"\w+", text.lower())
    if len(words) < 30:
        return 0.0, ["too few words"]
    uniq = len(set(words)) / len(words)
    if uniq < 0.12:
        reasons.append("very repetitive")
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) > 8 and len(set(lines)) / len(lines) < 0.5:
        reasons.append("many duplicate lines")
    lang = sum(w in _ID for w in words) + sum(w in _EN for w in words)
    if lang / len(words) < 0.04 and not re.search(r"[{};=]{3}|def |class |function ", text):
        reasons.append("no recognisable id/en function words")
    if sum(1 for l in lines if len(l) < 25) / max(1, len(lines)) > 0.8 and len(lines) > 20:
        reasons.append("mostly short lines (menu/list)")
    score = max(0.0, 1.0 - 0.3 * len(reasons))
    return score, reasons


# ---------------------------------------------------------------- the learner front door
class Ingestor:
    """learn(source) -> report. `inbox_dir` receives one .txt per document (+ .meta.json sidecar which lifelong ignores).
    A SHA1 index prevents re-learning identical text. `memory` (MemoryStore) optionally also gets the text as 'knowledge'."""

    def __init__(self, inbox_dir, memory=None, workspace="default", min_quality=0.6, max_chars_per_doc=400_000,
                 fetcher=fetch_url, searcher=search_urls, flag_injection=True, clock=time.time, truth=None):
        """truth: optional hoh.truthfilter.TruthFilter. With it, text is only released to the inbox (= only trained on) when
        the verdict allows; everything else waits in `<inbox>_quarantine/` and is re-checked as new documents arrive."""
        self.inbox = Path(inbox_dir)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.truth = truth
        self.quarantine = self.inbox.parent / (self.inbox.name + "_quarantine")
        self.memory, self.ws, self.minq, self.maxc = memory, workspace, min_quality, max_chars_per_doc
        self.fetcher, self.searcher, self.flag_inj, self.clock = fetcher, searcher, flag_injection, clock
        self.index_path = self.inbox / ".learned_index.json"
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    # --- reading ---
    def read(self, source):
        """source: http(s) URL, file path, or directory -> list[Document] (raises IngestError)."""
        if re.match(r"^https?://", source):
            data, ctype, final = self.fetcher(source)
            docs = parse_bytes(data, final or source, ctype)
            for d in docs:
                d.source = source
            return docs
        p = Path(source).expanduser()
        if p.is_dir():
            docs = []
            for f in sorted(p.rglob("*")):
                if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".meta.json"):
                    try:
                        docs += parse_bytes(f.read_bytes(), str(f))
                    except IngestError:
                        continue
            return docs
        if not p.is_file():
            raise IngestError(f"not found: {source}")
        return parse_bytes(p.read_bytes(), str(p))

    # --- filtering + storing ---
    def _store(self, doc):
        text, nsec = scrub_secrets(clean_text(doc.text))
        if len(text) > self.maxc:
            text = text[: self.maxc]
            doc.notes.append(f"truncated to {self.maxc} chars")
        score, why = quality(text)
        inj = bool(_INJECT.search(text))
        h = hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()
        rep = {"source": doc.source, "title": doc.title, "kind": doc.kind, "chars": len(text), "quality": round(score, 2),
               "secrets_redacted": nsec, "injection_flag": inj, "notes": doc.notes + why, "links": doc.links[:40]}
        if h in self.index:
            return {**rep, "status": "duplicate"}
        if score < self.minq:
            return {**rep, "status": "rejected", "reason": "; ".join(why) or "low quality"}
        if inj and self.flag_inj:
            text = _INJECT.sub("[removed]", text)       # keep the facts, drop the command-like phrasing
        name = f"{int(self.clock())}_{h[:10]}.txt"
        if self.truth is not None:
            user_supplied = not re.match(r"^https?://", doc.source or "")
            tr = self.truth.assess(text, doc.source, user_supplied=user_supplied)
            rep["truth"] = tr.meta()
            text = tr.text or text
            self.truth.register(text, doc.source, accepted=tr.learn)
            if not tr.learn:                               # NOT learned: kept aside, still searchable as an unverified note
                self.quarantine.mkdir(parents=True, exist_ok=True)
                (self.quarantine / name).write_text(text + "\n", encoding="utf-8")
                (self.quarantine / f"{name}.meta.json").write_text(
                    json.dumps({**rep, "source": doc.source, "quarantined_at": self.clock()}, ensure_ascii=False, indent=1), encoding="utf-8")
                self.index[h] = {"file": name, "source": doc.source, "quarantined": True}
                self.index_path.write_text(json.dumps(self.index), encoding="utf-8")
                if self.memory is not None:                # 'conversation' layer is never trained on by default
                    self.memory.add(self.ws, "conversation", f"[belum terverifikasi: {tr.verdict}] {(doc.title or doc.source)}: {text[:1500]}",
                                    {"quarantined": True, "source": doc.source, "verdict": tr.verdict}, importance=0.2)
                return {**rep, "status": "quarantined", "reason": f"{tr.verdict} (skor {tr.score}); " + "; ".join(tr.reasons),
                        "file": name}
        (self.inbox / name).write_text(text + "\n", encoding="utf-8")
        (self.inbox / f"{name}.meta.json").write_text(json.dumps({**rep, "learned_at": self.clock()}, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
        self.index[h] = {"file": name, "source": doc.source}
        self.index_path.write_text(json.dumps(self.index), encoding="utf-8")
        if self.memory is not None:
            self.memory.add_document(self.ws, doc.title or doc.source, text)
        return {**rep, "status": "accepted", "file": name}

    def recheck_quarantine(self):
        """Re-assess quarantined documents against the (now larger) fact index; release those that became corroborated."""
        if self.truth is None or not self.quarantine.is_dir():
            return {"released": 0, "still": 0}
        released = still = 0
        for f in sorted(self.quarantine.glob("*.txt")):
            meta_p = self.quarantine / f"{f.name}.meta.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            text = f.read_text(encoding="utf-8")
            tr = self.truth.assess(text, meta.get("source", ""), user_supplied=not re.match(r"^https?://", meta.get("source", "")))
            if tr.learn:
                (self.inbox / f.name).write_text((tr.text or text) + "\n", encoding="utf-8")
                meta.update(truth=tr.meta(), learned_at=self.clock(), released_from_quarantine=True)
                (self.inbox / f"{f.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
                self.truth.register(tr.text or text, meta.get("source", ""), accepted=True)
                f.unlink()
                meta_p.unlink(missing_ok=True)
                if self.memory is not None:
                    self.memory.add_document(self.ws, meta.get("title") or meta.get("source", ""), tr.text or text)
                released += 1
            else:
                still += 1
        return {"released": released, "still": still}

    def learn(self, source):
        try:
            docs = self.read(source)
        except IngestError as e:
            return {"source": source, "status": "error", "error": str(e), "documents": []}
        results = [self._store(d) for d in docs if d.text and d.text.strip()]
        ok = sum(r["status"] == "accepted" for r in results)
        if self.truth is not None and results:
            rc = self.recheck_quarantine()
            if rc["released"]:
                ok += rc["released"]
        return {"source": source, "status": "ok" if ok else "nothing_learned", "accepted": ok,
                "documents": results, "chars": sum(r["chars"] for r in results if r["status"] == "accepted")}

    def learn_search(self, query, n=5):
        try:
            urls = self.searcher(query, n)
        except IngestError as e:
            return {"query": query, "status": "error", "error": str(e), "results": []}
        res = [self.learn(u) for u in urls]
        return {"query": query, "status": "ok", "urls": urls, "accepted": sum(r.get("accepted", 0) for r in res), "results": res}

    def summary(self, rep):
        """One short human line for chat/CLI."""
        if "results" in rep:
            return f"cari '{rep['query']}': {len(rep.get('urls', []))} link, {rep.get('accepted', 0)} dokumen dipelajari" + \
                   (f" (error: {rep['error']})" if rep.get("error") else "")
        if rep["status"] == "error":
            return f"gagal: {rep['error']}"
        bad = [f"{d['status']}({d.get('reason', '')})" for d in rep["documents"] if d["status"] != "accepted"]
        return f"{rep['source']}: {rep['accepted']} dokumen diterima, {rep['chars']} karakter" + (f"; ditolak/dobel: {', '.join(bad)}" if bad else "")
