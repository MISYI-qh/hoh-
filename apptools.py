"""Alat bawaan aplikasi HOH: cari web, baca halaman, jalankan kode Python, buat berkas, ingatan ajaran.
Semua memakai pustaka standar Python (tanpa API/kunci). Pencarian & baca halaman butuh internet saat dipakai."""
import difflib, html, io, ipaddress, json, os, re, socket, subprocess, sys, tempfile, urllib.parse, urllib.request, zipfile

UA = "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Mobile Safari/537.36"


# ------------------------------------------------------------------ web
def _public_host(url):
    """Tolak alamat lokal/privat supaya alat ini tidak bisa dipakai membuka layanan di perangkat sendiri."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        infos = socket.getaddrinfo(u.hostname, None)
    except Exception:
        return None          # tidak bisa diterjemahkan (biasanya offline): bukan berarti dilarang
    for fam, _, _, _, sa in infos:
        ip = ipaddress.ip_address(sa[0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _get(url, timeout=15, max_bytes=3_000_000, data=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, "Accept-Language": "id,en;q=0.8", "Accept": "text/html,*/*;q=0.5"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(max_bytes), r.headers.get("Content-Type", ""), r.geturl()


def parse_ddg(page):
    """Hasil pencarian DuckDuckGo (versi HTML) -> [{title,url,snippet}]."""
    out = []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)', page, re.S):
        href, title, rest = m.group(1), m.group(2), m.group(3)
        href = html.unescape(href)
        if href.startswith("//"):
            href = "https:" + href
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        if "uddg" in q:
            href = q["uddg"][0]
        if not href.startswith("http") or "duckduckgo.com/y.js" in href:
            continue
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>', rest, re.S)
        clean = lambda x: re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip()
        out.append({"title": clean(title), "url": href, "snippet": clean(sn.group(1)) if sn else ""})
        if len(out) >= 6:
            break
    return out


def search(q):
    q = q.strip()[:200]
    if not q:
        return {"error": "empty"}
    try:
        body, _, _ = _get("https://html.duckduckgo.com/html/", data=urllib.parse.urlencode({"q": q}).encode(), timeout=15)
        res = parse_ddg(body.decode("utf-8", "replace"))
    except Exception as e:
        return {"error": "offline", "detail": type(e).__name__}
    return {"results": res} if res else {"error": "no_results"}


def page(url):
    from hoh import ingest
    ok = _public_host(url)
    if ok is False:
        return {"error": "blocked"}
    if ok is None:
        return {"error": "offline", "detail": "DNS"}
    try:
        data, ctype, final = _get(url, timeout=20)
    except Exception as e:
        return {"error": "offline", "detail": type(e).__name__}
    if _public_host(final) is False:
        return {"error": "blocked"}
    txt = ""
    try:
        if "html" in ctype.lower() or data[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
            m = re.search(rb"charset=([\w-]+)", data[:4000], re.I)
            txt = data.decode((m.group(1).decode() if m else "utf-8"), "replace")
            title, body = ingest.html_to_text(txt)
        else:
            docs = ingest.parse_bytes(data, final, ctype)
            title, body = os.path.basename(urllib.parse.urlparse(final).path) or final, "\n\n".join(d.text for d in docs)
    except Exception as e:
        return {"error": "unreadable", "detail": str(e)[:120]}
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if len(body) < 30:
        return {"error": "empty"}
    links = []
    if "<a" in (txt if "txt" in dir() else ""):
        seen = set()
        for m in re.finditer(r'<a\s[^>]*href=["\']([^"\'#][^"\']*)["\'][^>]*>(.*?)</a>', txt, re.S | re.I):
            u = urllib.parse.urljoin(final, html.unescape(m.group(1)))
            lbl = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m.group(2)))).strip()
            if u.startswith(("http://", "https://")) and u not in seen and 2 < len(lbl) < 90:
                seen.add(u)
                links.append({"url": u, "text": lbl})
            if len(links) >= 25:
                break
    return {"title": (title or final)[:200], "url": final, "text": body[:14000], "chars": len(body), "links": links}


# ------------------------------------------------------------------ kode
MAGICS = [(b"\x89PNG\r\n\x1a\n", "Gambar PNG"), (b"\xff\xd8\xff", "Gambar JPEG"), (b"GIF8", "Gambar GIF"), (b"%PDF-", "Dokumen PDF"),
          (b"PK\x03\x04", "Arsip ZIP (atau APK/DOCX/XLSX/JAR)"), (b"\x1f\x8b", "Arsip GZIP"), (b"7z\xbc\xaf\x27\x1c", "Arsip 7-Zip"), (b"Rar!", "Arsip RAR"),
          (b"\x7fELF", "Program ELF (Linux/Android)"), (b"MZ", "Program Windows (EXE/DLL)"), (b"SQLite format 3\x00", "Basis data SQLite"),
          (b"ID3", "Audio MP3"), (b"OggS", "Audio/Video OGG"), (b"fLaC", "Audio FLAC"), (b"\x00\x00\x00", "Kemungkinan MP4/MOV/HEIC"),
          (b"dex\n", "Kode Android (DEX)"), (b"\xca\xfe\xba\xbe", "Kelas Java / biner Mach-O"), (b"{\\rtf", "Dokumen RTF"), (b"\xd0\xcf\x11\xe0", "Dokumen Office lama (DOC/XLS/PPT)")]


def describe_bytes(data, name=""):
    """Deskripsi aman untuk berkas apa saja. Tidak ada yang dijalankan atau diekstrak ke disk."""
    import hashlib, math, io, zipfile, tarfile
    n = len(data)
    if n == 0:
        return {"text": "Berkas kosong (0 byte).", "generic": True}
    head = data[:4096]
    if b"\x00" not in head:                                   # kemungkinan teks
        for enc in ("utf-8", "utf-16"):
            try:
                t = data.decode(enc)
            except Exception:
                continue
            if sum(c < " " and c not in "\n\r\t" for c in t[:20000]) / max(1, min(len(t), 20000)) < 0.05:
                return {"text": t[:300000], "chars": len(t)}
    kind = next((lbl for m, lbl in MAGICS if data.startswith(m)), "")
    if data[4:8] in (b"ftyp",):
        kind = "Video/Audio MP4 atau gambar HEIC"
    if data[:4] == b"RIFF" and data[8:12] in (b"WAVE", b"WEBP", b"AVI "):
        kind = {b"WAVE": "Audio WAV", b"WEBP": "Gambar WebP", b"AVI ": "Video AVI"}[data[8:12]]
    cnt = [0] * 256
    for b_ in data[:2_000_000]:
        cnt[b_] += 1
    tot = sum(cnt)
    ent = -sum(c / tot * math.log2(c / tot) for c in cnt if c)
    lines = ["Tipe: " + (kind or "tidak dikenali (data biner)"), "Ukuran: %d byte (%.1f KB)" % (n, n / 1024),
             "SHA-256: " + hashlib.sha256(data).hexdigest(), "Entropi: %.2f bit/byte (%s)" % (ent, "tinggi: terkompresi atau terenkripsi" if ent > 7.5 else "sedang" if ent > 5 else "rendah: banyak pola berulang")]
    try:
        if data.startswith(b"PK"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = [i.filename for i in z.infolist()]
            lines.append("Isi arsip: %d berkas" % len(names))
            lines += ["- " + x for x in names[:25]]
            if "AndroidManifest.xml" in names:
                lines.append("Catatan: ini tampak seperti paket aplikasi Android (APK).")
        elif data[:2] == b"\x1f\x8b" or name.lower().endswith((".tar", ".tgz")):
            with tarfile.open(fileobj=io.BytesIO(data)) as tf:
                names = tf.getnames()
            lines.append("Isi arsip: %d berkas" % len(names))
            lines += ["- " + x for x in names[:25]]
    except Exception:
        pass
    import re
    strs = re.findall(rb"[ -~]{6,}", data[:500_000])[:15]
    if strs:
        lines.append("Teks yang terbaca di dalamnya:")
        lines += ["> " + x.decode("ascii", "replace")[:120] for x in strs]
    lines.append("Catatan: berkas ini tidak dijalankan dan tidak diubah. Aku hanya membaca ciri-cirinya.")
    return {"text": "\n".join(lines), "generic": True}


def _norm(s):
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def match_taught(q, taught, thr=0.86):
    nq = _norm(q)
    if not nq:
        return None
    best, ba = 0.0, None
    for t in taught[:400]:
        nt = _norm(str(t.get("q", "")))
        if not nt:
            continue
        r = 1.0 if nt == nq else difflib.SequenceMatcher(None, nq, nt).ratio()
        if r > best:
            best, ba = r, str(t.get("a", ""))
    return ba if best >= thr and ba else None
