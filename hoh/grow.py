"""Tumbuh: HOH belajar dan berkembang sendiri di latar belakang (tanpa harus disuruh tidur).

Saat tidak ada yang mengobrol (idle), satu siklus bisa melakukan:
  jelajah   - mencari di web untuk topik minat (maks `auto_searches` pencarian per siklus bila tidak disuruh), membaca halaman
              dengan sopan (robots.txt, jeda per situs, batas harian), menyaring teks, lalu menyimpannya sebagai catatan;
  riset     - bila pengguna menyuruh (`/riset topik`), HOH boleh mencari BEBAS (batas jauh lebih longgar) lalu menyusun laporan
              yang menunggu di kotak masuk dan dipakai untuk menjawab saat ada yang bertanya lagi;
  latih     - bila catatan baru / jempol-suka sudah cukup, melatih diri secara singkat. Hasilnya hanya dipakai jika tidak lebih
              buruk dari model sekarang pada soal jangkar (pengaman dari "lupa"); model lama tetap tersimpan sebagai slot;
  umpan balik - jempol suka -> pasangan tanya-jawab itu dilatihkan; jempol tidak suka -> tidak pernah dilatihkan dan topiknya
              dijadikan target riset.
Semua batasan: gerbang idle, jeda, kuota harian, tombol jeda. Jaringan TIDAK diverifikasi di tempat pembuatan (tes memakai fungsi
palsu); apakah HOH benar-benar makin pintar diukur oleh pengaman jangkar, bukan oleh siklus ini.

Penyimpanan: satu berkas JSON dengan kunci berkas lintas-proses (UI dan layanan latar belakang Android boleh berbeda proses)."""
import hashlib, json, os, re, threading, time, urllib.parse
from collections import Counter

DEFAULTS = dict(enabled=True, notify=True, auto_searches=5, auto_pages=8, idle_seconds=120, explore_interval=2700, train_interval=1800,
                min_train_chars=3000, rehearse_interval=21600, free_searches=25, free_pages=40, free_seconds=480, daily_pages=60, seconds_per_cycle=240)
_INJECT = re.compile(r"(?i)(ignore (all )?(previous|prior) instructions|abaikan (semua )?instruksi|you are now|system prompt|"
                     r"jailbreak|act as)")
_STOP = set("yang dan di ke dari untuk pada dengan adalah ini itu atau juga dalam tidak akan oleh sebagai karena para telah the of and to in is that for with was".split())


class Store:
    """JSON dengan kunci berkas lintas-proses; semua ubah lewat update(fn)."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def _lock(self):
        lp, t0 = self.path + ".lock", time.time()
        while True:
            try:
                fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return lp
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lp) > 10:
                        os.remove(lp)
                        continue
                except OSError:
                    pass
                if time.time() - t0 > 8:
                    raise TimeoutError("store busy")
                time.sleep(0.02)

    def read(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            s = {}
        base = dict(cfg=dict(DEFAULTS), goals=[], notes=[], likes=[], dislikes=[], tasks=[], inbox=[],
                    stats=dict(day="", searches=0, pages=0, last_explore=0, last_train=0, generation=0, trained_chars=0, last_notify=0), log=[])
        for k, v in base.items():
            s.setdefault(k, v)
        for k, v in DEFAULTS.items():
            s["cfg"].setdefault(k, v)
        return s

    def update(self, fn):
        lp = self._lock()
        try:
            s = self.read()
            r = fn(s)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            return r
        finally:
            try:
                os.remove(lp)
            except OSError:
                pass


def _words(text):
    return [w for w in re.findall(r"[a-zA-Z\u00c0-\u024f]{4,}", text.lower()) if w not in _STOP]


class Grower:
    def __init__(self, store, hooks, clock=time.time, sleep=time.sleep):
        self.store, self.h, self.clock, self.sleep = store, hooks, clock, sleep
        self.stop_flag = threading.Event()
        self.phase = "idle"

    # ---------------------------------------------------------------- perintah dari UI / chat
    def add_goal(self, topic, prio=1.0, src="user"):
        topic = " ".join(str(topic).split())[:120]
        if len(topic) < 3:
            return False

        def f(s):
            for g in s["goals"]:
                if g["t"].lower() == topic.lower():
                    g["p"] = max(g["p"], prio)
                    return True
            s["goals"].append(dict(t=topic, p=prio, src=src, n=0, last=0))
            return True
        return self.store.update(f)

    def queue_research(self, topic):
        topic = " ".join(str(topic).split())[:160]
        if len(topic) < 3:
            return None
        tid = hashlib.sha1((topic + str(self.clock())).encode()).hexdigest()[:8]
        self.store.update(lambda s: s["tasks"].append(dict(id=tid, topic=topic, state="queued", ts=self.clock())))
        self._log("riset", "Antrean riset bebas: " + topic)
        return tid

    def feedback(self, q, a, good):
        q, a = str(q).strip()[:300], str(a).strip()[:400]
        if not q or not a:
            return False

        def f(s):
            if good:
                s["likes"].append(dict(q=q, a=a, ts=self.clock()))
                s["likes"] = s["likes"][-200:]
                s["dislikes"] = [d for d in s["dislikes"] if not (d["q"] == q and d["a"] == a)]
            else:
                s["dislikes"].append(dict(q=q, a=a, ts=self.clock()))
                s["dislikes"] = s["dislikes"][-200:]
                s["likes"] = [d for d in s["likes"] if not (d["q"] == q and d["a"] == a)]
        self.store.update(f)
        if not good:
            self.add_goal(q[:100], 2.0, "tidak-suka")        # jawaban buruk -> cari jawaban yang lebih baik
        return True

    def set_cfg(self, **kw):
        def f(s):
            for k, v in kw.items():
                if k in DEFAULTS and isinstance(v, type(DEFAULTS[k])):
                    s["cfg"][k] = v
        self.store.update(f)

    def _log(self, ev, msg):
        def f(s):
            s["log"] = (s["log"] + [dict(t=self.clock(), ev=ev, msg=msg[:160])])[-60:]
        self.store.update(f)

    def _notify(self, title, body, force=False):
        s = self.store.read()
        if not s["cfg"]["notify"]:
            return
        if not force and self.clock() - s["stats"]["last_notify"] < 1200:
            return
        self.store.update(lambda st: st["stats"].__setitem__("last_notify", self.clock()))
        try:
            self.h["notify"](title, body)
        except Exception:
            pass

    # ---------------------------------------------------------------- penyaring teks
    @staticmethod
    def clean(text):
        text = _INJECT.sub(" ", text or "")
        text = re.sub(r"[ \t]+", " ", text)
        text = "\n".join(l.strip() for l in text.split("\n") if len(l.strip()) > 30)
        if len(text) < 400:
            return None
        letters = sum(c.isalpha() for c in text)
        if letters / len(text) < 0.55:
            return None
        w = text.lower().split()
        if len(w) < 60 or len(set(w)) / len(w) < 0.15:
            return None
        return text[:6000]

    # ---------------------------------------------------------------- jelajah / riset
    def _study(self, topic, max_searches, max_pages, seconds):
        t0, got, searches, pages = self.clock(), [], 0, 0
        s0 = self.store.read()
        seen = {n["url"] for n in s0["notes"]}
        queries = [topic] + [topic + " " + x for x in ("pengertian", "contoh", "sejarah", "makna", "manfaat", "penjelasan", "ulasan", "artikel", "filsafat",
                                                       "puisi", "penelitian", "fakta", "perspektif", "pandangan", "kutipan")]
        for q in queries:
            if searches >= max_searches or pages >= max_pages or self.clock() - t0 > seconds or self.stop_flag.is_set():
                break
            searches += 1
            try:
                urls = self.h["search"](q, 4) or []
            except Exception:
                urls = []
            for u in urls:
                if pages >= max_pages or self.clock() - t0 > seconds or self.stop_flag.is_set():
                    break
                if u in seen:
                    continue
                seen.add(u)
                pages += 1
                try:
                    txt = self.h["fetch"](u)
                except Exception:
                    txt = None
                txt = self.clean(txt) if txt else None
                if txt:
                    got.append(dict(id=hashlib.sha1(txt[:400].encode()).hexdigest()[:10], topic=topic, url=u,
                                    host=urllib.parse.urlparse(u).netloc, text=txt, ts=self.clock(), trained=False))
        return got, searches, pages

    def _store_notes(self, got, searches, pages):
        day = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))

        def f(s):
            if s["stats"]["day"] != day:
                s["stats"].update(day=day, searches=0, pages=0)
            have = {n["id"] for n in s["notes"]}
            new = [n for n in got if n["id"] not in have]
            s["notes"] = (s["notes"] + new)[-120:]
            s["stats"]["searches"] += searches
            s["stats"]["pages"] += pages
            return len(new)
        return self.store.update(f)

    def _reflect(self, got):
        """Kata sering muncul di bacaan baru -> target belajar baru (heuristik, bukan penilaian kebenaran)."""
        c = Counter(w for n in got for w in set(_words(n["text"])))
        s = self.store.read()
        known = " ".join(g["t"].lower() for g in s["goals"])
        added = 0
        for w, k in c.most_common(30):
            if k >= 3 and len(w) >= 5 and w not in known and len(s["goals"]) + added < 80 and added < 2:
                self.add_goal(w, 0.5, "rasa-ingin-tahu")
                added += 1
        return added

    def _next_goal(self, s):
        now = self.clock()
        live = [g for g in s["goals"] if now - g["last"] > 3600]
        if not live:
            return None
        return max(live, key=lambda g: g["p"] / (1 + g["n"]))

    def _summary(self, topic, notes):
        try:
            from .consolidate import extractive_summary
            return extractive_summary([n["text"] for n in notes], max_chars=600)
        except Exception:
            return " ".join(notes[0]["text"].split(". ")[:3])[:600] if notes else ""

    def _do_research(self, task):
        c = self.store.read()["cfg"]
        self.phase = "riset"
        self.store.update(lambda s: [t.__setitem__("state", "running") for t in s["tasks"] if t["id"] == task["id"]])
        self._log("riset", "Mulai riset bebas: " + task["topic"])
        got, se, pg = self._study(task["topic"], c["free_searches"], c["free_pages"], c["free_seconds"])
        n = self._store_notes(got, se, pg)
        summ = self._summary(task["topic"], got)
        if got and summ:
            srcs = sorted({g["host"] for g in got})[:6]
            self.store.update(lambda s: s["inbox"].append(dict(id=task["id"], title="Hasil riset: " + task["topic"], text=summ, sources=srcs,
                                                               topic=task["topic"], ts=self.clock(), read=False)))
            self._notify("HOH selesai meriset", task["topic"][:60], force=True)
        else:
            self._notify("HOH meriset", "Belum menemukan bahan yang layak untuk: " + task["topic"][:50], force=True)
        self.store.update(lambda s: [t.__setitem__("state", "done") for t in s["tasks"] if t["id"] == task["id"]])
        self._log("riset", f"Selesai: {se} pencarian, {pg} halaman, {n} catatan baru")
        return dict(phase="riset", pages=pg, notes=n)

    def _do_explore(self):
        s = self.store.read()
        c = s["cfg"]
        if s["stats"]["pages"] >= c["daily_pages"] and s["stats"]["day"] == time.strftime("%Y-%m-%d", time.gmtime(self.clock())):
            return dict(phase="idle", why="kuota harian")
        g = self._next_goal(s)
        if g is None:                                   # tak ada target: modelnya sendiri yang memilih apa yang ingin dipelajari
            self.add_goal(self.h["propose"](), 0.5, "model")
            g = self._next_goal(self.store.read())
            if g is None:
                return dict(phase="idle")
        self.phase = "jelajah"
        got, se, pg = self._study(g["t"], c["auto_searches"], c["auto_pages"], c["seconds_per_cycle"])
        n = self._store_notes(got, se, pg)

        def f(st):
            for x in st["goals"]:
                if x["t"] == g["t"]:
                    x["n"] += 1; x["last"] = self.clock()
            st["stats"]["last_explore"] = self.clock()
        self.store.update(f)
        self._reflect(got)
        if n:
            self._notify("HOH belajar sendiri", f"Membaca {n} bahan baru tentang: {g['t'][:50]}")
        self._log("jelajah", f"{g['t']}: {se} pencarian, {pg} halaman, {n} catatan baru")
        return dict(phase="jelajah", topic=g["t"], pages=pg, notes=n)

    # ---------------------------------------------------------------- latih diri
    def _do_train(self, force=False):
        s = self.store.read()
        notes = [n for n in s["notes"] if not n["trained"]]
        bad = {(d["q"], d["a"]) for d in s["dislikes"]}
        likes = [(l["q"], l["a"]) for l in s["likes"][-40:] if (l["q"], l["a"]) not in bad]
        chars = sum(len(n["text"]) for n in notes)
        if not force and chars < s["cfg"]["min_train_chars"] and len(likes) < 3:
            return None
        if self.h["busy"]():
            return None
        self.phase = "latih"
        texts = [n["text"] for n in notes][:12]
        self._log("latih", f"Berlatih sendiri: {chars} karakter, {len(likes)} jawaban disukai")
        res = self.h["train"](texts, likes, 45) or {}
        ok = bool(res.get("ok"))

        def f(st):
            st["stats"]["last_train"] = self.clock()
            if ok:
                ids = {n["id"] for n in notes[:12]}
                for n in st["notes"]:
                    if n["id"] in ids:
                        n["trained"] = True
                st["likes"] = [l for l in st["likes"] if (l["q"], l["a"]) not in set(likes)]
                st["stats"]["generation"] += 1
                st["stats"]["trained_chars"] += chars
        self.store.update(f)
        if ok:
            self._notify("HOH bertumbuh", "Selesai berlatih sendiri (versi %d)" % (s["stats"]["generation"] + 1), force=True)
            self._log("latih", "Diterima: " + str(res.get("detail", ""))[:100])
        else:
            self._log("latih", "Ditolak/gagal: " + str(res.get("detail", ""))[:100])
        return dict(phase="latih", ok=ok)

    # ---------------------------------------------------------------- satu keputusan
    def tick(self):
        s = self.store.read()
        c = s["cfg"]
        if not c["enabled"] or self.stop_flag.is_set():
            self.phase = "dijeda"
            return dict(phase="dijeda")
        if self.h["idle_for"]() < c["idle_seconds"] or self.h["busy"]():
            self.phase = "siaga"
            return dict(phase="siaga")
        self.h["wake"](True)
        try:
            return self._work(s, c)
        finally:
            self.h["wake"](False)

    def _work(self, s, c):
        task = next((t for t in s["tasks"] if t["state"] == "queued"), None)
        if task:
            return self._do_research(task)
        now = self.clock()
        out = dict(phase="diam")
        if now - s["stats"]["last_explore"] >= c["explore_interval"]:
            out = self._do_explore()
        if now - s["stats"]["last_train"] >= c["train_interval"]:
            r = self._do_train(force=now - s["stats"]["last_train"] >= c["rehearse_interval"])    # tanpa data baru/offline: latihan ulang berpengaman
            if r:
                out = r
        self.phase = "diam" if out["phase"] in ("diam", "idle") else out["phase"]
        return out

    def run_forever(self, poll=30.0):
        while not self.stop_flag.is_set():
            try:
                self.tick()
            except Exception as e:                       # daemon harus bertahan dari satu kegagalan
                try:
                    self._log("galat", f"{type(e).__name__}: {e}")
                except Exception:
                    pass
            self.sleep(poll)

    # ---------------------------------------------------------------- dipakai saat ada yang bertanya lagi
    def notes_about(self, topic):
        """Alat <ingat>: ringkasan dari kotak masuk riset/catatan yang paling mirip topik (dipanggil oleh model, bukan otomatis)."""
        qw = set(_words(topic))
        s = self.store.read()
        pool = [(r["topic"], r["text"], r.get("sources", [])) for r in s["inbox"]] + [(n["topic"], n["text"][:600], [n["host"]]) for n in s["notes"]]
        scored = sorted(((len(qw & set(_words(t + " " + x[:200]))) / max(1, len(qw)), t, x, src) for t, x, src in pool), reverse=True)
        best = scored[0] if scored else None
        return dict(text=best[2], sources=best[3], topic=best[1]) if best and best[0] > 0 else None

    def summary(self):
        s = self.store.read()
        return dict(cfg=s["cfg"], phase=self.phase, goals=[g["t"] for g in sorted(s["goals"], key=lambda g: -g["p"])][:8],
                    notes=len(s["notes"]), pending_chars=sum(len(n["text"]) for n in s["notes"] if not n["trained"]),
                    likes=len(s["likes"]), dislikes=len(s["dislikes"]), generation=s["stats"]["generation"],
                    today=dict(searches=s["stats"]["searches"], pages=s["stats"]["pages"]),
                    tasks=[dict(id=t["id"], topic=t["topic"], state=t["state"]) for t in s["tasks"][-5:]],
                    inbox=[dict(id=r["id"], title=r["title"], text=r["text"], sources=r["sources"], read=r["read"], ts=r["ts"]) for r in s["inbox"][-5:]],
                    log=s["log"][-12:])
