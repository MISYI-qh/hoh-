#!/usr/bin/env python3
"""Server lokal HOH. Obrolan, pelatihan, dan ekspor/impor model memakai paket `hoh` (NumPy).
Semua berjalan di perangkat sendiri: server hanya mendengarkan 127.0.0.1."""
import apptools, io, json, os, re, shutil, subprocess, sys, tempfile, threading, time, urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _home():
    want = os.environ.get("HOH_HOME") or os.path.join(ROOT, "hoh_home")
    try:
        os.makedirs(want, exist_ok=True)
        t = os.path.join(want, ".w")
        open(t, "w").close()
        os.remove(t)
        return want
    except OSError:
        alt = os.path.join(tempfile.gettempdir(), "hoh_home")
        os.makedirs(alt, exist_ok=True)
        return alt


HOME = _home()
MODEL_DIR = os.path.join(HOME, "model")
CUR = os.path.join(MODEL_DIR, "current.npz")
PREV = os.path.join(MODEL_DIR, "previous.npz")
JOBS = os.path.join(HOME, "jobs")
SETTINGS = os.path.join(HOME, "settings.json")
STARTER = os.path.join(ROOT, "model", "hoh-starter.hh")
SLOTS_DIR = os.path.join(MODEL_DIR, "slots")
SLOTS = os.path.join(MODEL_DIR, "slots.json")
SEED = os.path.join(ROOT, "model", "seed.jsonl")
STATIC = {"/": "index.html", "/index.html": "index.html", "/manifest.json": "manifest.json", "/icon.png": "icon.png",
          "/icon-1024.png": "icon-1024.png", "/icon.svg": "icon.svg", "/favicon.ico": "icon.png"}
MIME = {".html": "text/html; charset=utf-8", ".json": "application/manifest+json", ".png": "image/png", ".svg": "image/svg+xml"}
MAX_UPLOAD = 800 * 1024 * 1024
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(JOBS, exist_ok=True)


def payload_voice(payload):
    import io, numpy as np
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            return np.array(z["voice/profile"]) if "voice/profile" in z.files else None
    except Exception:
        return None


def add_voice(payload, vec):
    """Sisipkan/ganti profil suara di dalam payload .npz (vec=None -> hapus)."""
    import io, numpy as np
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "voice/profile"}
    if vec is not None:
        arrays["voice/profile"] = np.asarray(vec, np.float32)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def android_call(fn):
    """Jalankan fungsi yang memakai pyjnius; aman di desktop (diabaikan)."""
    if not os.environ.get("ANDROID_ARGUMENT"):
        return False
    try:
        fn()
        return True
    except Exception:
        return False


def notify(title, body):
    def go():
        from jnius import autoclass
        try:
            PA = autoclass("org.kivy.android.PythonActivity")
            act = PA.mActivity
        except Exception:
            act = None
        if act is None:
            PA = autoclass("org.kivy.android.PythonActivity")
            act = autoclass("org.kivy.android.PythonService").mService
        Ctx = autoclass("android.content.Context")
        NM = act.getSystemService(Ctx.NOTIFICATION_SERVICE)
        Build = autoclass("android.os.Build$VERSION")
        if Build.SDK_INT >= 26:
            Ch = autoclass("android.app.NotificationChannel")
            ch = Ch("hoh", "HOH", 3)
            NM.createNotificationChannel(ch)
            B = autoclass("android.app.Notification$Builder")(act, "hoh")
        else:
            B = autoclass("android.app.Notification$Builder")(act)
        Intent = autoclass("android.content.Intent")
        PI = autoclass("android.app.PendingIntent")
        it = Intent(act, PA)
        pi = PI.getActivity(act, 0, it, PI.FLAG_IMMUTABLE if Build.SDK_INT >= 23 else 0)
        R = autoclass(act.getPackageName() + ".R$mipmap") if False else None
        B.setContentTitle(str(title)).setContentText(str(body)).setAutoCancel(True).setContentIntent(pi)
        B.setSmallIcon(act.getApplicationInfo().icon)
        NM.notify(1, B.build())
    return android_call(go)


_WL = None


def wakelock(on):
    """Jaga CPU tetap hidup selama HOH bekerja di latar belakang (layar mati). Hanya Android."""
    def go():
        global _WL
        from jnius import autoclass
        try:
            ctx = autoclass("org.kivy.android.PythonService").mService
        except Exception:
            ctx = None
        ctx = ctx or autoclass("org.kivy.android.PythonActivity").mActivity
        pm = ctx.getSystemService(autoclass("android.content.Context").POWER_SERVICE)
        if on:
            _WL = _WL or pm.newWakeLock(1, "hoh:tumbuh")
            if not _WL.isHeld():
                _WL.acquire(15 * 60 * 1000)
        elif _WL is not None and _WL.isHeld():
            _WL.release()
    return android_call(go)


def ignore_battery_optimization():
    def go():
        from jnius import autoclass
        PA = autoclass("org.kivy.android.PythonActivity")
        act = PA.mActivity
        Intent, Uri, Settings = autoclass("android.content.Intent"), autoclass("android.net.Uri"), autoclass("android.provider.Settings")
        it = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
        it.setData(Uri.parse("package:" + act.getPackageName()))
        act.startActivity(it)
    return android_call(go)


def open_external(url):
    if not str(url).startswith(("http://", "https://")):
        return False
    if os.environ.get("ANDROID_ARGUMENT"):
        def go():
            from jnius import autoclass
            PA = autoclass("org.kivy.android.PythonActivity")
            Intent, Uri = autoclass("android.content.Intent"), autoclass("android.net.Uri")
            PA.mActivity.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(str(url))))
        return android_call(go)
    try:
        import webbrowser
        return bool(webbrowser.open(str(url)))
    except Exception:
        return False


def hh_mod():
    from hoh import hh
    return hh


def ensure_numpy():
    import importlib.util as u
    if u.find_spec("numpy") is None and not os.environ.get("ANDROID_ARGUMENT"):
        print("Mengunduh NumPy…")
        subprocess.call([sys.executable, "-m", "pip", "install", "-q", "numpy"])


ensure_numpy()


def load_settings():
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(s):
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f)
    os.replace(tmp, SETTINGS)


def export_dir():
    """Folder tujuan ekspor di perangkat (Download di Android bila bisa ditulis)."""
    for d in ("/storage/emulated/0/Download", os.path.expanduser("~/Downloads"), os.path.expanduser("~/Download")):
        if os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    d = os.path.join(HOME, "exports")
    os.makedirs(d, exist_ok=True)
    return d


def atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class Engine:
    def __init__(self):
        import importlib.util as u
        self.np = u.find_spec("numpy") is not None
        self.pkg = u.find_spec("hoh") is not None
        self.ok = self.np and self.pkg
        self.gen, self.gen_key = None, None
        self.lock = threading.RLock()
        self.proc, self.cancel = None, False
        self.st = self._idle()
        self._info, self._info_key = None, None
        if self.ok:
            self._seed_model()
            threading.Thread(target=self._warm, daemon=True).start()

    def _warm(self):
        """Muat model di latar belakang supaya pesan pertama tidak menunggu."""
        try:
            if self.ready():
                self._gen()
        except Exception:
            pass

    @staticmethod
    def _idle():
        return dict(running=False, phase="", step=0, total=0, loss=[], log=[], eta=None, done=False, error="", id=0)

    # ------------------------------------------------------------------ model files
    def _seed_model(self):
        """Pasang model bawaan pada jalan pertama (kalau belum ada model)."""
        if os.path.exists(CUR) or not os.path.exists(STARTER):
            return
        try:
            with open(STARTER, "rb") as f:
                _, payload = hh_mod().verify_and_extract(f.read())
            self._install(payload, "bawaan", "Bawaan", "bawaan")
        except Exception as e:
            print("Model bawaan gagal dipasang:", e)

    def ready(self):
        return os.path.exists(CUR)

    def status(self):
        return {"engine": self.ok, "numpy": self.np, "model": self.pkg, "ready": self.ready(), "training": self.st["running"],
                "app": bool(os.environ.get("ANDROID_ARGUMENT")), "grow": (self.grower().phase if self.ok else ""), "voice": self.voice_info().get("trained", False)}

    def info(self):
        if not self.ok or not self.ready():
            return {"ready": False}
        key = (os.path.getmtime(CUR), os.path.exists(PREV))
        if self._info_key == key:
            return self._info
        from hoh.checkpoint import load_checkpoint
        from hoh.config import ModelConfig
        ck = load_checkpoint(CUR)
        c = ModelConfig(**ck["model_config"])
        n = int(sum(int(v.size) for v in ck["model"].values()))
        tr = ck.get("train_config") or {}
        self._info = {"ready": True, "params": n, "layers": c.n_layer, "width": c.n_embd, "heads": c.n_head, "block": c.block_size,
                      "vocab": c.vocab_size, "step": int(ck["step"]), "bytes": os.path.getsize(CUR),
                      "updated": int(os.path.getmtime(CUR)), "has_prev": os.path.exists(PREV),
                      "tokenizer": (ck.get("tokenizer") or {}).get("type", "utf8-byte"), "mode": tr.get("mode", "")}
        self._info_key = key
        return self._info

    def _install(self, payload, kind="latih", name="", sid=None, keep_voice=True):
        """Pasang checkpoint .npz baru jadi model aktif (yang lama jadi 'previous') dan catat sebagai slot."""
        if keep_voice and payload_voice(payload) is None:      # hasil latih mewarisi suara model sebelumnya
            v = self.voice_vec()
            if v is not None:
                payload = add_voice(payload, v)
        with self.lock:
            if os.path.exists(CUR):
                shutil.copy2(CUR, PREV)
            atomic_write(CUR, payload)
            self.gen, self._info_key = None, None
        if kind:
            self._add_slot(payload, kind, name or ("Hasil latih " + time.strftime("%d/%m %H:%M")), True, sid)

    # ------------------------------------------------------------------ slot model: bawaan / hasil latih / impor
    def _slots(self):
        try:
            with open(SLOTS, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"active": "", "items": {}}

    def _add_slot(self, payload, kind, name, activate, sid=None):
        os.makedirs(SLOTS_DIR, exist_ok=True)
        sid = sid or "%s-%d" % (kind, int(time.time() * 1000))
        fp = os.path.join(SLOTS_DIR, sid + ".npz")
        atomic_write(fp, payload)
        with self.lock:
            s = self._slots()
            s["items"][sid] = {"name": name[:40], "kind": kind, "file": fp, "created": int(time.time()), "bytes": len(payload)}
            if activate:
                s["active"] = sid
            atomic_write(SLOTS, json.dumps(s).encode("utf-8"))
        return sid

    # ------------------------------------------------------------------ suara (bagian dari model yang sama)
    def voice_vec(self):
        try:
            mt = os.path.getmtime(CUR)
        except OSError:
            return None
        if getattr(self, "_vc", (0, None))[0] != mt:
            with open(CUR, "rb") as f:
                self._vc = (mt, payload_voice(f.read()))
        return self._vc[1]

    def _set_voice(self, vec):
        with open(CUR, "rb") as f:
            new = add_voice(f.read(), vec)
        with self.lock:
            atomic_write(CUR, new)
            s = self._slots()
            it = s["items"].get(s.get("active", ""))
            if it and os.path.exists(it["file"]):
                with open(it["file"], "rb") as f:
                    atomic_write(it["file"], add_voice(f.read(), vec))
            self._info_key = None

    def voice_info(self):
        from hoh import voice
        v = self.voice_vec()
        if v is None or not voice.valid_profile(v):
            return {"trained": False}
        p = voice.vec_to_params(v)
        return {"trained": True, "f0": round(p["f0"]), "scale": round(p["scale"], 2), "rate": round(p["rate"], 2), "seconds": round(p["seconds"])}

    def voice_train(self, wav):
        from hoh import voice
        if self.st["running"]:
            raise hh_mod().HhError("busy", "sedang melatih")
        x, sr = voice.read_wav(wav)
        vec, info = voice.estimate_profile(x, sr)
        self._set_voice(vec)
        return dict(self.voice_info(), **{k: info[k] for k in ("seconds",)})

    def voice_reset(self):
        self._set_voice(None)

    def voice_speak(self, text):
        from hoh import voice
        v = self.voice_vec()
        if v is None or not voice.valid_profile(v):
            raise hh_mod().HhError("no_voice", "suara belum dilatih")
        return voice.to_wav(voice.speak(text, v))

    def models(self):
        s = self._slots()
        if "bawaan" not in s["items"] and os.path.exists(STARTER):      # pastikan model bawaan selalu ada sebagai pilihan
            try:
                with open(STARTER, "rb") as f:
                    _, payload = hh_mod().verify_and_extract(f.read())
                self._add_slot(payload, "bawaan", "Bawaan", False, "bawaan")
                s = self._slots()
            except Exception:
                pass
        act = s.get("active") or ("bawaan" if "bawaan" in s["items"] else "")
        rows = [dict(id=k, name=v["name"], kind=v["kind"], created=v["created"], bytes=v.get("bytes", 0), active=(k == act))
                for k, v in sorted(s["items"].items(), key=lambda kv: (kv[0] != "bawaan", -kv[1]["created"]))]
        return {"models": rows, "active": act}

    def use(self, sid):
        if self.st["running"]:
            raise hh_mod().HhError("busy", "sedang melatih")
        s = self._slots(); it = s["items"].get(sid)
        if not it or not os.path.exists(it["file"]):
            raise hh_mod().HhError("missing", "model tidak ditemukan")
        with open(it["file"], "rb") as f:
            payload = f.read()
        self._install(payload, kind=None, keep_voice=False)
        with self.lock:
            s["active"] = sid
            atomic_write(SLOTS, json.dumps(s).encode("utf-8"))

    def delete(self, sid):
        s = self._slots(); it = s["items"].get(sid)
        if sid == "bawaan" or not it or sid == s.get("active"):
            return False
        try:
            os.remove(it["file"])
        except OSError:
            pass
        with self.lock:
            del s["items"][sid]
            atomic_write(SLOTS, json.dumps(s).encode("utf-8"))
        return True

    def export_bytes(self, with_opt=False, name="HOH", sid=""):
        hh = hh_mod()
        if not self.ready():
            raise hh.HhError("no_model", "belum ada model")
        it = self._slots()["items"].get(sid) if sid else None
        if it:
            return hh.pack(it["file"], include_optimizer=with_opt, name=it["name"])
        return hh.pack(CUR, include_optimizer=with_opt, name=name)

    def import_bytes(self, data):
        from hoh import hh
        if self.st["running"]:
            raise hh.HhError("busy", "sedang melatih")
        h, payload = hh.verify_and_extract(data)
        nm = str(h.get("name") or "Impor")
        if nm in {v["name"] for v in self._slots()["items"].values()}:
            nm += " (impor)"
        sid = self._add_slot(payload, "impor", nm, False)   # tidak otomatis dipakai: pengguna memilih
        return dict(hh.info_of(h), id=sid)

    def revert(self):
        with self.lock:
            if self.st["running"] or not os.path.exists(PREV):
                return False
            tmp = CUR + ".swap"
            shutil.copy2(CUR, tmp)
            shutil.copy2(PREV, CUR)
            os.replace(tmp, PREV)
            self.gen, self._info_key = None, None
        return True

    def reset(self):
        if self.st["running"]:
            return False
        self.models()
        try:
            self.use("bawaan")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ chat
    def _gen(self):
        with self.lock:
            key = os.path.getmtime(CUR)
            if self.gen is None or self.gen_key != key:
                from hoh.inference import HohGenerator
                self.gen = HohGenerator(CUR, "cpu")
                self.gen_key = key
            return self.gen

    # ------------------------------------------------------------------ Tumbuh (belajar mandiri di latar belakang)
    def touch(self):
        try:
            with open(os.path.join(HOME, "activity"), "w") as f:
                f.write(str(time.time()))
        except OSError:
            pass

    def _idle_for(self):
        try:
            return time.time() - os.path.getmtime(os.path.join(HOME, "activity"))
        except OSError:
            return 1e9

    def grower(self, thread=False):
        if getattr(self, "grow", None) is None:
            from hoh import grow
            hooks = dict(search=self._g_search, fetch=self._g_fetch, train=self._g_train, notify=notify, propose=self._g_propose, wake=wakelock,
                         idle_for=self._idle_for, busy=lambda: bool(self.st["running"]))
            self.grow = grow.Grower(grow.Store(os.path.join(HOME, "grow.json")), hooks)
        if thread and not getattr(self, "_grow_thread", None):
            self._grow_thread = threading.Thread(target=self.grow.run_forever, daemon=True)
            self._grow_thread.start()
        return self.grow

    def _g_propose(self):
        """Model memilih sendiri topik yang ingin dipelajari (sampling dari bobotnya, bukan daftar tetap)."""
        import numpy as np
        from hoh.sft import messages_to_prompt_ids
        from hoh.tokenizer import EOS
        g = self._gen()
        ids = messages_to_prompt_ids([{"role": "user", "content": "Apa yang ingin kamu pelajari sekarang?"}], g.tokenizer)
        out = g.model.generate(np.asarray([ids], dtype=np.int64), 40, 0.9, 30, EOS, 1.0, 1.0, True, rng=np.random.default_rng())
        txt = g.tokenizer.decode(out[0].tolist()[len(ids):])
        return re.sub(r"<[^>]*>", "", txt).strip().split("\n")[0][:80]

    def _g_search(self, q, n):
        from hoh import ingest
        return ingest.search_urls(q, n)

    def _g_fetch(self, url):
        from hoh import ingest
        from hoh.curiosity import Robots
        if not hasattr(self, "_robots"):
            self._robots, self._hit = Robots(lambda u: ingest.fetch_url(u, timeout=8, max_bytes=200000)), {}
        if not self._robots.allowed(url):
            return None
        host = urllib.parse.urlparse(url).netloc
        wait = 2.0 - (time.time() - self._hit.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self._hit[host] = time.time()
        r = self.fetch_link(url)
        return r.get("text")

    def _anchors(self):
        import random
        try:
            with open(SEED, encoding="utf-8") as f:
                rows = [json.loads(l) for l in f if l.strip()]
        except OSError:
            return []
        random.Random(11).shuffle(rows)
        return rows[:30]

    def anchor_nll(self, path):
        """Rata-rata NLL per byte jawaban pada soal jangkar (rendah = baik). Dipakai sebagai pengaman 'lupa'."""
        import numpy as np
        from hoh.inference import HohGenerator
        from hoh.sft import messages_to_prompt_ids
        g = HohGenerator(path, "cpu")
        tok, tot, n = g.tokenizer, 0.0, 0
        for r in self._anchors():
            pi = messages_to_prompt_ids([{"role": "user", "content": r["prompt"]}], tok)
            ans = tok.encode(r["response"])
            ids = (pi + ans)[-g.model.config.block_size:]
            k = min(len(ans), len(ids) - 1)
            logits, _ = g.model.forward(np.array([ids[:-1]], dtype=np.int64))
            lg = logits[0, -k:].astype(np.float64)
            lg -= lg.max(axis=1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
            tgt = np.array(ids[-k:])
            tot += float(-lp[np.arange(k), tgt].sum()); n += k
        return tot / max(1, n)

    def _g_train(self, texts, likes, seconds):
        qa = [{"q": q, "a": a} for q, a in likes]
        r = self.train_start({"texts": texts, "qa": qa, "preset": "quick", "advanced": {"seconds": seconds}, "auto": True})
        if r.get("error"):
            return {"ok": False, "detail": r["error"]}
        while self.st["running"]:
            time.sleep(1.0)
        ok = self.st.get("phase") == "done"
        return {"ok": ok, "detail": self.st.get("error") or ("lulus uji jangkar" if ok else self.st.get("phase", ""))}

    def chat(self, messages, style="seimbang", instructions="", taught=()):
        """Hasilkan potongan teks demi potongan (benar-benar streaming)."""
        self.touch()
        if self.st["running"] and self.st.get("auto"):          # ada yang mengobrol: latihan mandiri mengalah
            self.cancel = True
            for _ in range(40):
                if not self.st["running"]:
                    break
                time.sleep(0.1)
        last = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        if last and taught and not self.st["running"]:        # ajaran pengguna dijawab seketika, tanpa menunggu pelatihan
            q = str(last.get("content", "")).split("\n\n")[-1]
            hit = apptools.match_taught(q, taught)
            if hit:
                for i in range(0, len(hit), 24):
                    yield {"token": hit[i:i + 24]}
                return
        if not self.ok:
            yield {"err": "no_engine"}
            return
        if self.st["running"]:
            yield {"err": "training"}
            return
        if not self.ready():
            yield {"err": "no_model"}
            return
        import numpy as np
        from hoh.reasoning import split_thinking
        from hoh.sft import messages_to_prompt_ids
        from hoh.stopping import is_looping
        from hoh.tokenizer import EOS
        g = self._gen()
        # model ini dilatih hanya untuk satu giliran (tanya -> jawab); riwayat dan instruksi sistem membuatnya bingung, jadi hanya pesan terakhir yang dipakai
        msgs = [{"role": "user", "content": str(last["content"]).split("\n\n")[-1].strip()}] if last else []
        instructions = ""
        if instructions.strip():
            msgs.insert(0, {"role": "system", "content": instructions.strip()[:400]})
        ctx = g.model.config.block_size
        while True:   # potong riwayat dari depan sampai muat konteks (menyisakan ruang untuk jawaban)
            ids = messages_to_prompt_ids(msgs, g.tokenizer)
            if len(ids) <= ctx - 48 or len(msgs) <= 1:
                break
            msgs = msgs[1:] if msgs[0]["role"] != "system" or len(msgs) == 2 else [msgs[0]] + msgs[2:]
        if len(ids) > ctx - 16:                          # satu pesan saja sudah terlalu panjang: ambil ujung belakangnya
            ids = ids[:1] + ids[-(ctx - 16):]
        limit = {"ringkas": 120, "seimbang": 320, "rinci": 700}.get(style, 320)
        rng = np.random.default_rng()
        all_ids, start, t0, sent = list(ids), len(ids), time.time(), ""
        g.touch()
        while len(all_ids) - start < limit and time.time() - t0 < 180:
            idx = np.asarray([all_ids[-ctx:]], dtype=np.int64)
            out = g.model.generate(idx, 6 if len(all_ids) == start else 16, 0.2, 10, EOS, 1.0, 1.0, True, rng=rng)   # suhu rendah: model kecil jauh lebih rapi
            new = out[0].tolist()[idx.shape[1]:]
            if not new:
                break
            ended = new[-1] == EOS
            all_ids.extend(new)
            text = g.tokenizer.decode(all_ids[start:])
            cut = text.find("\nUser:")
            if cut >= 0:
                text, ended = text[:cut], True
            if not (text.endswith("\ufffd") and not ended) and len(text) > len(sent):
                yield {"token": text[len(sent):]}          # tag <think>/<cek>/<cari>... ikut terkirim apa adanya: UI membaca fase dari tag
                sent = text
            if ended or is_looping(all_ids[start:]):
                break
        text = g.tokenizer.decode(all_ids[start:])
        cut = text.find("\nUser:")
        text = text[:cut] if cut >= 0 else text
        if len(text) > len(sent):
            yield {"token": text[len(sent):]}
        elif not text.strip():
            yield {"err": "empty"}
        for m in re.finditer(r"<(cari|riset|ingat)>(.*?)</\1>", text, re.S):     # alat yang DIPILIH model; runtime hanya menjalankannya
            yield {"tool": m.group(1), "obs": self.TOOLS[m.group(1)](self, m.group(2).strip())}

    # ------------------------------------------------------------------ alat yang dipanggil model lewat tag
    def _tool_cari(self, q):
        from hoh import ingest
        from hoh.consolidate import extractive_summary
        try:
            urls = ingest.search_urls(q, 4)
        except Exception:
            return "Tidak ada koneksi internet."
        texts, hosts = [], []
        for u in urls[:3]:
            t = self._g_fetch(u)
            if t:
                texts.append(t[:4000]); hosts.append(urllib.parse.urlparse(u).netloc)
        return ("%s\n(Sumber: %s)" % (extractive_summary(texts, max_chars=500), ", ".join(hosts))) if texts else "Tidak menemukan bahan yang layak."

    def _tool_riset(self, topic):
        self.grower().queue_research(topic)
        return ""

    def _tool_ingat(self, topic):
        r = self.grower().notes_about(topic)
        return ("%s\n(Sumber: %s)" % (r["text"][:500], ", ".join(r["sources"][:3]))) if r else "Aku belum punya catatan tentang itu."

    TOOLS = {"cari": _tool_cari, "riset": _tool_riset, "ingat": _tool_ingat}

    # ------------------------------------------------------------------ files
    def extract(self, data, name):
        """Berkas APA SAJA -> teks untuk dianalisis. Dokumen/gambar/audio dibaca engine; sisanya dideskripsikan (tanpa dijalankan)."""
        try:
            from hoh import ingest
            docs = ingest.parse_bytes(data, name or "")
            text = "\n\n".join(d.text for d in docs if getattr(d, "text", "")).strip()
            if text:
                return {"text": text[:300000], "chars": len(text), "ocr": any("ocr" in (getattr(d, "notes", []) or []) for d in docs)}
        except Exception:
            pass
        return apptools.describe_bytes(data, name or "")

    def fetch_link(self, url):
        from hoh import ingest
        try:
            data, ctype, final = ingest.fetch_url(url, timeout=15.0, max_bytes=5_000_000)
            docs = ingest.parse_bytes(data, final, ctype)
        except ingest.IngestError as e:
            return {"error": "unreadable", "detail": str(e)[:300]}
        text = "\n\n".join(d.text for d in docs if getattr(d, "text", "")).strip()
        return {"text": text[:300000]} if text else {"error": "empty"}

    # ------------------------------------------------------------------ training
    def train_start(self, c):
        if not self.ok:
            return {"error": "no_engine"}
        if not self.ready():
            return {"error": "no_model"}
        with self.lock:
            if self.st["running"]:
                return {"error": "already_running"}
            qa = [(str(x.get("q", "")).strip(), str(x.get("a", "")).strip()) for x in c.get("qa", [])]
            qa = [(q, a) for q, a in qa if q and a]
            texts = [str(t).strip() for t in c.get("texts", []) if str(t).strip()]
            links = [str(u) for u in c.get("links", []) if str(u).startswith(("http://", "https://"))][:10]
            if not qa and not texts and not links:
                return {"error": "no_data"}
            preset = c.get("preset", "balanced")
            adv = c.get("advanced") or {}
            auto = bool(c.get("auto"))
            if auto and not qa:                                   # latihan mandiri selalu disertai ulangan materi jangkar
                qa = [(str(r["prompt"]), str(r["response"])) for r in self._anchors()[:24]]
            jid = int(time.time())
            self.st = dict(self._idle(), running=True, phase="prepare", log=["…"], id=jid, auto=auto)
            self.cancel = False
        threading.Thread(target=self._job, args=(jid, qa, texts, links, preset, adv, auto), daemon=True).start()
        return {"ok": True}

    def _log(self, line):
        self.st["log"] = (self.st["log"] + [line])[-200:]

    def _job(self, jid, qa, texts, links, preset, adv, auto=False):
        job = os.path.join(JOBS, str(jid))
        try:
            os.makedirs(job, exist_ok=True)
            for u in links:
                if self.cancel:
                    raise InterruptedError
                r = self.fetch_link(u)
                if r.get("text"):
                    texts.append(r["text"])
                    self._log("OK link: " + u[:70])
                else:
                    self._log("SKIP link: " + u[:70])
            phases = []
            tx = "\n\n".join(texts)
            if len(tx.encode("utf-8")) >= 32:
                p = os.path.join(job, "train.txt")
                base = tx
                while len(tx.encode("utf-8")) < 640:      # teks pendek diulang supaya cukup untuk dilatih
                    tx += "\n\n" + base
                with open(p, "w", encoding="utf-8") as f:
                    f.write(tx)
                phases.append(("pretrain", p, len(tx.encode("utf-8")) + 2))
            if qa:
                rows = [{"prompt": q, "response": a} for q, a in qa] * 4     # materi baru diulang agar benar-benar melekat
                if os.path.exists(SEED):            # ulangi materi dasar supaya kemampuan lama tidak hilang
                    with open(SEED, encoding="utf-8") as f:
                        seed = [json.loads(l) for l in f if l.strip()]
                    import random
                    random.Random(5).shuffle(seed)
                    rows += seed[:{"quick": 72, "balanced": 100, "deep": 140}.get(preset, 48) + len(rows)]
                while len(rows) < 4:
                    rows += rows
                p = os.path.join(job, "train.jsonl")
                with open(p, "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                phases.append(("sft", p, len(rows)))
            if not phases:
                raise ValueError("no_data")
            from hoh.checkpoint import load_checkpoint
            block = int(load_checkpoint(CUR)["model_config"]["block_size"])
            epochs = {"quick": (1, 12), "balanced": (2, 30), "deep": (4, 80)}.get(preset, (2, 30))
            min_sft = {"quick": 40, "balanced": 70, "deep": 160}.get(preset, 60)
            budget = {"quick": 25, "balanced": 45, "deep": 90}.get(preset, 45)      # detik: batas keras total, berlaku juga di HP lambat
            src, total_steps, plans = CUR, 0, []
            for kind, path, size in phases:
                batch = int(adv.get("batch") or 4)
                if kind == "sft":
                    steps = max(min_sft, min(1200, min_sft + 3 * len(qa)))
                else:
                    steps = max(12, min(500, int(epochs[0] * size / (batch * block) + 0.999)))
                if adv.get("steps"):
                    steps = max(1, min(20000, int(adv["steps"])))
                plans.append((kind, path, steps, batch))
                total_steps += steps
            if adv.get("seconds"):
                budget = max(5, min(1800, int(adv["seconds"])))
            self.st.update(total=total_steps, step=0, phase="train", log=self.st["log"] + ["Mulai: %d langkah" % total_steps])
            done = 0
            for n, (kind, path, steps, batch) in enumerate(plans):
                out = os.path.join(job, "p%d" % n)
                lr = float(adv.get("lr") or (3e-4 if kind == "pretrain" else 6e-4))
                args = ["--data", path, "--out-dir", out, "--steps", str(steps), "--batch-size", str(batch), "--grad-accum", "1",
                        "--lr", str(lr), "--min-lr", str(lr / 10), "--warmup-steps", str(max(1, steps // 10)),
                        "--eval-interval", str(steps), "--eval-batches", "3", "--save-interval", str(steps),
                        "--mode", kind, "--device", "cpu", "--init-from", src]
                if not self._run(args, done, steps, budget * steps / max(1, total_steps)):
                    raise InterruptedError if self.cancel else RuntimeError("train_failed")
                last = os.path.join(out, "last.npz")
                if not os.path.exists(last):
                    raise RuntimeError("train_failed")
                src, done = last, done + steps
                self.st["step"] = done
            with open(src, "rb") as f:
                payload = f.read()
            if auto:                                              # pengaman: tolak jika lebih buruk pada soal jangkar
                new, old = self.anchor_nll(src), self.anchor_nll(CUR)
                self._log("jangkar: baru %.3f, lama %.3f" % (new, old))
                if new > old * 1.15 + 0.02:
                    raise RuntimeError("ditolak: model baru lebih buruk pada soal jangkar (%.3f > %.3f)" % (new, old))
                self._install(payload, "tumbuh", "Tumbuh " + time.strftime("%d/%m %H:%M"))
            else:
                self._install(payload)
            self.st.update(step=total_steps, done=True, phase="done", eta=0)
            if not auto:
                notify("HOH", "Pelatihan selesai")
            self._log("OK")
        except InterruptedError:
            self.st.update(phase="cancelled")
        except Exception as e:
            msg = self.st.get("error") or str(e)
            self.st.update(error=msg[:300], phase="error")
            self._log("ERROR: " + msg[:300])
        finally:
            self.st["running"] = False
            self.proc = None
            shutil.rmtree(job, ignore_errors=True)

    def _run(self, args, base, steps, seconds=0):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=ROOT, HOH_MAX_SECONDS=str(seconds or ""))
        t0 = time.time()
        self.proc = subprocess.Popen([sys.executable, "-m", "hoh", "train"] + args, cwd=ROOT, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = []
        for line in self.proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-6:]
            self._log(line)
            m = re.search(r"step\s+(\d+) \| train_loss ([\d.]+)", line)
            if m:
                s = int(m.group(1)) + 1
                self.st["step"] = base + min(s, steps)
                self.st["loss"].append(float(m.group(2)))
                el = time.time() - t0
                if s > 3:
                    self.st["eta"] = int(el / s * max(0, steps - s))
            if self.cancel:
                self.proc.terminate()
        self.proc.wait()
        if self.proc.returncode != 0 and not self.cancel:
            self.st["error"] = " ".join(tail[-2:])[:300]
        return self.proc.returncode == 0 and not self.cancel

    def train_stop(self):
        self.cancel = True
        p = self.proc
        if p and self.st["running"]:
            try:
                p.terminate()
            except Exception:
                pass


E = Engine()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") else "[::1]"
        if host not in ("127.0.0.1", "localhost", "[::1]"):
            return False
        o = self.headers.get("Origin")
        if o and o != "null":
            oh = urllib.parse.urlparse(o).hostname or ""
            if oh not in ("127.0.0.1", "localhost", "::1"):
                return False
        return True

    def _json(self, o, code=200):
        b = json.dumps(o).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _body(self, limit=MAX_UPLOAD):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            self.close_connection = True
            return None
        return self.rfile.read(n) if n else b""

    def do_GET(self):
        if not self._host_ok():
            return self._json({"error": "forbidden"}, 403)
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p == "/api/status":
            return self._json(E.status())
        if p == "/api/model/info":
            try:
                return self._json(E.info())
            except Exception as e:
                return self._json({"ready": False, "error": str(e)[:200]})
        if p == "/api/models":
            return self._json(E.models())
        if p == "/api/voice":
            return self._json(E.voice_info())
        if p == "/api/grow":
            return self._json(E.grower().summary())
        if p == "/api/train/status":
            return self._json(E.st)
        if p == "/api/model/export":
            q = urllib.parse.parse_qs(u.query)
            try:
                data = E.export_bytes(with_opt=q.get("opt", ["0"])[0] == "1", name=(q.get("name", ["HOH"])[0] or "HOH")[:24], sid=q.get("id", [""])[0])
            except Exception as e:
                return self._json({"error": getattr(e, "code", "failed"), "detail": str(e)[:200]}, 400)
            fn = "hoh-%s.hh" % time.strftime("%Y%m%d-%H%M")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % fn)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if p in STATIC:
            fp = os.path.join(ROOT, STATIC[p])
            try:
                with open(fp, "rb") as f:
                    b = f.read()
            except OSError:
                return self._json({"error": "not found"}, 404)
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(os.path.splitext(fp)[1], "application/octet-stream"))
            if fp.endswith(".html") and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                import gzip
                b = gzip.compress(b, 6)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(b)
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._host_ok():
            return self._json({"error": "forbidden"}, 403)
        p = urllib.parse.urlparse(self.path).path
        if p == "/api/model/import":
            data = self._body()
            if data is None:
                return self._json({"error": "too_big"}, 413)
            try:
                return self._json({"ok": True, "info": E.import_bytes(data)})
            except Exception as e:
                code = getattr(e, "code", "failed")
                return self._json({"error": code, "detail": str(e)[:200]}, 400)
        if p == "/api/extract":
            data = self._body(60 * 1024 * 1024)
            if data is None:
                return self._json({"error": "too_big"}, 413)
            name = urllib.parse.unquote(self.headers.get("X-Filename") or "")
            return self._json(E.extract(data, name))
        if p == "/api/voice/train":
            data = self._body(40 * 1024 * 1024)
            if data is None:
                return self._json({"error": "too_big"}, 413)
            try:
                return self._json({"ok": True, "info": E.voice_train(data)})
            except ValueError as e:
                return self._json({"error": str(e)[:60]}, 400)
            except Exception as e:
                return self._json({"error": getattr(e, "code", "failed"), "detail": str(e)[:200]}, 400)
        raw = self._body(8 * 1024 * 1024)
        if raw is None:
            return self._json({"error": "too_big"}, 413)
        try:
            b = json.loads(raw or b"{}")
        except Exception:
            return self._json({"error": "bad_json"}, 400)
        if p == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for ev in E.chat(b.get("messages", []), b.get("style", "seimbang"), b.get("instructions", ""), b.get("taught", [])):
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self.wfile.write((json.dumps({"err": "failed", "detail": str(e)[:200]}) + "\n").encode())
                except Exception:
                    pass
            return
        if p == "/api/train/start":
            return self._json(E.train_start(b))
        if p == "/api/voice/speak":
            try:
                wav = E.voice_speak(str(b.get("text", "")))
            except Exception as e:
                return self._json({"error": getattr(e, "code", "failed")}, 409 if getattr(e, "code", "") == "no_voice" else 400)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(wav)
            return
        if p == "/api/voice/reset":
            E.voice_reset()
            return self._json({"ok": True})
        if p == "/api/grow/cfg":
            E.grower().set_cfg(**{k: v for k, v in b.items()})
            return self._json(E.grower().summary())
        if p == "/api/grow/research":
            return self._json({"ok": bool(E.grower().queue_research(str(b.get("topic", ""))))})
        if p == "/api/grow/goal":
            return self._json({"ok": bool(E.grower().add_goal(str(b.get("topic", "")), 1.5, "user"))})
        if p == "/api/grow/read":
            E.grower().store.update(lambda st: [r.__setitem__("read", True) for r in st["inbox"] if r["id"] == str(b.get("id", ""))])
            return self._json({"ok": True})
        if p == "/api/feedback":
            return self._json({"ok": E.grower().feedback(str(b.get("q", "")), str(b.get("a", "")), bool(b.get("good")))})
        if p == "/api/grow/battery":
            return self._json({"ok": ignore_battery_optimization()})
        if p == "/api/open":
            return self._json({"ok": open_external(str(b.get("url", "")))})
        if p == "/api/notify":
            return self._json({"ok": notify(str(b.get("title", "HOH"))[:60], str(b.get("body", ""))[:160])})
        if p == "/api/train/stop":
            E.train_stop()
            return self._json({"ok": True})
        if p in ("/api/model/use", "/api/model/delete"):
            try:
                if p.endswith("use"):
                    E.use(str(b.get("id", "")))
                    return self._json({"ok": True})
                return self._json({"ok": E.delete(str(b.get("id", "")))})
            except Exception as e:
                return self._json({"error": getattr(e, "code", "failed"), "detail": str(e)[:200]}, 400)
        if p == "/api/model/revert":
            return self._json({"ok": E.revert()})
        if p == "/api/model/reset":
            return self._json({"ok": E.reset()})
        if p == "/api/model/save":
            try:
                data = E.export_bytes(with_opt=bool(b.get("opt")), name=str(b.get("name") or "HOH")[:24], sid=str(b.get("id") or ""))
                fp = os.path.join(export_dir(), "hoh-%s.hh" % time.strftime("%Y%m%d-%H%M"))
                atomic_write(fp, data)
                return self._json({"ok": True, "path": fp, "bytes": len(data)})
            except Exception as e:
                return self._json({"error": getattr(e, "code", "failed"), "detail": str(e)[:200]}, 400)
        if p == "/api/search":
            return self._json(apptools.search(str(b.get("q", ""))))
        if p == "/api/page":
            return self._json(apptools.page(str(b.get("url", ""))))
        self._json({"error": "not found"}, 404)


def serve(port, block=True):
    ThreadingHTTPServer.daemon_threads = True
    if E.ok and not os.environ.get("ANDROID_ARGUMENT") and not os.environ.get("HOH_NO_GROW"):
        E.grower(thread=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    if block:
        srv.serve_forever()
    else:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print("HOH di http://localhost:%d" % port)
    serve(port)
