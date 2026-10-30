"""Format berkas bobot model `.hh` — hanya model buatan aplikasi HOH yang bisa diimpor.

Susunan berkas:
    8 byte   magic  b"HOHWT\\x01"
    4 byte   panjang header (uint32 big-endian)
    N byte   header JSON (UTF-8): format, versi, parameter, konfigurasi, sha256 isi, tanda tangan HMAC
    sisanya  isi = arsip .npz (tanpa pickle) berisi bobot `model/*` + `meta`

Pemeriksaan saat impor (semuanya wajib lolos):
  1. magic + header valid, ukuran masuk akal
  2. sha256 isi cocok dengan header (berkas tidak rusak/terpotong)
  3. tanda tangan HMAC cocok (berkas dibuat oleh HOH, bukan berkas acak/model lain)
  4. isi dibaca dengan allow_pickle=False (tidak ada kode yang bisa dijalankan)
  5. nama dan bentuk setiap tensor persis sama dengan yang diharapkan konfigurasi model
  6. semua nilai berhingga (tidak ada NaN/Inf)

Catatan jujur: kunci tanda tangan ada di kode sumber, jadi ini penyaring untuk berkas yang bukan dari HOH dan
perlindungan dari berkas rusak — bukan kunci kriptografi terhadap orang yang membaca kode dan sengaja memalsukan.
"""
import hashlib, hmac, io, json, struct, time, zipfile
import numpy as np

MAGIC = b"HOH-HH\x00\x01"
FORMAT = "hoh-hh"
VERSION = 1
_KEY = b"hoh/hh/v1/3c8a51e0d94b7f26"
MAX_BYTES = 800 * 1024 * 1024


class HhError(ValueError):
    """Pesan galat memakai kode singkat di `.code` supaya UI bisa menerjemahkannya."""

    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


def _sig(sha, header_core):
    return hmac.new(_KEY, (sha + "|" + header_core).encode("utf-8"), hashlib.sha256).hexdigest()


def _core(h):
    return json.dumps({k: h[k] for k in sorted(h) if k != "sig"}, sort_keys=True, separators=(",", ":"))


def pack(npz_path, include_optimizer=False, name="HOH"):
    """Checkpoint .npz HOH -> bytes berkas .hh."""
    from .checkpoint import load_checkpoint
    from .config import ModelConfig
    ck = load_checkpoint(npz_path)
    cfg = ModelConfig(**ck["model_config"]).validate()
    arrays = {f"model/{k}": np.ascontiguousarray(v, dtype=np.float32) for k, v in ck["model"].items()}
    meta = {"step": int(ck["step"]), "best_val": float(ck["best_val"]) if np.isfinite(ck["best_val"]) else 1e9,
            "model_config": dict(vars(cfg)), "train_config": ck.get("train_config", {}),
            "format_version": ck.get("format_version"), "tokenizer": ck.get("tokenizer"),
            "opt_names": ck.get("opt_names") if include_optimizer else None, "opt_steps": {}, "opt_groups": [], "rng": None}
    if include_optimizer and ck["optimizer"]["state"]:
        for i, st in ck["optimizer"]["state"].items():
            arrays[f"opt/{i}/exp_avg"], arrays[f"opt/{i}/exp_avg_sq"] = st["exp_avg"], st["exp_avg_sq"]
            meta["opt_steps"][str(i)] = int(st["step"])
        meta["opt_groups"] = json.loads(json.dumps(ck["optimizer"]["param_groups"], default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    arrays["meta"] = np.frombuffer(json.dumps(meta, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode("utf-8"), dtype=np.uint8)
    has_voice = False
    try:                                              # suara ikut tersimpan di dalam model (satu berkas)
        from .voice import valid_profile
        with np.load(npz_path, allow_pickle=False) as raw:
            if "voice/profile" in raw.files and valid_profile(raw["voice/profile"]):
                arrays["voice/profile"] = np.ascontiguousarray(raw["voice/profile"], dtype=np.float32)
                has_voice = True
    except Exception:
        pass
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()
    n_params = int(sum(int(np.prod(v.shape)) for k, v in ck["model"].items()))
    h = {"format": FORMAT, "version": VERSION, "name": name, "created": int(time.time()), "params": n_params,
         "step": int(ck["step"]), "layers": cfg.n_layer, "width": cfg.n_embd, "heads": cfg.n_head, "block": cfg.block_size,
         "vocab": cfg.vocab_size, "optimizer": bool(include_optimizer and meta["opt_steps"]), "voice": has_voice, "sha256": sha,
         "bytes": len(payload)}
    h["sig"] = _sig(sha, _core(h))
    hb = json.dumps(h, sort_keys=True).encode("utf-8")
    return MAGIC + struct.pack(">I", len(hb)) + hb + payload


def read_header(data):
    if len(data) < 16 or data[:8] != MAGIC:
        raise HhError("not_hh", "bukan berkas .hh dari HOH")
    (n,) = struct.unpack(">I", data[8:12])
    if n <= 0 or n > 65536 or 12 + n > len(data):
        raise HhError("corrupt", "header rusak")
    try:
        h = json.loads(data[12:12 + n].decode("utf-8"))
    except Exception:
        raise HhError("corrupt", "header tidak terbaca")
    if not isinstance(h, dict) or h.get("format") != FORMAT:
        raise HhError("not_hh", "format tidak dikenal")
    if h.get("version") != VERSION:
        raise HhError("version", f"versi {h.get('version')} tidak didukung")
    return h, data[12 + n:]


def verify_and_extract(data):
    """bytes .hh -> (header, bytes .npz checkpoint) setelah semua pemeriksaan lolos. Memunculkan HhError."""
    if len(data) > MAX_BYTES:
        raise HhError("too_big", "berkas terlalu besar")
    h, payload = read_header(data)
    sha = hashlib.sha256(payload).hexdigest()
    if sha != h.get("sha256") or len(payload) != h.get("bytes"):
        raise HhError("corrupt", "isi berkas rusak atau terpotong")
    if not hmac.compare_digest(str(h.get("sig", "")), _sig(sha, _core(h))):
        raise HhError("not_hoh", "bukan model dari aplikasi HOH")
    check_payload(payload)
    return h, payload


def check_payload(payload):
    """Pastikan isi .npz hanya berisi tensor yang diharapkan dan berhingga."""
    from .config import ModelConfig
    from .model import param_shapes
    try:
        zf = zipfile.ZipFile(io.BytesIO(payload))
        infos = {i.filename: i for i in zf.infolist()}
    except Exception:
        raise HhError("corrupt", "arsip isi tidak valid")
    if "meta.npy" not in infos:
        raise HhError("not_hoh", "metadata tidak ada")
    if infos["meta.npy"].file_size > 8_000_000:
        raise HhError("corrupt", "metadata terlalu besar")
    try:
        z = np.load(io.BytesIO(payload), allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        cfg = ModelConfig(**meta["model_config"]).validate()
    except HhError:
        raise
    except Exception as e:
        raise HhError("not_hoh", f"konfigurasi tidak valid ({type(e).__name__})")
    shapes = param_shapes(cfg)
    expect = {f"model/{k}": tuple(v) for k, v in shapes.items()}
    got = {k for k in z.files if k.startswith("model/")}
    if got != set(expect):
        raise HhError("mismatch", "daftar bobot tidak sesuai arsitektur")
    budget = sum(int(np.prod(s)) for s in expect.values()) * 4 * 6 + 20_000_000   # bobot + sisa moment optimizer
    if sum(i.file_size for i in infos.values()) > budget:
        raise HhError("corrupt", "ukuran isi tidak wajar")
    for k, shp in expect.items():
        a = z[k]
        if tuple(a.shape) != shp or a.dtype.kind != "f":
            raise HhError("mismatch", f"bentuk {k} tidak sesuai")
        if not np.isfinite(a).all():
            raise HhError("corrupt", f"{k} berisi NaN/Inf")
    if "voice/profile" in z.files:
        from .voice import valid_profile
        if not valid_profile(z["voice/profile"]):
            raise HhError("mismatch", "profil suara tidak valid")
    tok = meta.get("tokenizer")
    if tok is not None and int(tok.get("vocab_size", cfg.vocab_size)) != cfg.vocab_size:
        raise HhError("mismatch", "tokenizer tidak cocok dengan model")


def info_of(data_or_header):
    h = data_or_header if isinstance(data_or_header, dict) else read_header(data_or_header)[0]
    return {k: h.get(k) for k in ("name", "created", "params", "step", "layers", "width", "heads", "block", "vocab", "optimizer", "voice", "bytes")}
