"""Suara HOH: pembaca bahasa Indonesia berbasis NumPy (tanpa model terpisah, tanpa unduhan).

Cara kerja:
  teks -> fonem (aturan ejaan Indonesia) -> lintasan parameter (pitch, formant, noise) -> sintesis formant (STFT) -> WAV 16 kHz.
Suara "dilatih" dari rekaman penggunanya: `estimate_profile` mengukur pitch, panjang saluran suara (skala formant), kemiringan
spektrum, warna suara (EQ 16 pita), napas, dan kecepatan bicara. Hasilnya vektor 32 angka (`PROFILE_LEN`) yang disimpan DI DALAM
checkpoint model (`voice/profile`), jadi suara ikut berpindah bersama berkas .hh.

Ini sintesis formant: jelas bukan kloning suara dan terdengar seperti robot. Tujuannya ringan, cepat, dan berjalan di HP lambat."""
import io, math, re, struct
import numpy as np

SR = 16000
PROFILE_LEN = 32
HOP = 160                      # 10 ms
WIN = 320                      # 20 ms
NFFT = 512

# ----------------------------------------------------------------------------------------------- profil
DEFAULT = dict(f0=120.0, f0_sd=2.5, scale=1.0, tilt=0.0, rate=1.0, breath=0.05)


def profile_to_vec(f0, f0_sd, scale, tilt, rate, breath, eq, seconds, voiced):
    v = np.zeros(PROFILE_LEN, np.float32)
    v[0] = 1.0
    v[1:8] = [f0, f0_sd, scale, tilt, rate, breath, 0.0]
    v[8:24] = np.asarray(eq, np.float32)[:16]
    v[24], v[25] = seconds, voiced
    return v


def vec_to_params(v):
    v = np.asarray(v, np.float32)
    return dict(f0=float(v[1]), f0_sd=float(v[2]), scale=float(v[3]), tilt=float(v[4]), rate=float(v[5]),
                breath=float(v[6]), eq=v[8:24].astype(np.float32), seconds=float(v[24]), voiced=float(v[25]))


def valid_profile(v):
    v = np.asarray(v)
    if v.shape != (PROFILE_LEN,) or not np.isfinite(v).all() or v[0] != 1.0:
        return False
    return 60 <= v[1] <= 450 and 0.5 <= v[3] <= 1.6 and 0.4 <= v[5] <= 2.0 and abs(v[4]) <= 12 and v[8:24].min() >= -12 and v[8:24].max() <= 12


# ----------------------------------------------------------------------------------------------- teks -> fonem
_ONES = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]


def _num_words(n):
    if n < 10:
        return _ONES[n]
    if n == 10:
        return "sepuluh"
    if n == 11:
        return "sebelas"
    if n < 20:
        return _ONES[n - 10] + " belas"
    if n < 100:
        return _ONES[n // 10] + " puluh" + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 200:
        return "seratus" + (" " + _num_words(n - 100) if n > 100 else "")
    if n < 1000:
        return _ONES[n // 100] + " ratus" + (" " + _num_words(n % 100) if n % 100 else "")
    if n < 2000:
        return "seribu" + (" " + _num_words(n - 1000) if n > 1000 else "")
    if n < 1_000_000:
        return _num_words(n // 1000) + " ribu" + (" " + _num_words(n % 1000) if n % 1000 else "")
    if n < 1_000_000_000:
        return _num_words(n // 1_000_000) + " juta" + (" " + _num_words(n % 1_000_000) if n % 1_000_000 else "")
    return " ".join(_ONES[int(c)] for c in str(n))


def normalize(text):
    t = text.lower()
    t = re.sub(r"https?://\S+", " tautan ", t)
    t = re.sub(r"(\d+)[.,](\d+)", lambda m: m.group(1) + " koma " + " ".join(_ONES[int(c)] for c in m.group(2)), t)
    t = re.sub(r"\d+", lambda m: " " + (_num_words(int(m.group())) if len(m.group()) <= 9 else " ".join(_ONES[int(c)] for c in m.group())) + " ", t)
    t = t.replace("%", " persen ").replace("&", " dan ").replace("+", " tambah ").replace("=", " sama dengan ")
    t = t.replace("\n", ". ")
    t = re.sub(r"[^a-z\s.,?!;:\-]", " ", t)
    return t


VOWELS = "aiueo"


PREFIX = ("be", "me", "pe", "te", "ke", "se", "de")


def _schwa(word, i):
    """'e' taling (e) atau pepet (@): aturan perkiraan (ejaan Indonesia tidak membedakan keduanya)."""
    if i == len(word) - 1 or not any(c in VOWELS for c in word[i + 1:]):
        return "e"
    if i == 1 and word[:2] in PREFIX and len(word) > 3:
        return "@"
    after = word[i + 1:i + 3]
    if len(after) == 2 and after[0] not in VOWELS and after[1] not in VOWELS:
        return "@"                                  # suku kata tertutup: enam, sempat, tempat
    return "e"


def word_to_phones(word):
    out, i, w = [], 0, word
    while i < len(w):
        c = w[i]
        two = w[i:i + 2]
        if two == "ng":
            out.append("N"); i += 2
            if i < len(w) and w[i] == "g":
                i += 0                                  # 'ngg' -> ŋg
        elif two == "ny":
            out.append("Y"); i += 2
        elif two == "sy":
            out.append("S"); i += 2
        elif two == "kh":
            out.append("x"); i += 2
        elif c == "e":
            out.append(_schwa(w, i)); i += 1
        elif c in VOWELS:
            out.append(c); i += 1
        elif c == "c":
            out.append("C"); i += 1
        elif c == "j":
            out.append("J"); i += 1
        elif c == "y":
            out.append("y"); i += 1
        elif c == "q":
            out.append("k"); i += 1
        elif c == "v":
            out.append("f"); i += 1
        elif c == "x":
            out += ["k", "s"]; i += 1
        elif c.isalpha():
            out.append(c); i += 1
        else:
            i += 1
    return out


def text_to_units(text):
    """-> daftar unit: ('p', fonem, ditekan) | ('pause', detik) | ('end', 'q'/'.')"""
    units = []
    for tok in re.findall(r"[a-z]+|[.,?!;:\-]", normalize(text)):
        if tok.isalpha():
            ph = word_to_phones(tok)
            vidx = [k for k, p in enumerate(ph) if p in "aiueo@"]
            stress = vidx[-2] if len(vidx) >= 2 else (vidx[-1] if vidx else -1)
            if len(vidx) == 1:
                stress = vidx[0]
            for k, p in enumerate(ph):
                units.append(("p", p, k == stress))
            units.append(("pause", 0.045))
        elif tok in ",;:-":
            units.append(("pause", 0.20))
        else:
            units.append(("end", "q" if tok == "?" else "."))
            units.append(("pause", 0.34))
    if not units or units[-1][0] != "pause" or units[-1][1] < 0.3:
        units.append(("end", "."))
    return units


# ----------------------------------------------------------------------------------------------- tabel fonem
# (jenis, F1, F2, F3, durasi detik, gv, gn, noise (lo, hi))
V = {"a": (750, 1250, 2550), "i": (310, 2250, 3000), "u": (350, 850, 2350), "e": (530, 1950, 2600), "o": (510, 950, 2400), "@": (500, 1450, 2500)}
SON = {  # sonoran: formant, durasi, gv, gn
    "m": (250, 1000, 2200, .070, .50), "n": (260, 1650, 2600, .065, .50), "N": (260, 1400, 2400, .075, .50), "Y": (260, 2050, 2700, .075, .50),
    "l": (400, 1100, 2650, .060, .70), "r": (420, 1300, 2300, .055, .65), "w": (300, 750, 2250, .055, .75), "y": (280, 2150, 2900, .055, .75)}
FRIC = {"s": (.095, 4200, 7800, .55, 0), "S": (.095, 2600, 7000, .55, 0), "f": (.080, 1500, 7800, .22, 0), "h": (.075, 400, 4800, .20, 0),
        "x": (.085, 900, 3200, .35, 0), "z": (.085, 4200, 7800, .40, 1)}
STOP = {"p": (0, 400, 1600), "b": (1, 400, 1600), "t": (0, 3000, 6500), "d": (1, 3000, 6500), "k": (0, 1400, 3200), "g": (1, 1400, 3200),
        "C": (0, 2600, 6500), "J": (1, 2600, 6500)}
F1_STOP_VOICED = 200


def build_tracks(units, P):
    """Bangun lintasan parameter per frame (10 ms). Mengembalikan dict array (T,)."""
    rate = max(0.6, min(1.6, P["rate"]))
    segs = []     # (dur_frames, dict)
    def fr(sec):
        return max(1, int(round(sec / rate / 0.010)))
    seg_pitch_accent = []
    sf = P["scale"]
    for idx, u in enumerate(units):
        if u[0] == "pause":
            segs.append(dict(n=fr(u[1]), gv=0, gn=0, f=(500, 1500, 2500), nz=(0, 0), nasal=0, tag="pause", accent=0))
            continue
        if u[0] == "end":
            segs.append(dict(n=1, gv=0, gn=0, f=(500, 1500, 2500), nz=(0, 0), nasal=0, tag="end", accent=0, q=(u[1] == "q")))
            continue
        _, p, st = u
        if p in V:
            d = 0.095 + (0.045 if st else 0)
            segs.append(dict(n=fr(d), gv=1.0, gn=0, f=V[p], nz=(0, 0), nasal=0, tag="v", accent=2.2 if st else 0))
        elif p in SON:
            f1, f2, f3, d, g = SON[p]
            segs.append(dict(n=fr(d), gv=g, gn=0, f=(f1, f2, f3), nz=(0, 0), nasal=1 if p in "mnNY" else 0, tag="s", accent=0))
        elif p in FRIC:
            d, lo, hi, g, vc = FRIC[p]
            segs.append(dict(n=fr(d), gv=0.30 if vc else 0, gn=g, f=(300, 1500, 2500), nz=(lo, hi), nasal=0, tag="f", accent=0))
        elif p in STOP:
            vc, lo, hi = STOP[p]
            segs.append(dict(n=fr(0.055), gv=0.14 if vc else 0, gn=0, f=(F1_STOP_VOICED, 1200, 2400), nz=(0, 0), nasal=0, tag="c", accent=0))
            segs.append(dict(n=1, gv=0.1 if vc else 0, gn=0.9, f=(500, 1500, 2500), nz=(lo, hi), nasal=0, tag="b", accent=0))
            segs.append(dict(n=fr(0.014), gv=0.15 if vc else 0, gn=0.30, f=(500, 1500, 2500), nz=(400, 4500), nasal=0, tag="b", accent=0))
    T = sum(s["n"] for s in segs)
    A = {k: np.zeros(T, np.float32) for k in ("gv", "gn", "f1", "f2", "f3", "nlo", "nhi", "nasal", "accent")}
    t = 0
    bounds = []
    for s in segs:
        n = s["n"]
        sl = slice(t, t + n)
        A["gv"][sl] = s["gv"]; A["gn"][sl] = s["gn"]
        A["f1"][sl], A["f2"][sl], A["f3"][sl] = (x * sf for x in s["f"])
        A["nlo"][sl], A["nhi"][sl] = (x * sf for x in s["nz"])
        A["nasal"][sl] = s["nasal"]; A["accent"][sl] = s["accent"]
        bounds.append((t, t + n, s))
        t += n
    def sm(x, k):
        if k <= 1:
            return x
        pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
        return np.convolve(pad, np.ones(k, np.float32) / k, mode="valid").astype(np.float32)
    for k in ("f1", "f2", "f3"):
        A[k] = sm(A[k], 5)
    A["gv"] = sm(A["gv"], 3); A["accent"] = sm(A["accent"], 7)
    # f0: deklinasi + aksen + intonasi akhir
    voiced_any = A["gv"] > 0.05
    ph = np.arange(T) / max(1, T)
    semi = 2.0 * (1 - ph) - 2.0 * ph + A["accent"]
    for a, b, sg in bounds:
        if sg["tag"] == "end":
            lo = max(0, a - 22)
            semi[lo:a] += np.linspace(0, 1, a - lo, dtype=np.float32) * (4.0 if sg.get("q") else -3.0)
    f0 = P["f0"] * (2.0 ** ((semi * (P["f0_sd"] / 2.5)) / 12.0))
    A["f0"] = sm(f0.astype(np.float32), 5)
    return A


# ----------------------------------------------------------------------------------------------- sintesis
def _env_voiced(freqs, A, P, eqcurve):
    """Respons magnitudo cascade 3 formant (+F4 tetap) untuk setiap frame -> (T, nbins)."""
    f = freqs[None, :]
    sf = P["scale"]
    out = np.ones((len(A["f1"]), len(freqs)), np.float32)
    for key, bw0 in (("f1", 70.0), ("f2", 110.0), ("f3", 170.0)):
        F = np.maximum(A[key][:, None], 120.0)
        B = bw0 * (1.0 + 0.0 * F)
        out *= 1.0 / np.sqrt((1 - (f / F) ** 2) ** 2 + (f * B / F ** 2) ** 2)
    F4 = 3500.0 * sf
    out *= 1.0 / np.sqrt((1 - (f / F4) ** 2) ** 2 + (f * 250 / F4 ** 2) ** 2)
    # nasal: anti-resonansi + redam formant atas
    nz = A["nasal"][:, None]
    out *= 1.0 - nz * 0.75 * np.exp(-(((f - 1100 * sf) / 450.0) ** 2))
    out *= 1.0 / (1.0 + (f / 1500.0) ** 2) ** 0.5            # kemiringan glotis
    tilt = P["tilt"]
    out *= (np.maximum(f, 50.0) / 1000.0) ** (tilt / 6.02)
    out *= eqcurve[None, :]
    return out


def synth_units(units, P=None):
    P = dict(DEFAULT, **(P or {}))
    A = build_tracks(units, P)
    T = len(A["gv"])
    n_samp = T * HOP + WIN
    rng = np.random.default_rng(1234)
    # f0 per sampel -> pulsa
    f0s = np.interp(np.arange(n_samp), np.arange(T) * HOP, A["f0"]).astype(np.float64)
    phase = np.cumsum(f0s / SR)
    pulses = np.zeros(n_samp, np.float32)
    idx = np.nonzero(np.diff(np.floor(phase)) > 0)[0] + 1
    pulses[idx[idx < n_samp]] = 1.0
    jit = 1.0
    noise = rng.standard_normal(n_samp).astype(np.float32)
    # bingkai
    pad = np.zeros(WIN // 2, np.float32)
    def frames(x):
        xp = np.concatenate([pad, x, np.zeros(WIN, np.float32)])
        return np.lib.stride_tricks.sliding_window_view(xp, WIN)[::HOP][:T]
    win = np.hanning(WIN + 1)[:-1].astype(np.float32)
    FP = np.fft.rfft(frames(pulses) * win, NFFT, axis=1)
    FN = np.fft.rfft(frames(noise) * win, NFFT, axis=1)
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR).astype(np.float32)
    eq = P.get("eq")
    eqcurve = np.ones(len(freqs), np.float32)
    if eq is not None:
        centers = np.geomspace(150, 7500, len(eq))
        eqcurve = (10.0 ** (np.interp(np.log(np.maximum(freqs, 1)), np.log(centers), np.clip(eq, -6, 6)) / 20.0)).astype(np.float32)
    Hv = _env_voiced(freqs, A, P, eqcurve)
    Hv /= np.sqrt((Hv ** 2).mean(axis=1, keepdims=True)) + 1e-9
    # noise: pita bandpass halus atau aspirasi mengikuti formant
    lo, hi = A["nlo"][:, None], A["nhi"][:, None]
    f = freqs[None, :]
    band = (1 / (1 + np.exp(-(f - lo) / 250.0))) * (1 / (1 + np.exp(-(hi - f) / 250.0)))
    asp = np.minimum(Hv / (Hv.max(axis=1, keepdims=True) + 1e-9), 1.0) ** 0.5
    Hn = np.where(hi > 0, band, 0.0)
    breath = P["breath"]
    # gain
    gv = A["gv"][:, None]
    gn = A["gn"][:, None]
    voiced_pow = 2.2
    Y = FP * Hv * gv * voiced_pow
    Y += FN * (Hn * gn * 0.32)
    Y += FN * (asp * (A["gv"][:, None] > 0.05) * (breath * 1.8) * gv)
    y_fr = np.fft.irfft(Y, NFFT, axis=1)[:, :WIN].astype(np.float32)
    out = np.zeros((T + 1) * HOP, np.float32)           # Hann 20 ms, hop 10 ms: jumlah tumpang-tindih = 1
    out[:T * HOP] += y_fr[:, :HOP].reshape(-1)
    out[HOP:(T + 1) * HOP] += y_fr[:, HOP:].reshape(-1)
    out = out[WIN // 2: WIN // 2 + T * HOP]
    pk = np.abs(out).max() + 1e-9
    out = out / pk * 0.85
    return out.astype(np.float32)


def speak(text, profile_vec=None, max_chars=1200):
    P = vec_to_params(profile_vec) if profile_vec is not None else {}
    text = (text or "").strip()[:max_chars]
    if not text:
        return np.zeros(0, np.float32)
    return synth_units(text_to_units(text), P)


def to_wav(x, sr=SR):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    return b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + struct.pack("<I", len(pcm)) + pcm


def read_wav(data):
    """WAV PCM 16-bit mono/stereo -> float32 mono, sample rate. Hanya format sederhana (klien mengonversi sebelum kirim)."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("bukan_wav")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if not fmt or pcm is None or fmt[0] != 1 or fmt[5] != 16:
        raise ValueError("wav_tidak_didukung")
    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], "<i2").astype(np.float32) / 32768.0
    if fmt[1] == 2:
        x = x[:len(x) // 2 * 2].reshape(-1, 2).mean(axis=1)
    return x, int(fmt[2])


# ----------------------------------------------------------------------------------------------- "latih suara": ukur profil
def _resample(x, sr):
    if sr == SR:
        return x
    n = int(len(x) * SR / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _lpc(frame, order):
    r = np.correlate(frame, frame, "full")[len(frame) - 1:len(frame) + order]
    if r[0] <= 1e-9:
        return None
    a = np.zeros(order + 1); a[0] = 1.0; e = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        b = a.copy()
        b[1:i] = a[1:i] + k * a[i - 1:0:-1]
        b[i] = k
        a, e = b, e * (1 - k * k)
        if e <= 1e-12:
            return None
    return a


def _formants(frame, order=14):
    a = _lpc(frame * np.hanning(len(frame)), order)
    if a is None:
        return None
    rts = np.roots(a)
    rts = rts[np.imag(rts) > 0.01]
    ang = np.angle(rts) * SR / (2 * np.pi)
    bw = -np.log(np.maximum(np.abs(rts), 1e-6)) * SR / np.pi
    ok = (ang > 200) & (ang < 4200) & (bw < 450)
    fs = np.sort(ang[ok])
    return fs[:3] if len(fs) >= 3 else None


def _ltas(x, frames_idx):
    seg = np.stack([x[i:i + 512] for i in frames_idx]) * np.hanning(512)
    P = (np.abs(np.fft.rfft(seg, axis=1)) ** 2).mean(axis=0)
    return np.fft.rfftfreq(512, 1.0 / SR), P + 1e-12


def _tilt_and_eq(freqs, P):
    db = 10 * np.log10(P)
    m = (freqs >= 200) & (freqs <= 5000)
    slope, icpt = np.polyfit(np.log2(freqs[m]), db[m], 1)
    centers = np.geomspace(150, 7500, 16)
    edges = np.sqrt(centers[:-1] * centers[1:])
    edges = np.concatenate([[100], edges, [8000]])
    eq = np.array([db[(freqs >= edges[i]) & (freqs < edges[i + 1])].mean() if ((freqs >= edges[i]) & (freqs < edges[i + 1])).any() else np.nan for i in range(16)])
    eq = np.nan_to_num(eq, nan=0.0)
    fit = slope * np.log2(centers) + icpt
    return float(slope), eq - fit


def _syl_rate(x, speech_seconds):
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    i = np.arange(0, len(x) - 800, 160)
    env = np.sqrt((cs[i + 800] - cs[i]) / 800.0)
    env = env / (env.max() + 1e-9)
    pk = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > 0.12)
    idx = np.nonzero(pk)[0] + 1
    keep, last = 0, -99
    for j in idx:
        if j - last >= 12:
            keep += 1; last = j
    return keep / max(speech_seconds, 1.0)


def estimate_profile(x, sr=SR, max_speech_seconds=80, progress=None):
    """x: float32 mono. Mengembalikan (vektor_profil, ringkasan_dict). Waktu dibatasi (~detik di HP) dengan memotong rekaman."""
    x = _resample(np.asarray(x, np.float32), sr)
    if len(x) < SR * 3:
        raise ValueError("rekaman_terlalu_pendek")
    x = x - np.mean(x)
    x = x / (np.abs(x).max() + 1e-9)
    n = (len(x) - 640) // HOP
    st = np.arange(n) * HOP
    cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    e = np.sqrt((cs[st + 640] - cs[st]) / 640.0).astype(np.float32)
    thr = max(np.percentile(e, 30) * 1.5, e.max() * 0.04)
    act = np.nonzero(e > thr)[0]
    if len(act) < 200:
        raise ValueError("rekaman_terlalu_sunyi")
    rng = np.random.default_rng(7)
    limit = int(max_speech_seconds * 100)
    if len(act) > limit:
        act = np.sort(rng.choice(act, limit, replace=False))
    total_speech = len(np.nonzero(e > thr)[0]) / 100.0
    if progress: progress(0.15)
    # f0 lewat autokorelasi FFT
    fr = {int(i): x[st[i]:st[i] + 640] for i in act}
    seg = np.stack([fr[int(i)] for i in act]) * np.hanning(640)
    F = np.fft.rfft(seg, 1280, axis=1)
    ac = np.fft.irfft(np.abs(F) ** 2, axis=1)[:, :260]
    ac = ac / (ac[:, :1] + 1e-9)
    lags = np.arange(35, 230)
    sub = ac[:, 35:230]
    best = sub.argmax(axis=1)
    pk = sub[np.arange(len(sub)), best]
    voiced = pk > 0.5
    f0_all = SR / lags[best]
    f0v = f0_all[voiced]
    if len(f0v) < 60:
        raise ValueError("suara_tidak_terdeteksi")
    # buang kesalahan oktaf: pusatkan di median, lipat
    med = float(np.median(f0v))
    f0v = f0v[(f0v > med * 0.6) & (f0v < med * 1.6)]
    f0 = float(np.median(f0v))
    sd_st = float(np.std(12 * np.log2(f0v / f0)))
    if progress: progress(0.4)
    # formant dari frame bersuara yang kuat
    vi = act[voiced]
    strong = vi[e[vi] > np.percentile(e[vi], 55)]
    strong = strong[:700]
    F1, F2, F3 = [], [], []
    for i in strong:
        z = fr[int(i)][160:480]
        r = _formants(z - 0.97 * np.concatenate([[0], z[:-1]]))
        if r is not None:
            F1.append(r[0]); F2.append(r[1]); F3.append(r[2])
    if progress: progress(0.7)
    if len(F3) >= 30:
        s = 0.5 * (np.median(F2) / 1500.0) + 0.5 * (np.median(F3) / 2500.0)
    else:
        s = f0 / 120.0 ** 0.5 / 1.0                      # cadangan: perkirakan dari pitch
        s = 1.0 + 0.25 * np.log2(f0 / 120.0)
    scale = float(np.clip(s, 0.85, 1.45))
    # spektrum rata-rata
    fa, Pa = _ltas(x, st[act][:1500])
    slope, eq = _tilt_and_eq(fa, Pa)
    # kecepatan & napas
    rate = _syl_rate(x[: SR * 120], min(total_speech, 120.0))
    hf = (np.abs(np.fft.rfft(seg[voiced][:400], axis=1)) ** 2)
    fq = np.fft.rfftfreq(640, 1.0 / SR)
    breath = float(np.clip(hf[:, fq > 4000].sum() / (hf.sum() + 1e-9) * 1.5, 0.02, 0.18))
    # referensi sintesis bawaan (agar tilt/eq/rate relatif terhadap pemulih kita sendiri)
    ref = synth_units(text_to_units("Saya sedang belajar berbicara bahasa Indonesia dengan pelan dan jelas. Hari ini cuaca cerah."), dict(DEFAULT, f0=f0))
    rf, rP = _ltas(ref, np.arange(0, len(ref) - 512, 160)[:300])
    ref_slope, ref_eq = _tilt_and_eq(rf, rP)
    ref_rate = _syl_rate(ref, max(len(ref) / SR * 0.8, 1.0))
    tilt = float(np.clip(slope - ref_slope, -9, 9))
    eq = np.clip(eq - ref_eq, -5, 5)
    rate_mult = float(np.clip((rate / max(ref_rate, 1e-3)) * 0.92, 0.9, 1.3)) if rate > 0.5 else 1.0
    if progress: progress(0.95)
    vec = profile_to_vec(f0, max(1.2, min(6.0, sd_st * 1.3)), scale, tilt, rate_mult, breath, eq, float(total_speech), float(len(f0v)) / max(1, len(act)))
    info = dict(f0=round(f0), scale=round(scale, 2), tilt=round(tilt, 1), rate=round(rate_mult, 2), seconds=round(total_speech), formants=len(F3))
    return vec, info
