"""OCR + speech-to-text for HOH's "learn from anywhere" pipeline. Everything here is OPTIONAL and imports lazily:
a missing engine raises OcrUnavailable with an install hint, which ingest turns into a clear per-file error (the rest of a
folder/zip still gets learned).

Images / scanned PDFs : Tesseract through `pytesseract` + Pillow (+ `pypdfium2` or `pdf2image`+poppler to rasterise PDFs).
                        Install the engine and the Indonesian model:  apt install tesseract-ocr tesseract-ocr-ind
                        (Termux: pkg install tesseract; macOS: brew install tesseract tesseract-lang).
Audio                 : first available of faster-whisper, openai-whisper, vosk (all need their model weights downloaded).

OCR text is noisy. It goes through the same clean/quality/truth filters as everything else, words below `min_conf` are
dropped, and documents are tagged `ocr` in their notes so the provenance is visible in the .meta.json sidecar."""
import io, os, re, tempfile
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac", ".wma"}


AUDIO_BACKEND = None        # optional global override: callable(path, language) -> text


class OcrUnavailable(Exception):
    pass


def _tess():
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise OcrUnavailable("OCR needs `pip install pytesseract pillow` and the tesseract program") from e
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise OcrUnavailable("tesseract program not found (apt install tesseract-ocr tesseract-ocr-ind)") from e
    return pytesseract, Image


def pick_langs(wanted="ind+eng"):
    """Use the wanted languages that are actually installed; returns (lang_string, notes)."""
    pt, _ = _tess()
    have = set(pt.get_languages())
    use = [l for l in wanted.split("+") if l in have] or [l for l in ("eng",) if l in have]
    notes = [] if len(use) == len(wanted.split("+")) else [f"OCR languages missing: {sorted(set(wanted.split('+')) - have)} (using {use})"]
    if not use:
        raise OcrUnavailable("no tesseract language data installed")
    return "+".join(use), notes


def _prep(img):
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img).convert("L")
    if min(img.size) < 900:                                    # small text: upscale before recognition
        f = 2 if min(img.size) >= 450 else 3
        img = img.resize((img.width * f, img.height * f))
    return ImageOps.autocontrast(img)


def ocr_image(img, langs="ind+eng", min_conf=40):
    """PIL image -> text. Words with confidence < min_conf are dropped; line structure is kept."""
    pt, _ = _tess()
    lang, _ = pick_langs(langs)
    d = pt.image_to_data(_prep(img), lang=lang, output_type=pt.Output.DICT, config="--psm 3")
    lines, cur, key = [], [], None
    for i, w in enumerate(d["text"]):
        k = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
        if k != key and cur:
            lines.append((key, " ".join(cur)))
            cur = []
        key = k
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if w.strip() and conf >= min_conf:
            cur.append(w)
    if cur:
        lines.append((key, " ".join(cur)))
    out, prev = [], None
    for k, t in lines:
        if prev is not None and (k[0], k[1]) != (prev[0], prev[1]):
            out.append("")                                    # new paragraph
        out.append(t)
        prev = k
    return "\n".join(out).strip()


def ocr_image_bytes(data, langs="ind+eng", min_conf=40):
    _, Image = _tess()
    return ocr_image(Image.open(io.BytesIO(data)), langs, min_conf)


def pdf_page_count(data):
    try:
        import pypdfium2
        return len(pypdfium2.PdfDocument(data))
    except Exception:
        return len(re.findall(rb"/Type\s*/Page[^s]", data)) or 1


def _render_pdf(data, dpi, max_pages):
    try:
        import pypdfium2
        pdf = pypdfium2.PdfDocument(data)
        for i in range(min(len(pdf), max_pages)):
            yield pdf[i].render(scale=dpi / 72).to_pil()
        return
    except ImportError:
        pass
    try:
        from pdf2image import convert_from_bytes
    except ImportError as e:
        raise OcrUnavailable("scanned PDFs need `pip install pypdfium2` (or pdf2image + poppler)") from e
    for im in convert_from_bytes(data, dpi=dpi, last_page=max_pages):
        yield im


def ocr_pdf(data, langs="ind+eng", dpi=200, max_pages=200, min_conf=40):
    pages = []
    for im in _render_pdf(data, dpi, max_pages):
        pages.append(ocr_image(im, langs, min_conf))
    return "\n\n".join(p for p in pages if p)


def looks_scanned(extracted_text, data):
    """True when a PDF's text layer is (almost) empty compared with its page count -> needs OCR."""
    n = pdf_page_count(data)
    return len(re.sub(r"\s", "", extracted_text or "")) < 40 * max(1, n)


# ---------------------------------------------------------------- audio
def transcribe_audio(data, name="audio", language="id", backend=None):
    """-> text. `backend(path, language) -> str` can be injected (tests, custom engines); otherwise the first installed
    engine is used. Models must already be downloaded; HOH never downloads weights by itself."""
    backend = backend or AUDIO_BACKEND
    if backend is not None:
        return _with_tempfile(data, name, lambda p: backend(p, language))
    errs = []
    try:
        from faster_whisper import WhisperModel
        size = os.environ.get("HOH_WHISPER_MODEL", "small")
        m = WhisperModel(size, device="auto", compute_type="auto")
        return _with_tempfile(data, name, lambda p: " ".join(s.text.strip() for s in m.transcribe(p, language=language)[0]))
    except ImportError as e:
        errs.append(f"faster-whisper: {e}")
    try:
        import whisper
        m = whisper.load_model(os.environ.get("HOH_WHISPER_MODEL", "small"))
        return _with_tempfile(data, name, lambda p: m.transcribe(p, language=language)["text"].strip())
    except ImportError as e:
        errs.append(f"whisper: {e}")
    raise OcrUnavailable("audio needs `pip install faster-whisper` (or openai-whisper) with a downloaded model; " + "; ".join(errs))


def _with_tempfile(data, name, fn):
    suffix = Path(name).suffix or ".wav"
    fd, p = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return fn(p)
    finally:
        Path(p).unlink(missing_ok=True)
