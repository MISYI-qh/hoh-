"""Robust page fetching for the learner: retries, JavaScript pages, paywall/bot-wall detection.

Drop-in replacement for `ingest.fetch_url` (same call signature and return value): `Ingestor(..., fetcher=RobustFetcher())`.

What it does
  1. Plain HTTP with retry + exponential backoff on transient failures (timeouts, 429, 5xx); honours Retry-After when given.
  2. If the page is an empty "JavaScript shell" (React/Vue/Next apps), it first looks for text the page already ships inside
     JSON-LD / __NEXT_DATA__; if that is not enough it RENDERS the page in headless Chromium (Playwright, optional) and returns
     the resulting HTML. Images/media/fonts are not downloaded.
  3. Paywall detection (publisher flags such as isAccessibleForFree=false, "subscribe to continue" wording, locked-content meta,
     or a short teaser): raises `Paywalled`. The teaser is NOT learned (a truncated article teaches fragments).
  4. Bot walls / captchas / 403 (Cloudflare "Just a moment", "verify you are human"): raises `Blocked` and remembers the host
     for `block_cooldown` seconds so the explorer stops hammering it.

What it deliberately does NOT do: bypass paywalls or bot protection (no crawler-UA spoofing, cookie tricks, archive mirrors,
captcha solving or stealth plugins). Those break site terms and often the law. If you are entitled to the content, save it and
`hoh learn file.pdf` / `/learn file`. Robots.txt is enforced by the Explorer, not here.

Needs for JS rendering: `pip install playwright` and `playwright install chromium`. Without it, JS-only pages are reported as
'needs JavaScript rendering' instead of silently learning an empty shell."""
import html as _html, json, re, time, urllib.parse
from .ingest import IngestError, fetch_url, extract_links, html_to_text


class Paywalled(IngestError):
    pass


class Blocked(IngestError):
    pass


class NeedsJavaScript(IngestError):
    pass


_PAY_PHRASES = re.compile(
    r"(berlangganan untuk (membaca|melanjutkan)|khusus (pelanggan|member|anggota)|hanya untuk (pelanggan|member|anggota)|"
    r"artikel (ini )?(khusus|hanya)|login untuk membaca|masuk untuk membaca|subscribe to (continue|read|unlock)|"
    r"subscribers? only|this (article|content) is (for|available to) (paid )?(subscribers|members)|"
    r"sign in to (continue )?read|create a free account to (continue )?read|you have reached your (free )?(article )?limit|"
    r"unlock (this|the full) (article|story))", re.I)
_PAY_FLAGS = re.compile(r"(\"isAccessibleForFree\"\s*:\s*(false|\"false\"|\"False\")|article:content_tier[^>]*(locked|metered)|"
                        r"class=\"[^\"]*\b(paywall|premium-content|locked-content)\b)", re.I)
_BOT = re.compile(r"(just a moment\.\.\.|checking your browser|verify you are (a )?human|attention required|"
                  r"enable javascript and cookies to continue|are you a robot|captcha|access denied|unusual traffic)", re.I)
_SHELL = re.compile(r"(<div[^>]+id=\"(root|app|__next|__nuxt)\"|window\.__INITIAL_STATE__|__NEXT_DATA__|ng-version|data-reactroot|"
                    r"<noscript>[^<]*(enable|aktifkan)[^<]*javascript)", re.I)
_RETRY_STATUS = {429, 500, 502, 503, 504}


def visible_chars(html):
    try:
        return len(html_to_text(html)[1])
    except Exception:
        return len(re.sub(r"<[^>]+>", " ", html))


def embedded_text(html):
    """Text the publisher ships inside JSON-LD `articleBody` / `text` and Next.js __NEXT_DATA__ (no JS execution needed)."""
    out = []
    for m in re.finditer(r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", html, re.S | re.I):
        try:
            data = json.loads(_html.unescape(m.group(1).strip()))
        except ValueError:
            continue
        stack = [data]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                for k in ("articleBody", "text"):
                    if isinstance(o.get(k), str) and len(o[k]) > 300:
                        out.append(o[k])
                stack.extend(v for v in o.values() if isinstance(v, (dict, list)))
            elif isinstance(o, list):
                stack.extend(o)
    m = re.search(r"<script[^>]+id=[\"']__NEXT_DATA__[\"'][^>]*>(.*?)</script>", html, re.S | re.I)
    if m:
        try:
            stack = [json.loads(m.group(1))]
            while stack:
                o = stack.pop()
                if isinstance(o, str) and len(o) > 400 and " " in o and not o.startswith(("http", "{", "<")):
                    out.append(o)
                elif isinstance(o, dict):
                    stack.extend(o.values())
                elif isinstance(o, list):
                    stack.extend(o)
        except ValueError:
            pass
    seen, res = set(), []
    for t in out:
        if t[:80] not in seen:
            seen.add(t[:80])
            res.append(t)
    return "\n\n".join(res)


def analyze_html(html, min_text=400):
    vis = visible_chars(html)
    low = html[:400_000]
    info = {"visible_chars": vis, "bot_wall": False, "paywall": False, "js_shell": False, "reasons": []}
    title = re.search(r"<title[^>]*>(.*?)</title>", low, re.S | re.I)
    head = (title.group(1) if title else "") + " " + re.sub(r"<[^>]+>", " ", low)[:1500]
    if vis < 1500 and _BOT.search(head):
        info["bot_wall"] = True
        info["reasons"].append("bot/captcha wall")
    if _PAY_FLAGS.search(low):
        info["paywall"] = True
        info["reasons"].append("publisher marks content as locked")
    if _PAY_PHRASES.search(re.sub(r"<[^>]+>", " ", low)):
        info["paywall"] = True if vis < 6000 else info["paywall"]
        info["reasons"].append("subscription wording")
    if vis < min_text and _SHELL.search(low):
        info["js_shell"] = True
        info["reasons"].append("JavaScript app shell")
    return info


class JsRenderer:
    """Headless Chromium through Playwright (optional). One browser, reused; call close() when done."""

    def __init__(self, timeout=25.0, settle=1.0, block=("image", "media", "font")):
        self.timeout, self.settle, self.block, self._pw, self._browser = timeout, settle, set(block), None, None

    def available(self):
        try:
            import playwright.sync_api  # noqa
            return True
        except ImportError:
            return False

    def _start(self):
        if self._browser is None:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            self._browser = self._pw.chromium.launch()
        return self._browser

    def render(self, url, user_agent):
        try:
            b = self._start()
        except ImportError as e:
            raise NeedsJavaScript("page needs JavaScript rendering: pip install playwright && playwright install chromium") from e
        except Exception as e:
            raise NeedsJavaScript(f"cannot start Chromium ({type(e).__name__}): {e}") from e
        ctx = b.new_context(user_agent=user_agent, java_script_enabled=True)
        try:
            page = ctx.new_page()
            page.route("**/*", lambda r: r.abort() if r.request.resource_type in self.block else r.continue_())
            resp = page.goto(url, wait_until="networkidle", timeout=int(self.timeout * 1000))
            page.wait_for_timeout(int(self.settle * 1000))
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")      # lazy-loaded text
            page.wait_for_timeout(int(self.settle * 500))
            return page.content(), (resp.status if resp else 200), page.url
        except Exception as e:
            raise IngestError(f"render failed ({type(e).__name__}): {e}")
        finally:
            ctx.close()

    def close(self):
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        finally:
            self._browser = self._pw = None


class RobustFetcher:
    def __init__(self, base=fetch_url, renderer=None, render=True, retries=3, backoff=1.5, block_cooldown=86400.0,
                 min_text=400, user_agent="HOH-learner/1.5 (+personal research)", sleep=time.sleep, clock=time.time):
        self.base, self.render_enabled, self.retries, self.backoff = base, render, retries, backoff
        self.renderer = renderer if renderer is not None else (JsRenderer() if render else None)
        self.block_cooldown, self.min_text, self.ua, self.sleep, self.clock = block_cooldown, min_text, user_agent, sleep, clock
        self.blocked = {}                                    # host -> time it was blocked
        self.stats = {"retries": 0, "rendered": 0, "paywalled": 0, "blocked": 0, "embedded": 0}

    # ---- plain fetch with retry ----
    def _status(self, msg):
        m = re.search(r"HTTP Error (\d{3})", msg)
        return int(m.group(1)) if m else None

    def _get(self, url, **kw):
        last = None
        for attempt in range(self.retries + 1):
            try:
                return self.base(url, user_agent=self.ua, **kw)
            except IngestError as e:
                last, msg = e, str(e)
                code = self._status(msg)
                if code in (401, 403, 451):
                    raise Blocked(f"{url}: HTTP {code} (access refused; not bypassed)")
                transient = code in _RETRY_STATUS or code is None and re.search(r"timed out|Timeout|reset|URLError|temporar|refused|unreachable", msg, re.I)
                if not transient or attempt == self.retries:
                    break
                self.stats["retries"] += 1
                self.sleep(self.backoff ** attempt)
        raise last

    def __call__(self, url, timeout=20.0, max_bytes=20_000_000, user_agent=None):
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        t = self.blocked.get(host)
        if t is not None and self.clock() - t < self.block_cooldown:
            raise Blocked(f"{host} refused us earlier; skipping for now")
        try:
            data, ctype, final = self._get(url, timeout=timeout, max_bytes=max_bytes)
        except Blocked:
            self.blocked[host] = self.clock()
            self.stats["blocked"] += 1
            raise
        if "html" not in (ctype or "").lower() and not data.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
            return data, ctype, final
        html = data.decode(_charset(ctype, data), errors="replace")
        html = self._meta_refresh(html, final, timeout, max_bytes)
        info = analyze_html(html, self.min_text)
        if info["bot_wall"]:
            self.blocked[host] = self.clock()
            self.stats["blocked"] += 1
            raise Blocked(f"{url}: bot/captcha wall (not bypassed)")
        if info["js_shell"] or (info["visible_chars"] < self.min_text and not info["paywall"]):
            emb = embedded_text(html)
            if len(emb) > max(self.min_text, info["visible_chars"] * 2):
                self.stats["embedded"] += 1
                html = self._synthetic(html, emb)
                info = analyze_html(html, self.min_text)
            elif info["js_shell"] and self.render_enabled and self.renderer is not None:
                rendered, status, final2 = self.renderer.render(final or url, self.ua)
                self.stats["rendered"] += 1
                if status in (401, 403, 451):
                    self.blocked[host] = self.clock()
                    raise Blocked(f"{url}: HTTP {status} after rendering")
                html, final = rendered, final2 or final
                info = analyze_html(html, self.min_text)
                if info["bot_wall"]:
                    self.blocked[host] = self.clock()
                    self.stats["blocked"] += 1
                    raise Blocked(f"{url}: bot/captcha wall (not bypassed)")
                if info["js_shell"]:
                    raise NeedsJavaScript(f"{url}: still empty after rendering")
            elif info["js_shell"]:
                raise NeedsJavaScript(f"{url}: page needs JavaScript rendering (pip install playwright; playwright install chromium)")
        if info["paywall"]:
            self.stats["paywalled"] += 1
            raise Paywalled(f"{url}: paywalled ({'; '.join(info['reasons'])}); teaser not learned. "
                            f"If you may read it, save the text/PDF and use `hoh learn <file>`.")
        return html.encode("utf-8"), "text/html; charset=utf-8", final

    def _meta_refresh(self, html, base, timeout, max_bytes):
        m = re.search(r"<meta[^>]+http-equiv=[\"']refresh[\"'][^>]+content=[\"']\d+;\s*url=([^\"']+)", html[:5000], re.I)
        if not m or len(html) > 3000:
            return html
        data, ctype, final = self._get(urllib.parse.urljoin(base, _html.unescape(m.group(1))), timeout=timeout, max_bytes=max_bytes)
        return data.decode(_charset(ctype, data), errors="replace")

    @staticmethod
    def _synthetic(html, text):
        t = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        body = "".join(f"<p>{_html.escape(p)}</p>" for p in text.split("\n\n"))
        return f"<html><head><title>{t.group(1) if t else ''}</title></head><body><article>{body}</article></body></html>"

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def _charset(ctype, data):
    m = re.search(r"charset=([\w-]+)", ctype or "", re.I) or re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", data[:4000], re.I)
    if m:
        cs = m.group(1) if isinstance(m.group(1), str) else m.group(1).decode()
        try:
            "".encode(cs)
            return cs
        except LookupError:
            pass
    return "utf-8"
