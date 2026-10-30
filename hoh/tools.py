"""Tool registry with schema validation, sandboxed file access, safe calculator, risk gating.

Security notes: run_python is a resource-limited subprocess, NOT a security boundary (no network or
filesystem isolation). It is marked risky and refused unless the caller allows risky tools."""
import ast, base64, json, math, operator, os, subprocess, sys, tempfile, time, urllib.parse, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict            # {arg: {"type": str|integer|number|boolean|array, "required": bool, "description": str}}
    fn: object
    risky: bool = False
    timeout: float = 10.0

    def schema(self):
        return {"name": self.name, "description": self.description, "risky": self.risky,
                "parameters": self.params}


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list}


def validate_args(spec, args):
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    extra = set(args) - set(spec.params)
    if extra:
        raise ToolError(f"unexpected arguments: {sorted(extra)}")
    clean = {}
    for k, d in spec.params.items():
        if k not in args:
            if d.get("required"):
                raise ToolError(f"missing required argument '{k}'")
            continue
        v, t = args[k], d["type"]
        if t in ("integer", "number") and isinstance(v, bool):
            raise ToolError(f"argument '{k}' must be {t}")
        if not isinstance(v, _TYPES[t]):
            raise ToolError(f"argument '{k}' must be {t}, got {type(v).__name__}")
        clean[k] = v
    return clean


class ToolRegistry:
    def __init__(self, max_output_chars=4000):
        self.tools, self.max_out = {}, max_output_chars

    def register(self, spec):
        if spec.name in self.tools:
            raise ValueError(f"tool {spec.name} already registered")
        self.tools[spec.name] = spec

    def describe(self):
        return [t.schema() for t in self.tools.values()]

    def call(self, name, args, allow_risky=False, confirm=None):
        spec = self.tools.get(name)
        if spec is None:
            return ToolResult(False, error=f"unknown tool '{name}'; available: {sorted(self.tools)}")
        try:
            clean = validate_args(spec, args)
        except ToolError as e:
            return ToolResult(False, error=str(e))
        if spec.risky and not (allow_risky or (confirm and confirm(name, clean))):
            return ToolResult(False, error=f"tool '{name}' is risky and was not permitted")
        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(spec.fn, **clean)
        try:
            out = fut.result(timeout=spec.timeout)
        except FutTimeout:
            return ToolResult(False, error=f"tool '{name}' timed out after {spec.timeout}s (call abandoned)")
        except ToolError as e:
            return ToolResult(False, error=str(e))
        except Exception as e:  # tool bugs must not crash the agent
            return ToolResult(False, error=f"{type(e).__name__}: {e}")
        finally:
            pool.shutdown(wait=False)
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        if len(text) > self.max_out:
            text = text[:self.max_out] + f"... [truncated {len(text) - self.max_out} chars]"
        return ToolResult(True, output=text)


# ---- calculator ---------------------------------------------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
          "exp": math.exp, "abs": abs, "round": round, "min": min, "max": max}
_CONST = {"pi": math.pi, "e": math.e}


def safe_eval(expr):
    if len(expr) > 200:
        raise ToolError("expression too long")
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand) if isinstance(n.op, ast.USub) else ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 1000 or abs(a) > 1e100):
                raise ToolError("exponent/base too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ToolError(f"unsupported expression element: {type(n).__name__}")
    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except ToolError:
        raise
    except (SyntaxError, ZeroDivisionError, ValueError, OverflowError, TypeError) as e:
        raise ToolError(f"calculation error: {e}")


# ---- sandboxed filesystem -----------------------------------------------------
class Sandbox:
    def __init__(self, root, max_read_bytes=1_000_000, max_write_bytes=1_000_000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_read, self.max_write = max_read_bytes, max_write_bytes

    def resolve(self, rel):
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            raise ToolError("invalid path")
        if os.path.isabs(rel) or Path(rel).drive:
            raise ToolError("absolute paths are not allowed")
        p = (self.root / rel).resolve()  # resolves '..' and symlinks
        if p != self.root and self.root not in p.parents:
            raise ToolError("path escapes the sandbox")
        return p

    def read_file(self, path):
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        if p.stat().st_size > self.max_read:
            raise ToolError(f"file larger than {self.max_read} bytes")
        if p.suffix.lower() in {".pdf", ".docx", ".pptx", ".xlsx", ".epub", ".html", ".htm"}:
            from .ingest import parse_bytes, IngestError
            try:
                return "\n\n".join(d.text for d in parse_bytes(p.read_bytes(), str(p)))
            except IngestError as e:
                raise ToolError(str(e))
        if p.suffix.lower() not in {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".py", ".log", ".yaml", ".yml", ".toml"}:
            raise ToolError(f"unsupported document type '{p.suffix}'")
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ToolError("file is not valid UTF-8")

    def write_file(self, path, content, overwrite=False):
        p = self.resolve(path)
        if len(content.encode("utf-8")) > self.max_write:
            raise ToolError("content too large")
        if p.exists() and not overwrite:
            raise ToolError("file exists; pass overwrite=true to replace it")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, p)
        return f"wrote {len(content)} chars to {path}"

    def list_dir(self, path="."):
        p = self.resolve(path)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path}")
        return "\n".join(sorted(f"{c.name}{'/' if c.is_dir() else ''}" for c in p.iterdir()))

    def make_zip(self, paths, output):
        out = self.resolve(output)
        if out.suffix.lower() != ".zip":
            raise ToolError("output must end with .zip")
        if out.exists():
            raise ToolError("output exists")
        total = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in paths:
                p = self.resolve(rel)
                if not p.is_file():
                    raise ToolError(f"not a file: {rel}")
                total += p.stat().st_size
                if total > 20_000_000:
                    raise ToolError("archive too large")
                z.write(p, arcname=str(p.relative_to(self.root)))
        return f"created {output} with {len(paths)} files"


def run_python(code, sandbox_root, timeout=5.0):
    timeout = min(float(timeout), 20.0)
    def limits():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (int(timeout) + 1,) * 2)
            resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 2**20,) * 2)
        except Exception:
            pass
    with tempfile.TemporaryDirectory(dir=sandbox_root) as cwd:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=cwd, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": os.environ.get("PATH", "")},
                               preexec_fn=limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            raise ToolError(f"code timed out after {timeout}s")
    out = (r.stdout or "")[:3000]
    err = (r.stderr or "")[:1500]
    return f"exit={r.returncode}\nstdout:\n{out}\nstderr:\n{err}"


# ---- external providers (configured via environment; never faked) ------------
def _http(url, data=None, headers=None, timeout=10.0):
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ToolError("provider URL must be http(s)")
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(5_000_000), r.headers.get("Content-Type", "")
    except Exception as e:
        raise ToolError(f"provider request failed: {type(e).__name__}")  # no URL/key in message


def web_search(query, n=5, env=None):
    env = env if env is not None else os.environ
    url = env.get("HOH_WEB_SEARCH_URL")
    if not url:
        raise ToolError("web search not configured: set HOH_WEB_SEARCH_URL (and optionally HOH_WEB_SEARCH_KEY)")
    h = {"Authorization": f"Bearer {env['HOH_WEB_SEARCH_KEY']}"} if env.get("HOH_WEB_SEARCH_KEY") else {}
    body, _ = _http(f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode({'q': query, 'n': n})}", headers=h)
    return body.decode("utf-8", errors="replace")


def generate_image(prompt, output, sandbox, env=None):
    """Provider contract: POST {"prompt":...} -> image bytes (Content-Type image/*) or JSON {"image_base64": ...}."""
    env = env if env is not None else os.environ
    url = env.get("HOH_IMAGE_URL")
    if not url:
        raise ToolError("image generation not configured: set HOH_IMAGE_URL (and optionally HOH_IMAGE_KEY)")
    h = {"Content-Type": "application/json"}
    if env.get("HOH_IMAGE_KEY"):
        h["Authorization"] = f"Bearer {env['HOH_IMAGE_KEY']}"
    body, ctype = _http(url, json.dumps({"prompt": prompt}).encode(), h, timeout=60.0)
    if ctype.startswith("image/"):
        data = body
    else:
        try:
            data = base64.b64decode(json.loads(body)["image_base64"], validate=True)
        except Exception:
            raise ToolError("provider response is neither an image nor JSON with image_base64")
    p = sandbox.resolve(output)
    if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp") or p.exists():
        raise ToolError("output must be a new .png/.jpg/.webp path")
    p.write_bytes(data)
    return f"saved image ({len(data)} bytes) to {output}"


def build_default_registry(sandbox_root, memory=None, workspace="default", env=None, ingestor=None):
    sb = Sandbox(sandbox_root)
    r = ToolRegistry()
    S = lambda d, req=True: {"type": "string", "required": req, "description": d}
    r.register(ToolSpec("get_time", "Current UTC date/time (ISO 8601).", {},
                        lambda: datetime.now(timezone.utc).isoformat()))
    r.register(ToolSpec("calculator", "Evaluate an arithmetic expression.", {"expression": S("e.g. 2*(3+4)")},
                        lambda expression: str(safe_eval(expression))))
    r.register(ToolSpec("text_stats", "Count chars, words, lines of a text.", {"text": S("text")},
                        lambda text: json.dumps({"chars": len(text), "words": len(text.split()),
                                                 "lines": len(text.splitlines())})))
    r.register(ToolSpec("read_file", "Read a text file inside the workspace.", {"path": S("relative path")}, sb.read_file))
    r.register(ToolSpec("list_dir", "List a workspace directory.", {"path": S("relative path", False)}, sb.list_dir))
    r.register(ToolSpec("write_file", "Create/overwrite a text file in the workspace.",
                        {"path": S("relative path"), "content": S("text"),
                         "overwrite": {"type": "boolean", "required": False, "description": "replace existing"}},
                        sb.write_file, risky=True))
    r.register(ToolSpec("make_zip", "Zip workspace files into a new archive.",
                        {"paths": {"type": "array", "required": True, "description": "relative file paths"},
                         "output": S("new .zip path")}, sb.make_zip, risky=True))
    r.register(ToolSpec("run_python", "Run Python in a resource-limited subprocess (not a hard sandbox).",
                        {"code": S("python source"), "timeout": {"type": "number", "required": False, "description": "seconds, max 20"}},
                        lambda code, timeout=5.0: run_python(code, sb.root, timeout), risky=True, timeout=30.0))
    r.register(ToolSpec("web_search", "Web search through the configured provider.",
                        {"query": S("query"), "n": {"type": "integer", "required": False, "description": "results"}},
                        lambda query, n=5: web_search(query, n, env), timeout=15.0))
    r.register(ToolSpec("generate_image", "Generate an image via the configured provider.",
                        {"prompt": S("description"), "output": S("new image path in workspace")},
                        lambda prompt, output: generate_image(prompt, output, sb, env), risky=True, timeout=70.0))
    if ingestor is not None:
        from .ingest import IngestError
        def _page(url, max_chars=4000):
            docs = ingestor.read(url)
            return "\n\n".join(d.text for d in docs)[:max(200, min(int(max_chars), 20000))]
        def _learn(source):
            rep = ingestor.learn(source)
            return ingestor.summary(rep)
        r.register(ToolSpec("open_page", "Read a web page / file (URL or workspace-independent path) WITHOUT learning it.",
                            {"url": S("http(s) URL or file path"), "max_chars": {"type": "integer", "required": False, "description": "max chars"}},
                            _page, timeout=30.0))
        r.register(ToolSpec("learn", "Learn from a URL, file, folder or zip: cleaned text is queued for training and memory.",
                            {"source": S("http(s) URL or path")}, _learn, timeout=60.0))
        r.register(ToolSpec("learn_search", "Search the web and learn from the top results.",
                            {"query": S("search query"), "n": {"type": "integer", "required": False, "description": "results (1-8)"}},
                            lambda query, n=5: ingestor.summary(ingestor.learn_search(query, max(1, min(int(n), 8)))), timeout=120.0))
    if memory is not None:
        r.register(ToolSpec("memory_add", "Store a note in persistent memory.", {"content": S("text")},
                            lambda content: f"stored id={memory.add(workspace, 'persistent', content)[0]}"))
        r.register(ToolSpec("memory_search", "Search stored memory.",
                            {"query": S("query"), "k": {"type": "integer", "required": False, "description": "max results"}},
                            lambda query, k=5: [{"id": m["id"], "content": m["content"]}
                                                for m in memory.search(workspace, query, k=min(k, 10))]))
    return r
