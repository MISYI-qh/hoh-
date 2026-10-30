"""Reasoning support (backend-independent).

What this IS: (1) a fixed "think, then answer" format the model is trained to emit: <think>langkah...</think> jawaban;
(2) a generator of PROGRAMMATICALLY VERIFIED reasoning data in Indonesian (arithmetic, word problems, unit conversion,
comparison/ordering, sequences, syllogisms, scheduling) — every answer is computed, never guessed;
(3) inference-time helpers: strip hidden thinking, self-consistency voting, and tool-based verification.

What this is NOT: a guarantee of reasoning ability. Reasoning in a small model comes from scale + diverse data + training;
this module supplies verified practice material and test-time compute, not intelligence. Measure with `evaluate_reasoning`."""
import json, random, re
from collections import Counter
from .tools import safe_eval

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def split_thinking(text):
    """-> (thinking, visible_answer). An unterminated <think> is treated as all-thinking (answer empty)."""
    m = THINK_RE.findall(text)
    visible = THINK_RE.sub("", text)
    if "<think>" in visible:
        visible, tail = visible.split("<think>", 1)
        m.append(tail)
    return "\n".join(x.strip() for x in m), visible.strip()


def fmt(n):
    if isinstance(n, float) and n == int(n):
        n = int(n)
    return f"{n:,}".replace(",", ".") if isinstance(n, int) and abs(n) >= 10000 else (str(round(n, 4)) if isinstance(n, float) else str(n))


# ---------------------------------------------------------------- verified problem generators
def _arith(r):
    a, b, c = r.randint(2, 99), r.randint(2, 99), r.randint(2, 20)
    op1, op2 = r.choice("+-"), r.choice("*+-")
    expr = f"({a}{op1}{b}){op2}{c}"
    val = safe_eval(expr)
    inner = safe_eval(f"{a}{op1}{b}")
    th = f"Kerjakan kurung dulu: {a}{op1}{b} = {inner}. Lalu {inner}{op2}{c} = {val}."
    return f"Hitung {expr}.", th, str(val), expr


def _shop(r):
    item = r.choice(["buku", "pensil", "apel", "kue", "tiket", "baterai"])
    n, p, d = r.randint(2, 12), r.choice([1500, 2500, 3000, 4500, 7000, 12000]), r.choice([500, 1000, 2000])
    pay = r.choice([50000, 100000])
    total = n * p
    chg = pay - total
    if chg < 0:
        pay, chg = 200000, 200000 - total
    return (f"Dina membeli {n} {item} seharga Rp{p} per buah dan membayar dengan uang Rp{pay}. Berapa kembaliannya?",
            f"Total harga = {n} x {p} = {total}. Kembalian = {pay} - {total} = {chg}.", f"Rp{chg}", f"{pay}-{n}*{p}")


def _speed(r):
    v, t = r.choice([30, 40, 45, 60, 72, 80]), r.choice([2, 3, 4, 5])
    return (f"Sebuah mobil melaju {v} km/jam selama {t} jam. Berapa jarak yang ditempuh?",
            f"Jarak = kecepatan x waktu = {v} x {t} = {v * t} km.", f"{v * t} km", f"{v}*{t}")


def _unit(r):
    kind = r.choice(["m_cm", "kg_g", "jam_menit", "l_ml"])
    x = r.randint(2, 90)
    f, a, b = {"m_cm": (100, "meter", "sentimeter"), "kg_g": (1000, "kilogram", "gram"),
               "jam_menit": (60, "jam", "menit"), "l_ml": (1000, "liter", "mililiter")}[kind]
    return (f"Berapa {b} dalam {x} {a}?", f"1 {a} = {f} {b}. Jadi {x} x {f} = {x * f}.", f"{x * f} {b}", f"{x}*{f}")


def _compare(r):
    names = r.sample(["Ani", "Budi", "Citra", "Dewi", "Eko", "Fajar"], 3)
    h = r.sample(range(140, 190), 3)
    order = sorted(zip(h, names), reverse=True)
    return (f"{names[0]} tingginya {h[0]} cm, {names[1]} {h[1]} cm, {names[2]} {h[2]} cm. Siapa yang paling tinggi dan siapa yang paling pendek?",
            f"Bandingkan: {', '.join(f'{n}={x}' for x, n in zip(h, names))}. Terbesar {order[0][0]} milik {order[0][1]}; terkecil {order[-1][0]} milik {order[-1][1]}.",
            f"Paling tinggi {order[0][1]}, paling pendek {order[-1][1]}.", None)


def _seq(r):
    a, d = r.randint(1, 20), r.randint(2, 9)
    s = [a + i * d for i in range(5)]
    return (f"Lanjutkan pola: {', '.join(map(str, s))}, ...",
            f"Selisih tiap suku {s[1]} - {s[0]} = {d}, konsisten. Suku berikutnya {s[-1]} + {d} = {s[-1] + d}.", str(s[-1] + d), f"{s[-1]}+{d}")


def _syll(r):
    a, b, c = r.choice([("kucing", "mamalia", "hewan berdarah panas"), ("mawar", "tumbuhan", "makhluk hidup"),
                        ("paus", "mamalia", "hewan berdarah panas"), ("segitiga", "bangun datar", "bangun dua dimensi")])
    return (f"Semua {a} adalah {b}. Semua {b} adalah {c}. Apakah semua {a} adalah {c}?",
            f"{a} termasuk {b}, dan {b} termasuk {c}. Maka {a} termasuk {c}.", f"Ya, semua {a} adalah {c}.", None)


def _sched(r):
    h, m, dur = r.randint(7, 15), r.choice([0, 15, 30, 45]), r.choice([45, 90, 135, 150])
    tot = h * 60 + m + dur
    return (f"Rapat dimulai pukul {h:02d}.{m:02d} dan berlangsung {dur} menit. Pukul berapa selesai?",
            f"{h:02d}.{m:02d} = {h * 60 + m} menit sejak tengah malam. Tambah {dur} = {tot} menit = {tot // 60} jam {tot % 60} menit.",
            f"Pukul {tot // 60:02d}.{tot % 60:02d}", f"{h * 60 + m}+{dur}")


GENERATORS = {"arith": _arith, "shop": _shop, "speed": _speed, "unit": _unit, "compare": _compare,
              "seq": _seq, "syllogism": _syll, "schedule": _sched}
NUMERIC = {"arith", "shop", "speed", "unit", "seq", "schedule"}   # have a calculator expression to double-check


def make_problem(rng, kind=None):
    kind = kind or rng.choice(list(GENERATORS))
    q, th, ans, expr = GENERATORS[kind](rng)
    calc = expr                      # what the calculator tool can re-check (intermediate value)
    if kind == "schedule":           # final answer is a clock time, not the minute count -> graded as text
        expr = None
    return {"kind": kind, "question": q, "thinking": th, "answer": ans, "expr": expr, "calc": calc}


def to_sft(p, with_tool=False):
    """with_tool=True teaches: think -> call calculator -> use result. The tool result is real (safe_eval)."""
    if with_tool and p.get("calc"):
        call = json.dumps({"name": "calculator", "arguments": {"expression": p["calc"]}}, ensure_ascii=False)
        res = safe_eval(p["calc"])
        return {"messages": [
            {"role": "user", "content": p["question"]},
            {"role": "assistant", "content": f"<think>{p['thinking']} Saya periksa dengan kalkulator.</think>\n<tool_call>{call}</tool_call>"},
            {"role": "tool", "content": f"<tool_result name=\"calculator\">\n{res}\n</tool_result>"},
            {"role": "assistant", "content": f"{p['answer']}"}]}
    return {"messages": [{"role": "user", "content": p["question"]},
                         {"role": "assistant", "content": f"<think>{p['thinking']}</think>\n{p['answer']}"}]}


def _pick_kind(rng, kinds, weights):
    kinds = list(kinds) if kinds else list(GENERATORS)
    if not weights:
        return rng.choice(kinds)
    w = [max(0.0, float(weights.get(k, 1.0))) for k in kinds]
    return rng.choices(kinds, weights=w)[0] if sum(w) > 0 else rng.choice(kinds)


def generate_dataset(path, n=2000, seed=0, tool_fraction=0.25, kinds=None, exclude=(), kind_weights=None):
    """exclude: questions that must NOT appear (the fixed evaluation set -> no train/test leakage).
    kind_weights: {kind: weight}, e.g. emphasise the kinds the model currently gets wrong."""
    rng, seen, rows = random.Random(seed), set(exclude), 0
    with open(path, "w", encoding="utf-8") as f:
        tries = 0
        while rows < n and tries < n * 20:
            tries += 1
            p = make_problem(rng, _pick_kind(rng, kinds, kind_weights))
            if p["question"] in seen:
                continue
            seen.add(p["question"])
            f.write(json.dumps(to_sft(p, rng.random() < tool_fraction), ensure_ascii=False) + "\n")
            rows += 1
    return rows


# ---------------------------------------------------------------- inference-time helpers
def extract_final(text):
    """Visible answer after stripping hidden thinking; last non-empty line is the 'answer key' for voting."""
    _, vis = split_thinking(text)
    return vis.strip()


def _norm(a):
    return re.sub(r"[^0-9a-z.,]+", " ", a.lower()).strip()


def self_consistent(gen_fn, messages, n=5, temperature=0.8, key=extract_final, **kw):
    """Sample n reasoning paths, majority-vote the final answer. gen_fn(messages, temperature=..., seed=...) -> text.
    Returns {"answer", "votes", "agreement", "samples"}. Agreement < 0.5 means 'uncertain' — callers should say so."""
    outs = []
    for i in range(n):
        try:
            outs.append(gen_fn(messages, temperature=temperature, seed=1000 + i, **kw))
        except TypeError:
            outs.append(gen_fn(messages))
    answers = [key(o) for o in outs]
    cnt = Counter(_norm(a) for a in answers if a)
    if not cnt:
        return {"answer": "", "votes": 0, "agreement": 0.0, "samples": outs}
    top, v = cnt.most_common(1)[0]
    best = next(a for a in answers if _norm(a) == top)
    return {"answer": best, "votes": v, "agreement": v / len(answers), "samples": outs}


def _numbers(text):
    """All numbers in text under both readings of '.' (id thousands '12.000' vs en decimal '3.5') and ',' (id decimal)."""
    vals = set()
    for tok in re.findall(r"-?\d[\d.,]*", text):
        tok = tok.rstrip(".,")
        for t in (tok.replace(".", "").replace(",", "."), tok.replace(",", "")):
            try:
                vals.add(float(t))
            except ValueError:
                pass
    return vals


def verify_numeric(answer_text, expression):
    """Recompute `expression` with the calculator; True if that number appears in the answer text."""
    want = safe_eval(expression)
    return any(abs(v - want) < 1e-6 for v in _numbers(answer_text))


def fixed_eval_set(n=60, seed=777, kinds=None):
    """A FIXED list of problems (same questions every time) so scores of different model versions are comparable."""
    rng, seen, out, tries = random.Random(seed), set(), [], 0
    while len(out) < n and tries < n * 40:
        tries += 1
        p = make_problem(rng, rng.choice(kinds) if kinds else None)
        if p["question"] not in seen:
            seen.add(p["question"])
            out.append(p)
    return out


def grade(problem, output_text):
    out = extract_final(output_text)
    return bool(verify_numeric(out, problem["expr"]) if problem["expr"] else (_norm(problem["answer"]) in _norm(out)))


def evaluate_problems(gen_fn, problems, system=None):
    """gen_fn(messages) -> text. Returns accuracy overall + per kind on EXACTLY these problems."""
    per, ok_total = {}, 0
    for p in problems:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p["question"]}]
        try:
            ok = grade(p, gen_fn(msgs))
        except Exception:
            ok = False
        a = per.setdefault(p["kind"], [0, 0])
        a[0] += ok
        a[1] += 1
        ok_total += ok
    n = len(problems)
    return {"accuracy": ok_total / max(1, n), "n": n, "per_kind": {k: v[0] / v[1] for k, v in per.items()}}


def significant_change(cur, cand, n, z=2.0):
    """Difference cand-cur in accuracy, and whether it exceeds z standard errors (binomial) for n problems each."""
    import math
    pooled = max(1e-9, (cur + cand) / 2)
    se = math.sqrt(2 * pooled * (1 - pooled) / max(1, n)) if 0 < pooled < 1 else 0.0
    d = cand - cur
    return {"delta": d, "se": se, "significant": abs(d) > z * se if se > 0 else d != 0}


def evaluate_reasoning(gen_fn, n=100, seed=12345, kinds=None, system=None):
    """Held-out check with fresh seeded problems. gen_fn(messages)->text. Returns accuracy overall and per kind.
    Numeric kinds are graded by the calculator-verified expected value; text kinds by exact expected answer substring."""
    rng, per = random.Random(seed), {}
    for _ in range(n):
        p = make_problem(rng, rng.choice(kinds) if kinds else None)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p["question"]}]
        out = extract_final(gen_fn(msgs))
        ok = verify_numeric(out, p["expr"]) if p["expr"] else (_norm(p["answer"]) in _norm(out))
        a = per.setdefault(p["kind"], [0, 0])
        a[0] += ok
        a[1] += 1
    tot = sum(v[1] for v in per.values())
    return {"accuracy": sum(v[0] for v in per.values()) / max(1, tot), "n": tot,
            "per_kind": {k: v[0] / v[1] for k, v in per.items()}}
