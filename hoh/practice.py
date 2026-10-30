"""Automatic SFT "practice" inside the life loop (backend-independent orchestration).

Reading (lifelong cycle) teaches the model LANGUAGE and FACTS as plain text. Practice teaches it to ANSWER: it fine-tunes on
 (a) experience   - episodes you marked good (with their verified tool traces) and your corrections (hoh.experience.export_sft)
 (b) reasoning    - fresh calculator-verified <think> problems, weighted toward the kinds the model currently gets wrong,
                    never containing a question from the fixed evaluation set
 (c) replay       - earlier experience examples, so old skills are not overwritten
Then the candidate must pass BOTH gates before it replaces the live model:
 1. language gate  - held-out anchor bits/byte not worse than `sft_forget_tolerance`
 2. reasoning gate - score on the FIXED evaluation set not significantly lower than the current model (and not lower at all when
                     the evaluator is available); a real, significant gain bumps the MINOR version
Rejected practice leaves the live model untouched and halves the next practice learning rate.
HONESTY: this loop makes the model practise; whether it actually becomes better is decided by the measured scores, and a
small model has a ceiling. Needs a backend with train_sft/eval_reasoning (NumpyBackend has them; NOT RUN here without PyTorch)."""
import hashlib, json, math, random
from pathlib import Path
from .reasoning import generate_dataset


def _key(line):
    return hashlib.sha1(line.strip().encode("utf-8")).hexdigest()[:16]


class Practice:
    def __init__(self, lifelong, experience=None):
        self.ll, self.exp = lifelong, experience

    # ---- dataset ----
    def _weights(self):
        pk = ((self.ll.registry.get("reasoning_history") or [{}])[-1] or {}).get("per_kind") or {}
        return {k: (1.0 - acc) + 0.1 for k, acc in pk.items()} or None

    def build_dataset(self, work, seed):
        cfg, ll = self.ll.cfg, self.ll
        work.mkdir(parents=True, exist_ok=True)
        new_exp = []
        if self.exp is not None:
            tmp = work / "experience.jsonl"
            self.exp.export_sft(tmp)
            new_exp = [l for l in tmp.read_text(encoding="utf-8").splitlines() if l.strip()][-cfg.sft_experience_max:]
        pool_p = ll.dir / "sft_pool.jsonl"
        pool = [l for l in pool_p.read_text(encoding="utf-8").splitlines() if l.strip()] if pool_p.exists() else []
        known = {_key(l) for l in pool}
        fresh = [l for l in new_exp if _key(l) not in known]
        merged = (pool + fresh)[-cfg.sft_replay_max:]
        pool_p.write_text("\n".join(merged) + ("\n" if merged else ""), encoding="utf-8")
        rng = random.Random(seed)
        replay = rng.sample(pool, min(len(pool), max(len(fresh), 50)))
        excl = {p["question"] for p in ll.reasoning_problems()}
        rpath = work / "reasoning.jsonl"
        generate_dataset(rpath, cfg.sft_reasoning_n, seed, exclude=excl, kind_weights=self._weights())
        reasoning = [l for l in rpath.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = list(dict.fromkeys(fresh + replay + reasoning))
        rng.shuffle(rows)
        out = work / "sft.jsonl"
        out.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return out, {"experience_new": len(fresh), "replay": len(replay), "reasoning": len(reasoning), "total": len(rows)}

    # ---- one practice session ----
    def run(self, force=False):
        ll, cfg, st = self.ll, self.ll.cfg, self.ll.state
        if not hasattr(ll.backend, "train_sft"):
            return {"status": "skipped", "reason": "backend has no train_sft"}
        why = ll._gate(force)
        if why:
            return {"status": "skipped", "reason": why}
        cur = ll.registry["current_model"]
        if not cur:
            return {"status": "skipped", "reason": "no model yet (reading must produce the first model)"}
        if not ll._acquire():
            return {"status": "skipped", "reason": "another cycle is running"}
        try:
            return self._run(cur)
        except Exception as e:
            ll._log({"event": "practice_error", "error": f"{type(e).__name__}: {e}"})
            return {"status": "error", "error": str(e)}
        finally:
            ll._release()

    def _run(self, cur):
        ll, cfg, st = self.ll, self.ll.cfg, self.ll.state
        n = st.get("practice_runs", 0) + 1
        work = ll.dir / "work" / f"practice_{n:06d}"
        data, counts = self.build_dataset(work, 5000 + n)
        if counts["total"] < 4:
            return {"status": "skipped", "reason": "not enough practice data"}
        st["practice_runs"] = n
        per_step = cfg.batch_size * cfg.grad_accum
        steps = int(min(cfg.sft_steps_max, max(cfg.sft_steps_min, math.ceil(2 * counts["total"] / per_step))))
        rej = st.get("practice_rejections", 0)
        lr = cfg.sft_lr * (0.5 ** min(rej, 4))
        res = ll.backend.train_sft(str(data), str(work / "run"), steps, init_from=cur, lr=lr)
        cand = res["best_ckpt"]
        rep = {"event": "practice", "run": n, "steps": steps, "lr": lr, **counts, "train_loss": res.get("train_loss")}
        anchor = ll.dir / "anchor" / "anchor.txt"
        ok, why = True, "anchor too small to verify language quality"
        if anchor.exists() and anchor.stat().st_size >= 200:
            cb, ub = ll.backend.eval_bpb(cand, str(anchor)), ll.backend.eval_bpb(cur, str(anchor))
            ok = math.isfinite(cb) and cb <= ub * (1 + cfg.sft_forget_tolerance)
            why = f"anchor bpb {cb:.4f} vs {ub:.4f} (tolerance {cfg.sft_forget_tolerance:.0%})"
            rep.update(candidate_bpb=cb, current_bpb=ub)
        gate = ll.reasoning_gate(cand, cur) if ok else None
        if gate is not None:
            rep["reasoning"] = {k: gate.get(k) for k in ("cand", "cur", "delta", "gain", "ok", "applicable")}
            if not gate["ok"] or (gate.get("applicable") and gate["delta"] < 0):
                ok, why = False, f"reasoning {gate['cur']:.2f} -> {gate['cand']:.2f}; " + why
        rep["reason"] = why
        if not ok:
            st["practice_rejections"] = rej + 1
            rep["status"] = "rejected"
            ll._log(rep)
            ll._save()
            return rep
        st["practice_rejections"] = 0
        gen = ll.promote_checkpoint(cand, ll.registry["current_tokenizer"], {"practice": n})
        rep["status"], rep["generation"], rep["version"] = "promoted", gen, ll._register_version("practice", gate)
        ll._log(rep)
        ll._save()
        return rep
