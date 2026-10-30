"""Lifelong learning while idle (backend-independent orchestration).

A cycle: gate (idle? STOP? disk?) -> ingest new text -> maybe grow tokenizer -> continue training on
new + REPLAYED old data -> compare against the current model on a frozen-then-append-only ANCHOR set
-> promote only if it did not forget -> occasionally propose/run a parameter-growth experiment.

Nothing here edits code, reaches the network, or overwrites the live model: a rejected or crashed cycle
leaves the registry untouched. The numeric work is delegated to a backend object
(see hoh/lifelong_backend.py), so this logic is testable without PyTorch."""
import hashlib, json, math, os, random, re, shutil, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from .autotokenizer import TokenizerPolicy, grow_tokenizer, tokens_per_byte
from .growth_policy import propose, estimate_params, bootstrap_arch
from .tokenizer import BPETokenizer, ByteTokenizer, load_tokenizer, tokenizer_from_dict

TEXT_SUFFIXES = {".txt", ".md", ".jsonl"}


@dataclass
class LifelongConfig:
    state_dir: str = "runs/lifelong"
    inbox_dir: str = ""                      # files dropped here are learned once (.txt/.md/.jsonl)
    memory_db: str = ""                      # optional MemoryStore database to learn from
    memory_workspace: str = "default"
    memory_layers: tuple = ("knowledge", "persistent")   # add "conversation" to also learn from chats
    idle_seconds: float = 300.0              # learn only after this long without chat activity
    min_new_chars: int = 4000                # wait until this much new text is pending
    replay_ratio: float = 1.0                # old chars replayed per new char (capped by what exists)
    anchor_every: int = 20                   # 1 in N paragraphs is held out for evaluation, never trained on
    min_anchor_chars: int = 3000             # until the anchor has this much, hold out more aggressively
    epochs_new: float = 2.0
    min_steps: int = 20
    max_steps: int = 500
    batch_size: int = 8
    grad_accum: int = 4
    block_size: int = 512
    lr: float = 1e-4
    forget_tolerance: float = 0.02           # candidate may be at most 2% worse (bits/byte) on the anchor
    keep_models: int = 5
    growth_every_cycles: int = 3             # minimum cycles between growth attempts (doubles after a rejection)
    growth_steps: int = 200
    growth_min_improvement: float = 0.01
    max_params: int = 0                      # 0 = no ceiling; the hardware budget below still applies
    memory_budget_bytes: int = 0             # 0 = ~50% of free RAM (estimate)
    bootstrap_min_chars: int = 50_000        # no pretraining: the FIRST model waits until this much text was gathered
    auto_arch: bool = True                   # first model's size follows the amount of data (see growth_policy.bootstrap_arch)
    arch_tokens_per_param: float = 20.0
    growth_min_tokens_per_param: float = 8.0  # parameters are only added once unique data supports them
    hard_replay_fraction: float = 0.5        # share of replay chosen by LOSS (what the model still gets wrong)
    hard_pool: int = 300                     # candidates scored per cycle (scoring costs a forward pass each)
    noise_drop: float = 0.1                  # ignore the hardest 10% of the pool: usually gibberish, not knowledge
    # --- reasoning gate + versions + automatic SFT practice ---
    reasoning_eval_n: int = 60               # size of the FIXED evaluation set (same questions every time, never trained on)
    reasoning_tolerance: float = 0.05        # a candidate may not lose more than this much accuracy ...
    reasoning_z_drop: float = 2.0            # ... and only a drop beyond this many standard errors counts as real
    reasoning_z_gain: float = 1.0
    version_min_gain: float = 0.02           # minor version goes up only with at least this much (significant) reasoning gain
    reasoning_gate_reads: bool = False       # also evaluate reasoning on plain-text reading cycles (costs generation time)
    reasoning_gate_growth: bool = True       # a grown model must not reason worse than its parent
    sft_steps_min: int = 30
    sft_steps_max: int = 400
    sft_lr: float = 5e-5
    sft_forget_tolerance: float = 0.05       # practice may shift the language-model score a bit more than reading may
    sft_experience_max: int = 2000
    sft_reasoning_n: int = 400
    sft_replay_max: int = 3000
    practice_every_cycles: int = 1
    min_free_disk_bytes: int = 1 << 30
    lock_stale_seconds: float = 6 * 3600
    tokenizer: TokenizerPolicy = field(default_factory=TokenizerPolicy)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["tokenizer"] = TokenizerPolicy(**d.get("tokenizer", {}))
        d["memory_layers"] = tuple(d.get("memory_layers", ("knowledge", "persistent")))
        return cls(**d)


def _atomic_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _read_json(path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def touch_activity(state_dir):
    """Call whenever the model is being used for chat; the learner stays out of the way."""
    p = Path(state_dir) / "activity"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(time.time()), encoding="utf-8")


def available_ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 2 << 30


def _read_any(path):
    """Torch-free reader for .txt/.md/.jsonl (strict UTF-8; bad JSONL raises with file:line)."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        from .sft import read_jsonl, record_to_text
        return "\n\n".join(record_to_text(obj, f"{p}:{n}") for n, obj in read_jsonl(p))
    return p.read_text(encoding="utf-8-sig")


def _paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _phash(p):
    return hashlib.sha1(" ".join(p.split()).lower().encode("utf-8")).hexdigest()[:20]


class Lifelong:
    def __init__(self, cfg, backend, memory=None, clock=time.time):
        self.cfg, self.backend, self.memory, self.clock = cfg, backend, memory, clock
        self.dir = Path(cfg.state_dir)
        for sub in ("corpus", "anchor", "tokenizers", "models", "work"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.state = _read_json(self.dir / "state.json", {
            "cycle": 0, "ingested_files": {}, "memory_cursor": 0.0, "unlearned": [], "learned": [],
            "last_growth_cycle": 0, "growth_wait": cfg.growth_every_cycles, "rejections": 0, "last_skip": ""})
        self.registry = _read_json(self.dir / "registry.json", {
            "generation": 0, "current_model": "", "current_tokenizer": "", "milestones": [], "history": []})
        for k, v in (("version", [0, 0, 0]), ("best_reasoning", 0.0), ("reasoning", None), ("reasoning_history", [])):
            self.registry.setdefault(k, v)
        _atomic_json(self.dir / "config.json", cfg.to_dict())

    # ---------------- control files ----------------
    @property
    def stop_file(self):
        return self.dir / "STOP"

    def pause(self):
        self.stop_file.write_text(str(self.clock()), encoding="utf-8")

    def resume(self):
        self.stop_file.unlink(missing_ok=True)

    def idle_for(self):
        p = self.dir / "activity"
        return self.clock() - float(p.read_text()) if p.exists() else float("inf")

    def _save(self):
        _atomic_json(self.dir / "state.json", self.state)
        _atomic_json(self.dir / "registry.json", self.registry)

    def _log(self, row):
        row = {"time": self.clock(), **row}
        with (self.dir / "cycles.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------- locking ----------------
    def _acquire(self):
        lock = self.dir / "lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self.clock() - lock.stat().st_mtime > self.cfg.lock_stale_seconds:   # crashed earlier run
                lock.unlink(missing_ok=True)
                return self._acquire()
            return False
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True

    def _release(self):
        (self.dir / "lock").unlink(missing_ok=True)

    # ---------------- ingestion ----------------
    def _seen(self):
        p = self.dir / "seen.txt"
        return set(p.read_text().split()) if p.exists() else set()

    def _anchor_chars(self):
        return sum(f.stat().st_size for f in (self.dir / "anchor").glob("*.txt"))

    def _collect(self):
        """Gather candidate paragraphs WITHOUT changing any state; returns a commit payload."""
        cfg, seen = self.cfg, self._seen()
        found, files, cursor = [], {}, self.state["memory_cursor"]
        if cfg.inbox_dir and Path(cfg.inbox_dir).is_dir():
            for f in sorted(Path(cfg.inbox_dir).rglob("*")):
                if not (f.is_file() and f.suffix.lower() in TEXT_SUFFIXES):
                    continue
                sig = f"{f.stat().st_size}:{hashlib.sha1(f.read_bytes()).hexdigest()}"
                if self.state["ingested_files"].get(str(f)) == sig:
                    continue
                try:
                    found += _paragraphs(_read_any(f))
                    files[str(f)] = sig
                except Exception as e:                  # a bad file must not block learning from the rest
                    self._log({"event": "inbox_skip", "file": str(f), "error": str(e)})
        if self.memory is not None:
            rows = self.memory.iter_since(cfg.memory_workspace, cfg.memory_layers, cursor)
            for r in rows:
                found += _paragraphs(r["content"])
                cursor = max(cursor, r["created"])
        train, anchor, hashes, anchor_len = [], [], [], self._anchor_chars()
        for p in found:
            h = _phash(p)
            if h in seen:
                continue
            seen.add(h)
            hashes.append(h)
            n = int(h, 16)
            to_anchor = n % cfg.anchor_every == 0 or (anchor_len < cfg.min_anchor_chars and n % 4 == 0)
            if to_anchor:
                anchor.append(p)
                anchor_len += len(p)
            else:
                train.append(p)
        return {"train": train, "anchor": anchor, "hashes": hashes, "files": files, "cursor": cursor}

    def _commit(self, c):
        n = self.state["cycle"] + 1
        self.state["unique_bytes"] = self.state.get("unique_bytes", 0) + sum(len(p.encode("utf-8")) for p in c["train"] + c["anchor"])
        if c["train"]:
            name = f"shard_{n:06d}.txt"
            (self.dir / "corpus" / name).write_text("\n\n".join(c["train"]) + "\n", encoding="utf-8")
            self.state["unlearned"].append(name)
        if c["anchor"]:
            with (self.dir / "anchor" / "anchor.txt").open("a", encoding="utf-8") as f:
                f.write("\n\n".join(c["anchor"]) + "\n\n")
        with (self.dir / "seen.txt").open("a", encoding="utf-8") as f:
            f.write("\n".join(c["hashes"]) + ("\n" if c["hashes"] else ""))
        self.state["ingested_files"].update(c["files"])
        self.state["memory_cursor"] = c["cursor"]
        self._save()

    def _shard_text(self, names):
        return "\n\n".join((self.dir / "corpus" / n).read_text(encoding="utf-8") for n in names)

    def _replay(self, budget_chars, rng, cur_model=None):
        names = list(self.state["learned"])
        out, used, cache = [], 0, {}
        out, used = self._hard_replay(names, budget_chars, rng, cur_model, cache)
        while names and used < budget_chars:
            n = rng.choice(names)
            if n not in cache:
                cache[n] = _paragraphs((self.dir / "corpus" / n).read_text(encoding="utf-8"))
            if not cache[n]:
                names.remove(n)
                continue
            p = rng.choice(cache[n])
            out.append(p)
            used += len(p)
        return out

    def _hard_replay(self, names, budget_chars, rng, cur_model, cache):
        """Loss-prioritised replay: score a random pool with the CURRENT model and rehearse what it still gets wrong
        (excluding the very hardest slice, which is mostly noise). Needs backend.score_paragraphs; otherwise a no-op."""
        cfg = self.cfg
        if not (cur_model and names and cfg.hard_replay_fraction > 0 and hasattr(self.backend, "score_paragraphs")):
            return [], 0
        pool = []
        for _ in range(cfg.hard_pool * 3):
            n = rng.choice(names)
            if n not in cache:
                cache[n] = _paragraphs((self.dir / "corpus" / n).read_text(encoding="utf-8"))
            if cache[n]:
                pool.append(rng.choice(cache[n]))
            if len(pool) >= cfg.hard_pool:
                break
        pool = list(dict.fromkeys(pool))
        if not pool:
            return [], 0
        scores = self.backend.score_paragraphs(cur_model, pool)
        ranked = sorted((s, p) for s, p in zip(scores, pool) if math.isfinite(s))
        ranked = ranked[: max(1, int(len(ranked) * (1 - cfg.noise_drop)))]        # drop hardest slice (noise)
        out, used, want = [], 0, budget_chars * cfg.hard_replay_fraction
        for _, p in reversed(ranked):                                             # hardest of what is left first
            if used >= want:
                break
            out.append(p)
            used += len(p)
        return out, used

    # ---------------- helpers ----------------
    def _gate(self, force):
        if self.stop_file.exists():
            return "paused (STOP file)"
        if not force and self.idle_for() < self.cfg.idle_seconds:
            return "busy: chat activity within idle window"
        if shutil.disk_usage(self.dir).free < self.cfg.min_free_disk_bytes:
            return "disk nearly full"
        return ""

    def _budget(self):
        return self.cfg.memory_budget_bytes or int(available_ram_bytes() * 0.5)

    def _current_tokenizer(self):
        p = self.registry["current_tokenizer"]
        return load_tokenizer(p) if p else ByteTokenizer()

    # ---------------- reasoning gate + versions ----------------
    @property
    def version(self):
        return ".".join(str(x) for x in self.registry["version"])

    def reasoning_problems(self):
        """The fixed evaluation set (created once, then reused so versions stay comparable)."""
        p = self.dir / "reasoning_eval.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        from .reasoning import fixed_eval_set
        probs = fixed_eval_set(self.cfg.reasoning_eval_n)
        _atomic_json(p, probs)
        return probs

    def measure_reasoning(self, ckpt):
        """-> {'accuracy','n','per_kind'} or None when the backend cannot generate (e.g. fake backend without eval_reasoning)."""
        if not ckpt or not hasattr(self.backend, "eval_reasoning"):
            return None
        return self.backend.eval_reasoning(ckpt, self.reasoning_problems())

    def reasoning_gate(self, cand_ckpt, cur_ckpt=None, cur_score=None):
        """Decide whether the candidate reasons at least as well as the current model.
        -> {'ok','gain','cand','cur','delta','n'} ; ok=True when no evaluator exists (gate not applicable)."""
        from .reasoning import significant_change
        cand = self.measure_reasoning(cand_ckpt)
        if cand is None:
            return {"ok": True, "gain": False, "applicable": False}
        if cur_score is None:
            cur_score = self.registry.get("reasoning")
            if cur_score is None and cur_ckpt:
                m = self.measure_reasoning(cur_ckpt)
                cur_score = m["accuracy"] if m else 0.0
        cur_score = cur_score or 0.0
        n = cand["n"]
        d = significant_change(cur_score, cand["accuracy"], n, self.cfg.reasoning_z_drop)
        drop = d["delta"] < -self.cfg.reasoning_tolerance and d["significant"]
        g = significant_change(cur_score, cand["accuracy"], n, self.cfg.reasoning_z_gain)
        best = self.registry.get("best_reasoning", 0.0)
        gain = cand["accuracy"] - best >= self.cfg.version_min_gain and (cand["accuracy"] - best) > self.cfg.reasoning_z_gain * g["se"]
        return {"ok": not drop, "gain": bool(gain), "applicable": True, "cand": cand["accuracy"], "cur": cur_score,
                "delta": d["delta"], "n": n, "per_kind": cand["per_kind"]}

    def _register_version(self, kind, gate=None):
        """Bump the version of the model that was just promoted.
        patch +1 : any promoted update;  minor +1 (patch=0): reasoning improved by a real margin;  major +1: grown model that passed the gate."""
        v = self.registry["version"]
        if kind == "growth" and (gate is None or gate.get("ok")):
            v[0], v[1], v[2] = v[0] + 1, 0, 0
        elif gate and gate.get("applicable") and gate.get("gain"):
            v[1], v[2] = v[1] + 1, 0
        else:
            v[2] += 1
        if gate and gate.get("applicable"):
            self.registry["reasoning"] = gate["cand"]
            self.registry["best_reasoning"] = max(self.registry.get("best_reasoning", 0.0), gate["cand"])
            self.registry["reasoning_history"] = (self.registry["reasoning_history"] + [
                {"version": self.version, "accuracy": gate["cand"], "n": gate["n"], "per_kind": gate.get("per_kind")}])[-50:]
        return self.version

    def promote_checkpoint(self, cand, tok_path, extra_history=None, milestone=False):
        """Copy a candidate into models/, make it current. Returns the new generation number."""
        gen = self.registry["generation"] + 1
        dst = self.dir / "models" / f"model_{gen:04d}.npz"
        shutil.copy2(cand, dst)
        tok_dst = self.dir / "tokenizers" / f"tok_v{gen:04d}.json"
        if tok_path:
            shutil.copy2(tok_path, tok_dst)
            self.registry["current_tokenizer"] = str(tok_dst)
        self.registry.update(generation=gen, current_model=str(dst))
        if milestone:
            self.registry["milestones"].append(str(dst))
        self.registry["history"].append({"generation": gen, "cycle": self.state["cycle"], **(extra_history or {})})
        self._prune_models()
        return gen

    # ---------------- the cycle ----------------
    def run_cycle(self, force=False):
        """Run at most one cycle. Returns a dict with 'status' in {skipped, promoted, rejected, error}."""
        why = self._gate(force)
        if why:
            return self._skip(why)
        if not self._acquire():
            return self._skip("another cycle is running")
        try:
            return self._cycle(force)
        except Exception as e:                                # a failed cycle never touches the live model
            self._log({"event": "error", "error": f"{type(e).__name__}: {e}"})
            self.state["last_skip"] = f"error: {e}"
            self._save()
            return {"status": "error", "error": str(e)}
        finally:
            self._release()

    def _skip(self, why):
        self.state["last_skip"] = why
        self._save()
        return {"status": "skipped", "reason": why}

    def _cycle(self, force):
        cfg, st = self.cfg, self.state
        c = self._collect()
        pending = sum(len(p) for p in c["train"]) + sum(len(self._shard_text([n])) for n in st["unlearned"])
        if pending < cfg.min_new_chars and not force:
            return self._skip(f"only {pending} new chars pending (< {cfg.min_new_chars})")
        if pending == 0:
            return self._skip("nothing to learn")
        if not self.registry["current_model"] and pending < cfg.bootstrap_min_chars and not force:
            return self._skip(f"gathering first data: {pending} of {cfg.bootstrap_min_chars} chars needed to start the first model")
        self._commit(c)
        st["cycle"] += 1
        n = st["cycle"]
        work = self.dir / "work" / f"cycle_{n:06d}"
        work.mkdir(parents=True, exist_ok=True)
        rng = random.Random(1000 + n)
        new_paras = _paragraphs(self._shard_text(st["unlearned"]))
        new_chars = sum(len(p) for p in new_paras)
        replay = self._replay(int(new_chars * cfg.replay_ratio), rng, self.registry["current_model"])
        mix = new_paras + replay
        rng.shuffle(mix)
        train_path = work / "train.txt"
        train_path.write_text("\n\n".join(mix) + "\n", encoding="utf-8")
        anchor_path = self.dir / "anchor" / "anchor.txt"

        cur_model, cur_tok = self.registry["current_model"], self._current_tokenizer()
        rep = {"cycle": n, "new_chars": new_chars, "replay_chars": sum(len(p) for p in replay)}

        # 1) tokenizer: grow only when it measurably compresses held-out text better
        cand_tok, tok_rep = grow_tokenizer(cur_tok, mix, cfg.tokenizer)
        rep["tokenizer"] = tok_rep
        tok_path = work / "tokenizer.json"
        tok_path.write_text(json.dumps(cand_tok.to_dict()), encoding="utf-8")
        tok_changed = cand_tok.vocab_size != cur_tok.vocab_size

        # 2) starting point for the candidate
        if not cur_model:
            init = None
        elif tok_changed:
            init = str(work / "migrated.npz")
            self.backend.migrate_vocab(cur_model, str(tok_path), init)
        else:
            init = cur_model

        # 3) train on new + replay
        tpb = tokens_per_byte(cand_tok, mix[: max(10, min(len(mix), 200))]) or 0.3
        tokens = sum(len(p.encode("utf-8")) for p in mix) * tpb
        per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
        steps = int(min(cfg.max_steps, max(cfg.min_steps, math.ceil(cfg.epochs_new * tokens / per_step))))
        lr = cfg.lr * (0.5 ** min(st["rejections"], 4))      # repeated rejections -> gentler updates
        kw = {}
        if init is None and cfg.auto_arch:
            u_tokens = st.get("unique_bytes", 0) * tpb
            kw["arch"] = bootstrap_arch(u_tokens, cand_tok.vocab_size, cfg.block_size, cfg.arch_tokens_per_param)
            rep["arch"] = kw["arch"]
        st["tpb"] = tpb
        res = self.backend.train(str(train_path), str(work / "train_run"), steps, init_from=init,
                                 tokenizer_path=None if init else str(tok_path), lr=lr, **kw)
        cand = res["best_ckpt"]
        rep.update(steps=steps, lr=lr, train_loss=res.get("train_loss"), val_loss=res.get("val_loss"))

        # 4) judge on the anchor (bits/byte, comparable across tokenizers)
        if not anchor_path.exists() or anchor_path.stat().st_size < 200:
            verdict, why = (not cur_model), "anchor too small to verify; only the first model is accepted unverified"
            cand_bpb = cur_bpb = None
        else:
            cand_bpb = self.backend.eval_bpb(cand, str(anchor_path))
            cur_bpb = self.backend.eval_bpb(cur_model, str(anchor_path)) if cur_model else None
            if not math.isfinite(cand_bpb):
                verdict, why = False, "candidate loss is not finite"
            elif cur_bpb is None:
                verdict, why = True, "first model (no baseline)"
            else:
                verdict = cand_bpb <= cur_bpb * (1 + cfg.forget_tolerance)
                why = f"anchor bpb {cand_bpb:.4f} vs current {cur_bpb:.4f} (tolerance {cfg.forget_tolerance:.0%})"
        rep.update(candidate_bpb=cand_bpb, current_bpb=cur_bpb, reason=why)
        gate = None
        if verdict and cfg.reasoning_gate_reads and cur_model:
            gate = self.reasoning_gate(cand, cur_model)
            rep["reasoning"] = {k: gate.get(k) for k in ("cand", "cur", "delta", "ok")}
            if not gate["ok"]:
                verdict, rep["reason"] = False, f"reasoning dropped {gate['cur']:.2f} -> {gate['cand']:.2f} (gate); " + why

        if not verdict:
            st["rejections"] += 1
            rep["status"] = "rejected"
            self._log({"event": "cycle", **rep})
            self._save()
            return rep

        # 5) promote
        st["rejections"] = 0
        gen = self.registry["generation"] + 1
        dst = self.dir / "models" / f"model_{gen:04d}.npz"
        shutil.copy2(cand, dst)
        tok_dst = self.dir / "tokenizers" / f"tok_v{gen:04d}.json"
        shutil.copy2(tok_path, tok_dst)
        self.registry.update(generation=gen, current_model=str(dst), current_tokenizer=str(tok_dst))
        if tok_changed:
            self.registry["milestones"].append(str(dst))
        st["learned"] += st["unlearned"]
        st["unlearned"] = []
        self.registry["history"].append({"generation": gen, "cycle": n, "bpb": cand_bpb, "tokenizer_added": tok_rep["added"]})
        rep["status"], rep["generation"] = "promoted", gen
        rep["version"] = self._register_version("read", gate)
        self._prune_models()
        self._save()

        # 6) history for the growth policy + occasional growth experiment
        if cand_bpb is not None and res.get("train_loss") is not None:
            tr_bpb = res["train_loss"] * tpb / math.log(2)
            with (self.dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": n, "train_loss": tr_bpb, "val_loss": cand_bpb}) + "\n")
        rep["growth"] = self._maybe_grow(train_path, anchor_path)
        self._log({"event": "cycle", **rep})
        self._save()
        return rep

    def _prune_models(self):
        keep = set(self.registry["milestones"]) | {self.registry["current_model"]}
        files = sorted((self.dir / "models").glob("model_*.npz"))
        for f in files[:-self.cfg.keep_models]:
            if str(f) not in keep:
                f.unlink(missing_ok=True)

    def _maybe_grow(self, data_path, anchor_path):
        cfg, st = self.cfg, self.state
        if st["cycle"] - st["last_growth_cycle"] < st["growth_wait"]:
            return {"attempted": False, "why": "cooldown"}
        hist = self.dir / "history.jsonl"
        rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()][-8:] if hist.exists() else []
        mc = self.backend.model_config(self.registry["current_model"])
        u_tokens = st.get("unique_bytes", 0) * st.get("tpb", 0.3)
        prop = propose(rows, mc, cfg.max_params or 10 ** 15, cfg.batch_size, unique_tokens=u_tokens,
                       min_tokens_per_param=cfg.growth_min_tokens_per_param)
        if not prop.grow:
            return {"attempted": False, "why": prop.reason}
        if prop.est_train_mem_bytes > self._budget():
            return {"attempted": False, "why": f"estimated {prop.est_train_mem_bytes:,} B exceeds hardware budget "
                                                f"{self._budget():,} B"}
        st["last_growth_cycle"] = st["cycle"]
        report = self.backend.grow(self.registry["current_model"], str(data_path), str(self.dir / "growth"),
                                   prop.kind, prop.amount, cfg.growth_steps, cfg.growth_min_improvement,
                                   cfg.max_params or None, str(anchor_path))
        gate = None
        if report.get("decision") == "promoted" and cfg.reasoning_gate_growth:
            gate = self.reasoning_gate(report["promoted_to"], self.registry["current_model"])
            if not gate["ok"]:
                report = {**report, "decision": "rejected",
                          "reason": f"reasoning gate: {gate['cur']:.2f} -> {gate['cand']:.2f} (grown model reasons worse)"}
        if report.get("decision") == "promoted":
            gen = self.registry["generation"] + 1
            dst = self.dir / "models" / f"model_{gen:04d}.npz"
            shutil.copy2(report["promoted_to"], dst)
            self.registry.update(generation=gen, current_model=str(dst))
            self.registry["milestones"].append(str(dst))
            self.registry["history"].append({"generation": gen, "cycle": st["cycle"], "growth": prop.kind,
                                             "params_before": report.get("params_before"),
                                             "params_after": report.get("params_after")})
            st["growth_wait"] = cfg.growth_every_cycles
            self._register_version("growth", gate)
            hist.rename(self.dir / f"history_before_gen{gen:04d}.jsonl")     # capacity changed: new baseline
        else:
            st["growth_wait"] = min(st["growth_wait"] * 2, 64)               # back off after a rejection
        return {"attempted": True, "kind": prop.kind, **{k: report.get(k) for k in ("decision", "reason")}}

    def adopt(self, ckpt_path, tokenizer_dict):
        """Start from an existing checkpoint (e.g. a model you already trained) instead of from scratch."""
        if self.registry["current_model"]:
            raise ValueError("a model is already registered; adopt() is only for a fresh state dir")
        dst = self.dir / "models" / "model_0000.npz"
        shutil.copy2(ckpt_path, dst)
        tok = self.dir / "tokenizers" / "tok_v0000.json"
        tok.write_text(json.dumps(tokenizer_dict), encoding="utf-8")
        self.registry.update(generation=0, current_model=str(dst), current_tokenizer=str(tok))
        self.registry["milestones"].append(str(dst))
        self._save()

    # ---------------- loop + status ----------------
    def run_forever(self, poll_seconds=30.0, max_cycles=None, sleep=time.sleep):
        done = 0
        while max_cycles is None or done < max_cycles:
            r = self.run_cycle()
            if r["status"] != "skipped":
                done += 1
            sleep(poll_seconds if r["status"] in ("skipped", "error") else 1.0)

    def status(self):
        cur = self.registry["current_model"]
        return {"generation": self.registry["generation"], "current_model": cur,
                "vocab_size": self._current_tokenizer().vocab_size, "cycles": self.state["cycle"],
                "unlearned_shards": len(self.state["unlearned"]), "learned_shards": len(self.state["learned"]),
                "anchor_chars": self._anchor_chars(), "paused": self.stop_file.exists(),
                "idle_seconds_now": self.idle_for(), "last_skip": self.state["last_skip"],
                "growth_wait_cycles": self.state["growth_wait"], "version": self.version,
                "reasoning": self.registry.get("reasoning"), "best_reasoning": self.registry.get("best_reasoning")}
