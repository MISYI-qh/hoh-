"""HOH's life outside chat: explore (self-directed reading) -> sleep (consolidate into weights) -> reflect (decide what to study).

    awake   someone chatted recently  -> do nothing, never compete for the GPU
    explore idle + explore budget      -> Explorer reads the web/inbox for active goals (politely, budgeted, robots.txt)
    sleep   new text pending           -> Lifelong cycle: grow tokenizer if measurably useful, train on new + loss-prioritised
                                          replay, promote only if the held-out anchor did not get worse, grow parameters only
                                          if plateaued AND data supports it
    reflect after a cycle              -> failures and frequent unfamiliar terms become new goals; saturated goals retire

Chat is just one input. Everything is reversible (registry + STOP file) and bounded (daily budgets). Torch-free orchestration;
the real training is the backend's job. Whether it gets smarter is decided by the anchor numbers, not by this loop."""
import json, os, time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class AutonomyConfig:
    explore_interval: float = 3600.0      # seconds between exploration sessions
    explore: bool = True                  # False = only sleep on what is already in the inbox
    reflect: bool = True
    seed_from_text: int = 3               # curiosity goals per session
    max_goal_queue: int = 200             # stop inventing goals beyond this
    practice: bool = True                 # SFT on experience + verified reasoning after each reading cycle (needs a model)
    practice_interval: float = 21600.0    # ... and at least this often even when nothing new was read (experience may have piled up)
    practice_every_sessions: int = 1      # run practice every N sleep phases that produced/kept a model
    maintenance_interval: float = 86400.0 # memory consolidation + semantic re-training + quarantine re-check
    consolidate_days: float = 14.0
    semantic_path: str = ""               # where the RandomIndex embedder lives; empty = do not retrain it
    watch_paths: tuple = ()               # extra folders/files/URLs learned from every explore interval (dedup by hash)


class Autonomy:
    def __init__(self, lifelong, curriculum=None, explorer=None, experience=None, cfg=None, clock=time.time, sleep=time.sleep,
                 practice=None, memory=None, ingestor=None):
        self.ll, self.cur, self.explorer, self.exp = lifelong, curriculum, explorer, experience
        self.practice, self.memory, self.ing = practice, memory, ingestor
        self.cfg = cfg or AutonomyConfig()
        self.clock, self.sleep = clock, sleep
        self.path = Path(lifelong.dir) / "autonomy.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"last_explore": 0.0, "sessions": 0, "log": []}
        self.state.setdefault("last_maintenance", 0.0)
        self.state.setdefault("sleeps", 0)
        self.state.setdefault("last_practice", 0.0)

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def _note(self, row):
        self.state["log"] = (self.state["log"] + [{"t": self.clock(), **row}])[-50:]
        self._save()

    def step(self):
        """One decision. Returns {'phase': ..., ...}; never raises for ordinary failures (network, bad page, backend error)."""
        if self.ll.stop_file.exists():
            return {"phase": "paused"}
        if self.ll.idle_for() < self.ll.cfg.idle_seconds:
            return {"phase": "awake", "idle_for": self.ll.idle_for()}
        out = {"phase": "idle"}
        due = self.clock() - self.state["last_explore"] >= self.cfg.explore_interval
        if self.cfg.explore and self.explorer is not None and self.cur is not None and due:
            self.state["last_explore"] = self.clock()
            self.state["sessions"] += 1
            try:
                st = self.explorer.explore()
            except Exception as e:
                st = {"error": f"{type(e).__name__}: {e}", "accepted": 0, "files": []}
            st["watched"] = self._watch()
            out.update(phase="explore", explore={k: st.get(k) for k in ("pages", "accepted", "chars", "stopped", "goals", "error", "watched")})
            self._note({"event": "explore", **out["explore"]})
            if self.cfg.reflect:
                out["reflect"] = self.reflect(st)
        cyc = self.ll.run_cycle()
        if cyc["status"] != "skipped":
            out["phase"] = "sleep"
            out["cycle"] = {k: cyc.get(k) for k in ("status", "generation", "version", "reason", "candidate_bpb", "current_bpb", "arch", "growth")}
            self._note({"event": "sleep", **out["cycle"]})
            self.state["sleeps"] += 1
        else:
            out["sleep_skipped"] = cyc["reason"]
        slept = cyc["status"] != "skipped"
        due_p = self.clock() - self.state["last_practice"] >= self.cfg.practice_interval
        if self.practice is not None and self.cfg.practice and self.ll.registry["current_model"] and (
                (slept and self.state["sleeps"] % max(1, self.cfg.practice_every_sessions) == 0) or due_p):
            self.state["last_practice"] = self.clock()
            pr = self.practice.run()
            if pr["status"] != "skipped":
                out["phase"] = "sleep"
                out["practice"] = {k: pr.get(k) for k in ("status", "generation", "version", "reason", "reasoning", "total")}
                self._note({"event": "practice", **out["practice"]})
        if self.clock() - self.state["last_maintenance"] >= self.cfg.maintenance_interval and self.ll.idle_for() >= self.ll.cfg.idle_seconds:
            out["maintenance"] = self.maintain()
        return out

    def _watch(self):
        got = 0
        if self.ing is not None:
            for p in self.cfg.watch_paths:
                try:
                    got += self.ing.learn(str(p)).get("accepted", 0)
                except Exception as e:                       # a bad path must not stop the loop
                    self._note({"event": "watch_error", "path": str(p), "error": str(e)})
        return got

    def maintain(self):
        """'Deep sleep': consolidate old notes, re-learn word meanings, re-check quarantined documents. Never raises."""
        self.state["last_maintenance"] = self.clock()
        rep = {}
        try:
            if self.memory is not None:
                from .consolidate import consolidate
                rep["consolidate"] = consolidate(self.memory, self.ll.cfg.memory_workspace, self.cfg.consolidate_days)
                if self.cfg.semantic_path:
                    from .semantic import train_from_memory
                    rep["semantic"] = train_from_memory(self.memory, self.ll.cfg.memory_workspace, self.cfg.semantic_path)
            if self.ing is not None:
                rep["quarantine"] = self.ing.recheck_quarantine()
        except Exception as e:
            rep["error"] = f"{type(e).__name__}: {e}"
        self._note({"event": "maintenance", **{k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "summary_ids"}) for k, v in rep.items()}})
        return rep

    def reflect(self, explore_stats):
        """Turn what happened into the next goals. Heuristic; it adds work, it does not judge truth."""
        added = 0
        if self.cur is None or len(self.cur.goals("active")) >= self.cfg.max_goal_queue:
            return {"goals_added": 0}
        if self.exp is not None:
            added += self.cur.seed_from_experience(self.exp)
        inbox = Path(self.ll.cfg.inbox_dir) if self.ll.cfg.inbox_dir else None
        if inbox:
            for f in explore_stats.get("files", [])[:5]:
                try:
                    added += self.cur.seed_from_text((inbox / f).read_text(encoding="utf-8"), k=self.cfg.seed_from_text)
                except OSError:
                    pass
        return {"goals_added": added, "saturated": len(self.cur.goals("saturated"))}

    def run_forever(self, poll_seconds=60.0, max_steps=None):
        n = 0
        while max_steps is None or n < max_steps:
            try:
                r = self.step()
            except Exception as e:                       # the daemon must outlive any single failure
                self._note({"event": "error", "error": f"{type(e).__name__}: {e}"})
                r = {"phase": "error"}
            n += 1
            self.sleep(1.0 if r["phase"] in ("explore", "sleep") else poll_seconds)

    def status(self):
        d = {"autonomy": {"sessions": self.state["sessions"], "last_explore": self.state["last_explore"],
                          "recent": self.state["log"][-5:]}, "lifelong": self.ll.status()}
        if self.cur is not None:
            d["goals"] = {s: len(self.cur.goals(s)) for s in ("active", "saturated")}
            d["today"] = dict(zip(("day", "pages", "chars"), self.cur.day_usage()))
        return d
