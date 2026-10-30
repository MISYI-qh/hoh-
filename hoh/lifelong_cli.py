import argparse, json
from pathlib import Path
from .lifelong import Lifelong, LifelongConfig, touch_activity


def _build(a):
    sd = Path(a.state_dir)
    cfg_file = sd / "config.json"
    cfg = LifelongConfig.from_dict(json.loads(cfg_file.read_text())) if cfg_file.exists() else LifelongConfig()
    cfg.state_dir = a.state_dir
    for flag, attr in (("inbox", "inbox_dir"), ("memory_db", "memory_db"), ("workspace", "memory_workspace"),
                       ("idle_seconds", "idle_seconds"), ("max_params", "max_params"), ("block_size", "block_size")):
        v = getattr(a, flag, None)
        if v is not None:
            setattr(cfg, attr, v)
    if getattr(a, "memory_budget_mb", None) is not None:
        cfg.memory_budget_bytes = int(a.memory_budget_mb) << 20
    if getattr(a, "learn_conversations", False):
        cfg.memory_layers = ("knowledge", "persistent", "conversation")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hoh lifelong", description="Idle-time continual learning for HOH.")
    ap.add_argument("action", choices=["run", "once", "status", "pause", "resume", "touch", "life"])
    ap.add_argument("--no-explore", action="store_true", help="life: sleep on the inbox only, never browse")
    ap.add_argument("--explore-interval", type=float, default=3600.0)
    ap.add_argument("--max-pages", type=int, default=15); ap.add_argument("--daily-pages", type=int, default=150)
    ap.add_argument("--deny-host", action="append", default=[])
    ap.add_argument("--experience-db", default="")
    ap.add_argument("--state-dir", default="runs/lifelong")
    ap.add_argument("--inbox"); ap.add_argument("--memory-db"); ap.add_argument("--workspace")
    ap.add_argument("--learn-conversations", action="store_true",
                    help="also learn from the 'conversation' memory layer (off by default: chats may hold secrets/injected text)")
    ap.add_argument("--idle-seconds", type=float); ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-params", type=int); ap.add_argument("--memory-budget-mb", type=int)
    ap.add_argument("--block-size", type=int); ap.add_argument("--device", default="auto")
    ap.add_argument("--width", type=int, default=512); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8); ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--initial-checkpoint", default="", help="adopt an existing HOH checkpoint on first run")
    ap.add_argument("--threads", default="auto", help="BLAS threads for NumPy matrix products ('auto' = all CPUs; needs threadpoolctl)")
    ap.add_argument("--no-practice", action="store_true", help="life: do not run automatic SFT practice (experience + reasoning)")
    ap.add_argument("--practice-interval", type=float, default=21600.0)
    ap.add_argument("--maintenance-interval", type=float, default=86400.0)
    ap.add_argument("--watch", action="append", default=[], help="life: extra folder/file/URL to learn from on every explore interval")
    ap.add_argument("--semantic-path", default="", help="life: file for the word-meaning embedder (default <state-dir>/semantic.bin)")
    from .wiring import add_learning_flags
    add_learning_flags(ap)
    ap.add_argument("--force", action="store_true", help="ignore idle/min-new-chars gates (not STOP)")
    a = ap.parse_args(argv)
    if a.action == "touch":
        touch_activity(a.state_dir); return 0
    cfg = _build(a)
    memory = None
    if cfg.memory_db:
        from .memory import MemoryStore
        memory = MemoryStore(cfg.memory_db)
    backend = None
    if a.action in ("run", "once", "life"):
        from .lifelong_backend import NumpyBackend
        backend = NumpyBackend(a.device, "auto", cfg.batch_size, cfg.grad_accum, cfg.block_size,
                               a.layers, a.heads, a.kv_heads, a.width, threads=a.threads)
    ll = Lifelong(cfg, backend, memory)
    if a.action == "life":
        from .autonomy import Autonomy, AutonomyConfig
        from .curiosity import Curriculum, Explorer, Robots
        from .ingest import fetch_url
        from .wiring import build_ingestor, build_memory
        from .practice import Practice
        sem = a.semantic_path or str(Path(a.state_dir) / "semantic.bin")
        if cfg.memory_db:
            memory = build_memory(cfg.memory_db, sem)
        cur = Curriculum(Path(a.state_dir) / "curriculum.db")
        ing = build_ingestor(a, cfg.inbox_dir or Path(a.state_dir) / "inbox", memory, cfg.memory_workspace, a.state_dir)
        cfg.inbox_dir = str(ing.inbox)
        ll = Lifelong(cfg, backend, memory)
        exp = None
        if a.experience_db:
            from .experience import ExperienceStore
            exp = ExperienceStore(a.experience_db)
        ex = Explorer(ing, cur, Robots(fetch_url), a.max_pages, daily_pages=a.daily_pages, deny_hosts=a.deny_host)
        life = Autonomy(ll, cur, ex, exp,
                        AutonomyConfig(explore_interval=a.explore_interval, explore=not a.no_explore, practice=not a.no_practice,
                                       practice_interval=a.practice_interval, maintenance_interval=a.maintenance_interval,
                                       semantic_path=sem if memory is not None else "", watch_paths=tuple(a.watch)),
                        practice=Practice(ll, exp), memory=memory, ingestor=ing)
        if a.initial_checkpoint and not ll.registry["current_model"]:
            ll.adopt(a.initial_checkpoint, backend.tokenizer_dict(a.initial_checkpoint))
        try:
            life.run_forever(a.poll_seconds)
        except KeyboardInterrupt:
            print("stopped")
        return 0
    if a.action == "status":
        print(json.dumps(ll.status(), indent=2)); return 0
    if a.action == "pause":
        ll.pause(); print("paused (STOP file written)"); return 0
    if a.action == "resume":
        ll.resume(); print("resumed"); return 0
    if a.initial_checkpoint and not ll.registry["current_model"]:
        ll.adopt(a.initial_checkpoint, backend.tokenizer_dict(a.initial_checkpoint))
    if a.action == "once":
        print(json.dumps(ll.run_cycle(force=a.force), indent=2, default=str)); return 0
    try:
        ll.run_forever(a.poll_seconds)
    except KeyboardInterrupt:
        print("stopped")
    return 0
