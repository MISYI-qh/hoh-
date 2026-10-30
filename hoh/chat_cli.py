"""CLI: `hoh chat` (interactive), `hoh learn` (feed sources, no model needed), `hoh experience`, `hoh reasoning-data`."""
import argparse, json, sys


def chat_main(argv):
    p = argparse.ArgumentParser(prog="hoh chat")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint")
    g.add_argument("--state-dir", help="serve what `hoh lifelong` last promoted (auto-reloads between turns)")
    p.add_argument("--workdir", default="hoh_data")
    p.add_argument("--inbox", help="default: <workdir>/inbox (point it at the lifelong --inbox to learn while idle)")
    p.add_argument("--allow-risky", action="store_true", help="allow write_file/run_python/make_zip without asking")
    p.add_argument("--learn-conversations", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="auto")
    p.add_argument("--sleep-steps", type=int, default=5, help="training steps per /tidur (1-1000)")
    p.add_argument("--base-data", default="", help="extra corpus mixed into sleep training; defaults to packaged smoke corpus")
    a = p.parse_args(argv)
    from pathlib import Path
    from .inference import HohGenerator
    from .session import Session

    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = str(Path(a.checkpoint).resolve()) if a.checkpoint else ""
    marker = workdir / "sleep_latest.json"
    selected_checkpoint = a.checkpoint
    if a.checkpoint and marker.is_file():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
            requested = str(Path(a.checkpoint).resolve())
            if saved.get("base_checkpoint") == requested or str(Path(saved.get("checkpoint", "")).resolve()) == requested:
                latest = Path(saved.get("checkpoint", ""))
                if latest.is_file():
                    selected_checkpoint = str(latest)
        except (OSError, ValueError, TypeError):
            pass

    gen = HohGenerator.from_state(a.state_dir, a.device) if a.state_dir else HohGenerator(selected_checkpoint, a.device)
    active = {"gen": gen, "root_checkpoint": base_checkpoint, "latest": str(Path(gen.checkpoint).resolve())}

    def chat(msgs, **kw):
        current = active["gen"]
        if a.state_dir and current.state_dir:
            current.reload_if_changed()
        kw.setdefault("temperature", a.temperature)
        return current.chat(msgs, **kw)

    def sleep_train(corpus_path, out_dir, steps):
        from .train import train
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        steps = max(1, min(int(steps), 1000))
        current_checkpoint = active["latest"]
        train([
            "--data", str(corpus_path), "--out-dir", str(out), "--steps", str(steps),
            "--batch-size", "2", "--grad-accum", "1", "--warmup-steps", "0",
            "--eval-interval", str(max(1, steps // 2)), "--eval-batches", "1",
            "--save-interval", str(steps), "--precision", "fp32", "--device", "cpu",
            "--lr", "3e-5", "--min-lr", "1e-6", "--weight-decay", "0.01",
            "--init-from", current_checkpoint,
        ])
        checkpoint = out / "best.npz"
        if not checkpoint.is_file():
            raise RuntimeError("training selesai tetapi best.npz tidak ditemukan")
        # Load before promotion: if it cannot load, keep serving the previous model.
        new_gen = HohGenerator(str(checkpoint), a.device)
        active["gen"] = new_gen
        active["latest"] = str(checkpoint.resolve())
        marker.write_text(json.dumps({
            "base_checkpoint": active["root_checkpoint"],
            "checkpoint": active["latest"],
            "trained_at": __import__("time").time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(checkpoint)

    def confirm(name, args):
        return input(f"[izin] jalankan {name} {json.dumps(args, ensure_ascii=False)[:200]} ? (y/N) ").strip().lower() == "y"

    default_base = Path(__file__).resolve().parent.parent / "data" / "smoke.txt"
    base_data = Path(a.base_data).expanduser() if a.base_data else default_base
    s = Session(chat, a.workdir, inbox_dir=a.inbox, allow_risky=a.allow_risky, confirm=confirm,
                learn_conversations=a.learn_conversations, sleep_callback=sleep_train,
                base_data=base_data, sleep_steps=a.sleep_steps)
    print("HOH siap. Ketik pertanyaan, /ajari <materi>, /apa_yang_dipelajari, /koreksi <id> <materi>, /tidur [steps], /learn <link|file>, /search <kata>, /good, /bad <koreksi>, /status. Ctrl-D keluar.")
    while True:
        try:
            line = input("kamu> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        try:
            out = s.handle(line)
        except Exception as e:                       # a bad turn must not kill the session
            out = f"[error] {type(e).__name__}: {e}"
        if out:
            print("hoh>", out)


def learn_main(argv):
    p = argparse.ArgumentParser(prog="hoh learn", description="Feed sources into the inbox/memory (no model needed).")
    p.add_argument("sources", nargs="*", help="URLs, files, folders, zips")
    p.add_argument("--search", action="append", default=[], help="web search query (repeatable)")
    p.add_argument("-n", type=int, default=5, help="results per search")
    p.add_argument("--inbox", default="inbox")
    p.add_argument("--memory-db", default="")
    p.add_argument("--workspace", default="default")
    p.add_argument("--min-quality", type=float, default=0.6)
    p.add_argument("--state-dir", default="runs/life", help="where facts.db (truth index) lives")
    from .wiring import add_learning_flags, build_ingestor
    add_learning_flags(p)
    a = p.parse_args(argv)
    from .memory import MemoryStore
    mem = MemoryStore(a.memory_db) if a.memory_db else None
    ing = build_ingestor(a, a.inbox, mem, a.workspace, a.state_dir, a.min_quality)
    for s in a.sources:
        print(ing.summary(ing.learn(s)))
    for q in a.search:
        print(ing.summary(ing.learn_search(q, a.n)))
    return 0


def experience_main(argv):
    p = argparse.ArgumentParser(prog="hoh experience")
    p.add_argument("action", choices=["stats", "export"])
    p.add_argument("--db", default="hoh_data/experience.db")
    p.add_argument("--out", default="experience_sft.jsonl")
    p.add_argument("--include-unverified", action="store_true")
    a = p.parse_args(argv)
    from .experience import ExperienceStore
    st = ExperienceStore(a.db)
    print(json.dumps(st.stats() if a.action == "stats" else st.export_sft(a.out, a.include_unverified), ensure_ascii=False))
    return 0


def reasoning_data_main(argv):
    p = argparse.ArgumentParser(prog="hoh reasoning-data")
    p.add_argument("--out", default="data/reasoning_sft.jsonl")
    p.add_argument("-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tool-fraction", type=float, default=0.25)
    a = p.parse_args(argv)
    from .reasoning import generate_dataset
    print(f"wrote {generate_dataset(a.out, a.n, a.seed, a.tool_fraction)} verified examples to {a.out}")
    return 0


def goals_main(argv):
    p = argparse.ArgumentParser(prog="hoh goals", description="What HOH should study on its own.")
    p.add_argument("action", choices=["add", "list", "remove"])
    p.add_argument("topic", nargs="*")
    p.add_argument("--db", default="runs/lifelong/curriculum.db")
    p.add_argument("--priority", type=float, default=2.0)
    a = p.parse_args(argv)
    from .curiosity import Curriculum
    c = Curriculum(a.db)
    if a.action == "add":
        for t in a.topic:
            print(f"goal #{c.add_goal(t, a.priority, 'user')}: {t}")
    elif a.action == "remove":
        for t in a.topic:
            c.db.execute("DELETE FROM goals WHERE topic=?", (t,)); c.db.commit()
    else:
        for g in c.goals():
            print(f"#{g['id']} [{g['status']}] p={g['priority']:.1f} {g['source']:<9} dibaca={g['gained']} chars  {g['topic']}")
    return 0
