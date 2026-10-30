import argparse, json, math, os, random, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from .config import ModelConfig, TrainConfig, ConfigError, save_json
from .checkpoint import save_checkpoint as _save_ck, load_checkpoint, tokenizer_from_checkpoint
from .data import load_tokens, split_tokens, TokenBatcher, SFTBatcher
from .sft import load_sft_examples, split_items
from .model import HohModel
from .optim import AdamW, clip_grad_norm
from .tokenizer import ByteTokenizer, load_tokenizer
from .dist import resolve_threads, apply_threads
from .optstate import param_groups, restore_hparams, carry_state, names_for_config


def choose_device(request):
    """NumPy runs on the CPU only. 'auto' and 'cpu' are accepted; anything else is an error (no silent fallback)."""
    if request in ("auto", "cpu"):
        return "cpu"
    raise ConfigError(f"hoh runs on the CPU only (NumPy); --device {request!r} is not available. Use auto or cpu.")


def choose_precision(name, device="cpu"):
    """Training is float32. bf16/fp16 do not exist in NumPy; they are accepted and mapped to fp32 (a warning is printed)."""
    if name not in ("auto", "fp32"):
        print(f"WARNING: precision {name!r} is not available in NumPy; using fp32")
    return "fp32"


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, *args, **kw):
    """Atomic save; a failed write is reported but does not destroy the previous checkpoint."""
    try:
        _save_ck(path, *args, **kw)
    except OSError as e:
        tmp = Path(str(path) + ".tmp")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"WARNING: could not write checkpoint {path}: {e}")


def estimate_loss(model, batchers, n_batches, amp_dtype=None, device=None):
    model.eval()
    out = {}
    for name, batcher in batchers.items():
        losses = []
        for _ in range(max(1, n_batches)):
            x, y = batcher.get()
            losses.append(float(model(x, y)[1]))
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="Train HOH (pure NumPy) language model from scratch.")
    a = ap.add_argument
    a("--data", default="data/train.txt"); a("--out-dir", default="runs/hoh")
    a("--steps", type=int, default=10000); a("--batch-size", type=int, default=8)
    a("--grad-accum", type=int, default=4); a("--block-size", type=int, default=512)
    a("--layers", type=int, default=5); a("--heads", type=int, default=4)
    a("--kv-heads", type=int, default=2); a("--width", type=int, default=128)
    a("--lr", type=float, default=3e-4); a("--min-lr", type=float, default=3e-5)
    a("--weight-decay", type=float, default=0.1); a("--grad-clip", type=float, default=1.0)
    a("--warmup-steps", type=int, default=200); a("--eval-interval", type=int, default=250)
    a("--eval-batches", type=int, default=20); a("--save-interval", type=int, default=500)
    a("--device", default="auto", help="auto or cpu (NumPy has no GPU backend)")
    a("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto", help="always fp32 in NumPy")
    a("--threads", default="auto", help="BLAS threads for matrix products (needs threadpoolctl, else use OMP_NUM_THREADS)")
    a("--resume", default="", help="checkpoint to resume (model+optimizer+step)")
    a("--init-from", default="", help="checkpoint to take weights from only (e.g. pretrained -> SFT)")
    a("--seed", type=int, default=1337); a("--mode", choices=["pretrain", "sft"], default="pretrain")
    a("--dropout", type=float, default=0.0)
    a("--tokenizer", default="", help="tokenizer.json (byte or BPE); default: from checkpoint, else byte")
    a("--grad-checkpoint", action="store_true", help="recompute activations in backward (saves RAM, slower)")
    a("--carry-optimizer", action="store_true", help="with --init-from: keep AdamW moments of surviving parameters (by name)")
    return ap


def train(argv=None):
    o = build_parser().parse_args(argv)
    cfg = TrainConfig(data=o.data, out_dir=o.out_dir, steps=o.steps, batch_size=o.batch_size,
                      grad_accum=o.grad_accum, learning_rate=o.lr, min_lr=o.min_lr,
                      weight_decay=o.weight_decay, warmup_steps=o.warmup_steps,
                      eval_interval=o.eval_interval, eval_batches=o.eval_batches,
                      save_interval=o.save_interval, grad_clip=o.grad_clip, device=o.device,
                      precision=o.precision, resume=o.resume, init_from=o.init_from, seed=o.seed,
                      mode=o.mode, block_size=o.block_size).validate()
    random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = choose_device(cfg.device)
    precision = choose_precision(cfg.precision, device)
    if o.threads not in ("auto", "", None):
        apply_threads(resolve_threads(o.threads))

    ck = None
    src = cfg.resume or cfg.init_from
    if src:
        ck = load_checkpoint(src)
        mc = ModelConfig(**ck["model_config"]).validate()  # architecture always comes from the checkpoint
        if o.grad_checkpoint:
            mc = replace(mc, grad_checkpoint=True)
    tokenizer = load_tokenizer(o.tokenizer) if o.tokenizer else (
        tokenizer_from_checkpoint(ck) if ck is not None else ByteTokenizer())
    if ck is None:
        mc = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=o.block_size, n_layer=o.layers,
                         n_head=o.heads, n_kv_head=o.kv_heads, n_embd=o.width, dropout=o.dropout,
                         grad_checkpoint=o.grad_checkpoint).validate()
    elif tokenizer.vocab_size != mc.vocab_size:
        raise ConfigError(f"tokenizer vocab ({tokenizer.vocab_size}) != checkpoint vocab ({mc.vocab_size}); "
                          "run `python -m hoh migrate-vocab` first")
    cfg.block_size = mc.block_size
    log = print
    log(f"HOH training (NumPy) | device={device} precision={precision} mode={cfg.mode}")

    if cfg.mode == "sft":
        examples, dropped = load_sft_examples(cfg.data, tokenizer, mc.block_size)
        tr, va = split_items(examples, 0.02, cfg.seed)
        train_batcher = SFTBatcher(tr, device, cfg.batch_size, rng)
        val_batcher = SFTBatcher(va, device, max(1, min(cfg.batch_size, 4)), rng)
        log(f"SFT examples: {len(examples)} (dropped {dropped}) | train/val: {len(tr)}/{len(va)} | "
            f"loss only on assistant tokens")
    else:
        tokens = load_tokens(cfg.data, tokenizer)
        tr, va = split_tokens(tokens, mc.block_size)
        train_batcher = TokenBatcher(tr, mc.block_size, device, cfg.batch_size, rng)
        val_batcher = TokenBatcher(va, mc.block_size, device, max(1, min(cfg.batch_size, 4)), rng)
        log(f"tokens: {len(tokens):,} | train/val: {len(tr):,}/{len(va):,}")

    model = HohModel(mc, seed=cfg.seed)
    model.rng = rng                                      # dropout draws come from the same (checkpointed) generator
    groups, opt_names = param_groups(model, cfg.weight_decay)
    optimizer = AdamW(model.params, groups, lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    start, best_val = 0, float("inf")
    if ck is not None:
        model.load_state_dict(ck["model"])
        if cfg.resume:
            optimizer.load_state_dict(ck["optimizer"])
            start = int(ck["step"]) + 1
            best_val = float(ck.get("best_val", best_val))
            if ck.get("rng"):
                rng.bit_generator.state = ck["rng"]
            log(f"Resumed from {cfg.resume}: continuing at step {start}")
        else:
            msg = "optimizer and step start fresh"
            if o.carry_optimizer and ck.get("optimizer", {}).get("state"):
                try:
                    old_names = ck.get("opt_names") or list(names_for_config(ModelConfig(**ck["model_config"])))
                    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
                    res = carry_state(ck["optimizer"], old_names, opt_names, shapes)
                    fresh_groups = [{k: v for k, v in g.items() if k not in ("params", "names")} for g in optimizer.param_groups]
                    sd = optimizer.state_dict()
                    sd["state"] = res["state"]
                    optimizer.load_state_dict(sd)
                    restore_hparams(optimizer, fresh_groups)
                    msg = (f"optimizer moments carried for {res['carried']} tensors "
                           f"({res['fresh']} new, {len(res['dropped'])} dropped); training step restarts, Adam bias-correction count kept")
                except Exception as e:                       # never fail a run over a courtesy
                    msg = f"optimizer state NOT carried ({type(e).__name__}: {e}); starting fresh"
            log(f"Initialised weights from {cfg.init_from} ({msg})")
    if start >= cfg.steps:
        log(f"Nothing to do: checkpoint is already at step {start - 1} >= target {cfg.steps}.")
        return 0

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "model_config.json", mc)
    save_json(out / "train_config.json", cfg)
    tokenizer.save(out / "tokenizer.json")
    print(f"Parameters: {model.parameter_count():,}")

    model.train()
    grads = model.zero_grads()
    started = time.time()
    tokens_per_step = cfg.batch_size * cfg.grad_accum * mc.block_size
    last_t = time.time()
    max_s = float(os.environ.get("HOH_MAX_SECONDS") or 0)      # batas waktu (HP lambat): LR ikut turun sesuai waktu, lalu berhenti rapi
    for step in range(start, cfg.steps):
        stop = False
        eff = max(step, int(min(1.0, (time.time() - started) / max_s) * cfg.steps)) if max_s else step
        lr = lr_at(min(eff, cfg.steps - 1), cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        for g in grads.values():
            g.fill(0.0)
        running = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_batcher.get()
            loss, _ = model.forward_backward(x, y, scale=1.0 / cfg.grad_accum, grads=grads)
            running += loss / cfg.grad_accum
        if not math.isfinite(running):
            raise FloatingPointError(f"Non-finite training loss at step {step}; lower the learning rate.")
        gnorm = clip_grad_norm(grads, cfg.grad_clip)
        if not math.isfinite(gnorm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}; lower the learning rate.")
        optimizer.step(grads)
        now = time.time()
        stop = bool(max_s) and (now - started) >= max_s
        tps = tokens_per_step / max(now - last_t, 1e-9)
        last_t = now
        if step % 10 == 0 or step == cfg.steps - 1 or stop:
            print(f"step {step:7d} | train_loss {running:.4f} | grad_norm {gnorm:.3f} | lr {lr:.2e} | "
                  f"{tps:,.0f} tok/s | elapsed {now - started:.1f}s")
        if step % cfg.eval_interval == 0 or step == cfg.steps - 1 or stop:
            losses = estimate_loss(model, {"train": train_batcher, "val": val_batcher}, cfg.eval_batches)
            print(f"EVAL step {step}: train={losses['train']:.4f} val={losses['val']:.4f}")
            if math.isfinite(losses["val"]) and losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out / "best.npz", model, optimizer, None, step, best_val, mc, cfg, tokenizer, opt_names, rng=rng)
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "train_loss": losses["train"],
                                    "val_loss": losses["val"], "lr": lr, "tokens_per_sec": tps}) + "\n")
        if step % cfg.save_interval == 0 or step == cfg.steps - 1 or stop:
            save_checkpoint(out / "last.npz", model, optimizer, None, step, best_val, mc, cfg, tokenizer, opt_names, rng=rng)
        if stop:
            log(f"Batas waktu {max_s:.0f}s tercapai di langkah {step}; berhenti rapi.")
            break
    log(f"Training finished. Best validation loss: {best_val:.4f}. Checkpoints: {out}")
    return 0


if __name__ == "__main__":
    train()
