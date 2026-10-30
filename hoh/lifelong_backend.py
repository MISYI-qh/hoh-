"""NumPy backend for hoh.lifelong (training, SFT, evaluation, growth, vocabulary migration on the CPU)."""
import json, math
from pathlib import Path
from .checkpoint import load_checkpoint
from .config import ModelConfig


class NumpyBackend:
    def __init__(self, device="auto", precision="auto", batch_size=8, grad_accum=4, block_size=512,
                 layers=6, heads=8, kv_heads=4, width=512, grad_checkpoint=False, threads=None):
        self.device, self.precision = device, precision
        self.bs, self.ga, self.block = batch_size, grad_accum, block_size
        self.arch = dict(layers=layers, heads=heads, kv_heads=kv_heads, width=width)
        self.gc = grad_checkpoint
        self.threads = threads                  # BLAS threads (None/'auto' = leave NumPy's default)

    def model_config(self, ckpt):
        return ModelConfig(**load_checkpoint(ckpt)["model_config"]).validate()

    def tokenizer_dict(self, ckpt):
        from .checkpoint import tokenizer_from_checkpoint
        return tokenizer_from_checkpoint(load_checkpoint(ckpt)).to_dict()

    def train(self, data_path, out_dir, steps, init_from=None, tokenizer_path=None, lr=1e-4, arch=None):
        from .dist import run_train
        arch = arch or self.arch
        a = ["--data", data_path, "--out-dir", out_dir, "--steps", str(steps), "--batch-size", str(self.bs),
             "--grad-accum", str(self.ga), "--block-size", str(self.block), "--lr", str(lr),
             "--min-lr", str(lr / 10), "--warmup-steps", str(max(1, min(200, steps // 10))),
             "--eval-interval", str(max(1, steps // 4)), "--save-interval", str(max(1, steps // 4)),
             "--eval-batches", "5", "--device", self.device, "--precision", self.precision]
        if self.gc:
            a.append("--grad-checkpoint")
        if init_from:
            a += ["--init-from", init_from, "--carry-optimizer"]     # architecture + tokenizer + Adam moments come from the checkpoint
        else:
            a += ["--layers", str(arch["layers"]), "--heads", str(arch["heads"]),
                  "--kv-heads", str(arch["kv_heads"]), "--width", str(arch["width"])]
            if tokenizer_path:
                a += ["--tokenizer", tokenizer_path]
        run_train(a, self.threads)
        out = Path(out_dir)
        rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()]
        best = out / "best.npz"
        if not best.exists():
            raise RuntimeError(f"training produced no best.npz in {out} (validation loss never finite?)")
        return {"best_ckpt": str(best), "train_loss": rows[-1]["train_loss"], "val_loss": rows[-1]["val_loss"]}

    def train_sft(self, data_path, out_dir, steps, init_from, lr=5e-5):
        """Fine-tune `init_from` on a chat-format JSONL (assistant-only loss). """
        from .dist import run_train
        a = ["--mode", "sft", "--data", data_path, "--out-dir", out_dir, "--steps", str(steps), "--batch-size", str(self.bs),
             "--grad-accum", str(self.ga), "--lr", str(lr), "--min-lr", str(lr / 10),
             "--warmup-steps", str(max(1, min(50, steps // 10))), "--eval-interval", str(max(1, steps // 4)),
             "--save-interval", str(max(1, steps // 4)), "--eval-batches", "5", "--device", self.device,
             "--precision", self.precision, "--init-from", init_from, "--carry-optimizer"]
        if self.gc:
            a.append("--grad-checkpoint")
        run_train(a, self.threads)
        out = Path(out_dir)
        rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()]
        if not (out / "best.npz").exists():
            raise RuntimeError(f"SFT produced no best.npz in {out}")
        return {"best_ckpt": str(out / "best.npz"), "train_loss": rows[-1]["train_loss"], "val_loss": rows[-1]["val_loss"]}

    def eval_reasoning(self, ckpt, problems, max_new_tokens=256):
        """Greedy answers to the FIXED problem list -> {'accuracy','n','per_kind'}. """
        from .inference import HohGenerator
        from .reasoning import evaluate_problems
        g = HohGenerator(ckpt, self.device)
        return evaluate_problems(g.chat_fn(temperature=0, max_new_tokens=max_new_tokens), problems)

    def eval_bpb(self, ckpt, text_path, max_chars=400_000, max_windows=256):
        """Bits per byte on the (tail of the) anchor text; comparable across different tokenizers."""
        from .data import read_text_file
        from .evaluate import deterministic_loss
        from .generate import load_model
        model, dev, tok = load_model(ckpt, self.device, return_tokenizer=True)
        text = read_text_file(text_path)[-max_chars:]
        toks = tok.encode(text, add_bos=True, add_eos=True)
        loss = deterministic_loss(model, toks, model.config.block_size, dev, max_windows=max_windows)
        return loss / math.log(2) * len(toks) / max(1, len(text.encode("utf-8")))

    def score_paragraphs(self, ckpt, paragraphs):
        """Mean next-token loss per paragraph under `ckpt` (higher = the model knows it less). """
        import numpy as np
        from .generate import load_model
        model, dev, tok = load_model(ckpt, self.device, return_tokenizer=True)
        model.eval()
        out = []
        for p in paragraphs:
            ids = tok.encode(p, add_bos=True, add_eos=True)[: model.config.block_size + 1]
            if len(ids) < 3:
                out.append(float("nan"))
                continue
            t = np.asarray([ids], dtype=np.int64)
            _, loss = model(t[:, :-1], t[:, 1:])
            out.append(float(loss))
        return out

    def migrate_vocab(self, ckpt_in, tokenizer_path, ckpt_out):
        from .growth import migrate_vocab, fresh_checkpoint
        from .tokenizer import load_tokenizer
        ck = load_checkpoint(ckpt_in)
        mc = ModelConfig(**ck["model_config"]).validate()
        tok = load_tokenizer(tokenizer_path)
        m, new_mc = migrate_vocab(ck["model"], mc, tok)
        fresh_checkpoint(ckpt_out, m, new_mc, tok, carry_from=ck)

    def grow(self, parent, data_path, workdir, kind, amount, steps, min_improvement, max_params, test_data):
        from .growth import run_experiment
        extra = ["--batch-size", str(self.bs), "--grad-accum", str(self.ga)]
        return run_experiment(parent, data_path, workdir, kind, amount, steps, extra, min_improvement,
                              max_params, test_data, self.device, threads=self.threads)
