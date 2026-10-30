"""Adaptive growth: function-preserving model growth + controlled experiment + promote/reject.

The parent checkpoint is only ever COPIED. The decision compares the candidate against a CONTROL
(the parent trained for the same extra steps), so extra training alone cannot win the comparison."""
import argparse, json, shutil, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from .checkpoint import load_checkpoint, save_checkpoint, tokenizer_from_checkpoint
from .config import ModelConfig, TrainConfig
from .data import load_tokens, split_tokens
from .evaluate import deterministic_loss
from .generate import load_model, pick_device
from .optim import AdamW
from .growth_policy import estimate_params, propose
from .model import HohModel
from .tokenizer import BASE_VOCAB


def transplant(model, old_state):
    """Copy old arrays into the leading slice of the (equal or larger) new arrays."""
    for k, t in model.params.items():
        if k not in old_state:
            continue
        o = np.asarray(old_state[k])
        if o.shape == t.shape:
            t[...] = o
        elif o.ndim == t.ndim and all(a <= b for a, b in zip(o.shape, t.shape)):
            t[tuple(slice(0, s) for s in o.shape)] = o
        else:
            raise ValueError(f"cannot transplant {k}: {tuple(o.shape)} -> {tuple(t.shape)}")
    return model


def grow_layers(old_state, mc, add):
    """New top layers have zeroed residual outputs => the grown model computes the same function."""
    if add < 1:
        raise ValueError("add must be >= 1")
    new_mc = replace(mc, n_layer=mc.n_layer + add)
    m = transplant(HohModel(new_mc), old_state)
    for i in range(mc.n_layer, new_mc.n_layer):
        for name, p in m.params.items():
            if name.startswith(f"blocks.{i}.") and (".attn.proj." in name or ".mlp.down." in name):
                p[...] = 0
    return m, new_mc


def grow_mlp(old_state, mc, new_hidden):
    """Extra hidden units have zeroed down-projection columns => same function at init."""
    old_h = mc.hidden_dim()
    if new_hidden <= old_h:
        raise ValueError(f"new_hidden ({new_hidden}) must exceed current hidden ({old_h})")
    new_mc = replace(mc, mlp_hidden=new_hidden)
    m = transplant(HohModel(new_mc), old_state)
    for i in range(new_mc.n_layer):
        m.params[f"blocks.{i}.mlp.down.weight"][:, old_h:] = 0
    return m, new_mc


def migrate_vocab(old_state, mc, tokenizer):
    """Grow embeddings to tokenizer.vocab_size. Old rows are kept; new merged token rows start as
    the mean of the embeddings of the two tokens it merges (so it begins near its parts)."""
    if tokenizer.vocab_size < mc.vocab_size:
        raise ValueError("vocabulary shrink is not supported")
    new_mc = replace(mc, vocab_size=tokenizer.vocab_size)
    m = transplant(HohModel(new_mc), old_state)
    E = m.params["tok_emb.weight"]  # tied with the output projection
    for j in range(mc.vocab_size - BASE_VOCAB, len(getattr(tokenizer, "merges", []))):
        if BASE_VOCAB + j < mc.vocab_size:
            continue
        a, b = tokenizer.merges[j]
        E[BASE_VOCAB + j] = (E[a] + E[b]) / 2
    return m, new_mc


def fresh_checkpoint(path, model, mc, tokenizer, note="", carry_from=None):
    """Write a step-0 checkpoint so train --init-from can use it. `carry_from` (the PARENT checkpoint dict) makes the saved
    optimizer keep the parent's AdamW moments for every surviving parameter (see optstate); otherwise it is fresh."""
    from .optstate import param_groups, carry_state, names_for_config
    tc = TrainConfig()
    groups, names = param_groups(model, tc.weight_decay)
    opt = AdamW(model.params, groups, lr=1e-4, betas=(0.9, 0.95))
    if carry_from is not None and carry_from.get("optimizer", {}).get("state"):
        try:
            old_names = carry_from.get("opt_names") or list(names_for_config(ModelConfig(**carry_from["model_config"])))
            shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
            res = carry_state(carry_from["optimizer"], old_names, names, shapes)
            sd = opt.state_dict()
            sd["state"] = res["state"]
            opt.load_state_dict(sd)
        except Exception as e:
            print(f"WARNING: optimizer state not carried ({type(e).__name__}: {e})")
    save_checkpoint(path, model, opt, None, -1, float("inf"), mc, tc, tokenizer, opt_names=names)


def _atomic_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(p)


def run_experiment(parent, data, workdir, kind, amount=0, steps=200, train_args=(), min_improvement=0.01,
                   max_params=None, test_data=None, device="cpu", threads=None):
    from .dist import run_train
    workdir = Path(workdir)
    exp = workdir / f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{kind}"
    exp.mkdir(parents=True, exist_ok=False)
    parent_copy = exp / "parent.npz"
    shutil.copy2(parent, parent_copy)                       # the original is never opened for writing
    ck = load_checkpoint(parent_copy)
    mc = ModelConfig(**ck["model_config"]).validate()
    tok = tokenizer_from_checkpoint(ck)
    if kind == "add_layers":
        cand, cmc = grow_layers(ck["model"], mc, amount or 1)
    elif kind == "grow_mlp":
        cand, cmc = grow_mlp(ck["model"], mc, amount or ((int(mc.hidden_dim() * 1.5) + 63) // 64) * 64)
    else:
        raise ValueError(f"unsupported growth kind {kind!r}")
    p_old, p_new = estimate_params(mc), estimate_params(cmc)
    if max_params and p_new > max_params:
        report = {"decision": "rejected", "reason": f"params {p_new:,} exceed max_params {max_params:,}"}
        _atomic_json(exp / "report.json", report)
        return report
    fresh_checkpoint(exp / "cand_init.npz", cand, cmc, tok, carry_from=ck)
    common = ["--data", str(data), "--steps", str(steps), "--device", device, "--precision", "fp32", "--carry-optimizer", *train_args]
    run_train(common + ["--out-dir", str(exp / "control"), "--init-from", str(parent_copy)], threads)
    run_train(common + ["--out-dir", str(exp / "candidate"), "--init-from", str(exp / "cand_init.npz")], threads)
    eval_tokens = load_tokens(test_data or data, tok)
    if not test_data:
        _, eval_tokens = split_tokens(eval_tokens, mc.block_size)
    dev = pick_device(device)
    losses = {}
    for name in ("control", "candidate"):
        m, _ = load_model(exp / name / "best.npz", device)
        losses[name] = deterministic_loss(m, eval_tokens, mc.block_size, dev)
    rel = (losses["control"] - losses["candidate"]) / max(losses["control"], 1e-9)
    accept = rel >= min_improvement
    report = {"kind": kind, "amount": amount, "steps": steps, "params_before": p_old, "params_after": p_new,
              "param_increase": p_new / p_old - 1, "control_loss": losses["control"],
              "candidate_loss": losses["candidate"], "relative_improvement": rel,
              "min_improvement": min_improvement, "eval_data": "test_data" if test_data else "held-out tail of data",
              "decision": "promoted" if accept else "rejected",
              "reason": f"relative improvement {rel:.2%} {'>=' if accept else '<'} required {min_improvement:.2%}"}
    reg_path = workdir / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"current": str(Path(parent).resolve()), "history": []}
    if accept:
        dst = workdir / "promoted" / f"{exp.name}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exp / "candidate" / "best.npz", dst)
        report["promoted_to"] = str(dst)
        reg["current"] = str(dst)
    reg["history"].append({"experiment": exp.name, "parent": str(Path(parent).resolve()), **report})
    _atomic_json(exp / "report.json", report)
    _atomic_json(reg_path, reg)  # current stays the parent when rejected
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Adaptive growth experiment (candidate vs control).")
    ap.add_argument("--parent", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--workdir", default="runs/growth")
    ap.add_argument("--kind", choices=["auto", "add_layers", "grow_mlp"], default="auto")
    ap.add_argument("--amount", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--min-improvement", type=float, default=0.01)
    ap.add_argument("--max-params", type=int, default=0)
    ap.add_argument("--test-data", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--propose-only", action="store_true")
    ap.add_argument("--train-arg", action="append", default=[], help="extra arg for hoh.train, e.g. --train-arg=--batch-size=2")
    a = ap.parse_args(argv)
    kind, amount = a.kind, a.amount
    if kind == "auto" or a.propose_only:
        ck = load_checkpoint(a.parent)
        mc = ModelConfig(**ck["model_config"]).validate()
        mpath = Path(a.parent).parent / "metrics.jsonl"
        rows = [json.loads(l) for l in mpath.read_text().splitlines() if l.strip()] if mpath.exists() else []
        prop = propose(rows, mc, a.max_params or estimate_params(mc) * 4)
        print(json.dumps(prop.to_dict(), indent=2))
        if a.propose_only or not prop.grow:
            return prop.to_dict()
        kind, amount = prop.kind, prop.amount
    extra = []
    for x in a.train_arg:
        extra += x.split("=", 1) if x.startswith("--") and "=" in x else [x]
    report = run_experiment(a.parent, a.data, a.workdir, kind, amount, a.steps, extra,
                            a.min_improvement, a.max_params or None, a.test_data or None, a.device)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
