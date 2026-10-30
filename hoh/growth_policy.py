"""Torch-free growth policy: evidence -> proposal. Never changes a model by itself."""
import math
from dataclasses import dataclass, field, asdict, replace
from .config import ModelConfig


def estimate_params(mc):
    d, L, hd = mc.n_embd, mc.n_layer, mc.n_embd // mc.n_head
    kvd = mc.n_kv_head * hd
    b = 1 if mc.bias else 0
    attn = d * d + 2 * d * kvd + d * d + b * (d + 2 * kvd + d)
    mlp = 3 * d * mc.hidden_dim() + b * (2 * mc.hidden_dim() + d)
    return mc.vocab_size * d + L * (attn + mlp + 2 * d) + d  # tied embeddings counted once


def estimate_train_memory_bytes(mc, batch_size, bytes_per_param=4):
    """Rough rule of thumb (NOT a measurement): weights+grads+AdamW(2 states) + coarse activations."""
    p = estimate_params(mc)
    states = p * bytes_per_param * 4
    acts = batch_size * mc.block_size * mc.n_embd * mc.n_layer * 34 * 2 // (4 if mc.grad_checkpoint else 1)
    logits = batch_size * mc.block_size * mc.vocab_size * 4 * 2
    return int(states + acts + logits)


@dataclass
class Proposal:
    grow: bool
    kind: str = "none"            # none | add_layers | grow_mlp
    amount: int = 0                # layers to add, or new mlp hidden size
    reason: str = ""
    params_before: int = 0
    params_after: int = 0
    est_train_mem_bytes: int = 0
    risks: list = field(default_factory=list)
    how_to_test: str = ""

    def to_dict(self):
        return asdict(self)


# (width, layers, heads, kv_heads): every rung is a valid ModelConfig (even head dim, divisible heads)
LADDER = [(128, 5, 4, 2), (192, 4, 6, 3), (256, 6, 8, 4), (384, 8, 8, 4), (512, 10, 8, 4), (768, 12, 12, 4)]


def bootstrap_arch(unique_tokens, vocab_size=260, block_size=512, tokens_per_param=20.0):
    """Cold start WITHOUT pretraining: begin as small as the data justifies (params ~ tokens / tokens_per_param, never below
    the smallest rung). More data later -> evidence-gated growth, not a big empty model that memorizes a few pages."""
    best = LADDER[0]
    for w, L, h, kv in LADDER:
        mc = ModelConfig(vocab_size=vocab_size, block_size=block_size, n_layer=L, n_head=h, n_kv_head=kv, n_embd=w).validate()
        if estimate_params(mc) <= max(1.0, unique_tokens) / tokens_per_param:
            best = (w, L, h, kv)
    w, L, h, kv = best
    return {"width": w, "layers": L, "heads": h, "kv_heads": kv}


def propose(rows, mc, max_params, batch_size=8, min_rows=4, plateau_tol=0.01, gap_tol=0.15,
            unique_tokens=None, min_tokens_per_param=8.0):
    """rows: metrics.jsonl dicts (step, train_loss, val_loss). Conservative: 'no growth' is the default."""
    p0 = estimate_params(mc)
    none = lambda why: Proposal(False, "none", 0, why, p0, p0, estimate_train_memory_bytes(mc, batch_size))
    if len(rows) < min_rows:
        return none(f"insufficient evidence: {len(rows)} eval points < {min_rows}")
    tr = [r["train_loss"] for r in rows]
    va = [r["val_loss"] for r in rows]
    if not all(math.isfinite(v) for v in tr + va):
        return none("non-finite losses: investigate bug/learning rate before changing capacity")
    w = max(3, len(va) // 3)
    first, last = va[-w], va[-1]
    improvement = (first - last) / max(abs(first), 1e-9)
    if improvement > plateau_tol:
        return none(f"validation loss still improving ({improvement:.1%} over last {w} evals): train longer")
    gap = (va[-1] - tr[-1]) / max(abs(tr[-1]), 1e-9)
    if gap > gap_tol:
        return none(f"val/train gap {gap:.1%} > {gap_tol:.0%}: data-limited/overfitting; add data or regularize, not capacity")
    if unique_tokens is not None and unique_tokens < min_tokens_per_param * p0:
        return none(f"data-limited: {int(unique_tokens):,} unique tokens < {min_tokens_per_param:g} x {p0:,} params; "
                    f"need about {int(min_tokens_per_param * p0 - unique_tokens):,} more tokens before more capacity is justified")
    cands = []
    add = 1 if mc.n_layer < 4 else max(1, mc.n_layer // 4)
    cands.append(("add_layers", add, replace(mc, n_layer=mc.n_layer + add)))
    h = mc.hidden_dim()
    nh = ((int(h * 1.5) + 63) // 64) * 64
    cands.append(("grow_mlp", nh, replace(mc, mlp_hidden=nh)))
    for kind, amount, new in cands:
        p1 = estimate_params(new)
        if p1 <= max_params:
            return Proposal(
                True, kind, amount,
                f"val loss plateaued ({improvement:.1%} over last {w} evals), small val/train gap ({gap:.1%}): "
                f"consistent with capacity limit (also possible: LR/steps; the experiment's control arm tests this)",
                p0, p1, estimate_train_memory_bytes(new, batch_size),
                ["candidate may not beat a control trained for the same extra steps",
                 "optimizer state is not migrated (fresh AdamW)", "extra compute/memory per step"],
                "python -m hoh grow --parent <ckpt> --data <corpus> --kind " + kind)
    return none(f"capacity growth would exceed max_params={max_params:,}")
