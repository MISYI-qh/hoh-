"""HOH model: decoder-only causal Transformer in pure NumPy (no PyTorch, no autograd library).

Same design as NOQ (RoPE, grouped-query attention, RMSNorm, SwiGLU, tied input/output embeddings, KV cache, sliding window,
function-preserving growth) but adapted to NumPy:

* parameters live in one ordered dict {name: ndarray}; names and (out, in) weight layout are identical to the PyTorch layout, so
  growth / optimizer-state-by-name logic works unchanged;
* the backward pass is written by hand (see ``forward_backward``) and verified against finite differences in the tests;
* ``grad_checkpoint`` recomputes a block's activations in backward (stores only the block input) to save RAM;
* attention probabilities are materialised per layer, so memory grows with block_size**2 -- use ``grad_checkpoint`` on small RAM;
* float32 by default (``dtype=np.float64`` is available and is what the gradient tests use); there is no bf16/fp16 and no GPU.
"""
import math
from collections import OrderedDict

import numpy as np

from .config import ModelConfig

NEG_INF = -np.inf


# --------------------------------------------------------------------------------------------------------------------------
# parameter layout
# --------------------------------------------------------------------------------------------------------------------------
def param_shapes(cfg):
    """OrderedDict name -> shape in registration order, WITHOUT allocating anything (tied lm_head is not listed)."""
    C, H, Hkv = cfg.n_embd, cfg.n_head, cfg.n_kv_head
    D = C // H
    hid = cfg.hidden_dim()
    s = OrderedDict()
    s["tok_emb.weight"] = (cfg.vocab_size, C)
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        s[p + "norm1.weight"] = (C,)
        for nm, out in (("q", H * D), ("k", Hkv * D), ("v", Hkv * D), ("proj", C)):
            s[p + f"attn.{nm}.weight"] = (out, C)
            if cfg.bias:
                s[p + f"attn.{nm}.bias"] = (out,)
        s[p + "norm2.weight"] = (C,)
        for nm, (o, n) in (("gate", (hid, C)), ("up", (hid, C)), ("down", (C, hid))):
            s[p + f"mlp.{nm}.weight"] = (o, n)
            if cfg.bias:
                s[p + f"mlp.{nm}.bias"] = (o,)
    s["norm.weight"] = (C,)
    return s


def count_params(cfg):
    return int(sum(int(np.prod(sh)) for sh in param_shapes(cfg).values()))


# --------------------------------------------------------------------------------------------------------------------------
# primitive ops (forward returns what backward needs)
# --------------------------------------------------------------------------------------------------------------------------
def scatter_add_rows(dst, idx, rows):
    """dst[idx[i]] += rows[i] for every i (duplicates summed). Much faster than np.add.at."""
    idx = np.asarray(idx).reshape(-1)
    rows = rows.reshape(-1, rows.shape[-1])
    order = np.argsort(idx, kind="stable")
    si, sr = idx[order], rows[order]
    starts = np.flatnonzero(np.concatenate(([True], si[1:] != si[:-1])))
    dst[si[starts]] += np.add.reduceat(sr, starts, axis=0)


def rms_fwd(x, w, eps=1e-6):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    xn = x * r
    return w * xn, (xn, r, w)


def rms_bwd(dy, cache, gw):
    xn, r, w = cache
    gw += (dy * xn).reshape(-1, xn.shape[-1]).sum(0)
    g = dy * w
    return r * (g - xn * (g * xn).mean(-1, keepdims=True))


def lin_fwd(x, W, b):
    y = x @ W.T
    return y if b is None else y + b


def lin_bwd(dy, x, W, grads, name, has_bias):
    x2 = x.reshape(-1, x.shape[-1])
    d2 = dy.reshape(-1, dy.shape[-1])
    grads[name + ".weight"] += d2.T @ x2
    if has_bias:
        grads[name + ".bias"] += d2.sum(0)
    return dy @ W


def rope_tables(offset, T, d, dtype):
    pos = np.arange(offset, offset + T, dtype=np.float64)
    inv = 1.0 / (10000.0 ** (np.arange(0, d, 2, dtype=np.float64) / d))
    f = np.outer(pos, inv)
    return np.cos(f).astype(dtype), np.sin(f).astype(dtype)


def rope_fwd(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)


def rope_bwd(dy, cos, sin):
    half = dy.shape[-1] // 2
    d1, d2 = dy[..., :half], dy[..., half:]
    return np.concatenate((d1 * cos + d2 * sin, -d1 * sin + d2 * cos), axis=-1)


def _dropout_mask(rng, shape, p, dtype):
    return ((rng.random(shape) >= p) / (1.0 - p)).astype(dtype)


# --------------------------------------------------------------------------------------------------------------------------
# attention / mlp / block
# --------------------------------------------------------------------------------------------------------------------------
def attn_fwd(x, P, pre, cfg, brng, training, past=None):
    B, T, C = x.shape
    H, Hkv = cfg.n_head, cfg.n_kv_head
    D, rep = C // H, H // Hkv
    q = lin_fwd(x, P[pre + "q.weight"], P.get(pre + "q.bias")).reshape(B, T, H, D).transpose(0, 2, 1, 3)
    k = lin_fwd(x, P[pre + "k.weight"], P.get(pre + "k.bias")).reshape(B, T, Hkv, D).transpose(0, 2, 1, 3)
    v = lin_fwd(x, P[pre + "v.weight"], P.get(pre + "v.bias")).reshape(B, T, Hkv, D).transpose(0, 2, 1, 3)
    offset = 0 if past is None else past[0].shape[2]
    if offset and T != 1:
        raise ValueError("cached attention supports one new token at a time")
    cos, sin = rope_tables(offset, T, D, x.dtype)
    q, k = rope_fwd(q, cos, sin), rope_fwd(k, cos, sin)
    if past is not None:
        k_all, v_all = np.concatenate([past[0], k], axis=2), np.concatenate([past[1], v], axis=2)
    else:
        k_all, v_all = k, v
    S = k_all.shape[2]
    qg = q.reshape(B, Hkv, rep, T, D)
    scale = 1.0 / math.sqrt(D)
    sc = np.matmul(qg, k_all[:, :, None].transpose(0, 1, 2, 4, 3)) * scale        # B,Hkv,rep,T,S
    if T > 1:
        sc[..., np.triu(np.ones((T, S), dtype=bool), k=1 + offset)] = NEG_INF
    sc -= sc.max(-1, keepdims=True)
    p = np.exp(sc)
    p /= p.sum(-1, keepdims=True)
    pm, pd = None, p
    if training and cfg.dropout > 0:
        pm = _dropout_mask(brng, p.shape, cfg.dropout, p.dtype)
        pd = p * pm
    o = np.matmul(pd, v_all[:, :, None])                                            # B,Hkv,rep,T,D
    y_attn = o.reshape(B, H, T, D).transpose(0, 2, 1, 3).reshape(B, T, C)
    out = lin_fwd(y_attn, P[pre + "proj.weight"], P.get(pre + "proj.bias"))
    return out, (x, y_attn, qg, k_all, v_all, p, pm, pd, cos, sin, scale), (k_all, v_all)


def attn_bwd(dout, cache, P, pre, cfg, grads):
    x, y_attn, qg, k_all, v_all, p, pm, pd, cos, sin, scale = cache
    B, T, C = x.shape
    H, Hkv = cfg.n_head, cfg.n_kv_head
    D, rep = C // H, H // Hkv
    dy = lin_bwd(dout, y_attn, P[pre + "proj.weight"], grads, pre + "proj", cfg.bias)
    do = dy.reshape(B, T, H, D).transpose(0, 2, 1, 3).reshape(B, Hkv, rep, T, D)
    dpd = np.matmul(do, v_all[:, :, None].transpose(0, 1, 2, 4, 3))                 # B,Hkv,rep,T,S
    dv = np.matmul(pd.transpose(0, 1, 2, 4, 3), do).sum(2)                          # B,Hkv,S,D
    dp = dpd * pm if pm is not None else dpd
    ds = p * (dp - (dp * p).sum(-1, keepdims=True))
    ds *= scale
    dq = np.matmul(ds, k_all[:, :, None]).reshape(B, H, T, D)
    dk = np.matmul(ds.transpose(0, 1, 2, 4, 3), qg).sum(2)                          # B,Hkv,S,D
    dq, dk = rope_bwd(dq, cos, sin), rope_bwd(dk, cos, sin)
    flat = lambda a, h: a.transpose(0, 2, 1, 3).reshape(B, T, h * D)
    dx = lin_bwd(flat(dq, H), x, P[pre + "q.weight"], grads, pre + "q", cfg.bias)
    dx = dx + lin_bwd(flat(dk, Hkv), x, P[pre + "k.weight"], grads, pre + "k", cfg.bias)
    dx = dx + lin_bwd(flat(dv, Hkv), x, P[pre + "v.weight"], grads, pre + "v", cfg.bias)
    return dx


def mlp_fwd(x, P, pre, cfg):
    a = lin_fwd(x, P[pre + "gate.weight"], P.get(pre + "gate.bias"))
    b = lin_fwd(x, P[pre + "up.weight"], P.get(pre + "up.bias"))
    sg = 1.0 / (1.0 + np.exp(-a))
    s = a * sg
    h = s * b
    return lin_fwd(h, P[pre + "down.weight"], P.get(pre + "down.bias")), (x, a, b, sg, s, h)


def mlp_bwd(dout, cache, P, pre, cfg, grads):
    x, a, b, sg, s, h = cache
    dh = lin_bwd(dout, h, P[pre + "down.weight"], grads, pre + "down", cfg.bias)
    db = dh * s
    da = dh * b * (sg * (1.0 + a * (1.0 - sg)))
    dx = lin_bwd(da, x, P[pre + "gate.weight"], grads, pre + "gate", cfg.bias)
    return dx + lin_bwd(db, x, P[pre + "up.weight"], grads, pre + "up", cfg.bias)


def block_fwd(x, P, i, cfg, brng, training, past=None, keep=True):
    pre = f"blocks.{i}."
    n1, c1 = rms_fwd(x, P[pre + "norm1.weight"])
    a, ca, kv = attn_fwd(n1, P, pre + "attn.", cfg, brng, training, past)
    ma = None
    if training and cfg.dropout > 0:
        ma = _dropout_mask(brng, a.shape, cfg.dropout, a.dtype)
        a = a * ma
    x1 = x + a
    n2, c2 = rms_fwd(x1, P[pre + "norm2.weight"])
    m, cm = mlp_fwd(n2, P, pre + "mlp.", cfg)
    mm = None
    if training and cfg.dropout > 0:
        mm = _dropout_mask(brng, m.shape, cfg.dropout, m.dtype)
        m = m * mm
    return x1 + m, ((c1, ca, ma, c2, cm, mm) if keep else None), kv


def block_bwd(dout, cache, P, i, cfg, grads):
    pre = f"blocks.{i}."
    c1, ca, ma, c2, cm, mm = cache
    dm = dout * mm if mm is not None else dout
    dn2 = mlp_bwd(dm, cm, P, pre + "mlp.", cfg, grads)
    dx1 = dout + rms_bwd(dn2, c2, grads[pre + "norm2.weight"])
    da = dx1 * ma if ma is not None else dx1
    dn1 = attn_bwd(da, ca, P, pre + "attn.", cfg, grads)
    return dx1 + rms_bwd(dn1, c1, grads[pre + "norm1.weight"])


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def cross_entropy(logits, targets, ignore_index=-100):
    """Mean over non-ignored targets (float). nan if everything is ignored (why every SFT example keeps >= 1 trained token)."""
    lg = logits.reshape(-1, logits.shape[-1]).astype(np.float64)
    tg = np.asarray(targets).reshape(-1)
    valid = tg != ignore_index
    if not valid.any():
        return float("nan")
    lg, tg = lg[valid], tg[valid]
    m = lg.max(-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(lg - m).sum(-1))
    return float((lse - lg[np.arange(len(tg)), tg]).mean())


# --------------------------------------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------------------------------------
class HohModel:
    def __init__(self, cfg: ModelConfig, seed=None, dtype=np.float32):
        self.config = cfg.validate()
        self.dtype = np.dtype(dtype)
        self.training = True
        self.rng = np.random.default_rng(seed)
        self.params = OrderedDict()
        proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, shape in param_shapes(cfg).items():
            if name.endswith("bias"):
                arr = np.zeros(shape)
            elif "norm" in name:
                arr = np.ones(shape)
            elif name.endswith(("proj.weight", "down.weight")):
                arr = self.rng.normal(0.0, proj_std, shape)
            else:
                arr = self.rng.normal(0.0, 0.02, shape)
            self.params[name] = arr.astype(self.dtype)

    # ---- nn.Module-like helpers -------------------------------------------------------------------------------------------
    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device=None):          # NumPy runs on the CPU only; kept so call sites stay the same
        return self

    def named_parameters(self):
        return iter(self.params.items())

    def parameters(self):
        return iter(self.params.values())

    def parameter_count(self):
        return int(sum(p.size for p in self.params.values()))

    def state_dict(self):
        return OrderedDict((k, v.copy()) for k, v in self.params.items())

    def load_state_dict(self, sd, strict=True):
        keys, mine = set(sd), set(self.params)
        if strict and (keys - mine or mine - keys):
            raise ValueError(f"state mismatch: missing={sorted(mine - keys)[:5]} unexpected={sorted(keys - mine)[:5]}")
        for k, v in sd.items():
            if k in self.params:
                v = np.asarray(v)
                if v.shape != self.params[k].shape:
                    raise ValueError(f"{k}: shape {v.shape} != {self.params[k].shape}")
                self.params[k][...] = v.astype(self.dtype, copy=False)
        return self

    def zero_grads(self):
        return OrderedDict((k, np.zeros_like(v)) for k, v in self.params.items())

    def _check_idx(self, idx):
        idx = np.asarray(idx)
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError(f"idx must be a non-empty (batch, time) array, got shape {tuple(idx.shape)}")
        return idx.astype(np.int64, copy=False)

    # ---- training: forward + hand-written backward ------------------------------------------------------------------------
    def forward_backward(self, idx, targets, scale=1.0, grads=None):
        """One micro-batch. Adds scale * d(loss)/d(param) into `grads` (like zero_grads(); created if None).
        Returns (loss, grads). Targets equal to -100 are ignored (mean over the rest)."""
        cfg, P = self.config, self.params
        idx = self._check_idx(idx)
        targets = np.asarray(targets, dtype=np.int64)
        if idx.shape[1] > cfg.block_size:
            idx, targets = idx[:, -cfg.block_size:], targets[:, -cfg.block_size:]
        if grads is None:
            grads = self.zero_grads()
        train = self.training
        E = P["tok_emb.weight"]
        x = E[idx]
        em = None
        if train and cfg.dropout > 0:
            em = _dropout_mask(self.rng, x.shape, cfg.dropout, x.dtype)
            x = x * em
        ckpt = bool(cfg.grad_checkpoint and train)
        saved = []
        for i in range(cfg.n_layer):
            seed = int(self.rng.integers(1 << 62)) if (train and cfg.dropout > 0) else 0
            out, cache, _ = block_fwd(x, P, i, cfg, np.random.default_rng(seed), train, keep=not ckpt)
            saved.append((x, seed) if ckpt else cache)
            x = out
        xn, cn = rms_fwd(x, P["norm.weight"])
        B, T, C = xn.shape
        logits = xn.reshape(-1, C) @ E.T                                            # N,V
        tg = targets.reshape(-1)
        valid = tg != -100
        n = int(valid.sum())
        if n == 0:                                                                  # same as torch: undefined mean -> nan
            return float("nan"), grads
        logits = logits.astype(np.float64) if self.dtype == np.float64 else logits
        logits -= logits.max(-1, keepdims=True)
        ex = np.exp(logits)
        z = ex.sum(-1, keepdims=True)
        rows = np.flatnonzero(valid)
        loss = float((np.log(z[rows, 0]) - logits[rows, tg[rows]]).sum() / n)
        ex /= z
        ex[rows, tg[rows]] -= 1.0
        ex[~valid] = 0.0
        ex *= scale / n                                                             # d(scaled mean loss)/d logits
        grads["tok_emb.weight"] += ex.T @ xn.reshape(-1, C)
        dx = rms_bwd((ex @ E).reshape(B, T, C), cn, grads["norm.weight"])
        del ex, logits
        for i in reversed(range(cfg.n_layer)):
            if ckpt:
                xin, seed = saved[i]
                _, cache, _ = block_fwd(xin, P, i, cfg, np.random.default_rng(seed), True, keep=True)
            else:
                cache = saved[i]
            dx = block_bwd(dx, cache, P, i, cfg, grads)
        if em is not None:
            dx = dx * em
        scatter_add_rows(grads["tok_emb.weight"], idx, dx)
        return loss, grads

    # ---- evaluation / inference ---------------------------------------------------------------------------------------------
    def forward(self, idx, targets=None):
        """No gradients, no dropout. Returns (logits, loss-or-None)."""
        cfg, P = self.config, self.params
        idx = self._check_idx(idx)
        if idx.shape[1] > cfg.block_size:
            idx = idx[:, -cfg.block_size:]
            if targets is not None:
                targets = np.asarray(targets)[:, -cfg.block_size:]
        x = P["tok_emb.weight"][idx]
        for i in range(cfg.n_layer):
            x, _, _ = block_fwd(x, P, i, cfg, None, False, keep=False)
        xn, _ = rms_fwd(x, P["norm.weight"])
        logits = xn @ P["tok_emb.weight"].T
        return logits, (cross_entropy(logits, targets) if targets is not None else None)

    __call__ = forward

    def hidden_states(self, idx):
        """Final normalised hidden states (B, T, C): used by semantic.ModelEmbedder."""
        cfg, P = self.config, self.params
        x = P["tok_emb.weight"][self._check_idx(idx)[:, -cfg.block_size:]]
        for i in range(cfg.n_layer):
            x, _, _ = block_fwd(x, P, i, cfg, None, False, keep=False)
        return rms_fwd(x, P["norm.weight"])[0]

    def forward_cached(self, idx, caches=None):
        """Inference only. idx: full prompt when caches is None, else exactly one new token."""
        cfg, P = self.config, self.params
        x = P["tok_emb.weight"][self._check_idx(idx)]
        new = []
        for i in range(cfg.n_layer):
            x, _, kv = block_fwd(x, P, i, cfg, None, False, None if caches is None else caches[i], keep=False)
            new.append(kv)
        xn, _ = rms_fwd(x, P["norm.weight"])
        return xn @ P["tok_emb.weight"].T, new

    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50, eos_token_id=258,
                 top_p=1.0, repetition_penalty=1.0, use_cache=True, rng=None):
        """temperature<=0 means greedy. KV cache is used while the context fits block_size, then a sliding-window recompute."""
        idx = self._check_idx(idx)
        if temperature < 0 or not (0.0 < top_p <= 1.0) or repetition_penalty <= 0:
            raise ValueError("need temperature >= 0, 0 < top_p <= 1, repetition_penalty > 0")
        rng = rng if rng is not None else self.rng
        self.eval()
        bs = self.config.block_size
        caches = None
        for _ in range(max_new_tokens):
            if use_cache and caches is not None and idx.shape[1] <= bs:
                logits, caches = self.forward_cached(idx[:, -1:], caches)
            else:
                logits, caches = self.forward_cached(idx[:, -bs:], None)
                if not use_cache:
                    caches = None
            logits = logits[:, -1, :].astype(np.float64)
            if repetition_penalty != 1.0:
                for b in range(idx.shape[0]):
                    seen = np.unique(idx[b])
                    sel = logits[b, seen]
                    logits[b, seen] = np.where(sel > 0, sel / repetition_penalty, sel * repetition_penalty)
            if temperature == 0:
                nxt = logits.argmax(-1)[:, None]
            else:
                logits = logits / temperature
                V = logits.shape[-1]
                if top_k and top_k < V:
                    kth = np.partition(logits, -top_k, axis=-1)[:, -top_k][:, None]
                    logits = np.where(logits < kth, NEG_INF, logits)
                if top_p < 1.0:
                    si = np.argsort(-logits, axis=-1)
                    sl = np.take_along_axis(logits, si, axis=-1)
                    pr = _softmax(sl)
                    sl = np.where((np.cumsum(pr, -1) - pr) > top_p, NEG_INF, sl)
                    logits = np.full_like(logits, NEG_INF)
                    np.put_along_axis(logits, si, sl, axis=-1)
                pr = _softmax(logits)
                c = np.cumsum(pr, axis=-1)
                u = rng.random((pr.shape[0], 1)) * c[:, -1:]
                nxt = np.minimum((c < u).sum(-1), V - 1)[:, None]
            idx = np.concatenate((idx, nxt.astype(np.int64)), axis=1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
        return idx
