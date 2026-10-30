"""Checkpoints as a single .npz (arrays) + one JSON metadata entry. No pickle anywhere: np.load(allow_pickle=False), so loading
an untrusted file cannot execute code (the NumPy equivalent of torch.load(weights_only=True)).

Layout:  model/<param name>            weights
         opt/<index>/exp_avg|exp_avg_sq  AdamW moments (optional)
         meta                          uint8 JSON: step, best_val, model_config, train_config, rng, tokenizer, opt_names,
                                       opt_steps, opt_groups, format_version"""
import json, os
from pathlib import Path
import numpy as np
from .config import ModelConfig

REQUIRED = {"model", "optimizer", "step", "model_config"}
FORMAT_VERSION = 3


def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def save_checkpoint(path, model, optimizer, scaler, step, best_val, model_cfg, train_cfg, tokenizer=None, opt_names=None, rng=None):
    """`scaler` is accepted for call-site compatibility with the PyTorch build and ignored (no loss scaling in fp32 NumPy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    arrays = {f"model/{k}": v for k, v in model.params.items()}
    opt_sd = optimizer.state_dict() if optimizer is not None else {"state": {}, "param_groups": []}
    steps = {}
    for i, st in opt_sd["state"].items():
        arrays[f"opt/{i}/exp_avg"], arrays[f"opt/{i}/exp_avg_sq"] = st["exp_avg"], st["exp_avg_sq"]
        steps[str(i)] = int(st["step"])
    meta = {"step": int(step), "best_val": float(best_val), "model_config": dict(vars(model_cfg)),
            "train_config": dict(vars(train_cfg)), "format_version": FORMAT_VERSION, "numpy_version": np.__version__,
            "tokenizer": tokenizer.to_dict() if tokenizer is not None else None, "opt_names": opt_names,
            "opt_steps": steps, "opt_groups": _to_jsonable(opt_sd["param_groups"]),
            "rng": _to_jsonable(rng.bit_generator.state) if rng is not None else None}
    arrays["meta"] = np.frombuffer(json.dumps(_to_jsonable(meta)).encode("utf-8"), dtype=np.uint8)
    with open(tmp, "wb") as f:                    # file object: np.savez must not append its own ".npz"
        np.savez(f, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    try:
        z = np.load(p, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
    except Exception as e:
        raise ValueError(f"{p}: not a HOH checkpoint ({type(e).__name__}: {e})") from e
    model = {k[6:]: z[k] for k in z.files if k.startswith("model/")}
    state = {}
    for i, step in meta.get("opt_steps", {}).items():
        state[int(i)] = {"step": int(step), "exp_avg": z[f"opt/{i}/exp_avg"], "exp_avg_sq": z[f"opt/{i}/exp_avg_sq"]}
    ck = {"model": model, "optimizer": {"state": state, "param_groups": meta.get("opt_groups", [])},
          "step": meta["step"], "best_val": meta.get("best_val", float("inf")), "model_config": meta["model_config"],
          "train_config": meta.get("train_config", {}), "tokenizer": meta.get("tokenizer"),
          "opt_names": meta.get("opt_names"), "rng": meta.get("rng"), "scaler": {},
          "format_version": meta.get("format_version")}
    missing = REQUIRED - set(ck)
    if missing:
        raise ValueError(f"{p}: not a HOH checkpoint (missing keys: {sorted(missing)})")
    ModelConfig(**ck["model_config"]).validate()
    return ck


def tokenizer_from_checkpoint(ck):
    from .tokenizer import ByteTokenizer, tokenizer_from_dict
    return tokenizer_from_dict(ck["tokenizer"]) if ck.get("tokenizer") else ByteTokenizer()
