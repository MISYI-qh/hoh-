"""Carry AdamW state (exp_avg, exp_avg_sq, step) across model growth and vocabulary migration.

Before: after add_layers / grow_mlp / migrate_vocab the optimizer started from scratch, throwing away what Adam had learned
about every gradient scale. Now each parameter that survives keeps its moments:

  same shape                 -> copied
  grown (all dims >=)        -> old values in the leading slice, ZEROS in the new region (new weights start with no history;
                                with a large `step` Adam's bias correction is ~1, so their first updates are small, not spiky)
  new parameter (new layer)  -> no entry: AdamW creates it lazily on the first step
  incompatible shape         -> dropped and counted (never guessed)

Parameter identity is by NAME. The grouping rule (weight decay for matrices, none for vectors/norms) lives here so train.py,
growth.py and this module cannot disagree. The numeric core works on anything with slicing and a zero-filled constructor
(numpy arrays)."""


def split_names(named):
    """named: iterable of (name, param) with .ndim/.requires_grad -> (decay_names, no_decay_names), in model order."""
    decay, no_decay = [], []
    for n, p in named:
        if getattr(p, "requires_grad", True):
            (no_decay if p.ndim < 2 or "norm" in n else decay).append(n)
    return decay, no_decay


def param_groups(model, weight_decay):
    """-> (optimizer_groups, [decay_names, no_decay_names]) for hoh.optim.AdamW."""
    decay, no_decay = split_names(model.named_parameters())
    groups = [{"names": decay, "weight_decay": weight_decay}, {"names": no_decay, "weight_decay": 0.0}]
    return groups, [decay, no_decay]


class _Shape:
    def __init__(self, shape):
        self.shape, self.ndim, self.requires_grad = tuple(shape), len(shape), True


def names_for_config(model_config):
    """Reconstruct the optimizer's parameter order from a config alone (nothing is allocated)."""
    from .model import param_shapes
    return split_names((n, _Shape(sh)) for n, sh in param_shapes(model_config).items())


def _zeros_like_shape(ref, shape):
    if hasattr(ref, "new_zeros"):
        return ref.new_zeros(tuple(shape))
    import numpy as np
    return np.zeros(tuple(shape), dtype=ref.dtype)


def _clone(x):
    return x.clone() if hasattr(x, "clone") else x.copy()


def _is_array(x):
    return hasattr(x, "shape") and len(getattr(x, "shape", ())) > 0


def resize_state(old, new_shape):
    """Return `old` re-embedded in `new_shape`, or None when the shapes are not compatible."""
    new_shape = tuple(new_shape)
    old_shape = tuple(old.shape)
    if old_shape == new_shape:
        return _clone(old)
    if len(old_shape) == len(new_shape) and all(a <= b for a, b in zip(old_shape, new_shape)):
        out = _zeros_like_shape(old, new_shape)
        out[tuple(slice(0, s) for s in old_shape)] = old
        return out
    return None


def carry_state(old_opt_sd, old_names, new_names, new_shapes):
    """old_opt_sd: optimizer.state_dict() of the old run; old_names/new_names: [decay_names, no_decay_names];
    new_shapes: {name: shape} of the NEW model. -> {"state": {new_index: {...}}, "carried", "fresh", "dropped"}"""
    old_flat = [n for g in old_names for n in g]
    new_flat = [n for g in new_names for n in g]
    old_index = {n: i for i, n in enumerate(old_flat)}
    state, carried, fresh, dropped = {}, 0, 0, []
    for j, name in enumerate(new_flat):
        i = old_index.get(name)
        st = old_opt_sd.get("state", {}).get(i) if i is not None else None
        if not st:
            fresh += 1
            continue
        new_st, ok = {}, True
        for k, v in st.items():
            if _is_array(v):
                r = resize_state(v, new_shapes[name])
                if r is None:
                    ok = False
                    break
                new_st[k] = r
            else:
                new_st[k] = _clone(v) if hasattr(v, "clone") or hasattr(v, "copy") else v      # `step`
        if ok:
            state[j] = new_st
            carried += 1
        else:
            dropped.append(name)
    return {"state": state, "carried": carried, "fresh": fresh, "dropped": dropped}


def restore_hparams(optimizer, fresh_groups):
    """After optimizer.load_state_dict(), put THIS run's lr/weight_decay/betas back (the loaded ones belong to the old run)."""
    for g, f in zip(optimizer.param_groups, fresh_groups):
        g.update(f)
