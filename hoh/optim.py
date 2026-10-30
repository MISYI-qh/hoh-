"""AdamW and gradient clipping in NumPy. The state_dict layout mirrors torch.optim.AdamW
({"state": {index: {"step", "exp_avg", "exp_avg_sq"}}, "param_groups": [{..., "params": [indices]}]}), so optstate.carry_state
(Adam moments follow parameters BY NAME across growth / vocab migration) works unchanged."""
import numpy as np


class AdamW:
    def __init__(self, params, groups, lr=3e-4, betas=(0.9, 0.95), eps=1e-8):
        """params: {name: ndarray} (updated in place). groups: [{"names": [...], "weight_decay": float}, ...]."""
        self.params = params
        self.names = [n for g in groups for n in g["names"]]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.param_groups, start = [], 0
        for g in groups:
            ids = list(range(start, start + len(g["names"])))
            start += len(g["names"])
            self.param_groups.append({"names": list(g["names"]), "params": ids, "lr": lr, "betas": tuple(betas),
                                      "eps": eps, "weight_decay": g["weight_decay"]})
        self.state = {}

    def step(self, grads):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for name in g["names"]:
                i = self.index[name]
                p, gr = self.params[name], grads[name]
                st = self.state.get(i)
                if st is None:
                    st = self.state[i] = {"step": 0, "exp_avg": np.zeros_like(p), "exp_avg_sq": np.zeros_like(p)}
                st["step"] += 1
                t = st["step"]
                if g["weight_decay"]:
                    p *= 1.0 - g["lr"] * g["weight_decay"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m *= b1; m += (1.0 - b1) * gr
                v *= b2; v += (1.0 - b2) * gr * gr
                bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t
                p -= (g["lr"] / bc1) * m / (np.sqrt(v / bc2) + g["eps"])

    def state_dict(self):
        return {"state": {i: {k: (v.copy() if hasattr(v, "copy") else v) for k, v in st.items()}
                          for i, st in self.state.items()},
                "param_groups": [{k: v for k, v in g.items() if k != "names"} | {"params": list(g["params"])}
                                 for g in self.param_groups]}

    def load_state_dict(self, sd):
        self.state = {}
        for i, st in sd.get("state", {}).items():
            i = int(i)
            if i >= len(self.names):
                continue
            shape = self.params[self.names[i]].shape
            if tuple(st["exp_avg"].shape) != tuple(shape):
                raise ValueError(f"optimizer state for {self.names[i]} has shape {st['exp_avg'].shape}, parameter {shape}")
            self.state[i] = {"step": int(st["step"]), "exp_avg": st["exp_avg"].astype(self.params[self.names[i]].dtype),
                             "exp_avg_sq": st["exp_avg_sq"].astype(self.params[self.names[i]].dtype)}
        for g, saved in zip(self.param_groups, sd.get("param_groups", [])):
            for k in ("lr", "betas", "eps", "weight_decay"):
                if k in saved:
                    g[k] = tuple(saved[k]) if k == "betas" else saved[k]


def clip_grad_norm(grads, max_norm):
    """Scale all gradients in place so their global L2 norm is <= max_norm. Returns the norm BEFORE clipping."""
    total = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
    if np.isfinite(total) and total > max_norm:
        s = max_norm / (total + 1e-6)
        for g in grads.values():
            g *= s
    return total
