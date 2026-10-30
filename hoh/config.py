from dataclasses import dataclass, asdict
import json
from pathlib import Path


class ConfigError(ValueError):
    pass


def _check(cond, msg):
    if not cond:
        raise ConfigError(msg)


@dataclass
class ModelConfig:
    vocab_size: int = 260
    block_size: int = 512
    n_layer: int = 5
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False
    mlp_hidden: int = 0           # 0 = auto (8/3 * n_embd rounded up to 64); >0 = explicit (used by growth)
    grad_checkpoint: bool = False  # recompute block activations in backward to save memory

    def hidden_dim(self):
        if self.mlp_hidden > 0:
            return self.mlp_hidden
        h = int(8 * self.n_embd / 3)
        return ((h + 63) // 64) * 64

    def validate(self):
        _check(self.mlp_hidden >= 0, "mlp_hidden must be >= 0")
        _check(self.vocab_size >= 260, "vocab_size must be >= 260 (256 bytes + 4 special tokens)")
        _check(self.block_size >= 8, "block_size must be >= 8")
        _check(self.n_layer >= 1, "n_layer must be >= 1")
        _check(self.n_head >= 1 and self.n_kv_head >= 1, "n_head and n_kv_head must be >= 1")
        _check(self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head")
        _check(self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head")
        _check((self.n_embd // self.n_head) % 2 == 0, "head dimension (n_embd/n_head) must be even for RoPE")
        _check(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        return self


@dataclass
class TrainConfig:
    data: str = "data/train.txt"
    out_dir: str = "runs/hoh"
    steps: int = 10000
    batch_size: int = 8
    grad_accum: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    eval_interval: int = 250
    eval_batches: int = 20
    save_interval: int = 500
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "auto"
    precision: str = "auto"  # auto, fp32, bf16, fp16
    resume: str = ""
    init_from: str = ""  # weights-only init (e.g. SFT starting from a pretrained checkpoint)
    mode: str = "pretrain"  # pretrain or sft
    block_size: int = 512

    def validate(self):
        _check(self.steps > 0 and self.batch_size > 0 and self.grad_accum > 0,
               "steps, batch_size and grad_accum must be > 0")
        _check(self.learning_rate > 0 and 0 <= self.min_lr <= self.learning_rate,
               "need 0 <= min_lr <= learning_rate and learning_rate > 0")
        _check(self.warmup_steps >= 0, "warmup_steps must be >= 0")
        _check(self.grad_clip > 0, "grad_clip must be > 0")
        _check(self.eval_interval > 0 and self.save_interval > 0 and self.eval_batches > 0,
               "eval_interval, save_interval and eval_batches must be > 0")
        _check(self.mode in {"pretrain", "sft"}, "mode must be 'pretrain' or 'sft'")
        _check(self.precision in {"auto", "fp32", "bf16", "fp16"}, "precision must be auto/fp32/bf16/fp16")
        _check(not (self.resume and self.init_from), "use either --resume or --init-from, not both")
        return self


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(obj), indent=2), encoding="utf-8")


def load_model_config(path):
    return ModelConfig(**json.loads(Path(path).read_text(encoding="utf-8"))).validate()
