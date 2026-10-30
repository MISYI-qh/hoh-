import argparse
from .checkpoint import load_checkpoint, tokenizer_from_checkpoint
from .config import ModelConfig, ConfigError
from .model import HohModel


def pick_device(device="auto"):
    """NumPy = CPU only. Returns the string 'cpu' (kept so call sites look like the PyTorch build)."""
    if device in ("auto", "cpu"):
        return "cpu"
    raise ConfigError(f"hoh runs on the CPU only (NumPy); device {device!r} is not available")


def load_model(checkpoint, device="auto", return_tokenizer=False):
    device = pick_device(device)
    ck = load_checkpoint(checkpoint)
    cfg = ModelConfig(**ck["model_config"]).validate()
    tok = tokenizer_from_checkpoint(ck)
    if tok.vocab_size != cfg.vocab_size:
        raise ValueError(f"tokenizer vocab {tok.vocab_size} != model vocab {cfg.vocab_size}")
    model = HohModel(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return (model, device, tok) if return_tokenizer else (model, device)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate text with a trained HOH checkpoint.")
    ap.add_argument("--checkpoint", default="runs/hoh/best.npz")
    ap.add_argument("--prompt", default="Halo,")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    from .inference import HohGenerator
    text = HohGenerator(a.checkpoint, a.device).generate(
        a.prompt, a.tokens, a.temperature, a.top_k, a.top_p, a.repetition_penalty, a.seed)
    print(text)
    return text


if __name__ == "__main__":
    main()
