import argparse, json, math
import numpy as np
from .generate import load_model
from .data import load_tokens, TokenBatcher, SFTBatcher
from .sft import load_sft_examples, DatasetError


def deterministic_loss(model, tokens, block_size, device=None, batch_size=4, max_windows=64):
    """Mean loss over non-overlapping windows in fixed order: repeatable, same data for any model."""
    t = np.asarray(tokens, dtype=np.int64)
    n = (len(t) - 1) // block_size
    if n < 1:
        raise DatasetError(f"need > block_size ({block_size}) tokens for evaluation, got {len(t)}")
    starts = [i * block_size for i in range(min(n, max_windows))]
    model.eval()
    losses = []
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        x = np.stack([t[s:s + block_size] for s in chunk])
        y = np.stack([t[s + 1:s + block_size + 1] for s in chunk])
        _, loss = model(x, y)
        losses.extend([float(loss)] * len(chunk))  # equal-size windows -> batch mean applies to each
    return sum(losses) / len(losses)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate loss/perplexity on a held-out file.")
    ap.add_argument("--checkpoint", default="runs/hoh/best.npz")
    ap.add_argument("--compare", default="", help="second checkpoint, evaluated on exactly the same windows")
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    model, device, tok = load_model(a.checkpoint, a.device, return_tokenizer=True)
    if a.mode == "sft":
        if a.compare:
            raise SystemExit("--compare is only implemented for pretrain-mode evaluation")
        examples, _ = load_sft_examples(a.data, tok, model.config.block_size)
        batcher = SFTBatcher(examples, device, a.batch_size)
        losses = []
        for _ in range(a.batches):
            x, y = batcher.get()
            losses.append(float(model(x, y)[1]))
        mean = sum(losses) / len(losses)
        result = {"mean_loss": mean, "perplexity": math.exp(min(mean, 80)), "batches": len(losses)}
    else:
        toks = load_tokens(a.data, tok)
        mean = deterministic_loss(model, toks, model.config.block_size, device, a.batch_size, a.batches)
        result = {"mean_loss": mean, "perplexity": math.exp(min(mean, 80)), "checkpoint": a.checkpoint}
        if a.compare:
            m2, _, t2 = load_model(a.compare, a.device, return_tokenizer=True)
            if t2.to_dict() != tok.to_dict() or m2.config.block_size != model.config.block_size:
                raise SystemExit("compared checkpoints must share tokenizer and block_size")
            other = deterministic_loss(m2, toks, m2.config.block_size, device, a.batch_size, a.batches)
            result.update({"compare_checkpoint": a.compare, "compare_mean_loss": other,
                           "delta_compare_minus_first": other - mean})
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
