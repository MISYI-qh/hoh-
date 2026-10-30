import argparse, json
from pathlib import Path
from .tokenizer import BPETokenizer, load_tokenizer, ByteTokenizer
from .data import read_text_file


def learn_main(argv=None):
    ap = argparse.ArgumentParser(description="Append BPE merges to a tokenizer (ids of old tokens never change).")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="", help="existing tokenizer.json to extend (default: fresh byte tokenizer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--min-freq", type=int, default=4)
    a = ap.parse_args(argv)
    base = load_tokenizer(a.base) if a.base else ByteTokenizer()
    tok = BPETokenizer(getattr(base, "merges", []))
    before = tok.vocab_size
    n = tok.learn_merges([read_text_file(a.data)], a.new_tokens, a.min_freq)
    Path(a.out).write_text(json.dumps(tok.to_dict()), encoding="utf-8")
    print(f"vocab {before} -> {tok.vocab_size} (+{n}); saved {a.out}")
    return tok


def migrate_main(argv=None):
    from .checkpoint import load_checkpoint
    from .config import ModelConfig
    from .growth import migrate_vocab, fresh_checkpoint
    ap = argparse.ArgumentParser(description="Grow a checkpoint's embeddings to match an extended tokenizer.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ck = load_checkpoint(a.checkpoint)
    mc = ModelConfig(**ck["model_config"]).validate()
    tok = load_tokenizer(a.tokenizer)
    m, new_mc = migrate_vocab(ck["model"], mc, tok)
    fresh_checkpoint(a.out, m, new_mc, tok)
    print(f"migrated vocab {mc.vocab_size} -> {new_mc.vocab_size}; optimizer state NOT carried over; saved {a.out}")
