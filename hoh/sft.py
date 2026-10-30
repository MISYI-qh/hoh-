"""Torch-free SFT / JSONL parsing and loss-mask construction (unit-testable without PyTorch)."""
import json
import random
from pathlib import Path
from .tokenizer import BOS, EOS

IGNORE_INDEX = -100
ROLES = {"system", "user", "assistant", "tool"}


class DatasetError(ValueError):
    pass


def _flatten(content, where):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict) and isinstance(x.get("text"), str):
                parts.append(x["text"])
            else:
                raise DatasetError(f"{where}: unsupported content part {x!r}")
        return " ".join(parts)
    raise DatasetError(f"{where}: content must be a string or list, got {type(content).__name__}")


def read_jsonl(path):
    """Return [(lineno, obj)]; raises DatasetError with file:line on any bad line."""
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"JSONL file not found: {p}")
    records = []
    try:
        with p.open("r", encoding="utf-8-sig") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise DatasetError(
                        f"{p}:{lineno}: invalid JSON ({e.msg}). Each line must hold exactly one JSON object."
                    ) from e
                if not isinstance(obj, dict):
                    raise DatasetError(f"{p}:{lineno}: each line must be a JSON object, got {type(obj).__name__}")
                records.append((lineno, obj))
    except UnicodeDecodeError as e:
        raise DatasetError(f"{p}: not valid UTF-8 ({e})") from e
    return records


def record_to_segments(obj, where="record", require_answer=True):
    """-> list of (text, trained, add_eos). Only assistant text (+EOS) is trained."""
    if "messages" in obj:
        msgs = obj["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise DatasetError(f"{where}: 'messages' must be a non-empty list")
        segs, has_answer = [], False
        for i, m in enumerate(msgs):
            if not isinstance(m, dict):
                raise DatasetError(f"{where}: message {i} must be an object")
            role = str(m.get("role", "")).strip().lower()
            if role not in ROLES:
                raise DatasetError(f"{where}: message {i} has invalid role {m.get('role')!r}")
            content = _flatten(m.get("content", ""), f"{where} message {i}")
            if role == "assistant":
                if not content.strip():
                    raise DatasetError(f"{where}: assistant message {i} is empty")
                segs.append(("Assistant: ", False, False))
                segs.append((content + "\n", True, True))
                has_answer = True
            else:
                segs.append((f"{role.capitalize()}: {content}\n", False, False))
        if require_answer and not has_answer:
            raise DatasetError(f"{where}: no assistant message to learn from")
        return segs
    if "prompt" in obj and "response" in obj:
        if not isinstance(obj["prompt"], str) or not isinstance(obj["response"], str):
            raise DatasetError(f"{where}: 'prompt' and 'response' must be strings")
        if not obj["response"].strip():
            raise DatasetError(f"{where}: empty 'response'")
        return [(f"User: {obj['prompt']}\nAssistant: ", False, False), (obj["response"] + "\n", True, True)]
    if "text" in obj:
        if not isinstance(obj["text"], str) or not obj["text"].strip():
            raise DatasetError(f"{where}: 'text' must be a non-empty string")
        return [(obj["text"], True, True)]
    raise DatasetError(f"{where}: unrecognized record; need 'messages', 'prompt'+'response', or 'text'")


def record_to_text(obj, where="record"):
    return "".join(t for t, _, _ in record_to_segments(obj, where))


def build_example(segments, tokenizer, block_size):
    """-> (ids, mask) with mask[i]=True iff token i is a training target; None if nothing learnable."""
    ids, mask = [BOS], [False]
    for text, trained, add_eos in segments:
        t = tokenizer.encode(text)
        ids += t
        mask += [trained] * len(t)
        if add_eos:
            ids.append(EOS)
            mask.append(True)
    ids, mask = ids[: block_size + 1], mask[: block_size + 1]
    if not any(mask[1:]):
        return None
    return ids, mask


def example_to_xy(ids, mask):
    """Next-token shift: x=ids[:-1]; y[i]=ids[i+1] if that token is trained else IGNORE_INDEX."""
    x = ids[:-1]
    y = [ids[i + 1] if mask[i + 1] else IGNORE_INDEX for i in range(len(ids) - 1)]
    return x, y


def load_sft_examples(path, tokenizer, block_size):
    p = Path(path)
    if p.suffix.lower() != ".jsonl":
        raise DatasetError(f"SFT mode needs a .jsonl file, got {p}")
    examples, dropped = [], 0
    for lineno, obj in read_jsonl(p):
        ex = build_example(record_to_segments(obj, f"{p}:{lineno}"), tokenizer, block_size)
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    if not examples:
        raise DatasetError(f"{p}: no usable SFT examples (dropped {dropped} with nothing to learn after truncation)")
    if len(examples) < 2:
        raise DatasetError(f"{p}: need at least 2 examples (1 train + 1 validation); got {len(examples)}")
    return examples, dropped


def split_items(items, val_fraction=0.02, seed=1337):
    """Deterministic example-level split; val and train never share an item."""
    n = len(items)
    if n < 2:
        raise DatasetError("need at least 2 items to split into train/validation")
    n_val = max(1, int(n * val_fraction))
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    val = [items[i] for i in idx[:n_val]]
    train = [items[i] for i in idx[n_val:]]
    return train, val


def messages_to_prompt_ids(messages, tokenizer):
    """Inference-time prompt: identical serialization to SFT training, ending with the
    'Assistant: ' header so the model continues with its answer. messages: [{'role','content'}]."""
    segs = record_to_segments({"messages": messages}, "prompt", require_answer=False)
    ids = [BOS]
    for text, _trained, add_eos in segs:
        ids += tokenizer.encode(text)
        if add_eos:
            ids.append(EOS)
    return ids + tokenizer.encode("Assistant: ")
