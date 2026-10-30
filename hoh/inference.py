"""Python inference API (no UI)."""
import json, time
from pathlib import Path
import numpy as np
from .generate import load_model
from .sft import messages_to_prompt_ids
from .stopping import is_looping
from .tokenizer import BOS, EOS

CHUNK = 256  # tokens per round when generating without a token limit


class HohGenerator:
    def __init__(self, checkpoint, device="auto", state_dir=None):
        self.checkpoint, self.requested_device, self.state_dir = str(checkpoint), device, state_dir
        self.model, self.device, self.tokenizer = load_model(checkpoint, device, return_tokenizer=True)

    @classmethod
    def from_state(cls, state_dir, device="auto"):
        """Serve whatever the lifelong learner last promoted; call reload_if_changed() between chats."""
        reg = json.loads((Path(state_dir) / "registry.json").read_text(encoding="utf-8"))
        if not reg.get("current_model"):
            raise FileNotFoundError(f"no promoted model yet in {state_dir}")
        return cls(reg["current_model"], device, state_dir)

    def reload_if_changed(self):
        if not self.state_dir:
            return False
        reg = json.loads((Path(self.state_dir) / "registry.json").read_text(encoding="utf-8"))
        if reg.get("current_model") and reg["current_model"] != self.checkpoint:
            self.checkpoint = reg["current_model"]
            self.model, self.device, self.tokenizer = load_model(self.checkpoint, self.requested_device,
                                                                 return_tokenizer=True)
            return True
        return False

    def touch(self):
        """Tell the idle learner that someone is chatting right now."""
        if self.state_dir:
            from .lifelong import touch_activity
            touch_activity(self.state_dir)

    def generate(self, prompt="", max_new_tokens=None, temperature=0.8, top_k=50, top_p=1.0,
                 repetition_penalty=1.0, seed=None, stop=None, use_cache=True, max_seconds=600.0):
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        ids = self.tokenizer.encode(prompt, add_bos=True)  # empty prompt -> just BOS
        return self._run(ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, seed, stop, use_cache,
                         max_seconds)

    def _run_until_done(self, ids, temperature, top_k, top_p, rep, seed, stop, use_cache, max_seconds):
        """No token limit: keep going until the model emits end-of-turn. Context beyond block_size slides.
        Safety valves (not token caps): a repetition-loop detector and `max_seconds` (None disables)."""
        rng = np.random.default_rng(seed)
        ctx, all_ids, start, t0 = self.model.config.block_size, list(ids), len(ids), time.time()
        while True:
            idx = np.asarray([all_ids[-ctx:]], dtype=np.int64)
            out = self.model.generate(idx, CHUNK, temperature, top_k, EOS, top_p, rep, use_cache, rng=rng)
            new = out[0].tolist()[idx.shape[1]:]
            all_ids.extend(new)
            text = self.tokenizer.decode(all_ids[start:])
            cuts = [text.index(s) for s in (stop or []) if s and s in text]
            if cuts:
                return text[:min(cuts)]
            if not new or new[-1] == EOS:
                return text
            if max_seconds and time.time() - t0 > max_seconds:
                return text
            if is_looping(all_ids[start:]):
                return text

    def _run(self, ids, max_new_tokens, temperature, top_k, top_p, rep, seed, stop, use_cache, max_seconds=600.0):
        self.touch()
        if max_new_tokens is None:
            return self._run_until_done(ids, temperature, top_k, top_p, rep, seed, stop, use_cache, max_seconds)
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer or None (no limit)")
        rng = np.random.default_rng(seed)
        idx = np.asarray([ids], dtype=np.int64)
        out = self.model.generate(idx, max_new_tokens, temperature, top_k, EOS, top_p, rep, use_cache, rng=rng)
        text = self.tokenizer.decode(out[0].tolist()[idx.shape[1]:])
        for s in (stop or []):
            if s and s in text:
                text = text[:text.index(s)]
        return text

    def chat(self, messages, **kw):
        """messages: [{'role': 'system|user|assistant|tool', 'content': str}], same format as SFT data."""
        stop = kw.pop("stop", None)
        ids = messages_to_prompt_ids(messages, self.tokenizer)
        return self._run(ids, kw.get("max_new_tokens"), kw.get("temperature", 0.7), kw.get("top_k", 50),
                         kw.get("top_p", 1.0), kw.get("repetition_penalty", 1.0), kw.get("seed"), stop,
                         kw.get("use_cache", True), kw.get("max_seconds", 600.0))

    def chat_fn(self, **kw):
        """Adapter for hoh.agent.Agent."""
        return lambda messages: self.chat(messages, **kw)
