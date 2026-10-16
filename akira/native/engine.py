"""Run Akira's native brains on the device.

    engine = NativeEngine.load("models/core-pro.pt")
    for piece in engine.stream_chat([{"role": "user", "content": "hi"}]):
        print(piece, end="")

Streams text as it is generated, stops at the end-of-turn marker, and keeps
the conversation inside the model's context window by dropping the oldest
turns first (the system turn is always kept).
"""
import os
import threading

import torch
import torch.nn.functional as F

from .chat import END, EOT, format_chat
from .model import GPT, Config


class NativeEngine:
    def __init__(self, ck, path=""):
        from tokenizers import Tokenizer
        self.path = path
        self.brain = ck.get("brain", os.path.basename(path).replace(".pt", ""))
        self.model = GPT(Config(**ck["config"]))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.tok = Tokenizer.from_str(ck["tokenizer"])
        self.stop_ids = {self.tok.token_to_id(END), self.tok.token_to_id(EOT)}
        self.block = self.model.cfg.block_size
        self.params = self.model.num_params()
        self.trained = ck.get("state", {})
        self.lock = threading.Lock()   # one generation at a time per engine

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=False), path)

    def info(self):
        hist = self.trained.get("history") or [{}]
        return {"brain": self.brain, "parameters": self.params, "context": self.block,
                "trained_tokens": self.trained.get("tokens"), "last_eval": hist[-1]}

    def _prompt_ids(self, messages, max_new):
        budget = self.block - max_new
        system = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        while True:
            ids = self.tok.encode(format_chat(system + rest, add_generation_prompt=True)).ids
            if len(ids) <= budget or len(rest) <= 1:
                return ids[-budget:]
            rest = rest[1:]

    @torch.no_grad()
    def stream_chat(self, messages, max_new_tokens=200, temperature=0.7, top_k=40, top_p=0.9,
                    repetition_penalty=1.15, stop=None):
        """Yield decoded text pieces. `stop` is an optional threading.Event to cancel."""
        max_new_tokens = min(max_new_tokens, self.block // 2)
        ids = self._prompt_ids(messages, max_new_tokens)
        with self.lock:
            caches = [{} for _ in self.model.blocks]
            logits, _ = self.model(torch.tensor([ids]), caches=caches, start=0)
            out, shown = [], ""
            pos = len(ids)
            for _ in range(max_new_tokens):
                if stop is not None and stop.is_set():
                    break
                nxt = self._sample(logits[0, -1], out, temperature, top_k, top_p, repetition_penalty)
                if nxt in self.stop_ids:
                    break
                out.append(nxt)
                text = self.tok.decode(out)
                if not text.endswith("�"):   # wait for a whole UTF-8 character
                    yield text[len(shown):]
                    shown = text
                if pos >= self.block:
                    break
                logits, _ = self.model(torch.tensor([[nxt]]), caches=caches, start=pos)
                pos += 1

    def chat(self, messages, **kw):
        return "".join(self.stream_chat(messages, **kw)).strip()

    @staticmethod
    def _sample(logits, history, temperature, top_k, top_p, rep):
        logits = logits.float().clone()
        if rep and history:
            seen = torch.tensor(sorted(set(history[-64:])))
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / rep, vals * rep)
        if temperature <= 0:
            return int(logits.argmax())
        logits = logits / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits[logits < kth] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sp, si = torch.sort(probs, descending=True)
            keep = torch.cumsum(sp, 0) - sp < top_p
            probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1))


LANGUAGE_BRAINS = ("core-", "una-")   # the families that run on this engine


class NativeBrains:
    """Finds and lazily loads whichever native brains have been trained."""

    def __init__(self, models_dir="models"):
        self.dir = models_dir
        self._loaded = {}

    def available(self):
        if not os.path.isdir(self.dir):
            return []
        return sorted(f[:-3] for f in os.listdir(self.dir)
                      if f.endswith(".pt") and not f.endswith(".last.pt") and f.startswith(LANGUAGE_BRAINS))

    def get(self, brain_id):
        if not brain_id.startswith(LANGUAGE_BRAINS):
            return None
        path = os.path.join(self.dir, f"{brain_id}.pt")
        if not os.path.exists(path):
            return None
        mtime = os.path.getmtime(path)
        cached = self._loaded.get(brain_id)
        if cached is None or cached[0] != mtime:   # pick up newer checkpoints while training runs
            cached = (mtime, NativeEngine.load(path))
            self._loaded[brain_id] = cached
        return cached[1]

    def best_for(self, family, rank):
        """The native brain for this family and tier, or the closest smaller one that exists."""
        from .. import brains
        tiers = brains.FAMILIES[family]["tiers"]
        for r in range(rank, -1, -1):
            e = self.get(f"{family}-{tiers[r]}")
            if e:
                return e
        if family != "core":
            return self.best_for("core", rank)
        for r in range(rank + 1, 3):
            e = self.get(f"core-{tiers[r]}")
            if e:
                return e
        return None
