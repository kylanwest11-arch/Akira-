"""Akira's router brain: a small transformer encoder trained from scratch.

It reads a request and predicts, in one pass of a few milliseconds:
    family   core | arc | una           (Live and Cast are chosen by input modality)
    rank     0 | 1 | 2                  Standard/Pro/Ultra or Lite/General/Max
    intents  which on-device sources the request needs (messages, calendar, ...)

The Alternate Protocol uses it when a trained checkpoint exists and is confident,
and falls back to its hand-written rules otherwise.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

FAMILIES = ["core", "arc", "una"]
INTENTS = ["messages", "contacts", "calendar", "files", "photos", "settings", "calculator", "clock", "relations"]
MAX_LEN = 96


class RouterNet(nn.Module):
    def __init__(self, vocab_size, d=192, layers=3, heads=4):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(MAX_LEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.family = nn.Linear(d, len(FAMILIES))
        self.rank = nn.Linear(d, 3)
        self.intents = nn.Linear(d, len(INTENTS))

    def forward(self, ids, mask):
        pos = torch.arange(ids.size(1), device=ids.device)
        h = self.enc(self.tok(ids) + self.pos(pos), src_key_padding_mask=~mask)
        h = self.norm(h)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.family(pooled), self.rank(pooled), self.intents(pooled)


class Router:
    """Loads models/router.pt and classifies requests."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.tok = Tokenizer.from_str(ck["tokenizer"])
        self.net = RouterNet(self.tok.get_vocab_size(), **ck["dims"])
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.metrics = ck.get("metrics", {})

    @staticmethod
    def load_default(models_dir="models"):
        path = os.path.join(models_dir, "router.pt")
        return Router(path) if os.path.exists(path) else None

    def encode(self, texts):
        rows = [self.tok.encode(t.lower()).ids[:MAX_LEN] or [0] for t in texts]
        n = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), n, dtype=torch.long)
        mask = torch.zeros(len(rows), n, dtype=torch.bool)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = True
        return ids, mask

    @torch.no_grad()
    def predict(self, text):
        fam, rank, intents = self.net(*self.encode([text]))
        fp = F.softmax(fam[0], -1)
        rp = F.softmax(rank[0], -1)
        ip = torch.sigmoid(intents[0])
        f = int(fp.argmax())
        r = int(rp.argmax())
        return {
            "family": FAMILIES[f], "family_confidence": round(float(fp[f]), 3),
            "rank": r, "rank_confidence": round(float(rp[r]), 3),
            "intents": [INTENTS[i] for i in range(len(INTENTS)) if ip[i] > 0.5],
            "intent_scores": {INTENTS[i]: round(float(ip[i]), 3) for i in range(len(INTENTS))},
        }
