"""Live: Akira's speech brain, a speech recogniser written from scratch.

Audio (16 kHz) -> log-mel spectrogram -> convolutional downsampling ->
transformer encoder -> a letter (or blank) for every 40 ms -> CTC decoding
collapses repeats and blanks into text.

Trained on LibriSpeech (read English audiobooks). It hears clear English
best; noisy rooms, accents and names it never heard are harder.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
N_MELS = 80
HOP = 160          # 10 ms
WIN = 400          # 25 ms
VOCAB = ["_", " ", "'"] + [chr(c) for c in range(ord("a"), ord("z") + 1)]   # "_" is the CTC blank
CHAR = {c: i for i, c in enumerate(VOCAB)}
SIZES = {
    # brain:       width, layers, heads
    "live-lite": (144, 6, 4),
    "live-general": (256, 8, 4),
    "live-max": (384, 12, 6),
}


def _mel_filters(n_fft=512, n_mels=N_MELS, sr=SR, fmin=20, fmax=7600):
    def hz_to_mel(f):
        return 2595 * math.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = np.array([mel_to_hz(m) for m in mels])
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            fb[i - 1, k] = (k - l) / max(1, c - l)
        for k in range(c, r):
            fb[i - 1, k] = (r - k) / max(1, r - c)
    return torch.from_numpy(fb)


_FB = _mel_filters()
_WINDOW = torch.hann_window(WIN)


def features(audio):
    """float waveform at 16 kHz (1-D numpy or tensor) -> (frames, 80) normalised log-mel."""
    x = torch.as_tensor(audio, dtype=torch.float32)
    if x.numel() < WIN:
        x = F.pad(x, (0, WIN - x.numel()))
    spec = torch.stft(x, n_fft=512, hop_length=HOP, win_length=WIN, window=_WINDOW, return_complex=True)
    power = spec.abs() ** 2
    mel = torch.log(torch.clamp(_FB @ power, min=1e-10)).T
    return (mel - mel.mean(0)) / (mel.std(0) + 1e-5)


def encode_text(t):
    t = t.lower()
    return [CHAR[c] for c in t if c in CHAR and c != "_"]


class LiveNet(nn.Module):
    def __init__(self, d=144, layers=6, heads=4):
        super().__init__()
        self.sub = nn.Sequential(                       # 4x fewer frames: one output per 40 ms
            nn.Conv2d(1, d, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GELU())
        self.proj = nn.Linear(d * ((N_MELS + 3) // 4), d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, len(VOCAB))
        self.d = d

    def pos(self, n, device):
        pos = torch.arange(n, device=device)[:, None]
        div = torch.exp(torch.arange(0, self.d, 2, device=device) * (-math.log(10000.0) / self.d))
        pe = torch.zeros(n, self.d, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    @staticmethod
    def out_lengths(lengths):
        for _ in range(2):
            lengths = (lengths - 1) // 2 + 1
        return lengths

    def forward(self, feats, lengths):
        """feats (B, T, 80), lengths (B,) -> log-probs (B, T/4, vocab), out lengths."""
        h = self.sub(feats.unsqueeze(1))                 # B, d, T/4, 20
        B, C, T, M = h.shape
        h = self.proj(h.permute(0, 2, 1, 3).reshape(B, T, C * M))
        h = h + self.pos(T, h.device)
        out_len = self.out_lengths(lengths)
        mask = torch.arange(T, device=h.device)[None] >= out_len[:, None]
        h = self.norm(self.enc(h, src_key_padding_mask=mask))
        return F.log_softmax(self.out(h).float(), dim=-1), out_len


def greedy_decode(logp, length):
    ids = logp[:length].argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(VOCAB[i])
        prev = i
    return " ".join("".join(out).split())


class Live:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        d, layers, heads = SIZES[ck["brain"]]
        self.brain = ck["brain"]
        self.net = LiveNet(d, layers, heads)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.info = {"brain": self.brain, "parameters": sum(p.numel() for p in self.net.parameters()),
                     "steps": ck["state"]["step"], "hours_heard": round(ck["state"]["seconds_of_audio"] / 3600, 1),
                     "word_error_rate": ck["state"].get("wer")}

    @torch.no_grad()
    def transcribe(self, audio):
        """16 kHz float waveform -> text."""
        f = features(audio)
        logp, n = self.net(f[None], torch.tensor([len(f)]))
        return greedy_decode(logp[0], int(n[0]))


def load_best(models_dir="models"):
    for name in ("live-max", "live-general", "live-lite"):
        path = os.path.join(models_dir, f"{name}.pt")
        if os.path.exists(path):
            return Live(path)
    return None


def word_error_rate(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))
