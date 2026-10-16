"""Arc: Akira's image-generation brain, a diffusion model written from scratch.

A U-Net learns to remove noise from images. To draw, Arc starts from pure
noise and removes it step by step (DDIM sampling), steered toward the
requested subject with classifier-free guidance.

Arc is trained on CIFAR-10, so it draws 32x32 images of ten subjects:
airplane, car, bird, cat, deer, dog, frog, horse, ship and truck.
"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SYNONYMS = {
    "plane": "airplane", "jet": "airplane", "aeroplane": "airplane", "aircraft": "airplane",
    "automobile": "car", "sports car": "car", "vehicle": "car",
    "kitten": "cat", "kitty": "cat", "puppy": "dog", "doggo": "dog", "pup": "dog",
    "boat": "ship", "sailboat": "ship", "yacht": "ship", "lorry": "truck", "pickup": "truck",
    "pony": "horse", "toad": "frog", "fawn": "deer", "reindeer": "deer", "sparrow": "bird", "parrot": "bird",
}
SIZES = {"arc-lite": 48, "arc-general": 64, "arc-max": 96}   # base channel width


def subject_of(text):
    """Find which of Arc's subjects a request asks for, or None."""
    t = text.lower()
    for phrase, cls in sorted(SYNONYMS.items(), key=lambda kv: -len(kv[0])):
        if phrase in t:
            return CLASSES.index(cls)
    for i, c in enumerate(CLASSES):
        if c in t or c + "s" in t:
            return i
    return None


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.e(F.silu(e))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.o = nn.Conv2d(c, c, 1)
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        y = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.o(y.transpose(-1, -2).reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, base=64, mults=(1, 2, 2), n_classes=len(CLASSES)):
        super().__init__()
        emb = base * 4
        self.base = base
        self.t_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cls = nn.Embedding(n_classes + 1, emb)          # last index = "no subject" for guidance
        self.inp = nn.Conv2d(3, base, 3, padding=1)
        chans = [base * m for m in mults]
        self.down = nn.ModuleList()
        c, skips = base, [base]
        for i, ch in enumerate(chans):
            self.down.append(nn.ModuleList([ResBlock(c, ch, emb), ResBlock(ch, ch, emb)]))
            c = ch
            skips += [ch, ch]
            if i < len(chans) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                skips.append(c)
        self.mid1, self.mid_attn, self.mid2 = ResBlock(c, c, emb), Attn(c), ResBlock(c, c, emb)
        self.up = nn.ModuleList()
        for i, ch in reversed(list(enumerate(chans))):
            blocks = nn.ModuleList()
            for _ in range(3):
                blocks.append(ResBlock(c + skips.pop(), ch, emb))
                c = ch
            self.up.append(blocks)
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1))

    def forward(self, x, t, y):
        e = self.t_mlp(timestep_embedding(t, self.base)) + self.cls(y)
        h = self.inp(x)
        hs = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):
                for b in layer:
                    h = b(h, e)
                    hs.append(h)
            else:
                h = layer(h)
                hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                for b in layer:
                    h = b(torch.cat([h, hs.pop()], dim=1), e)
            else:
                h = layer(h)
        return self.out(h)


class Diffusion:
    """Cosine noise schedule; the network predicts the added noise."""

    def __init__(self, T=1000):
        self.T = T
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        self.abar = (f[1:] / f[0]).clamp(1e-5, 0.9999).float()   # cumulative signal kept at each step

    def noisy(self, x0, t, noise):
        a = self.abar.to(x0.device)[t][:, None, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, labels, steps=50, guidance=3.0, seed=None, progress=None):
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        n = len(labels)
        x = torch.randn(n, 3, 32, 32, generator=g)
        y = torch.tensor(labels)
        null = torch.full_like(y, len(CLASSES))
        ts = torch.linspace(self.T - 1, 0, steps).long()
        for i, t in enumerate(ts):
            tt = t.repeat(n)
            eps_c, eps_u = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, null])).chunk(2)
            eps = eps_u + guidance * (eps_c - eps_u)
            a = self.abar[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            if progress:
                progress(i + 1, steps)
        return x


class Arc:
    """Loads a trained Arc brain and draws images."""

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.brain = ck["brain"]
        self.model = UNet(base=ck["base"])
        self.model.load_state_dict(ck["ema"])
        self.model.eval()
        self.diff = Diffusion()
        self.info = {"brain": self.brain, "parameters": sum(p.numel() for p in self.model.parameters()),
                     "steps": ck["state"]["step"], "images_seen": ck["state"]["images"]}

    def draw(self, subject, n=4, seed=None, upscale=4):
        """Returns a list of PNG bytes."""
        import io

        from PIL import Image
        x = self.diff.sample(self.model, [subject] * n, seed=seed)
        imgs = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
        out = []
        for a in imgs:
            im = Image.fromarray(a).resize((32 * upscale, 32 * upscale), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out


def load_best(models_dir="models"):
    for name in ("arc-max", "arc-general", "arc-lite"):
        path = os.path.join(models_dir, f"{name}.pt")
        if os.path.exists(path):
            return Arc(path)
    return None


def load_cifar(path):
    """CIFAR parquet -> uint8 array (N, 32, 32, 3) and labels, cached as .npz next to it."""
    cache = path.replace(".parquet", ".npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"]
    import io

    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path).to_pylist()
    x = np.stack([np.array(Image.open(io.BytesIO(r["img"]["bytes"])).convert("RGB")) for r in t])
    y = np.array([r["label"] for r in t], dtype=np.int64)
    np.savez(cache, x=x, y=y)
    return x, y
