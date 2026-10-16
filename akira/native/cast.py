"""Cast: Akira watching your screen, with a screen reader written from scratch.

Cast is real-time audio (Live) plus screen viewing. To understand a screen,
Cast finds every line of text in a screenshot (find_lines) and reads each
one with a text-line recogniser (CastNet): a convolutional network that turns
a 32-pixel-tall strip into a sequence of features, a bidirectional LSTM that
reads them left to right and right to left, and CTC decoding into characters.

The recogniser is trained only on synthetic screen text that synth_line()
renders on the fly: sentences, menus, buttons, links, paths, numbers, prices
and code, in the fonts installed on the machine, at many sizes, in light and
dark themes.
"""
import glob
import io
import os
import random
import re
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HEIGHT = 32
MAX_WIDTH = 512
CHARS = [chr(c) for c in range(32, 127)]           # printable ASCII
VOCAB = ["_"] + CHARS                              # "_" at 0 is the CTC blank (the real "_" is CHARS index)
INDEX = {c: i + 1 for i, c in enumerate(CHARS)}
SIZES = {
    # brain:        conv widths,            LSTM hidden
    "cast-lite": ((32, 64, 128, 128), 128),
    "cast-general": ((64, 128, 192, 256), 192),
    "cast-max": ((64, 128, 256, 384), 320),
}

# --- text ---------------------------------------------------------------------
ASCII_FOLD = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
              "\u2026": "...", "\u2022": "*", "\u00a0": " ", "\u00d7": "x", "\u2192": "->"}


def to_ascii(text):
    text = "".join(ASCII_FOLD.get(c, c) for c in text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip()


def encode(text):
    return [INDEX[c] for c in text if c in INDEX]


def greedy_decode(logp, length):
    ids = logp[:length].argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(VOCAB[i])
        prev = i
    return "".join(out).strip()


def char_error_rate(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


# --- synthetic screen text ------------------------------------------------------
UI_WORDS = ["File", "Edit", "View", "Help", "Settings", "Save", "Cancel", "OK", "Open", "Close", "Search",
            "Sign in", "Sign out", "Log in", "Submit", "Next", "Back", "Home", "Profile", "Messages", "Inbox",
            "Send", "Reply", "Share", "Download", "Upload", "Delete", "Undo", "Redo", "New tab", "Bookmarks",
            "Notifications", "Account", "Privacy", "Dashboard", "Projects", "Calendar", "Contacts", "Photos",
            "Today", "Yesterday", "Unread", "Archive", "Compose", "Add to cart", "Checkout", "Learn more",
            "Continue", "Get started", "Pricing", "Terms of Service", "Forgot password?", "Remember me"]
FIRST = ["Thomas", "Maya", "Jonas", "Sarah", "Alex", "Priya", "Liam", "Emma", "Noah", "Olivia", "Chris"]
DOMAINS = ["github.com", "google.com", "example.com", "riverbook.io", "news.ycombinator.com", "wikipedia.org"]


def _ui_string(r):
    k = r.random()
    if k < 0.2:
        return " ".join(r.sample(UI_WORDS, r.randint(1, 4)))
    if k < 0.35:
        path = "/".join(r.choice(["docs", "api", "v1", "users", "settings", "blog", "2026", "search"]) for _ in range(r.randint(1, 3)))
        return f"https://{r.choice(DOMAINS)}/{path}" + (f"?q={r.choice(FIRST).lower()}" if r.random() < 0.3 else "")
    if k < 0.45:
        return f"{r.choice(FIRST).lower()}.{r.choice(FIRST).lower()}@{r.choice(DOMAINS)}"
    if k < 0.6:
        return r.choice([f"${r.randint(1, 9999)}.{r.randint(0, 99):02d}", f"{r.randint(1, 12)}:{r.randint(0, 59):02d} {r.choice(['AM', 'PM'])}",
                         f"{r.randint(1, 28)}/{r.randint(1, 12)}/20{r.randint(10, 30)}", f"{r.randint(0, 100)}%",
                         f"{r.randint(1, 999)} unread", f"Oct {r.randint(1, 31)}, 2026", f"+1 555 {r.randint(1000, 9999)}",
                         f"{r.randint(1, 64)} GB of {r.choice([128, 256, 512])} GB used", f"v{r.randint(0, 9)}.{r.randint(0, 20)}.{r.randint(0, 9)}"])
    if k < 0.7:
        return f"{r.choice(FIRST)} {r.choice(['Reed', 'Chen', 'Patel', 'Garcia', 'Kim', 'Smith'])}"
    return r.choice([f"C:\\Users\\{r.choice(FIRST)}\\Documents\\report_{r.randint(1, 99)}.docx",
                     f"~/projects/{r.choice(['akira', 'web', 'api'])}/src/{r.choice(['main', 'app', 'utils'])}.py",
                     f"Error {r.choice([404, 500, 403, 401])}: {r.choice(['Not Found', 'Internal Server Error', 'Forbidden', 'Unauthorized'])}",
                     f"Last edited {r.randint(2, 59)} minutes ago", f"Page {r.randint(1, 20)} of {r.randint(20, 40)}"])


class TextSource:
    """Sentences, chat lines and code from the corpora, plus generated UI strings."""

    def __init__(self, raw_dir="data/raw", cache="data/cast/text_pool.txt", seed=0):
        self.r = random.Random(seed)
        if not os.path.exists(cache):
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            lines = []
            import pyarrow.parquet as pq
            wiki = pq.read_table(os.path.join(raw_dir, "simplewiki.parquet"), columns=["text"]).slice(0, 40000).to_pylist()
            for w in wiki:
                lines += [s.strip() for s in re.split(r"(?<=[.!?])\s+", w["text"][:1500]) if 3 < len(s.strip()) < 200]
            for r in pq.read_table(os.path.join(raw_dir, "everyday.parquet"), columns=["messages"]).to_pylist():
                lines += [m["content"].strip() for m in r["messages"] if len(m["content"]) < 300]
            for r in pq.read_table(os.path.join(raw_dir, "py18k.parquet"), columns=["output"]).to_pylist()[:8000]:
                lines += [ln.rstrip() for ln in r["output"].splitlines() if 2 < len(ln.strip()) < 120]
            lines = [to_ascii(s) for s in lines]
            with open(cache, "w") as f:
                f.write("\n".join(s for s in lines if s))
        self.lines = open(cache).read().splitlines()

    def sample(self, r=None):
        r = r or self.r
        if r.random() < 0.4:
            return _ui_string(r)
        s = r.choice(self.lines)
        if len(s) > 12 and r.random() < 0.7:                   # a window of the line, like a cut-off screen row
            n = r.randint(5, min(60, len(s)))
            start = r.randint(0, len(s) - n)
            s = s[start:start + n].strip()
        return s or "OK"


def system_fonts():
    fonts = []
    for pat in ("/usr/share/fonts/**/*.ttf", "/usr/share/fonts/**/*.otf", "/usr/local/share/fonts/**/*.ttf"):
        fonts += glob.glob(pat, recursive=True)
    skip = ("IPA", "Symbol", "Emoji", "Dingbat", "Math", "Icons")
    return sorted(f for f in fonts if not any(s.lower() in f.lower() for s in skip))


def _colors(r):
    if r.random() < 0.55:      # light theme
        bg = tuple(r.randint(215, 255) for _ in range(3))
        fg = tuple(r.randint(0, 90) for _ in range(3))
    else:                      # dark theme
        bg = tuple(r.randint(0, 50) for _ in range(3))
        fg = tuple(r.randint(170, 255) for _ in range(3))
    if r.random() < 0.15:      # coloured button or banner
        base = [r.randint(0, 200) for _ in range(3)]
        bg = tuple(base)
        fg = (255, 255, 255) if sum(base) < 380 else (0, 0, 0)
    return bg, fg


def synth_line(text, font_path, r, font_cache={}):
    """Render one line of screen text -> (32 x W uint8 grayscale strip, text actually shown)."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    size = r.randint(11, 30)
    key = (font_path, size)
    if key not in font_cache:
        font_cache[key] = ImageFont.truetype(font_path, size)
    font = font_cache[key]
    bg, fg = _colors(r)
    left, top, right, bottom = font.getbbox(text)
    while len(text) > 1 and (right - left) * HEIGHT / max(1, bottom - top + 4) > MAX_WIDTH:
        text = text[:max(1, int(len(text) * 0.85))].rstrip() or text[:1]   # too wide for one strip: show less
        left, top, right, bottom = font.getbbox(text)
    w, h = max(1, right - left), max(1, bottom - top)
    pad_x, pad_y = r.randint(1, 8), r.randint(1, max(2, size // 3))
    im = Image.new("RGB", (w + 2 * pad_x, h + 2 * pad_y), bg)
    ImageDraw.Draw(im).text((pad_x - left, pad_y - top), text, font=font, fill=fg)
    if r.random() < 0.3:                                   # screen scaling and compression softness
        s = r.uniform(0.6, 0.9)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.Resampling.BILINEAR)
    if r.random() < 0.15:
        im = im.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.8)))
    return to_strip(im), text


def to_strip(im, max_width=MAX_WIDTH):
    """Any PIL image of one text line -> 32-pixel-tall grayscale uint8 array, at most max_width wide."""
    from PIL import Image
    g = im.convert("L")
    w = max(1, round(g.width * HEIGHT / max(1, g.height)))
    g = g.resize((min(w, max_width), HEIGHT), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.uint8)
    if a.mean() < 128:                                      # dark theme: make text dark on light, like everything else
        a = 255 - a
    return a


def batch_tensor(strips):
    """List of (32, w) uint8 -> (B, 1, 32, W) float in [-1, 1] padded with white, and widths."""
    W = max(s.shape[1] for s in strips)
    W = (W + 3) // 4 * 4
    x = np.full((len(strips), 1, HEIGHT, W), 255, dtype=np.uint8)
    for i, s in enumerate(strips):
        x[i, 0, :, :s.shape[1]] = s
    return torch.from_numpy(x).float() / 127.5 - 1, torch.tensor([s.shape[1] for s in strips])


# --- model ----------------------------------------------------------------------
class CastNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 128), hidden=128):
        super().__init__()
        c1, c2, c3, c4 = widths

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())
        self.cnn = nn.Sequential(
            block(1, c1), nn.MaxPool2d(2, 2),                  # 16 x W/2
            block(c1, c2), nn.MaxPool2d(2, 2),                 # 8 x W/4
            block(c2, c3), block(c3, c3), nn.MaxPool2d((2, 1)),  # 4 x W/4
            block(c3, c4), nn.MaxPool2d((4, 1)),               # 1 x W/4
        )
        self.rnn = nn.LSTM(c4, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.out = nn.Linear(2 * hidden, len(VOCAB))

    def forward(self, x, widths):
        h = self.cnn(x).squeeze(2).transpose(1, 2)            # B, W/4, C
        h, _ = self.rnn(h)
        return F.log_softmax(self.out(h).float(), -1), (widths + 3) // 4


# --- finding text on a screen ------------------------------------------------------
def find_lines(gray, min_h=7, max_h=70):
    """Grayscale screenshot (H, W uint8) -> boxes [(x0, y0, x1, y1)] around lines of text, top to bottom."""
    from scipy import ndimage
    g = gray.astype(np.int16)
    edge = np.zeros_like(g)
    edge[:, 1:] = np.abs(np.diff(g, axis=1))
    edge[1:, :] = np.maximum(edge[1:, :], np.abs(np.diff(g, axis=0)))
    mask = edge > 40
    # join the letters of a word and the words of a line, but not lines to each other
    mask = ndimage.binary_closing(mask, structure=np.ones((1, 9)))
    mask = ndimage.binary_dilation(mask, structure=np.ones((1, 7)))
    labels, n = ndimage.label(mask)
    boxes = []
    for sl in ndimage.find_objects(labels):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        h, w = y1 - y0, x1 - x0
        if min_h <= h <= max_h and w >= h * 0.8 and w > 10:
            fill = mask[y0:y1, x0:x1].mean()
            if 0.15 < fill < 0.98:
                boxes.append((x0, y0, x1, y1))
    # a box around other boxes is a button or panel border, not text
    def inside(a, b):
        return a != b and b[0] <= a[0] and b[1] <= a[1] and a[2] <= b[2] and a[3] <= b[3]
    boxes = [b for b in boxes if not any(inside(o, b) for o in boxes)]
    boxes.sort(key=lambda b: (b[1] // 8, b[0]))
    return boxes


class Cast:
    """Loads a trained Cast brain and reads screens."""

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.brain = ck["brain"]
        widths, hidden = SIZES[self.brain]
        self.net = CastNet(widths, hidden)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        st = ck["state"]
        self.info = {"brain": self.brain, "parameters": sum(p.numel() for p in self.net.parameters()),
                     "char_error_rate": st.get("cer"), "lines_seen": st.get("lines")}

    @torch.no_grad()
    def read_strips(self, strips):
        out = []
        for k in range(0, len(strips), 32):
            x, w = batch_tensor(strips[k:k + 32])
            logp, n = self.net(x, w)
            out += [greedy_decode(logp[i], int(n[i])) for i in range(len(logp))]
        return out

    def read(self, image_bytes, max_lines=120):
        """Screenshot bytes -> [{"text", "box"}] for each line of text, top to bottom."""
        from PIL import Image
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if im.width > 2000:
            im = im.resize((2000, round(im.height * 2000 / im.width)), Image.Resampling.BILINEAR)
        gray = np.asarray(im.convert("L"))
        boxes = find_lines(gray)[:max_lines]
        strips = []
        for x0, y0, x1, y1 in boxes:
            p = max(2, (y1 - y0) // 5)
            crop = im.crop((max(0, x0 - p), max(0, y0 - p), min(im.width, x1 + p), min(im.height, y1 + p)))
            strips.append(to_strip(crop, max_width=4096))   # the network reads any width; only training is capped
        texts = self.read_strips(strips) if strips else []
        return [{"text": t, "box": b} for t, b in zip(texts, boxes) if len(t.strip()) > 1]


def load_best(models_dir="models"):
    for name in ("cast-max", "cast-general", "cast-lite"):
        path = os.path.join(models_dir, f"{name}.pt")
        if os.path.exists(path):
            return Cast(path)
    return None
