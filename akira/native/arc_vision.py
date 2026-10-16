"""Arc's eyes: an image recogniser written from scratch (a small ResNet).

Arc handles images: it draws them (arc.py) and looks at them (this file).
This network looks at a photo and says what it shows, from 100 everyday
categories of mini-ImageNet (animals, foods, vehicles, objects).
"""
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {"arc-vision-lite": (32, 64, 128, 256), "arc-vision-general": (48, 96, 192, 384), "arc-vision-max": (64, 128, 256, 512)}
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Sequential() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class VisionNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), n_classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        layers, c = [], widths[0]
        for i, w in enumerate(widths):
            layers += [Block(c, w, 1 if i == 0 else 2), Block(w, w, 1)]
            c = w
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x):
        h = self.layers(self.stem(x))
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))


def to_tensor(img_uint8):
    """(H, W, 3) uint8 -> normalised (3, H, W) float."""
    x = torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


class ArcVision:
    def __init__(self, path, classes_path="data/arc_vision/classes.json"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.brain = ck["brain"]
        self.classes = ck.get("classes") or json.load(open(classes_path))
        self.net = VisionNet(SIZES[self.brain], len(self.classes))
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.info = {"brain": self.brain, "parameters": sum(p.numel() for p in self.net.parameters()),
                     "accuracy": ck["state"].get("acc"), "top5_accuracy": ck["state"].get("acc5"),
                     "categories": len(self.classes)}

    @torch.no_grad()
    def see(self, image_bytes, k=3):
        """Image file bytes (JPEG/PNG) -> [(label, probability), ...] best first."""
        from PIL import Image
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((64, 64), Image.Resampling.BICUBIC)
        x = to_tensor(np.array(im))[None]
        probs = (F.softmax(self.net(x), -1)[0] + F.softmax(self.net(x.flip(-1)), -1)[0]) / 2
        top = torch.topk(probs, k)
        return [(self.classes[i], round(float(p), 3)) for p, i in zip(top.values, top.indices)]


def load_best(models_dir="models"):
    for name in ("arc-vision-max", "arc-vision-general", "arc-vision-lite"):
        path = os.path.join(models_dir, f"{name}.pt")
        if os.path.exists(path):
            return ArcVision(path)
    return None
