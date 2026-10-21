"""Put a newly trained Core/Una checkpoint into service only if it beats the one in service.

    python3 scripts/promote.py models/core-pro-next.pt models/core-pro.pt

Both are scored on the same validation batches (weighted like train.py's --score). The better
one ends up at the second path; the old one is kept as <name>.prev.pt.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.native.model import GPT, Config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("new")
p.add_argument("current")
p.add_argument("--data", default="data/core")
p.add_argument("--score", default="chat:0.7,know:0.3")
p.add_argument("--batches", type=int, default=40)
args = p.parse_args()
weights = {k: float(v) for k, v in (x.split(":") for x in args.score.split(","))}


def score(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    total = 0.0
    for source, w in weights.items():
        d = np.memmap(os.path.join(args.data, f"{source}_val.bin"), dtype=np.uint16, mode="r")
        rng = np.random.default_rng(0)                       # same batches for both checkpoints
        losses = []
        with torch.no_grad():
            for _ in range(args.batches):
                ix = rng.integers(0, len(d) - 513, size=8)
                x = torch.from_numpy(np.stack([d[i:i + 512] for i in ix]).astype(np.int64))
                y = torch.from_numpy(np.stack([d[i + 1:i + 513] for i in ix]).astype(np.int64))
                losses.append(model(x, y)[1].item())
        print(f"  {os.path.basename(path)} {source}: {np.mean(losses):.3f}")
        total += w * float(np.mean(losses))
    return total


new = score(args.new)
cur = score(args.current) if os.path.exists(args.current) else float("inf")
print(f"new {new:.3f} vs current {cur:.3f}")
if new < cur:
    if os.path.exists(args.current):
        shutil.copy2(args.current, args.current.replace(".pt", ".prev.pt"))
    shutil.copy2(args.new, args.current)
    print(f"promoted {args.new} -> {args.current}")
else:
    print("kept the current brain")
