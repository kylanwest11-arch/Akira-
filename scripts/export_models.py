"""Copy the trained brains into release/ for serving: weights only, stored in bfloat16 (half the size).

    python3 scripts/export_models.py

Training checkpoints keep optimizer state and full-precision weights so training can resume;
a server needs neither. Brains still training are exported from their best checkpoint so far.
"""
import glob
import os

import torch

SRC, DST = "models", "release"
os.makedirs(DST, exist_ok=True)


def shrink(obj):
    if torch.is_tensor(obj):
        return obj.to(torch.bfloat16) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {k: shrink(v) for k, v in obj.items()}
    return obj


for path in sorted(glob.glob(os.path.join(SRC, "*.pt"))):
    name = os.path.basename(path)
    if name.endswith(".last.pt") or name.startswith("router-"):
        continue
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck.pop("optimizer", None)
    if "state" in ck and isinstance(ck["state"], dict):
        ck["state"] = {k: v for k, v in ck["state"].items() if k != "history"}
    for key in ("model", "ema"):
        if key in ck:
            ck[key] = shrink(ck[key])
    out = os.path.join(DST, name)
    torch.save(ck, out + ".tmp")
    os.replace(out + ".tmp", out)
    print(f"{name}: {os.path.getsize(path) / 1e6:.1f} MB -> {os.path.getsize(out) / 1e6:.1f} MB")
