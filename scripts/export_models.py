"""Copy the trained brains into release/ for serving: weights only, stored in bfloat16 (half the size).

    python3 scripts/export_models.py

Training checkpoints keep optimizer state and full-precision weights so training can resume;
a server needs neither. Brains still training are skipped unless --all is given, so a half-trained
brain never replaces a finished one on the server.
"""
import argparse
import glob
import os

import torch

SRC, DST = "models", "release"
ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="also export brains still training (from their best checkpoint so far)")
opts = ap.parse_args()
os.makedirs(DST, exist_ok=True)


def finished(brain):
    """Done in the training queue, or trained outside it (no queue log at all)."""
    jobs = os.path.join(SRC, "jobs")
    if os.path.exists(os.path.join(jobs, brain + ".done")):
        return True
    return not os.path.exists(os.path.join(jobs, brain + ".log"))


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
    if not opts.all and not finished(name[:-3]):
        print(f"{name}: still training, skipped (use --all to include it)")
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
