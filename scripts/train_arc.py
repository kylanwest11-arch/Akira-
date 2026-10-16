"""Train an Arc brain (image generation) from scratch on CIFAR-10.

    python3 scripts/train_arc.py --brain arc-lite --minutes 240

Resumable: rerun the same command to continue from models/<brain>.last.pt.
Keeps an exponential moving average (EMA) of the weights, which draws much
cleaner images than the raw weights, and saves a sample grid at each checkpoint.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.native.accel import DEVICE, autocast, describe  # noqa: E402
from akira.native.arc import CLASSES, SIZES, Diffusion, UNet, load_cifar  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--brain", default="arc-lite", choices=list(SIZES))
p.add_argument("--minutes", type=float, required=True)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--save_every", type=float, default=15, help="minutes")
p.add_argument("--models", default="models")
args = p.parse_args()

x_all, y_all = load_cifar("data/arc/train.parquet")
data = torch.from_numpy(x_all).permute(0, 3, 1, 2).contiguous()   # uint8 N,3,32,32
labels = torch.from_numpy(y_all)
model = UNet(base=SIZES[args.brain]).to(DEVICE)
ema = copy.deepcopy(model).eval()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
diff = Diffusion()
state = {"step": 0, "seconds": 0.0, "images": 0, "history": []}
last = os.path.join(args.models, f"{args.brain}.last.pt")
best = os.path.join(args.models, f"{args.brain}.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    ema.load_state_dict(ck["ema"])
    opt.load_state_dict(ck["optimizer"])
    state = ck["state"]
    print(f"resuming {args.brain} at step {state['step']}", flush=True)
n_params = sum(p_.numel() for p_ in model.parameters())
print(f"{args.brain}: {n_params / 1e6:.1f}M parameters, {len(data):,} images, {args.minutes:g} min budget, {describe()}", flush=True)


def save():
    body = {"brain": args.brain, "base": SIZES[args.brain], "ema": ema.state_dict(), "state": state}
    torch.save({**body, "model": model.state_dict(), "optimizer": opt.state_dict()}, last + ".tmp")
    os.replace(last + ".tmp", last)
    torch.save(body, best + ".tmp")
    os.replace(best + ".tmp", best)
    # one sample per subject, for a quick look at progress
    from PIL import Image
    xs = diff.sample(ema, list(range(len(CLASSES))), steps=40, seed=0)
    grid = ((xs.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).numpy()
    Image.fromarray(np.concatenate(list(grid), axis=1)).resize((32 * 10 * 3, 32 * 3), Image.Resampling.NEAREST) \
        .save(os.path.join(args.models, f"{args.brain}.samples.png"))


t0 = time.time() - state["seconds"]
next_save = state["seconds"] + args.save_every * 60
losses = []
while True:
    elapsed = time.time() - t0
    if elapsed >= args.minutes * 60:
        break
    ix = torch.randint(0, len(data), (args.batch,))
    x0 = (data[ix].float() / 127.5 - 1).to(DEVICE)
    flip = (torch.rand(args.batch) < 0.5).to(DEVICE)
    x0[flip] = x0[flip].flip(-1)
    y = labels[ix].clone()
    y[torch.rand(args.batch) < 0.1] = len(CLASSES)        # learn the unguided case too
    y = y.to(DEVICE)
    t = torch.randint(0, diff.T, (args.batch,)).to(DEVICE)
    noise = torch.randn_like(x0)
    lr = args.lr * min(1.0, (state["step"] + 1) / 500)
    for g in opt.param_groups:
        g["lr"] = lr
    with autocast():
        pred = model(diff.noisy(x0, t, noise), t, y)
    loss = F.mse_loss(pred.float(), noise)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    decay = min(0.9995, (1 + state["step"]) / (10 + state["step"]))
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - decay)
    state["step"] += 1
    state["images"] += args.batch
    losses.append(loss.item())
    if state["step"] % 200 == 0:
        avg = sum(losses) / len(losses)
        losses = []
        state["history"].append({"step": state["step"], "minutes": round(elapsed / 60, 1), "loss": round(avg, 4)})
        print(f"step {state['step']:6d} | loss {avg:.4f} | {state['images'] / 1e6:.2f}M images | {elapsed / 60:.1f}/{args.minutes:g} min", flush=True)
    if elapsed >= next_save:
        state["seconds"] = elapsed
        save()
        json.dump(state["history"], open(os.path.join(args.models, f"{args.brain}.log.json"), "w"))
        next_save = elapsed + args.save_every * 60

state["seconds"] = time.time() - t0
save()
print(f"done: {args.brain} after {state['step']} steps", flush=True)
