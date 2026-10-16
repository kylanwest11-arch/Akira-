"""Train Arc's image recogniser (visual processing) from scratch on mini-ImageNet (100 categories, 64x64).

    python3 scripts/train_arc_vision.py --brain arc-vision-lite --minutes 240

Resumable. Random crops, flips and colour jitter; label smoothing; one-cycle
learning rate over the time budget. Measures accuracy on held-out images.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.native.accel import DEVICE, autocast, describe  # noqa: E402
from akira.native.arc_vision import MEAN, SIZES, STD, VisionNet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--brain", default="arc-vision-lite", choices=list(SIZES))
p.add_argument("--minutes", type=float, required=True)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--save_every", type=float, default=15)
p.add_argument("--models", default="models")
args = p.parse_args()

tr = np.load("data/arc_vision/train.npz")
va = np.load("data/arc_vision/validation.npz")
xtr = torch.from_numpy(tr["x"]).permute(0, 3, 1, 2).contiguous()   # uint8 N,3,72,72
ytr = torch.from_numpy(tr["y"])
xva = torch.from_numpy(va["x"]).permute(0, 3, 1, 2).contiguous()[:, :, 4:68, 4:68]
yva = torch.from_numpy(va["y"])
classes = json.load(open("data/arc_vision/classes.json"))
net = VisionNet(SIZES[args.brain], len(classes)).to(DEVICE, memory_format=torch.channels_last)
opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
state = {"step": 0, "seconds": 0.0, "images": 0, "acc": None, "acc5": None, "best": 0.0, "history": []}
last = os.path.join(args.models, f"{args.brain}.last.pt")
best = os.path.join(args.models, f"{args.brain}.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    state = ck["state"]
    print(f"resuming {args.brain} at step {state['step']}", flush=True)
print(f"{args.brain}: {sum(p_.numel() for p_ in net.parameters()) / 1e6:.1f}M parameters, "
      f"{len(xtr):,} training images, {len(classes)} categories, {describe()}", flush=True)


def augment(xb):
    n = len(xb)
    i, j = np.random.randint(0, 9, 2)
    x = xb[:, :, i:i + 64, j:j + 64].float() / 255
    flip = torch.rand(n) < 0.5
    x[flip] = x[flip].flip(-1)
    x = x * (0.8 + 0.4 * torch.rand(n, 1, 1, 1)) + 0.1 * (torch.rand(n, 3, 1, 1) - 0.5)   # brightness + tint
    return ((x.clamp(0, 1) - MEAN) / STD).contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def evaluate():
    net.eval()
    hit1 = hit5 = 0
    for k in range(0, len(xva), 256):
        x = ((xva[k:k + 256].float() / 255 - MEAN) / STD).contiguous(memory_format=torch.channels_last).to(DEVICE)
        with autocast():
            top = net(x).float().topk(5, -1).indices.cpu()
        y = yva[k:k + 256]
        hit1 += (top[:, 0] == y).sum().item()
        hit5 += (top == y[:, None]).any(-1).sum().item()
    net.train()
    return hit1 / len(xva), hit5 / len(xva)


def save(acc):
    torch.save({"brain": args.brain, "model": net.state_dict(), "optimizer": opt.state_dict(), "state": state,
                "classes": classes}, last + ".tmp")
    os.replace(last + ".tmp", last)
    if acc >= state["best"]:
        state["best"] = acc
        torch.save({"brain": args.brain, "model": net.state_dict(), "state": state, "classes": classes}, best)


t0 = time.time() - state["seconds"]
total = args.minutes * 60
next_save = state["seconds"] + args.save_every * 60
losses = []
while True:
    elapsed = time.time() - t0
    done = elapsed >= total
    if elapsed >= next_save or done:
        state["seconds"] = elapsed
        acc, acc5 = evaluate()
        state["acc"], state["acc5"] = round(acc, 4), round(acc5, 4)
        state["history"].append({"step": state["step"], "minutes": round(elapsed / 60, 1), "acc": state["acc"], "acc5": state["acc5"]})
        print(f"checkpoint | accuracy {acc:.1%} | top-5 {acc5:.1%} | {state['images'] / len(xtr):.1f} epochs", flush=True)
        save(acc)
        json.dump(state["history"], open(os.path.join(args.models, f"{args.brain}.log.json"), "w"))
        next_save = elapsed + args.save_every * 60
        if done:
            break
    progress = elapsed / total
    # one-cycle: warm up for the first 5% of the time, then cosine down
    lr = args.lr * (progress / 0.05 if progress < 0.05 else 0.5 * (1 + np.cos(np.pi * (progress - 0.05) / 0.95)))
    for g in opt.param_groups:
        g["lr"] = max(lr, 1e-5)
    ix = torch.randint(0, len(xtr), (args.batch,))
    x, y = augment(xtr[ix]).to(DEVICE), ytr[ix].to(DEVICE)
    with autocast():
        loss = F.cross_entropy(net(x).float(), y, label_smoothing=0.1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    state["step"] += 1
    state["images"] += args.batch
    losses.append(loss.item())
    if state["step"] % 100 == 0:
        print(f"step {state['step']:6d} | loss {sum(losses) / len(losses):.3f} | {elapsed / 60:.1f}/{args.minutes:g} min", flush=True)
        losses = []
print(f"done: {args.brain}, best accuracy {state['best']:.1%}", flush=True)
