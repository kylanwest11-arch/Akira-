"""Train a Cast brain (screen reading) from scratch on synthetic screen text.

    python3 scripts/train_cast.py --brain cast-lite --minutes 240

Every batch is freshly rendered (new text, font, size and theme), so the model
never sees the same line twice. Resumable. Measures the character error rate
on a fixed set of held-out rendered lines at each checkpoint.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.native.accel import DEVICE, autocast, describe  # noqa: E402
from akira.native.cast import (SIZES, CastNet, TextSource, batch_tensor, char_error_rate, encode,  # noqa: E402
                               greedy_decode, synth_line, system_fonts)

p = argparse.ArgumentParser()
p.add_argument("--brain", default="cast-lite", choices=list(SIZES))
p.add_argument("--minutes", type=float, required=True)
p.add_argument("--batch", type=int, default=48)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--save_every", type=float, default=15)
p.add_argument("--models", default="models")
args = p.parse_args()

fonts = system_fonts()
source = TextSource()
held_out_fonts = fonts[::10]                               # every 10th font is never trained on
train_fonts = [f for f in fonts if f not in held_out_fonts]
vr = random.Random(12345)
val = [synth_line(source.sample(vr), vr.choice(held_out_fonts), vr) for _ in range(400)]
print(f"{len(train_fonts)} training fonts, {len(held_out_fonts)} held-out fonts, {len(source.lines):,} text lines", flush=True)

widths, hidden = SIZES[args.brain]
net = CastNet(widths, hidden).to(DEVICE)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
state = {"step": 0, "seconds": 0.0, "lines": 0, "cer": None, "best": 9.9, "history": []}
last = os.path.join(args.models, f"{args.brain}.last.pt")
best = os.path.join(args.models, f"{args.brain}.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    state = ck["state"]
    print(f"resuming {args.brain} at step {state['step']}", flush=True)
print(f"{args.brain}: {sum(p_.numel() for p_ in net.parameters()) / 1e6:.1f}M parameters, {describe()}", flush=True)
rng = random.Random(state["step"])


@torch.no_grad()
def evaluate():
    net.eval()
    errs, chars, examples = 0.0, 0, []
    for k in range(0, len(val), 32):
        chunk = val[k:k + 32]
        x, w = batch_tensor([s for s, _ in chunk])
        logp, n = net(x.to(DEVICE), w.to(DEVICE))
        for i, (_, text) in enumerate(chunk):
            hyp = greedy_decode(logp[i], int(n[i]))
            errs += char_error_rate(text, hyp) * len(text)
            chars += len(text)
            if len(examples) < 3:
                examples.append({"shown": text, "read": hyp})
    net.train()
    return errs / chars, examples


def save():
    torch.save({"brain": args.brain, "model": net.state_dict(), "optimizer": opt.state_dict(), "state": state}, last + ".tmp")
    os.replace(last + ".tmp", last)


t0 = time.time() - state["seconds"]
total = args.minutes * 60
next_save = state["seconds"] + args.save_every * 60
losses = []
while True:
    elapsed = time.time() - t0
    done = elapsed >= total
    if elapsed >= next_save or done:
        state["seconds"] = elapsed
        cer, ex = evaluate()
        state["cer"] = round(cer, 4)
        state["history"].append({"step": state["step"], "minutes": round(elapsed / 60, 1), "cer": state["cer"]})
        print(f"checkpoint | character error rate {cer:.1%} | {ex}", flush=True)
        save()
        if cer < state["best"]:
            state["best"] = cer
            torch.save({"brain": args.brain, "model": net.state_dict(), "state": state}, best)
        json.dump(state["history"], open(os.path.join(args.models, f"{args.brain}.log.json"), "w"))
        next_save = elapsed + args.save_every * 60
        if done:
            break
    progress = elapsed / total
    lr = args.lr * min(1.0, (state["step"] + 1) / 500) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress)))
    for g in opt.param_groups:
        g["lr"] = lr
    lines = [synth_line(source.sample(rng), rng.choice(train_fonts), rng) for _ in range(args.batch)]
    x, w = batch_tensor([s for s, _ in lines])
    ys = [torch.tensor(encode(t) or [1], device=DEVICE) for _, t in lines]
    with autocast():
        logp, n = net(x.to(DEVICE), w.to(DEVICE))
    loss = F.ctc_loss(logp.transpose(0, 1), torch.cat(ys), n, torch.tensor([len(y) for y in ys], device=DEVICE),
                      blank=0, zero_infinity=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    state["step"] += 1
    state["lines"] += len(lines)
    losses.append(loss.item())
    if state["step"] % 100 == 0:
        print(f"step {state['step']:6d} | ctc {sum(losses) / len(losses):.3f} | {state['lines'] / 1e6:.2f}M lines | "
              f"{elapsed / 60:.1f}/{args.minutes:g} min", flush=True)
        losses = []
print(f"done: {args.brain}, best character error rate {state['best']:.1%}", flush=True)
