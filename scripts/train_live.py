"""Train a Live brain (speech recognition) from scratch with CTC.

    python3 scripts/train_live.py --brain live-lite --minutes 300

Resumable. Batches are built from utterances of similar length, with
SpecAugment (random time and frequency masks) so the model doesn't memorise.
Measures word error rate on held-out speech at each checkpoint.
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
from akira.native.live import ARCH, N_MELS, SIZES, encode_text, greedy_decode, word_error_rate  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--brain", default="live-lite", choices=list(SIZES))
p.add_argument("--minutes", type=float, required=True)
p.add_argument("--frames_per_batch", type=int, default=12000, help="about 2 minutes of audio per step")
p.add_argument("--lr", type=float, default=1.5e-3)
p.add_argument("--warmup", type=int, default=300, help="steps of learning-rate warm-up")
p.add_argument("--augment_after", type=float, default=0.2,
               help="share of the run before SpecAugment starts: first learn which sounds are which letters")
p.add_argument("--save_every", type=float, default=20)
p.add_argument("--dir", default="data/live")
p.add_argument("--models", default="models")
p.add_argument("--arch", default="transformer", choices=list(ARCH))
p.add_argument("--width", type=int, default=None, help="override the brain's width")
p.add_argument("--layers", type=int, default=None, help="override the brain's layer count")
args = p.parse_args()


def load(split):
    idx = json.load(open(os.path.join(args.dir, f"{split}.json")))["items"]
    feats = np.memmap(os.path.join(args.dir, f"{split}.feats"), dtype=np.float16, mode="r").reshape(-1, N_MELS)
    return idx, feats


train_idx, train_feats = load("train")
val_idx, val_feats = load("val")
random.seed(0)
val_idx = random.sample(val_idx, 200)
print(f"{len(train_idx):,} training utterances ({sum(i['frames'] for i in train_idx) / 360000:.1f} h)", flush=True)

d, layers, heads = SIZES[args.brain]
d, layers = args.width or d, args.layers or layers
net = ARCH[args.arch](d, layers, heads).to(DEVICE)
shape = {"arch": args.arch, "width": d, "layers": layers}
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
state = {"step": 0, "seconds": 0.0, "seconds_of_audio": 0.0, "history": [], "wer": None, "best_wer": 9.9}
last = os.path.join(args.models, f"{args.brain}.last.pt")
best = os.path.join(args.models, f"{args.brain}.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    state = ck["state"]
    print(f"resuming {args.brain} at step {state['step']}", flush=True)
print(f"{args.brain}: {sum(p_.numel() for p_ in net.parameters()) / 1e6:.1f}M parameters, {describe()}", flush=True)

order = sorted(range(len(train_idx)), key=lambda i: train_idx[i]["frames"])


def make_batch(items, feats, augment):
    xs = [torch.from_numpy(np.array(feats[it["offset"]:it["offset"] + it["frames"]], dtype=np.float32)) for it in items]
    if augment:
        for x in xs:
            for _ in range(2):
                f = random.randint(0, 15)
                f0 = random.randint(0, N_MELS - f)
                x[:, f0:f0 + f] = 0
            for _ in range(max(1, len(x) // 300)):
                t = random.randint(0, min(40, len(x) // 5))
                t0 = random.randint(0, max(0, len(x) - t))
                x[t0:t0 + t] = 0
    lengths = torch.tensor([len(x) for x in xs]).to(DEVICE)
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True).to(DEVICE)
    ys = [torch.tensor(encode_text(it["text"]), device=DEVICE) for it in items]
    return x, lengths, ys


def sample_batch():
    start = random.randint(0, len(order) - 1)
    items, frames = [], 0
    for j in order[start:] + order[:start]:
        it = train_idx[j]
        if items and (len(items) + 1) * max(frames, it["frames"]) > args.frames_per_batch:
            break
        items.append(it)
        frames = max(frames, it["frames"])
    return items


@torch.no_grad()
def evaluate():
    net.eval()
    errs, words = 0.0, 0
    examples = []
    for k in range(0, len(val_idx), 16):
        items = val_idx[k:k + 16]
        x, lengths, _ = make_batch(items, val_feats, False)
        logp, n = net(x, lengths)
        for i, it in enumerate(items):
            hyp = greedy_decode(logp[i], int(n[i]))
            w = len(it["text"].split())
            errs += word_error_rate(it["text"], hyp) * w
            words += w
            if len(examples) < 3:
                examples.append({"heard": it["text"], "wrote": hyp})
    net.train()
    return errs / words, examples


def save():
    body = {"brain": args.brain, "model": net.state_dict(), "state": state, **shape}
    torch.save({**body, "optimizer": opt.state_dict()}, last + ".tmp")
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
        wer, ex = evaluate()
        state["wer"] = round(wer, 4)
        state["history"].append({"step": state["step"], "minutes": round(elapsed / 60, 1), "wer": state["wer"]})
        print(f"checkpoint | word error rate {wer:.1%} | {ex}", flush=True)
        save()
        if wer < state["best_wer"]:
            state["best_wer"] = wer
            torch.save({"brain": args.brain, "model": net.state_dict(), "state": state, **shape}, best)
        json.dump(state["history"], open(os.path.join(args.models, f"{args.brain}.log.json"), "w"))
        next_save = elapsed + args.save_every * 60
        if done:
            break
    progress = elapsed / total
    lr = args.lr * min(1.0, (state["step"] + 1) / args.warmup) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress)))
    for g in opt.param_groups:
        g["lr"] = lr
    items = sample_batch()
    x, lengths, ys = make_batch(items, train_feats, progress >= args.augment_after)
    with autocast():
        logp, n = net(x, lengths)
    targets = torch.cat(ys)
    loss = F.ctc_loss(logp.transpose(0, 1), targets, n, torch.tensor([len(y) for y in ys], device=DEVICE),
                      blank=0, zero_infinity=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    state["step"] += 1
    state["seconds_of_audio"] += float(lengths.sum().item()) / 100
    losses.append(loss.item())
    if state["step"] % 100 == 0:
        print(f"step {state['step']:6d} | ctc {sum(losses) / len(losses):.3f} | "
              f"{state['seconds_of_audio'] / 3600:.0f} h heard | {elapsed / 60:.1f}/{args.minutes:g} min", flush=True)
        losses = []
print(f"done: {args.brain}, best word error rate {state['best_wer']:.1%}", flush=True)
