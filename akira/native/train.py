"""Train one of Akira's native language brains.

    python3 -m akira.native.train --brain core-standard --minutes 45
    python3 -m akira.native.train --brain una-lite --minutes 90 --init models/core-standard.pt \
        --mix code:0.7,chat:0.3 --final_mix code:0.7,chat:0.3 --score code:0.7,chat:0.3

Core brains train from scratch. Specialist brains (Una) start from a Core
brain's weights (--init) and continue on their own data mix, so they keep
Akira's voice and general knowledge.

The learning-rate schedule and the data mix follow the wall clock, so a run
always finishes on time: warm up, cosine decay, and for the last 15% of the
time switch to --final_mix (by default, mostly conversations, so the brain
ends in Akira's voice). Resumable: rerun the same command and it continues
from models/<brain>.last.pt.

Distillation: with --teacher, a smaller brain also learns from a bigger brain's predictions (every
next-token probability, not just the one word in the text), so Pro and Standard pick up what Ultra
knows. The loss is (1 - d) * text loss + d * KL(teacher || student) at temperature --distill_temp:

    python3 -m akira.native.train --brain core-pro --name core-pro-next --init models/core-pro.pt \
        --teacher models/core-ultra.pt --minutes 300 --lr_scale 0.5

A second round continues a finished brain at a lower learning rate under its own file name,
so the first round stays in service until the second one is done:

    python3 -m akira.native.train --brain core-pro --name core-pro-r2 --init models/core-pro.pt \
        --minutes 180 --lr_scale 0.5
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .accel import DEVICE, autocast, describe
from .model import GPT, Config

SIZES = {
    # brain id:        layers, heads, width, peak lr
    "core-standard": (4, 4, 256, 1.5e-3),
    "core-pro": (6, 6, 384, 1e-3),
    "core-ultra": (8, 8, 512, 7e-4),
    "una-lite": (4, 4, 256, 6e-4),
    "una-general": (6, 6, 384, 4e-4),
    "una-max": (8, 8, 512, 3e-4),
}


def parse_mix(s):
    parts = dict((k, float(v)) for k, v in (x.split(":") for x in s.split(",")))
    total = sum(parts.values())
    return {k: v / total for k, v in parts.items()}


p = argparse.ArgumentParser()
p.add_argument("--brain", required=True, choices=list(SIZES))
p.add_argument("--data", default="data/core")
p.add_argument("--models", default="models")
p.add_argument("--minutes", type=float, required=True)
p.add_argument("--block", type=int, default=512)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--mix", default="chat:0.65,know:0.35", help="data sources and their shares of batches")
p.add_argument("--final_mix", default="chat:0.9,know:0.1", help="mix for the last 15%% of the time")
p.add_argument("--score", default="chat:0.7,know:0.3", help="how validation losses combine to pick the best checkpoint")
p.add_argument("--init", default=None, help="start from this checkpoint's weights (first run only)")
p.add_argument("--name", default=None, help="file name for this run's checkpoints (default: the brain id)")
p.add_argument("--lr_scale", type=float, default=1.0, help="multiply the brain's peak learning rate (lower for a second round)")
p.add_argument("--teacher", default=None, help="learn from this bigger brain's predictions too (distillation)")
p.add_argument("--distill", type=float, default=0.5, help="share of the loss that comes from the teacher")
p.add_argument("--distill_temp", type=float, default=2.0, help="softens both brains' predictions before comparing")
p.add_argument("--eval_every", type=float, default=5, help="minutes between evaluations")
args = p.parse_args()

mix, final_mix, score_w = parse_mix(args.mix), parse_mix(args.final_mix), parse_mix(args.score)
torch.manual_seed(0)
os.makedirs(args.models, exist_ok=True)
name = args.name or args.brain
best_path = os.path.join(args.models, f"{name}.pt")
last_path = os.path.join(args.models, f"{name}.last.pt")
log_path = os.path.join(args.models, f"{name}.log.json")

tokenizer = open(os.path.join(args.data, "tokenizer.json"), encoding="utf-8").read()
load = lambda n: np.memmap(os.path.join(args.data, n), dtype=np.uint16, mode="r")
sources = sorted(set(mix) | set(final_mix) | set(score_w))
data = {s: (load(f"{s}.bin"), load(f"{s}_val.bin")) for s in sources}

L, H, E, peak_lr = SIZES[args.brain]
peak_lr *= args.lr_scale
from tokenizers import Tokenizer  # noqa: E402
cfg = Config(vocab_size=Tokenizer.from_str(tokenizer).get_vocab_size(), block_size=args.block,
             n_layer=L, n_head=H, n_embd=E, dropout=0.0)
model = GPT(cfg).to(DEVICE)
decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
no_decay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=peak_lr, betas=(0.9, 0.95))

state = {"step": 0, "seconds": 0.0, "tokens": 0, "best": float("inf"), "history": []}
if os.path.exists(last_path):
    ck = torch.load(last_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    state = ck["state"]
    print(f"resuming {args.brain} at step {state['step']} ({state['seconds'] / 60:.1f} min done)", flush=True)
elif args.init:
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    if Config(**ck["config"]) != cfg:
        raise SystemExit(f"{args.init} has a different shape than {args.brain}")
    model.load_state_dict(ck["model"])
    state["init"] = args.init
    print(f"{args.brain} starts from {args.init}", flush=True)
teacher = None
if args.teacher:
    tck = torch.load(args.teacher, map_location="cpu", weights_only=False)
    if tck["config"]["vocab_size"] != cfg.vocab_size:
        raise SystemExit(f"{args.teacher} uses a different tokenizer than {args.brain}")
    teacher = GPT(Config(**tck["config"])).to(DEVICE)
    teacher.load_state_dict({k: v.float() for k, v in tck["model"].items()})
    teacher.eval().requires_grad_(False)
    del tck
    print(f"{name} learns from {args.teacher} ({teacher.num_params() / 1e6:.1f}M parameters), "
          f"distill {args.distill:g} at temperature {args.distill_temp:g}", flush=True)
print(f"{args.brain}: {model.num_params() / 1e6:.1f}M parameters, {args.minutes:g} min budget, mix {mix}, {describe()}", flush=True)

total = args.minutes * 60


def batch(source, split):
    d = data[source][0 if split == "train" else 1]
    ix = np.random.randint(0, len(d) - args.block - 1, size=args.batch)
    x = np.stack([d[i:i + args.block] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1:i + args.block + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)


@torch.no_grad()
def evaluate(n=12):
    model.eval()
    out = {}
    for source in sources:
        losses = []
        for _ in range(n):
            with autocast():
                losses.append(model(*batch(source, "val"))[1].item())
        out[source] = sum(losses) / n
    model.train()
    return out


def lr_at(progress):
    if progress < 0.02:
        return peak_lr * progress / 0.02
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def checkpoint(path, extra=None):
    body = {"model": model.state_dict(), "config": cfg.__dict__, "tokenizer": tokenizer,
            "brain": args.brain, "state": state, **(extra or {})}
    torch.save(body, path + ".tmp")
    os.replace(path + ".tmp", path)


def pick_source(m):
    r, acc = np.random.rand(), 0.0
    for s, w in m.items():
        acc += w
        if r < acc:
            return s
    return s


np.random.seed(state["step"])
t0 = time.time() - state["seconds"]
next_eval = state["seconds"]
while True:
    elapsed = time.time() - t0
    progress = elapsed / total
    if elapsed >= next_eval or progress >= 1:
        losses = evaluate()
        score = sum(score_w.get(s, 0) * v for s, v in losses.items())
        state["seconds"] = elapsed
        state["history"].append({"step": state["step"], "minutes": round(elapsed / 60, 1),
                                 "tokens": state["tokens"], **{k: round(v, 4) for k, v in losses.items()}})
        print(f"step {state['step']:6d} | " + " | ".join(f"{k} {v:.3f}" for k, v in losses.items())
              + f" | {state['tokens'] / 1e6:.1f}M tokens | {elapsed / 60:.1f}/{args.minutes:g} min", flush=True)
        if score < state["best"]:
            state["best"] = score
            checkpoint(best_path)
        checkpoint(last_path, {"optimizer": opt.state_dict()})
        json.dump(state["history"], open(log_path, "w"), indent=1)
        next_eval = elapsed + args.eval_every * 60
        if progress >= 1:
            break

    for g in opt.param_groups:
        g["lr"] = lr_at(progress)
    x, y = batch(pick_source(mix if progress < 0.85 else final_mix), "train")
    with autocast():
        logits, loss = model(x, y)
    if teacher is not None:
        with torch.no_grad(), autocast():
            t_logits, _ = teacher(x)
        T = args.distill_temp
        kl = F.kl_div(F.log_softmax(logits.float().view(-1, logits.size(-1)) / T, -1),
                      F.log_softmax(t_logits.float().view(-1, t_logits.size(-1)) / T, -1),
                      log_target=True, reduction="batchmean") * T * T
        loss = (1 - args.distill) * loss + args.distill * kl
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    state["step"] += 1
    state["tokens"] += x.numel()

print(f"done: {name}, best score {state['best']:.3f}, saved {best_path}", flush=True)
