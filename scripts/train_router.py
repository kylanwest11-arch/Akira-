"""Build training data for Akira's router brain and train it from scratch.

Labels come from two places:
  * real requests (first user turns of the chat corpus and coding prompts),
    labelled by the Alternate Protocol's rules, which teaches the router the
    long tail of how people actually phrase things;
  * generated requests for the cases the rules miss: on-device lookups,
    image creation and editing, code problems described without code words,
    and requests whose difficulty is clear from meaning rather than keywords.

    python3 scripts/train_router.py --minutes 8
"""
import argparse
import os
import random
import sys
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira import protocol  # noqa: E402
from akira.native.accel import DEVICE, describe  # noqa: E402
from akira.native.router import FAMILIES, INTENTS, MAX_LEN, RouterNet  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--raw", default="data/raw")
p.add_argument("--tokenizer", default="data/core/tokenizer.json")
p.add_argument("--out", default="models/router.pt")
p.add_argument("--minutes", type=float, default=8)
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
random.seed(1)
torch.manual_seed(1)
torch.set_num_threads(args.threads)

R = random.choice
PEOPLE = ["Thomas", "Maya", "Jonas", "Mom", "Dad", "Sarah", "Alex", "my boss", "Priya", "Liam", "Grandma", "Chris", "Noah", "Emma"]
TOPICS = ["the movies", "dinner", "the trip", "the project", "the keyboard", "rent", "the party", "the game", "the deadline",
          "the budget", "the launch", "the meeting notes", "the concert", "the flight", "the invoice", "Riverbook"]
TIMES = ["", "", "yesterday", "the other day", "last week", "this morning", "on Friday", "recently", "this week", "last month"]
PREFIX = ["", "", "", "Akira, ", "hey akira ", "Akira can you ", "please ", "um, "]


def ex(text, family="core", rank=0, intents=()):
    return {"text": text, "family": family, "rank": rank, "intents": list(intents)}


def local_examples():
    out = []
    for _ in range(9000):
        who, topic, when, pre = R(PEOPLE), R(TOPICS), R(TIMES), R(PREFIX)
        kind = R(["messages", "contacts", "calendar", "files", "photos", "settings", "calculator", "clock", "relations", "mixed"])
        if kind == "messages":
            t = R([f"what was that message {who} sent me about {topic} {when}", f"did {who} text me {when}",
                   f"show my messages from {who}", f"find the text about {topic}", f"what did {who} say about {topic}",
                   f"read me {who}'s last message", f"any new texts from {who}?", f"search my chats for {topic}"])
        elif kind == "contacts":
            t = R([f"what's {who}'s number", f"call {who}", f"find {who} in my contacts", f"email address for {who}",
                   f"give me {who}'s phone number", f"how do I reach {who}"])
        elif kind == "calendar":
            t = R([f"what's on my calendar {R(['today', 'tomorrow', 'this week', 'on Friday'])}", f"when is my meeting about {topic}",
                   f"am I free {R(['tonight', 'on Sunday', 'tomorrow at 3'])}", f"what's my schedule like {R(['today', 'tomorrow'])}",
                   f"when's {topic.replace('the ', '')} on my calendar", f"do I have anything with {who} this week"])
        elif kind == "files":
            t = R([f"find the file about {topic}", f"open the {topic.replace('the ', '')} document", f"where are my notes on {topic}",
                   f"show me the spreadsheet for {topic}", f"find the pdf {who} sent", f"which folder has the {topic.replace('the ', '')} spec"])
        elif kind == "photos":
            t = R([f"show my photos from {R(['the beach', 'last summer', 'the trip', 'the party'])}", f"find pictures of {who}",
                   f"photos of my dog", f"search my camera roll for {R(['sunsets', 'the whiteboard', 'receipts'])}"])
        elif kind == "settings":
            t = R(["turn on dark mode", "turn off wifi", "set brightness to 40", "enable do not disturb", "turn bluetooth off",
                   "make the volume louder", "switch dark mode off", "lower the brightness", "disable wifi", "mute notifications"])
        elif kind == "calculator":
            a, b = random.randint(2, 999), random.randint(2, 99)
            t = R([f"what's {a} times {b}", f"{a} * {b}", f"calculate {b}% of {a}", f"what is {a} divided by {b}",
                   f"square root of {a}", f"how much is {a} + {b}", f"{a}^2", f"tip of {b}% on ${a}"])
        elif kind == "clock":
            t = R(["what time is it", "what's the date today", "what day is it", "what's today's date", "time please"])
        elif kind == "relations":
            t = R([f"show me the documents related to the {topic.replace('the ', '')} decision we made after that meeting",
                   f"what's connected to the {topic.replace('the ', '')} project", f"files linked to my meeting with {who}",
                   f"everything related to {topic}", f"what came out of that meeting with {who}",
                   f"which tasks follow from the decision about {topic}"])
        else:
            t = R([f"check if {who} messaged me about {topic} and add it to my calendar",
                   f"find the file {who} mentioned in their text {when}",
                   f"did {who} send the {topic.replace('the ', '')} notes before our meeting"])
            kind = None
        intents = [kind] if kind else [i for i, pat in protocol.LOCAL.items() if pat.search(t)]
        rank = 1 if kind is None or kind == "relations" else 0
        out.append(ex((pre + t).strip(), "core", rank, intents))
    return out


def arc_examples():
    subj = ["a fox in the snow", "a city at night", "my dog", "a mountain lake", "a coffee shop", "a robot", "a sunset over the sea",
            "a cozy bedroom", "a sports car", "an astronaut", "a logo for my bakery", "a forest path", "a cat wearing a hat"]
    style = ["", "in watercolor", "photorealistic", "as a pixel art sprite", "in the style of a 90s anime", "minimal flat design",
             "with dramatic lighting", "as an oil painting", "cinematic", "neon cyberpunk"]
    out = []
    for _ in range(4000):
        s, st = R(subj), R(style)
        t = R([f"draw {s} {st}", f"generate an image of {s} {st}", f"make a picture of {s} {st}", f"create {s} {st}",
               f"paint {s} {st}", "make this look more cinematic", "remove the background from this photo",
               "make the colors warmer in this picture", "can you upscale this image", "turn this photo into a sketch",
               f"design a poster for {R(TOPICS)}", f"I need a thumbnail for my video about {R(TOPICS)}",
               "make it look like a polaroid", "edit this so the sky is purple", f"give me a wallpaper of {s}",
               "brighten this photo", "fix the lighting in this picture", f"visualize {s} {st}"])
        rank = 2 if len(t.split()) > 9 or "cinematic" in t else 1 if st else 0
        out.append(ex((R(PREFIX) + t).strip(), "arc", rank))
    return out


def una_examples():
    lang = ["Python", "JavaScript", "TypeScript", "Rust", "Go", "SQL", "Java", "C++", "Swift", "Kotlin"]
    out = []
    for _ in range(4000):
        l = R(lang)
        t = R([f"why does my loop never stop", "my app crashes when I click save", "this keeps returning undefined",
               "it says index out of range, what does that mean", f"how do I read a csv in {l}",
               f"write a {l} function that reverses a linked list", "the build fails with a missing module error",
               "fix this function and explain why it was failing", f"convert this to {l}", "my query is really slow",
               f"set up a REST endpoint in {l}", "why is my website layout broken on mobile",
               "explain what a race condition is and how to avoid it", "help me write unit tests for my parser",
               f"design the architecture for a chat app backend in {l} with auth, queues and scaling",
               f"refactor this {l} module to be easier to test", "my git merge has conflicts, what do I do",
               "the server returns 500 errors under load", "segfault when I free the buffer"])
        rank = 2 if any(w in t for w in ("architecture", "scaling", "refactor", "race condition")) else 1 if len(t.split()) > 7 else 0
        out.append(ex(t, "una", rank))
    return out


def core_tier_examples():
    easy = ["what's the capital of Peru", "how tall is Mount Everest", "define photosynthesis", "translate hello to Spanish",
            "who wrote Hamlet", "give me a synonym for happy", "how many ounces in a pound", "is a tomato a fruit",
            "what does RSVP mean", "tell me a joke", "good morning", "thanks!", "recommend a movie", "how do I boil an egg"]
    mid = ["write a short email asking my landlord to fix the heater", "summarize the causes of World War I",
           "explain how vaccines work", "give me a 3 day itinerary for Rome", "help me write a cover letter for a barista job",
           "what are some tips for better sleep", "explain inflation like I'm 12", "write a birthday poem for my sister"]
    hard = ["compare the economic policies of Keynes and Hayek and evaluate which fits a recession better",
            "design a go-to-market strategy for a hardware keyboard startup with a limited budget",
            "analyze the trade-offs between nuclear and solar for a small country's grid in depth",
            "write a detailed business plan for a neighborhood coffee roastery",
            "prove that there are infinitely many primes and explain each step",
            "critique this argument and identify every logical fallacy in it",
            "help me plan my thesis research on urban heat islands, with methods and risks",
            "explain why the Roman Republic collapsed, weighing the competing theories"]
    out = []
    for rank, pool in enumerate((easy, mid, hard)):
        for _ in range(1500):
            out.append(ex((R(PREFIX) + R(pool)).strip(), "core", rank))
    return out


def more_real_examples():
    """Coding requests (Una) and everyday knowledge questions (Core) phrased by real people."""
    out = []
    for r in pq.read_table(os.path.join(args.raw, "py18k.parquet"), columns=["instruction"]).to_pylist()[:6000]:
        rank = 1 if len(r["instruction"].split()) > 15 else 0
        out.append(ex(r["instruction"][:400], "una", rank))
    for r in pq.read_table(os.path.join(args.raw, "dolly.parquet"), columns=["instruction", "category"]).to_pylist():
        if r["category"] in ("open_qa", "general_qa", "classification", "brainstorming", "creative_writing"):
            text = r["instruction"][:400]
            if protocol.CODE.search(text) or protocol.ARC_MAKE.search(text):
                continue
            n = len(text.split())
            rank = 2 if protocol.REASONING.search(text) and n > 12 else 1 if r["category"] in ("brainstorming", "creative_writing") else 0
            out.append(ex(text, "core", rank, [i for i, pat in protocol.LOCAL.items() if pat.search(text)]))
    return out


def real_examples():
    out = []
    t = pq.read_table(os.path.join(args.raw, "smoltalk_0.parquet"), columns=["messages", "source"]).to_pylist()
    random.shuffle(t)
    for r in t[:40000]:
        text = r["messages"][0]["content"] if r["messages"][0]["role"] == "user" else r["messages"][1]["content"]
        text = text[:600]
        route = protocol.route(protocol.Request(text=text))
        fam = route.brain.family if route.brain.family in FAMILIES else "core"
        if r["source"] == "self-oss-instruct":
            fam = "una"
        out.append(ex(text, fam, route.brain.rank, [i for i in route.local_intents if i in INTENTS]))
    return out


print("building router data...", flush=True)
data = local_examples() + arc_examples() + una_examples() + core_tier_examples() + real_examples() + more_real_examples()
random.shuffle(data)
val, train = data[:3000], data[3000:]
counts = {f: sum(d["family"] == f for d in data) for f in FAMILIES}
print(f"{len(train):,} train / {len(val):,} val examples, families {counts}", flush=True)

from tokenizers import Tokenizer  # noqa: E402
tok_json = open(args.tokenizer, encoding="utf-8").read()
tok = Tokenizer.from_str(tok_json)


def tensors(rows):
    enc = tok.encode_batch([r["text"].lower() for r in rows])
    n = min(MAX_LEN, max(len(e.ids) for e in enc))
    ids = torch.zeros(len(rows), n, dtype=torch.long)
    mask = torch.zeros(len(rows), n, dtype=torch.bool)
    for i, e in enumerate(enc):
        k = min(n, len(e.ids)) or 1
        ids[i, :k] = torch.tensor((e.ids or [0])[:k])
        mask[i, :k] = True
    fam = torch.tensor([FAMILIES.index(r["family"]) for r in rows])
    rank = torch.tensor([r["rank"] for r in rows])
    intents = torch.tensor([[float(i in r["intents"]) for i in INTENTS] for r in rows])
    return ids.to(DEVICE), mask.to(DEVICE), fam.to(DEVICE), rank.to(DEVICE), intents.to(DEVICE)


dims = {"d": 192, "layers": 3, "heads": 4}
net = RouterNet(tok.get_vocab_size(), **dims).to(DEVICE)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
print(f"router: {sum(p_.numel() for p_ in net.parameters()) / 1e6:.2f}M parameters, {describe()}", flush=True)


@torch.no_grad()
def evaluate():
    net.eval()
    ids, mask, fam, rank, intents = tensors(val)
    f, r, i = net(ids, mask)
    net.train()
    pred_i = torch.sigmoid(i) > 0.5
    return {"family_acc": round(float((f.argmax(-1) == fam).float().mean()), 4),
            "rank_acc": round(float((r.argmax(-1) == rank).float().mean()), 4),
            "intent_exact": round(float((pred_i == intents.bool()).all(-1).float().mean()), 4)}


start, step, best = time.time(), 0, None
while time.time() - start < args.minutes * 60:
    rows = random.sample(train, 64)
    ids, mask, fam, rank, intents = tensors(rows)
    f, r, i = net(ids, mask)
    loss = F.cross_entropy(f, fam) + 0.5 * F.cross_entropy(r, rank) + F.binary_cross_entropy_with_logits(i, intents)
    opt.zero_grad()
    loss.backward()
    opt.step()
    step += 1
    if step % 200 == 0:
        m = evaluate()
        print(f"step {step} | loss {loss.item():.3f} | {m} | {(time.time() - start) / 60:.1f} min", flush=True)
        if best is None or m["family_acc"] + m["intent_exact"] + 0.5 * m["rank_acc"] > best[0]:
            best = (m["family_acc"] + m["intent_exact"] + 0.5 * m["rank_acc"], m)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"model": net.state_dict(), "dims": dims, "tokenizer": tok_json, "metrics": m}, args.out)
print(f"done: best {best[1]}, saved {args.out}" if best else "done: no training steps ran", flush=True)
