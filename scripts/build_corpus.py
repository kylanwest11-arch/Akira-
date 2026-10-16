"""Build the training corpus for Akira's native Core brains.

Reads the raw downloads in data/raw, writes chat-formatted text, trains one
byte-level BPE tokenizer shared by all native brains, and saves token files:

    data/core/tokenizer.json
    data/core/chat.bin  chat_val.bin     conversations (Akira's voice)
    data/core/know.bin  know_val.bin     Simple Wikipedia leads + short stories

Every conversation is written in Akira's chat format (see akira/native/chat.py).
"""
import argparse
import os
import random
import re
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.native.chat import EOT, SPECIAL_TOKENS, format_chat  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--raw", default="data/raw")
p.add_argument("--out", default="data/core")
p.add_argument("--vocab", type=int, default=8192)
p.add_argument("--max_chat_chars", type=int, default=2400, help="skip longer conversations; small brains learn short ones better")
p.add_argument("--tokenizer", default="release/tokenizer.json",
               help="reuse this tokenizer if it exists, so new brains match the ones already trained")
args = p.parse_args()
random.seed(0)
os.makedirs(args.out, exist_ok=True)

# Other assistants' names in the source data become Akira.
RENAMES = [(re.compile(r"\b(SmolLM2?|SmolLM|Llama|LLaMA|ChatGPT|GPT-4|OpenAssistant|Hermes|Qwen|Claude)\b"), "Akira"),
           (re.compile(r"\b(Hugging ?Face|Meta AI|OpenAI|Nous Research|Anthropic|Alibaba)\b"), "the Akira team")]


def rename(text):
    for pat, rep in RENAMES:
        text = pat.sub(rep, text)
    return text


def conversations():
    out = []
    for i in range(4):
        path = os.path.join(args.raw, f"smoltalk_{i}.parquet")
        for r in pq.read_table(path, columns=["messages", "source"]).to_pylist():
            if r["source"] in ("self-oss-instruct", "longalign"):   # code goes to Una; very long docs don't fit
                continue
            msgs = [{"role": m["role"], "content": rename(m["content"])} for m in r["messages"]]
            if sum(len(m["content"]) for m in msgs) <= args.max_chat_chars:
                out.append(msgs)
    for r in pq.read_table(os.path.join(args.raw, "everyday.parquet"), columns=["messages"]).to_pylist():
        out.append([{"role": m["role"], "content": rename(m["content"])} for m in r["messages"]])
    for r in pq.read_table(os.path.join(args.raw, "dolly.parquet")).to_pylist():
        q = r["instruction"] if not r["context"] else f"{r['context']}\n\n{r['instruction']}"
        if len(q) + len(r["response"]) <= args.max_chat_chars:
            out.append([{"role": "user", "content": q}, {"role": "assistant", "content": rename(r["response"])}])
    return out


IDENTITY_Q = ["Who are you?", "What is your name?", "What's your name?", "Are you ChatGPT?", "Who made you?",
              "What are you?", "Introduce yourself.", "hi, who am I talking to?", "Are you a human?",
              "What can you do?", "Tell me about yourself.", "What should I call you?", "Are you Gemini?",
              "What kind of AI are you?", "Hello! What's your name?", "Who built you?"]
IDENTITY_A = [
    "I'm Akira, your AI assistant. I can answer questions, help you write, work through problems, and use what's on your device, like messages, calendar and files, when you ask me to.",
    "My name is Akira. I'm an AI assistant built to help with everyday questions, writing, planning, code and more.",
    "I'm Akira! I'm an AI assistant. Ask me anything, and I'll do my best to help.",
    "You're talking to Akira, an AI assistant. I'm not a person, but I'm happy to help with whatever you need.",
    "I'm Akira, an AI assistant made by the Akira team. I can chat, explain things, help with writing and code, and work with the information on your device.",
]
GREET_Q = ["hi", "Hello", "hey akira", "Good morning!", "yo", "Hi Akira, how are you?", "hello there"]
GREET_A = ["Hi! I'm Akira. What can I help you with today?", "Hello! How can I help?",
           "Hey! What's on your mind?", "Good to see you! What would you like to do today?"]


def identity():
    convs = []
    for _ in range(3000):
        if random.random() < 0.7:
            convs.append([{"role": "user", "content": random.choice(IDENTITY_Q)},
                          {"role": "assistant", "content": random.choice(IDENTITY_A)}])
        else:
            convs.append([{"role": "user", "content": random.choice(GREET_Q)},
                          {"role": "assistant", "content": random.choice(GREET_A)}])
    return convs


def knowledge():
    docs = []
    t = pq.read_table(os.path.join(args.raw, "simplewiki.parquet"), columns=["title", "text"]).to_pylist()
    for r in t:
        lead = r["text"].strip().split("\n\n")[0].strip()   # the lead paragraph carries the key facts
        if 80 < len(lead) < 2000:
            docs.append(f"{r['title']}\n{lead}")
    stories_path = os.path.join(args.raw, "tinystories_train.txt")
    if not os.path.exists(stories_path):                    # where the first experiments kept it
        stories_path = "/home/user/mini-gpt/data/tinystories/train.txt"
    if os.path.exists(stories_path):
        stories = [s.strip() for s in open(stories_path, encoding="utf-8").read().split(EOT) if s.strip()][:-1]
        docs += stories[:60_000]
    return docs


print("reading conversations...", flush=True)
convs = conversations() + identity()
random.shuffle(convs)
chat_texts = [format_chat(c) for c in convs]
print(f"{len(chat_texts):,} conversations", flush=True)

print("reading knowledge...", flush=True)
know_texts = knowledge()
random.shuffle(know_texts)
print(f"{len(know_texts):,} knowledge documents", flush=True)

if os.path.exists(args.tokenizer):
    print(f"reusing tokenizer {args.tokenizer}", flush=True)
    tok = Tokenizer.from_file(args.tokenizer)
else:
    print("training tokenizer...", flush=True)
    tok = None
if tok is None:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    sample = random.sample(chat_texts, min(80_000, len(chat_texts))) + random.sample(know_texts, min(60_000, len(know_texts)))
    tok.train_from_iterator(sample, trainer=trainer)
tok.save(os.path.join(args.out, "tokenizer.json"))
eot = tok.token_to_id(EOT)


def write(texts, name):
    ids = []
    for i in range(0, len(texts), 5000):
        for enc in tok.encode_batch(texts[i:i + 5000]):
            ids.extend(enc.ids)
            ids.append(eot)
    np.array(ids, dtype=np.uint16).tofile(os.path.join(args.out, name))
    return len(ids)


n_val = 2000
for name, texts in (("chat", chat_texts), ("know", know_texts)):
    nv = write(texts[:n_val], f"{name}_val.bin")
    nt = write(texts[n_val:], f"{name}.bin")
    print(f"{name}: {nt:,} train tokens, {nv:,} val tokens", flush=True)
print("vocab", tok.get_vocab_size())
