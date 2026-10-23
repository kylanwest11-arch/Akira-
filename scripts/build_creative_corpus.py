"""Build Core's creativity corpus: stories and poems written by people, in Akira's chat format.

    python3 scripts/build_creative_corpus.py      # -> data/core/creative.bin, creative_val.bin

Sources (data/raw), all human-written:
    writingprompts_*.parquet   short stories written for Reddit's r/WritingPrompts (euclaise/writingprompts)
    poetry.parquet             573 classic poems with titles and themes (merve/poetry)
    gutenberg_poetry.parquet   3 million lines of public-domain poetry (biglam/gutenberg-poetry-corpus)
Stories with explicit or hateful language are left out, and long ones are skipped: small brains
learn short pieces better.
"""
import glob
import os
import random
import re
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.native.chat import EOT, format_chat  # noqa: E402

RAW, OUT = "data/raw", "data/core"
MAX_CHARS = 2400
random.seed(0)

# a story with any of these is left out
BLOCK = re.compile(r"\b(fuck\w*|shit\w*|cunt|bitch\w*|whore|slut|nigg\w*|fag\w*|retard\w*|rape\w*|raping|porn\w*|sex\w*|"
                   r"naked|nude|penis|vagina|dick|cock|pussy|orgasm\w*|erotic\w*|torture\w*|suicide|gore|incest|"
                   r"nsfw)\b", re.I)
STORY_ASKS = ["Write a short story: {p}", "Write a story about this: {p}", "Can you write me a short story? {p}",
              "Story idea: {p} Write it.", "{p} Write a short story based on that."]
POEM_ASKS = ["Write a poem called \"{t}\".", "Write a {k} poem.", "Write me a poem titled \"{t}\".",
             "Can you write a {k} poem called \"{t}\"?"]
VERSE_ASKS = ["Write a few lines of poetry.", "Write a short poem.", "Write some verse in an old-fashioned style.",
              "Write a short rhyming poem.", "Give me a few lines of poetry."]


def detok(text):
    """WritingPrompts text is pre-tokenized: "You 've finally ... , and says , ``" -> normal spacing."""
    text = text.replace("``", '"').replace("''", '"').replace("<newline>", "\n")
    text = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text)
    text = re.sub(r" ([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([($]) ", r"\1", text)
    text = re.sub(r"[ \t]*\n[ \t\n]*\n[ \t]*", "\n\n", text)                # paragraphs
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def stories(paths):
    out = []
    for path in paths:
        for r in pq.read_table(path).to_pylist():
            prompt = re.sub(r"^\s*\[\s*\w+\s*\]\s*", "", detok(r["prompt"]))     # "[ WP ] ..." tag
            story = detok(r["story"])
            if not (200 <= len(story) <= MAX_CHARS) or BLOCK.search(prompt + " " + story) or "http" in story:
                continue
            out.append((random.choice(STORY_ASKS).format(p=prompt), story))
    return out


def poems():
    out = []
    for r in pq.read_table(f"{RAW}/poetry.parquet").to_pylist():
        body = r["content"].replace("\r", "").strip()
        title = (r["poem name"] or "").strip() or "Untitled"
        kind = (r["type"] or "").split("&")[0].strip().lower() or "short"
        if 80 <= len(body) <= MAX_CHARS and not BLOCK.search(body):
            for _ in range(4):                                       # few poems: show each a few ways
                out.append((random.choice(POEM_ASKS).format(t=title, k=kind), body))
    return out


def verses(n=60_000):
    t = pq.read_table(f"{RAW}/gutenberg_poetry.parquet")
    lines, books = t.column("line").to_pylist(), t.column("gutenberg_id").to_pylist()
    out, i = [], 0
    while i < len(lines) - 12 and len(out) < n * 3:
        size = random.randint(4, 10)
        chunk = lines[i:i + size]
        verse_like = all(15 < len(x) <= 62 for x in chunk) and sum(x[:1].isupper() for x in chunk) >= len(chunk) - 1
        if len(set(books[i:i + size])) == 1 and verse_like and not BLOCK.search(" ".join(chunk)):
            out.append(chunk)
        i += size + random.randint(20, 120)                          # spread over many books
    random.shuffle(out)
    pairs = []
    for chunk in out[:n]:
        if random.random() < 0.5:
            pairs.append((random.choice(VERSE_ASKS), "\n".join(chunk)))
        else:
            pairs.append(("Continue this poem:\n" + "\n".join(chunk[:2]), "\n".join(chunk[2:])))
    return pairs


def to_texts(pairs):
    return [format_chat([{"role": "user", "content": q}, {"role": "assistant", "content": a}]) for q, a in pairs]


train = stories(sorted(glob.glob(f"{RAW}/writingprompts_[0-9].parquet"))) + poems() + verses()
val = stories([f"{RAW}/writingprompts_val.parquet"])[:2000]
random.shuffle(train)
print(f"{len(train):,} creative pieces, {len(val):,} for validation", flush=True)

tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
eot = tok.token_to_id(EOT)


def write(texts, name):
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eot)
    np.array(ids, dtype=np.uint16).tofile(os.path.join(OUT, name))
    return len(ids)


print(f"creative: {write(to_texts(train), 'creative.bin'):,} train tokens, "
      f"{write(to_texts(val), 'creative_val.bin'):,} val tokens")
