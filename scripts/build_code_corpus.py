"""Build Una's code corpus with the Core tokenizer (so Una can start from Core's weights).

    data/core/code.bin, data/core/code_val.bin

Sources: coding conversations from smol-smoltalk (self-oss-instruct), the
Python instructions 18k set (both in Akira's chat format), and real Python
files from CodeParrot (as plain files, so Una also learns how code looks).
"""
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
random.seed(0)
tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
eot = tok.token_to_id(EOT)

docs = []
for i in range(4):
    for r in pq.read_table(os.path.join(RAW, f"smoltalk_{i}.parquet"), columns=["messages", "source"]).to_pylist():
        if r["source"] == "self-oss-instruct" and sum(len(m["content"]) for m in r["messages"]) < 4000:
            docs.append(format_chat(r["messages"]))
n_chat = len(docs)
for r in pq.read_table(os.path.join(RAW, "py18k.parquet")).to_pylist():
    q = r["instruction"] + (f"\n\nInput: {r['input']}" if r["input"].strip() else "")
    a = r["output"].strip()
    if not a.startswith("```"):
        a = "```python\n" + re.sub(r"^# Python code\n", "", a) + "\n```"
    docs.append(format_chat([{"role": "user", "content": q}, {"role": "assistant", "content": a}]))
n_instr = len(docs) - n_chat
files = 0
for r in pq.read_table(os.path.join(RAW, "codeparrot_0.parquet"), columns=["content", "path"]).to_pylist():
    c = r["content"]
    if 200 < len(c) < 6000 and "auto-generated" not in c[:500].lower():
        docs.append(f"# {r['path']}\n{c}")
        files += 1
print(f"{n_chat:,} coding conversations, {n_instr:,} instructions, {files:,} Python files", flush=True)
random.shuffle(docs)


def write(texts, name):
    ids = []
    for i in range(0, len(texts), 5000):
        for enc in tok.encode_batch(texts[i:i + 5000]):
            ids.extend(enc.ids)
            ids.append(eot)
    np.array(ids, dtype=np.uint16).tofile(os.path.join(OUT, name))
    return len(ids)


print(f"code_val.bin {write(docs[:1500], 'code_val.bin'):,} tokens", flush=True)
print(f"code.bin {write(docs[1500:], 'code.bin'):,} tokens", flush=True)
