"""Build Core's reasoning corpus: word problems solved step by step, in Akira's chat format.

    python3 scripts/build_reason_corpus.py      # -> data/core/reason.bin, reason_val.bin

Sources (data/raw): GSM8K (openai/gsm8k) and the grade-school parts of MetaMathQA, which
rephrase and re-solve GSM8K problems many ways. Validation is GSM8K's test split, so the
score measures problems the brains have never seen.
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
MAX_CHARS = 1500
random.seed(0)


def clean_gsm(answer):
    """GSM8K marks calculator steps as <<48/2=24>> and the result as '#### 72'."""
    answer = re.sub(r"<<[^>]*>>", "", answer)
    steps, _, final = answer.partition("####")
    return f"{steps.strip()}\nThe answer is {final.strip()}."


def clean_metamath(response):
    response = re.sub(r"\n#### [^\n]*", "", response)
    response = re.sub(r"\\boxed\{([^}]*)\}", r"\1", response)
    return re.sub(r"\nThe answer is: (.+)$", r"\nThe answer is \1.", response.strip())


def convs(pairs):
    return [format_chat([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
            for q, a in pairs if len(q) + len(a) <= MAX_CHARS]


train = [(r["question"], clean_gsm(r["answer"])) for r in pq.read_table(f"{RAW}/gsm8k_train.parquet").to_pylist()]
mm = pq.read_table(f"{RAW}/metamath.parquet", columns=["type", "query", "response"]).to_pylist()
train += [(r["query"], clean_metamath(r["response"])) for r in mm if r["type"] in ("GSM_AnsAug", "GSM_Rephrased", "GSM_SV", "GSM_FOBAR")]
val = [(r["question"], clean_gsm(r["answer"])) for r in pq.read_table(f"{RAW}/gsm8k_test.parquet").to_pylist()]
random.shuffle(train)

tok = Tokenizer.from_file(os.path.join(OUT, "tokenizer.json"))
eot = tok.token_to_id(EOT)


def write(texts, name):
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eot)
    np.array(ids, dtype=np.uint16).tofile(os.path.join(OUT, name))
    return len(ids)


n_train = write(convs(train), "reason.bin")
n_val = write(convs(val), "reason_val.bin")
print(f"reason: {n_train:,} train tokens, {n_val:,} val tokens")
