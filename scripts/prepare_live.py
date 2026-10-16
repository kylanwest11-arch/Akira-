"""Turn LibriSpeech parquet files into log-mel features for training Live.

    python3 scripts/prepare_live.py            # every data/live/raw/train_*.parquet + validation

Writes data/live/<split>.feats (float16, all utterances end to end) and
<split>.json (offset, frames and transcript of each utterance). Deletes each
parquet after converting it unless --keep is given, to save disk.
"""
import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from akira.native.live import features  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dir", default="data/live")
p.add_argument("--keep", action="store_true")
p.add_argument("--max_seconds", type=float, default=17.0)
args = p.parse_args()


def convert(files, split):
    feats_path = os.path.join(args.dir, f"{split}.feats")
    index_path = os.path.join(args.dir, f"{split}.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"done_files": [], "items": []}
    offset = sum(it["frames"] for it in index["items"])
    with open(feats_path, "ab") as out:
        for f in files:
            name = os.path.basename(f)
            if name in index["done_files"]:
                continue
            pf = pq.ParquetFile(f)
            n = 0
            for rb in pf.iter_batches(batch_size=200, columns=["audio", "text"]):
                for r in rb.to_pylist():
                    audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                    if sr != 16000 or len(audio) > args.max_seconds * sr:
                        continue
                    m = features(audio).numpy().astype(np.float16)
                    out.write(m.tobytes())
                    index["items"].append({"offset": offset, "frames": len(m), "text": r["text"].lower()})
                    offset += len(m)
                    n += 1
            out.flush()
            index["done_files"].append(name)
            json.dump(index, open(index_path, "w"))
            hours = sum(it["frames"] for it in index["items"]) / 100 / 3600
            print(f"{split}: {name} -> {n} utterances, {hours:.1f} h total", flush=True)
            if not args.keep:
                os.remove(f)


raw = os.path.join(args.dir, "raw")
convert(sorted(glob.glob(os.path.join(raw, "validation.parquet"))), "val")
convert(sorted(glob.glob(os.path.join(raw, "train_*.parquet"))), "train")
