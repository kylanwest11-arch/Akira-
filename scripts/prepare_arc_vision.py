"""Download mini-ImageNet (100 everyday object categories) and shrink it to 64x64 for Arc's visual processing.

    python3 scripts/prepare_arc_vision.py

Writes data/arc_vision/train.npz, val.npz (uint8 images + labels) and classes.json
(readable names). Each parquet shard is converted and deleted as it arrives.
"""
import io
import json
import os
import subprocess

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OUT = "data/arc_vision"
BASE = "https://huggingface.co/api/datasets/timm/mini-imagenet/parquet/default"
SIZE = 72          # stored a little larger than the 64 the model sees, for random crops
os.makedirs(OUT, exist_ok=True)


def shrink(b):
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.array(im.resize((SIZE, SIZE), Image.Resampling.BICUBIC))


def convert(split, shards):
    path = os.path.join(OUT, f"{split}.npz")
    if os.path.exists(path):
        print(f"{split}: already done")
        return
    xs, ys = [], []
    for i in range(shards):
        f = os.path.join(OUT, f"{split}_{i}.parquet")
        if not os.path.exists(f):
            subprocess.run(["curl", "-sSL", "-o", f, f"{BASE}/{split}/{i}.parquet"], check=True)
        pf = pq.ParquetFile(f)
        for rb in pf.iter_batches(batch_size=256):
            for r in rb.to_pylist():
                xs.append(shrink(r["image"]["bytes"]))
                ys.append(r["label"])
        os.remove(f)
        print(f"{split}: shard {i + 1}/{shards}, {len(xs):,} images", flush=True)
    np.savez(path, x=np.stack(xs), y=np.array(ys, dtype=np.int64))


# readable names for the WordNet ids that label the classes
meta = json.loads(pq.read_metadata(os.path.join(OUT, "val.parquet")).metadata[b"huggingface"]) \
    if os.path.exists(os.path.join(OUT, "val.parquet")) else None
if meta and not os.path.exists(os.path.join(OUT, "classes.json")):
    wnids = meta["info"]["features"]["label"]["names"]
    index = json.load(open(os.environ.get("IMAGENET_INDEX", "data/arc_vision/imagenet_class_index.json")))
    names = {wnid: name.replace("_", " ") for wnid, name in index.values()}
    json.dump([names.get(w, w) for w in wnids], open(os.path.join(OUT, "classes.json"), "w"), indent=0)
    os.rename(os.path.join(OUT, "val.parquet"), os.path.join(OUT, "validation_0.parquet"))

convert("validation", 1)
convert("train", 13)
