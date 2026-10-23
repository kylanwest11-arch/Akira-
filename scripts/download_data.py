"""Download the training data for Akira's brains from Hugging Face.

    python3 scripts/download_data.py                       # everything (about 9 GB)
    python3 scripts/download_data.py --brains core,una     # only what those brains need

Files that already exist are skipped, so it is safe to rerun after an interruption.
Arc's recogniser (mini-ImageNet) is downloaded by scripts/prepare_arc_vision.py itself,
which shrinks each file as it arrives.
"""
import argparse
import os
import subprocess

HF = "https://huggingface.co/api/datasets"
FILES = {
    "core": [
        *[(f"data/raw/smoltalk_{i}.parquet", f"{HF}/HuggingFaceTB/smol-smoltalk/parquet/default/train/{i}.parquet") for i in range(4)],
        ("data/raw/everyday.parquet", f"{HF}/HuggingFaceTB/everyday-conversations-llama3.1-2k/parquet/default/train_sft/0.parquet"),
        ("data/raw/dolly.parquet", f"{HF}/databricks/databricks-dolly-15k/parquet/default/train/0.parquet"),
        ("data/raw/simplewiki.parquet", f"{HF}/wikimedia/wikipedia/parquet/20231101.simple/train/0.parquet"),
        ("data/raw/gsm8k_train.parquet", f"{HF}/openai/gsm8k/parquet/main/train/0.parquet"),
        ("data/raw/gsm8k_test.parquet", f"{HF}/openai/gsm8k/parquet/main/test/0.parquet"),
        ("data/raw/metamath.parquet", f"{HF}/meta-math/MetaMathQA/parquet/default/train/0.parquet"),
        ("data/raw/writingprompts_0.parquet", f"{HF}/euclaise/writingprompts/parquet/default/train/0.parquet"),
        ("data/raw/writingprompts_val.parquet", f"{HF}/euclaise/writingprompts/parquet/default/validation/0.parquet"),
        ("data/raw/poetry.parquet", f"{HF}/merve/poetry/parquet/default/train/0.parquet"),
        ("data/raw/gutenberg_poetry.parquet", f"{HF}/biglam/gutenberg-poetry-corpus/parquet/default/train/0.parquet"),
        ("data/raw/py18k.parquet", f"{HF}/iamtarun/python_code_instructions_18k_alpaca/parquet/default/train/0.parquet"),
        # the first 200 MB of TinyStories is plenty for fluency
        ("data/raw/tinystories_train.txt", "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt", "0-209715199"),
    ],
    "una": [
        ("data/raw/codeparrot_0.parquet", f"{HF}/codeparrot/codeparrot-clean-valid/parquet/default/train/0.parquet"),
    ],
    "arc": [
        ("data/arc/train.parquet", f"{HF}/uoft-cs/cifar10/parquet/plain_text/train/0.parquet"),
        ("data/arc/test.parquet", f"{HF}/uoft-cs/cifar10/parquet/plain_text/test/0.parquet"),
    ],
    "live": [
        ("data/live/raw/validation.parquet", f"{HF}/openslr/librispeech_asr/parquet/clean/validation/0.parquet"),
        *[(f"data/live/raw/train_{i}.parquet", f"{HF}/openslr/librispeech_asr/parquet/clean/train.100/{i}.parquet") for i in range(7)],
    ],
    "arc-vision": [
        ("data/arc_vision/imagenet_class_index.json",
         "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"),
        ("data/arc_vision/val.parquet", f"{HF}/timm/mini-imagenet/parquet/default/validation/0.parquet"),
    ],
    "cast": [],   # Cast renders its own screens from the Core text (download "core" too)
}
NEEDS = {"una": ["core"], "cast": ["core"]}   # Una builds on Core's tokenizer; Cast reads Core's text

p = argparse.ArgumentParser()
p.add_argument("--brains", default=",".join(FILES))
args = p.parse_args()
wanted = []
for b in args.brains.split(","):
    for dep in NEEDS.get(b.strip(), []) + [b.strip()]:
        if dep not in wanted:
            wanted.append(dep)

for brain in wanted:
    for item in FILES[brain]:
        path, url = item[0], item[1]
        done_marker = path + ".done"
        if os.path.exists(done_marker) or (os.path.exists(path) and not path.endswith(".txt") and os.path.getsize(path) > 0):
            print(f"have {path}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cmd = ["curl", "-sSL", "--fail", "--retry", "3", "-o", path, url]
        if len(item) > 2:
            cmd[1:1] = ["-r", item[2]]
        print(f"downloading {path}", flush=True)
        subprocess.run(cmd, check=True)
        open(done_marker, "w").close()
print("data ready for:", ", ".join(wanted))
