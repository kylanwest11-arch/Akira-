# Akira on a Mac (Apple M1, M2, M3, M4)

Akira runs on your Mac and can train its smaller brains on the Mac's built-in GPU.

## 1. One-time setup

1. Open **Terminal** (Spotlight: press Cmd+Space, type "Terminal").
2. Install Apple's developer tools (includes Git). A window pops up; click **Install**:
   ```
   xcode-select --install
   ```
3. Install Python 3.12 from https://www.python.org/downloads/macos/ (the "macOS 64-bit universal2 installer").
4. Download Akira and install what it needs:
   ```
   cd ~
   git clone https://github.com/kylanwest11-arch/Akira-.git
   cd Akira-
   python3 -m pip install torch numpy tokenizers fastapi uvicorn pillow scipy pyarrow soundfile pytest
   ```

## 2. Use Akira

```
cd ~/Akira-
git pull
AKIRA_MODELS=release python3 -m akira.server
```

Then open **http://localhost:8000/app** in Safari or Chrome. Press Ctrl+C in Terminal to stop.

## 3. Train on the Mac's GPU (optional)

The M1's GPU suits Akira's smaller brains. Keep the Mac plugged in and awake (System Settings → Battery →
prevent automatic sleeping when the display is off), because training runs for hours.

```
cd ~/Akira-
python3 scripts/download_data.py --brains core,cast          # about 1.5 GB
python3 scripts/build_corpus.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_cast.py --brain cast-lite --minutes 240
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/train_router.py --minutes 30
```

Each trainer prints `GPU: Apple silicon (mps)` when it is using the Mac's GPU. Training stops and resumes
where it left off if you press Ctrl+C and run the same command again.

Good fits for an M1 with 8–16 GB of memory: Cast Lite, the router, Arc vision Lite, Core Standard.
The bigger brains (Core Pro/Ultra, Live General/Max, Arc Max) are better trained on a GPU in the cloud
(`notebooks/train_on_colab.ipynb`, free).

When a brain finishes, ship it so your Akira server uses it:

```
python3 scripts/export_models.py
git add release && git commit -m "New trained brains" && git push
```
