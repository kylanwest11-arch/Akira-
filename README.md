# Akira

One personal assistant with different brains for different jobs. You talk to Akira; Akira decides which
brain does the work, whether it runs on your device or in the cloud, and starts working while you are
still talking.

```
                     you ──▶  First-Hand Processing (fhp.py)      starts lookups while you type or speak
                                        │
                     Alternate Protocol (protocol.py)             which brain? which tier? local, cloud or both?
                       router brain (trained) + rules
                         │                        │
      LAC: Local Access Center (lac.py)     CAC: Cloud Access Center
      messages, contacts, calendar,         the Akira server's larger brains
      files, photos, settings, maths,       (Pro, Ultra, Max) and the internet
      Relations & References graph          (web.py: search, Wikipedia, pages)

      Every brain is Akira's own model, trained from scratch in akira/native. No third-party AI.
```

## Brains

| Family | Tiers | Job | Native brain (trained from scratch here) |
|---|---|---|---|
| Router | – | Alternate Protocol: picks family, tier and device sources | transformer encoder, 2.9M params |
| Core | Standard → Pro → Ultra | general intelligence | GPT language models: 5.3M / 13.8M / 29.4M params |
| Una | Lite → General → Max | code | Core's weights fine-tuned on 57M tokens of code |
| Arc | Lite → General → Max | images: drawing and recognising | drawing: diffusion U-Net, 3.6M / 6.3M / 14.2M params, 10 subjects (CIFAR-10); recognising: ResNet, 2.8M / 6.3M / 11.2M params, 100 categories (mini-ImageNet) |
| Live | Lite → General → Max | speech to text | CTC transformer, 2.1M / 8.2M / 25.6M params, 51 h of LibriSpeech |
| Cast | Lite → General → Max | live screen viewing + voice | screen reader: finds text lines and reads them with a CNN + LSTM (CTC), 1.1M / 2.7M / 6.2M params, trained on synthetic screen text in 80 fonts; plus Live |

**Internet access:** questions about anything current (news, prices, "search for…", a link) and factual questions
go online. Akira searches the web (DuckDuckGo, with Bing as a backup) and Wikipedia itself (`akira/web.py`) and
answers with links to its sources. It also **browses websites**: "go to bbc.com" opens the page and shows its main
text, sections and numbered links; "open 3" follows a link; "take me to wikipedia" finds the site first; "search
python.org for asyncio" searches within a site. No API keys. `AKIRA_WEB=off` turns it off.

**Akira uses no third-party AI and no API keys.** Every answer comes from Akira's own brains, the device tools, or
the internet. Answer quality grows with training: bigger brains and more training (a GPU) make Akira smarter.

## Run it

```bash
pip install -r requirements.txt
python3 -m akira.server                    # http://localhost:8000: demo page with the widget
python3 -m pytest -q tests
```

## Put Akira in a website

```html
<script src="https://YOUR-AKIRA-SERVER/embed.js" defer></script>
```

The widget has buttons to add an image, share your screen (Cast) and talk (Live).
Optional: `data-key` (API key), `data-position="left"`, `data-open="true"`, `data-launcher="false"`.
From your own code: `Akira.open()`, `Akira.send("…")`, `Akira.chat({messages, onEvent})`, `Akira.listen("partial text")`.

## Put Akira in an app

Any language that can make HTTP requests:

| Endpoint | Body | Returns |
|---|---|---|
| `POST /v1/listen` | `{"session", "text"}`, sent as the user types or speaks | what Akira has understood so far |
| `POST /v1/chat` | `{"messages", "images"?, "modality"?: "text" \| "audio" \| "screen", "session"?, "stream"?}` | Server-Sent Events: `route`, `understood`, `tool`, `text`, `done` (or one JSON object with `"stream": false`) |
| `POST /v1/transcribe` | `{"audio"}`: base64 16 kHz mono 16-bit PCM | `{"text"}` from Live |
| `GET /v1/status` | – | brains, which are trained, whether the cloud is connected |
| `GET /v1/training` | – | the training queue: done, training, waiting |

Protect a public server with `AKIRA_API_KEYS=key1,key2` and `AKIRA_ORIGINS=https://yoursite.com`.
In a Python app, use the `Akira` class directly. Connect it to the real device by subclassing
`DeviceData` in `akira/device.py`; the bundled `DemoDevice` is sample data.

## Train the native brains

**Fastest: a free GPU.** Open `notebooks/train_on_colab.ipynb` in Google Colab
(colab.research.google.com → File → Open notebook → GitHub → `kylanwest11-arch/Akira-`), switch the
runtime to a T4 GPU, and run the cells. Checkpoints are saved to your Google Drive, and every brain resumes
after a disconnect.

**On any machine:**


```bash
python3 scripts/download_data.py        # training data from Hugging Face (--brains core,una,... for a subset)
python3 scripts/build_corpus.py         # Core: chat + knowledge corpus -> data/core (reuses release/tokenizer.json)
python3 scripts/build_code_corpus.py    # Una: code corpus with the same tokenizer
python3 scripts/prepare_live.py         # Live: LibriSpeech -> log-mel features (download parquet into data/live/raw)
python3 scripts/prepare_arc_vision.py   # Arc's recogniser: mini-ImageNet -> 72x72 images
# Cast renders its own training screens; it only needs the corpora in data/raw
# Arc reads CIFAR-10 from data/arc/train.parquet

scripts/start_all.sh                    # training queue + API server; run it again after any restart
```

The queue runs the jobs in `models/queue.json` one at a time; edit that file to add, remove or reorder
jobs while it runs. Every trainer saves checkpoints and resumes where it stopped, and the server picks up
newer checkpoints automatically. Training uses bf16 on CPUs that support it. On a GPU, raise the sizes.

Data: smol-smoltalk, Dolly 15k, everyday-conversations, Simple Wikipedia and TinyStories (Core); CodeParrot
and Python Code Instructions 18k (Una); CIFAR-10 and mini-ImageNet (Arc); LibriSpeech (Live); the fonts on the machine plus the text corpora (Cast). All from
Hugging Face.
