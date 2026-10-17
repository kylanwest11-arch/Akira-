# Akira: project rules

- **No Anthropic API key, ever, and no third-party AI APIs.** Every answer comes from Akira's own models
  (trained from scratch in `akira/native`), the device tools (`akira/lac.py`), or the internet
  (`akira/web.py`, which needs no keys). Do not add the `anthropic` package, `ANTHROPIC_API_KEY`, or calls to
  any hosted model, and do not suggest them as a way to improve answers. Better answers come from better
  training (more data, bigger brains, a GPU: see `notebooks/train_on_colab.ipynb`).
- "Cloud" in Akira's design (the Cloud Access Center) means the Akira server's larger brains (Pro, Ultra, Max)
  plus the internet, not an external AI service.
- Brains: Core (Standard/Pro/Ultra), Arc (images: drawing + recognising), Una (code), Live (speech),
  Cast (screen viewing + voice), and the router (Alternate Protocol). Only finished brains go in `release/`
  (`scripts/export_models.py`).
- Run tests with `python3 -m pytest -q tests`.
