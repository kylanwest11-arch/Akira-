# Put Akira on your website

Akira has two parts:

- **The Akira server** (this repository). It runs Akira's brains and answers requests. It must run somewhere
  public, so visitors' browsers can reach it.
- **The front end** your visitors use. Pick one:
  - **A. The full Akira app** (`web/app.html`): Home, Workspace, Research, Image and Code chats, memory,
    Relations and settings.
  - **B. The chat widget** (`/embed.js`): one line on any existing page, which adds a chat bubble in the corner.

## 1. Run the Akira server

The repository includes a `Dockerfile`, so any host that builds from a GitHub repository with a Dockerfile
works: Render, Railway, Fly.io, Google Cloud Run, DigitalOcean App Platform, or your own server.

1. Create a new service from the GitHub repository `kylanwest11-arch/Akira-`.
2. Choose at least **1 GB of memory**. A plan with 2 GB is more comfortable once the larger brains are trained.
3. Set the **port** to `8000`.
4. Add these **environment variables**:

| Variable | Value | Why |
|---|---|---|
| `AKIRA_ORIGINS` | `https://yourwebsite.com` | Only your site may call the server. Separate several sites with commas. |
| `AKIRA_WEB` | `on` (default) or `off` | Internet access: Akira searches the web and Wikipedia and reads pages itself. No API key. |
| `AKIRA_CHATS_PER_MINUTE` | `12` (default) | How many messages one visitor can send per minute, so nobody can run up your cloud bill. |

5. Deploy. Your host gives you an address such as `https://akira-xyz.onrender.com`. That is your
   **Akira server address**.

Check it works: open `https://YOUR-SERVER/v1/status` in a browser. You should see a list of brains.

To run it on your own machine instead:

```bash
docker build -t akira .
docker run -p 8000:8000 -e AKIRA_ORIGINS=https://yourwebsite.com akira
```

## 2A. Use the full Akira app

**Easiest:** link to `https://YOUR-SERVER/app`. The server hosts the app and connects it to itself, so there is
nothing else to set up. You can put that address behind a button or menu item on your website, or point a
subdomain such as `akira.yourwebsite.com` at it in your host's custom-domain settings.

**On your own website instead:** upload `web/app.html` to your site (rename it to `index.html` if it should be
the page at that address), and fill in your server address on this line near the top:

```html
<meta name="akira-server" content="https://YOUR-SERVER">
```

Visitors can also enter or change the address themselves in **Settings → Intelligence → Akira server**.

## 2B. Or add the chat widget to existing pages

Paste this before `</body>` on any page:

```html
<script src="https://YOUR-SERVER/embed.js" defer></script>
```

Options go on the same tag: `data-position="left"`, `data-open="true"`, and `data-launcher="false"` (hide the
bubble and open it from your own button with `Akira.open()`).

## 3. Keep Akira up to date

Akira's brains keep training in the training environment. To ship newer brains:

```bash
python3 scripts/export_models.py      # copies the latest trained brains into release/
git add release && git commit -m "Update Akira's brains" && git push
```

Most hosts redeploy automatically when the repository changes.

## For apps and other sites: the API

| Endpoint | What it does |
|---|---|
| `POST /v1/chat` | Ask Akira. Body: `{"messages": [{"role": "user", "content": "Hi"}], "stream": true}`. Streams Server-Sent Events: `route`, `understood`, `tool`, `text`, `image`, `done`. With `"stream": false` it returns one JSON object with `text`. |
| `POST /v1/listen` | First-Hand Processing: send the request as the user types (`{"session", "text"}`) to get what Akira understood so far. |
| `POST /v1/transcribe` | Live: base64 16 kHz mono 16-bit PCM audio in, text out. |
| `GET /v1/status` | Which brains exist and which are trained. |

## What visitors get today

| Brain | Status |
|---|---|
| Core (chat) | Core Standard is trained; Core Pro is training. Small, so it answers simple things well and makes mistakes on harder ones. Factual and current questions are answered from the internet. |
| Router | Trained. Picks the brain for each request. |
| Arc, Una, Live, Cast | Training is queued. Until a brain is trained, Akira says plainly that it can't do that yet. |
| Calculator, clock, search of your own chats | Work fully in the visitor's browser. |
