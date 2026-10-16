"""Akira's HTTP API, for websites and apps.

    python3 -m akira.server                 # http://localhost:8000
    AKIRA_API_KEYS=key1,key2 AKIRA_ORIGINS=https://mysite.com python3 -m akira.server

Endpoints
    GET  /v1/status             which brains exist, which are trained, whether the cloud is connected
    POST /v1/transcribe         Live: {"audio": base64 16 kHz mono 16-bit PCM} -> {"text": ...}
    POST /v1/listen             First-Hand Processing: send the request as it grows
                                {"session": "...", "text": "what was that message Tho"}
    POST /v1/chat               {"messages": [...], "images": [...], "modality": "text"|"audio"|"screen",
                                 "session": "...", "stream": true}
                                streams Server-Sent Events: route, understood, tool, text, done
    GET  /embed.js              drop-in chat widget for any website
    GET  /                      demo page
"""
import asyncio
import base64
import json
import os
import threading

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .assistant import Akira

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
API_KEYS = {k.strip() for k in os.environ.get("AKIRA_API_KEYS", "").split(",") if k.strip()}
ORIGINS = [o.strip() for o in os.environ.get("AKIRA_ORIGINS", "*").split(",") if o.strip()]
MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

app = FastAPI(title="Akira", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Authorization", "Content-Type"])
akira = Akira(models_dir=os.environ.get("AKIRA_MODELS", "models"))


def check_key(request: Request):
    if not API_KEYS:
        return
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] not in API_KEYS:
        raise HTTPException(401, "Missing or wrong API key. Send Authorization: Bearer <key>.")


class Message(BaseModel):
    role: str
    content: str = Field(max_length=100_000)


class Image(BaseModel):
    data: str                       # base64, no data: prefix
    media_type: str = "image/jpeg"


class ChatBody(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=200)
    images: list[Image] = Field(default_factory=list, max_length=8)
    modality: str = "text"
    family: str = "auto"
    tier: str = "auto"
    session: str | None = None
    stream: bool = True
    prefer: str = "auto"


class TranscribeBody(BaseModel):
    audio: str = Field(max_length=3_000_000)   # about 90 s


class ListenBody(BaseModel):
    session: str = Field(min_length=4, max_length=100)
    text: str = Field(max_length=10_000)


@app.get("/v1/status")
def status(request: Request):
    check_key(request)
    return akira.status()


@app.get("/v1/training")
def training(request: Request):
    """Progress of the training queue (scripts/train_queue.py): what's done, running and waiting."""
    check_key(request)
    models = os.environ.get("AKIRA_MODELS", "models")
    try:
        jobs = json.load(open(os.path.join(models, "queue.json")))
    except (OSError, ValueError):
        return {"jobs": []}
    out, running_seen = [], False
    for job in jobs:
        mark = os.path.join(models, "jobs", job["name"])
        minutes = float(job["cmd"].split("--minutes ")[1].split()[0]) if "--minutes " in job["cmd"] else None
        if os.path.exists(mark + ".done"):
            status = "done"
        elif os.path.exists(mark + ".failed"):
            status = "failed"
        elif not running_seen and os.path.exists(mark + ".log"):
            status, running_seen = "training", True
        else:
            status = "waiting"
        item = {"name": job["name"], "status": status, "minutes": minutes}
        if status in ("training", "done", "failed") and os.path.exists(mark + ".log"):
            lines = [line.strip() for line in open(mark + ".log", errors="ignore").read().splitlines() if line.strip()]
            item["latest"] = lines[-1][:300] if lines else ""
        out.append(item)
    return {"jobs": out}


@app.post("/v1/listen")
def listen(body: ListenBody, request: Request):
    check_key(request)
    return akira.listen(body.session, body.text)


@app.post("/v1/transcribe")
async def transcribe(body: TranscribeBody, request: Request):
    check_key(request)
    try:
        pcm = base64.b64decode(body.audio, validate=True)
    except Exception:
        raise HTTPException(400, "audio isn't valid base64.")
    if len(pcm) % 2:
        pcm = pcm[:-1]
    text = await asyncio.to_thread(akira.transcribe, pcm)
    if text is None:
        raise HTTPException(503, "Akira's speech brain (Live) isn't trained yet. Use the browser's speech recognition.")
    live = akira.live.get()
    return {"text": text, "engine": f"native:{live.brain}" if live else None}


def _images(body):
    out = []
    for img in body.images:
        if img.media_type not in IMAGE_TYPES:
            raise HTTPException(400, f"Unsupported image type {img.media_type}. Use JPEG, PNG, WebP or GIF.")
        try:
            raw = base64.b64decode(img.data, validate=True)
        except Exception:
            raise HTTPException(400, "An image isn't valid base64.")
        if len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Images must be 5 MB or smaller.")
        out.append({"data": img.data, "media_type": img.media_type})
    return out


@app.post("/v1/chat")
async def chat(body: ChatBody, request: Request):
    check_key(request)
    if body.messages[-1].role != "user":
        raise HTTPException(400, "The last message must be from the user.")
    if body.modality not in ("text", "audio", "screen"):
        raise HTTPException(400, "modality must be text, audio or screen.")
    images = _images(body)
    msgs = [m.model_dump() for m in body.messages if m.role in ("user", "assistant")]
    stop = threading.Event()
    kwargs = dict(images=images, modality=body.modality, family=body.family, tier=body.tier,
                  session=body.session, stop=stop, prefer=body.prefer)

    if not body.stream:
        events = await asyncio.to_thread(lambda: list(akira.respond(msgs, **kwargs)))
        return JSONResponse({"text": "".join(e["text"] for e in events if e["type"] == "text"),
                             "route": next((e for e in events if e["type"] == "route"), None),
                             "engine": next((e["engine"] for e in events if e["type"] == "done"), None),
                             "events": [e for e in events if e["type"] != "text"]})

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce():
        try:
            for ev in akira.respond(msgs, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=produce, daemon=True).start()

    async def sse():
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        finally:
            stop.set()   # client went away: stop generating

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/embed.js")
def embed():
    return FileResponse(os.path.join(WEB, "akira.js"), media_type="application/javascript")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/")
def demo():
    return FileResponse(os.path.join(WEB, "demo.html"))


def main():
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
