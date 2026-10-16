"""Run with: python3 -m pytest -q tests"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("AKIRA_CLOUD", "off")

from akira import protocol  # noqa: E402
from akira.assistant import Akira  # noqa: E402
from akira.fhp import Listener  # noqa: E402
from akira.lac import LocalAccessCenter, calculate  # noqa: E402
from akira.relations import words  # noqa: E402


@pytest.fixture(scope="module")
def akira():
    return Akira(models_dir="/nonexistent")   # rules + LAC only, no trained models needed


def text_of(events):
    return "".join(e["text"] for e in events if e["type"] == "text")


@pytest.mark.parametrize("text,family", [
    ("fix this function and explain why it was failing", "una"),
    ("make this look more cinematic", "arc"),
    ("what's the capital of Peru", "core"),
])
def test_rules_pick_the_brain(text, family):
    assert protocol.route(protocol.Request(text=text)).brain.family == family


def test_modality_picks_live_and_cast():
    assert protocol.route(protocol.Request(text="hi", modality="audio")).brain.family == "live"
    assert protocol.route(protocol.Request(text="what's this", modality="screen")).brain.family == "cast"


def test_simple_device_lookup_stays_local():
    r = protocol.route(protocol.Request(text="what did Thomas text me yesterday"))
    assert r.placement == "local" and "messages" in r.local_intents


def test_no_cloud_means_local():
    r = protocol.route(protocol.Request(text="compare keynes and hayek in depth", cloud_available=False))
    assert r.placement == "local"


def test_plurals_fold():
    assert words("movies") == words("movie")
    assert words("parties") == words("party")
    assert words("Thomas") == ["thomas"]


def test_calculator():
    assert calculate("15% of 80") == 12
    assert calculate("2^10") == 1024
    with pytest.raises(Exception):
        calculate("__import__('os')")


def test_first_hand_processing_narrows_as_you_speak():
    li = Listener(LocalAccessCenter())
    li.update("what was that message")
    li.update("what was that message Thomas sent me")
    li.update("what was that message Thomas sent me about the movies the other day")
    notes = [n["note"] for n in li.understood]
    assert notes[0] == "Messages is relevant"
    assert "narrowed to Thomas Reed" in notes
    assert "time: the other day" in notes
    result = li.finish()["results"]["messages"]
    assert {m["id"] for m in result} == {"m1", "m2"}


def test_thomas_message_answer(akira):
    out = text_of(akira.respond([{"role": "user", "content": "what was that message Thomas sent me about the movies the other day"}]))
    assert "Dune" in out and "Thomas Reed" in out


def test_relations(akira):
    out = text_of(akira.respond([{"role": "user", "content":
                                  "show me the documents related to the Riverbook keyboard decision we made after that meeting"}]))
    assert "review notes.docx" in out and "split layout spec" in out
    assert "Tax return" not in out


def test_settings_change(akira):
    events = list(akira.respond([{"role": "user", "content": "turn on dark mode"}]))
    assert akira.lac.get_settings()["dark_mode"] is True
    assert events[-1] == {"type": "done", "engine": "device"}


def test_api():
    from fastapi.testclient import TestClient
    from akira import server
    c = TestClient(server.app)
    assert c.get("/v1/status").status_code == 200
    snap = c.post("/v1/listen", json={"session": "test1", "text": "did Maya text me"}).json()
    assert any("Maya" in n["note"] for n in snap["understood"])
    r = c.post("/v1/chat", json={"messages": [{"role": "user", "content": "what's 6 times 7"}], "stream": False}).json()
    assert "42" in r["text"]
    assert c.post("/v1/chat", json={"messages": [{"role": "assistant", "content": "hi"}]}).status_code == 400
    assert c.post("/v1/chat", json={"messages": [{"role": "user", "content": "x"}],
                                    "images": [{"data": "not base64!", "media_type": "image/png"}]}).status_code == 400


def test_arc_understands_subjects():
    from akira.native.arc import CLASSES, subject_of
    assert CLASSES[subject_of("draw me a cute puppy")] == "dog"
    assert CLASSES[subject_of("paint a sailboat at sunset")] == "ship"
    assert subject_of("design a logo") is None


def test_arc_untrained_model_draws_an_image():
    import torch
    from akira.native.arc import Diffusion, UNet
    x = Diffusion().sample(UNet(base=16), [3], steps=2, seed=0)
    assert x.shape == (1, 3, 32, 32) and torch.isfinite(x).all()


def test_live_pieces():
    import numpy as np
    import torch
    from akira.native.live import VOCAB, LiveNet, encode_text, features, greedy_decode, word_error_rate
    f = features(np.zeros(16000, dtype=np.float32))
    assert f.shape[1] == 80 and 95 <= f.shape[0] <= 105          # 10 ms frames
    logp, n = LiveNet(32, 1, 2)(f[None], torch.tensor([len(f)]))
    assert logp.shape[-1] == len(VOCAB) and int(n[0]) == logp.shape[1]
    assert "".join(VOCAB[i] for i in encode_text("Hi, it's me!")) == "hi it's me"
    ids = torch.full((6, len(VOCAB)), -9.0)
    for t, ch in enumerate(["h", "h", "_", "i", "_", "_"]):
        ids[t, VOCAB.index(ch)] = 0
    assert greedy_decode(ids, 6) == "hi"
    assert word_error_rate("the cat sat", "the bat sat") == 1 / 3


def test_arc_vision_sees_an_image():
    import io

    import torch
    from PIL import Image
    from akira.native.arc_vision import ArcVision, VisionNet
    path = "/tmp/akira-test-vision.pt"
    torch.save({"brain": "arc-vision-lite", "model": VisionNet((32, 64, 128, 256), 3).state_dict(), "state": {},
                "classes": ["cat", "dog", "fox"]}, path)
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (200, 100, 50)).save(buf, format="JPEG")
    seen = ArcVision(path).see(buf.getvalue(), k=2)
    assert len(seen) == 2 and seen[0][0] in ("cat", "dog", "fox") and 0 <= seen[0][1] <= 1


def _fake_screen():
    from PIL import Image, ImageDraw, ImageFont
    from akira.native.cast import system_fonts
    font = ImageFont.truetype(next(f for f in system_fonts() if "DejaVuSans.ttf" in f or "Inter" in f), 18)
    im = Image.new("RGB", (640, 300), (250, 250, 250))
    d = ImageDraw.Draw(im)
    rows = ["Inbox", "Thomas Reed: movies on Friday?", "Maya Chen: notes from the keyboard review"]
    for i, row in enumerate(rows):
        d.text((24, 30 + i * 60), row, font=font, fill=(20, 20, 20))
    d.rectangle((24, 230, 140, 266), fill=(40, 90, 200))
    d.text((44, 238), "Reply", font=font, fill=(255, 255, 255))
    return im, rows + ["Reply"]


def test_cast_finds_every_line_on_a_screen():
    import numpy as np
    from akira.native.cast import find_lines
    im, rows = _fake_screen()
    boxes = find_lines(np.asarray(im.convert("L")))
    assert len(boxes) == len(rows)
    assert [b[1] for b in boxes] == sorted(b[1] for b in boxes)


def test_cast_reads_a_screen_end_to_end():
    import io

    import torch
    from akira.native.cast import Cast, CastNet, SIZES
    path = "/tmp/akira-test-cast.pt"
    torch.save({"brain": "cast-lite", "model": CastNet(*SIZES["cast-lite"]).state_dict(), "state": {}}, path)
    im, rows = _fake_screen()
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    out = Cast(path).read(buf.getvalue())   # untrained, so the text is noise; the pipeline must still run
    assert isinstance(out, list) and all("box" in o and "text" in o for o in out)


def test_cast_training_text_fits_a_strip():
    import random

    from akira.native.cast import MAX_WIDTH, synth_line, system_fonts
    r = random.Random(0)
    fonts = system_fonts()
    for _ in range(30):
        strip, shown = synth_line("A very long line of screen text that would never fit in one strip " * 3, r.choice(fonts), r)
        assert strip.shape[0] == 32 and strip.shape[1] <= MAX_WIDTH and shown
