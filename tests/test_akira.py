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
    assert protocol.route(protocol.Request(text="what's this", modality="video")).brain.family == "cast"


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


def test_cast_sees_an_image():
    import io

    import torch
    from PIL import Image
    from akira.native.cast import Cast, CastNet
    path = "/tmp/akira-test-cast.pt"
    torch.save({"brain": "cast-lite", "model": CastNet((32, 64, 128, 256), 3).state_dict(), "state": {},
                "classes": ["cat", "dog", "fox"]}, path)
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (200, 100, 50)).save(buf, format="JPEG")
    seen = Cast(path).see(buf.getvalue(), k=2)
    assert len(seen) == 2 and seen[0][0] in ("cat", "dog", "fox") and 0 <= seen[0][1] <= 1
