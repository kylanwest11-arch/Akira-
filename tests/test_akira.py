"""Run with: python3 -m pytest -q tests"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


def test_website_mode_has_no_personal_tools():
    from akira.device import DeviceData
    names = {t["name"] for t in LocalAccessCenter(DeviceData()).tool_definitions()}
    assert names == {"calculator", "clock"}
    assert "search_messages" in {t["name"] for t in LocalAccessCenter().tool_definitions()}


def test_rate_limit():
    from fastapi.testclient import TestClient
    from akira import server
    server.LIMITS["/v1/listen"] = 3
    c = TestClient(server.app)
    codes = [c.post("/v1/listen", json={"session": "rate1", "text": "hi there"},
                    headers={"x-forwarded-for": "203.0.113.9"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert c.post("/v1/listen", json={"session": "rate2", "text": "hi"},
                  headers={"x-forwarded-for": "203.0.113.10"}).status_code == 200


def test_app_is_served_pointed_at_this_server():
    from fastapi.testclient import TestClient
    from akira import server
    r = TestClient(server.app).get("/app")
    assert r.status_code == 200 and 'name="akira-server" content="http://testserver"' in r.text


def test_website_mode_explains_instead_of_going_silent():
    from akira.device import DeviceData
    a = Akira(device=DeviceData(), models_dir="/nonexistent")
    out = text_of(a.respond([{"role": "user", "content": "what did Thomas text me?"}]))
    assert "can't see your messages" in out


def test_web_routing_and_keywords():
    from akira import web
    assert protocol.route(protocol.Request(text="what's the latest news about NASA?")).web
    assert protocol.route(protocol.Request(text="search the web for pizza near me")).web
    assert not protocol.route(protocol.Request(text="write a poem about rain")).web
    assert web.clean_query("What's the latest news about the James Webb telescope?") == "latest news about the James Webb telescope"
    assert web.clean_query("Who won the World Series in 2025?") == "Who won the World Series in 2025"
    assert web.PERSONAL.search("who are you?") and not web.PERSONAL.search("what is photosynthesis?")


def test_web_answers_without_a_cloud_key(monkeypatch):
    from akira import web
    from akira.device import DeviceData
    monkeypatch.setattr(web, "search", lambda q, n=5: [{"title": "NASA Webb news", "url": "https://science.nasa.gov/webb", "snippet": "New image."}])
    monkeypatch.setattr(web, "wikipedia", lambda q: None)
    a = Akira(device=DeviceData(), models_dir="/nonexistent")
    evs = list(a.respond([{"role": "user", "content": "latest news about the Webb telescope"}]))
    assert any(e.get("where") == "internet" for e in evs)
    assert "science.nasa.gov/webb" in text_of(evs) and evs[-1]["engine"] == "web"


def test_bing_links_are_decoded():
    import base64

    from akira import web
    target = "https://www.timeout.com/chicago/pizza"
    u = "a1" + base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    page = f'<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;u={u}&amp;ntb=1">Best <b>pizza</b></a></h2><p>Deep dish.</p></li>'

    class R:
        headers = None
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n): return page.encode()
    import urllib.request
    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: R()
    try:
        r = web._bing("pizza", 3)
    finally:
        urllib.request.urlopen = orig
    assert r == [{"title": "Best pizza", "url": target, "snippet": "Deep dish."}]


FAKE_PAGE = """<html><head><title>Riverbook News</title></head><body>
<nav><a href="/">Home</a><a href="/about">About us</a></nav>
<svg><title>logo</title></svg>
<h1>Today at Riverbook</h1>
<p>The split keyboard ships next month, the team announced on Friday after the final review was signed off.</p>
<a href="/news/keyboard">Keyboard ships next month</a> <a href="https://example.com/x#top">Partner page</a>
<a href="javascript:void(0)">Menu</a><a href="mailto:a@b.c">Email</a><a href="/news/keyboard">Keyboard ships next month</a>
</body></html>"""


def test_browse_reads_a_page_and_its_links(monkeypatch):
    from akira import web
    monkeypatch.setattr(web, "_get", lambda url, accept="text/html": FAKE_PAGE)
    page = web.visit("riverbook.io")
    assert page["title"] == "Riverbook News" and page["url"] == "https://riverbook.io"
    assert "split keyboard ships" in page["excerpt"] and page["headings"] == ["Today at Riverbook"]
    urls = [l["url"] for l in page["links"]]
    assert "https://riverbook.io/news/keyboard" in urls and "https://example.com/x" in urls
    assert len(urls) == len(set(urls)) and not any(u.startswith(("javascript", "mailto")) for u in urls)


def test_browse_targets_and_following_links(monkeypatch):
    from akira import web
    assert web.browse_target("go to bbc.com") == {"url": "https://bbc.com"}
    assert web.browse_target("search bbc.com for climate") == {"site_search": ("bbc.com", "climate")}
    assert web.browse_target("take me to wikipedia") == {"find_site": "wikipedia"}
    assert web.browse_target("go to the gym tomorrow") is None
    assert web.browse_target("write a poem about rain") is None
    shown = "Links:\n1. [Home](https://riverbook.io/)\n2. [News](https://riverbook.io/news)"
    hist = [{"role": "user", "content": "go to riverbook.io"}, {"role": "assistant", "content": shown}]
    assert web.browse_target("open 2", hist) == {"url": "https://riverbook.io/news"}
    assert web.browse_target("open 7", hist) == {"missing": 7, "count": 2}
    monkeypatch.setattr(web, "_get", lambda url, accept="text/html": FAKE_PAGE)
    from akira.device import DeviceData
    out = text_of(Akira(device=DeviceData(), models_dir="/nonexistent").respond(hist + [{"role": "user", "content": "open 2"}]))
    assert "Riverbook News" in out and "1. [" in out


def test_reasoning_check_fixes_arithmetic_and_carries_it():
    from akira.reasoning import check, is_word_problem
    fixed, n = check("She has 50 - 10 - 8 = 30 left.\nSplit 16 ways: 30 / 16 = 2.\n#### 2\nThe answer is 30.")
    assert n == 1 and "= 32 left" in fixed and "32 / 16" in fixed and "####" not in fixed and "The answer is 32." in fixed
    assert is_word_problem("Karen bakes 50 cookies and gives away 18. How many are left?")
    assert not is_word_problem("What is the capital of France?")


def test_research_plans_and_picks_sentences():
    from akira import research
    assert research.plan("Compare solar power and wind power") == ["solar power", "wind power"]
    assert research.plan("solar vs wind") == ["solar", "wind"]
    page = ("Skip to content. Wind power is the use of wind to make electricity with turbines on farms. "
            "Our newsletter is great, subscribe now to get updates every week from us. "
            "Wind power is variable, because the wind does not always blow at the same speed.")
    got = [s for _, s in research.best_sentences(page, "wind power", question="why is wind power variable")]
    assert got and all("subscribe" not in s for s in got)
    assert research.wants_research("Why do cats purr?") and not research.wants_research("who are you?")


def test_who_wrote_is_not_about_messages():
    assert protocol.route(protocol.Request(text="Who wrote Hamlet?")).local_intents == []
    assert "messages" in protocol.route(protocol.Request(text="what did Maya write to me?")).local_intents


def test_knowledge_answers_from_a_tiny_encyclopedia(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from akira import knowledge
    src = tmp_path / "wiki.parquet"
    pq.write_table(pa.table({"title": ["Japan", "Tokyo", "Spider"], "text": [
        "Japan is an island country in East Asia. Tokyo is the capital of Japan and its biggest city.",
        "Tokyo is the capital city of Japan, on the island of Honshu.",
        "Spiders are small animals. They have eight legs, and most spiders make silk."]}), src)
    db = str(tmp_path / "k.db")
    assert knowledge.build(str(src), db) == 3
    k = knowledge.Knowledge(db)
    assert "Tokyo" in k.answer("What is the capital of Japan?")[0]
    assert "eight legs" in k.answer("How many legs does a spider have?")[0]
    assert k.answer("Who painted the Mona Lisa?") == (None, None)
