"""Akira: one assistant on top of many brains.

    akira = Akira()
    for event in akira.respond([{"role": "user", "content": "what did Thomas say about the movies?"}]):
        print(event)

respond() yields small dict events, in order:
    {"type": "route", ...}        which brain, where it runs, and why
    {"type": "understood", ...}   what First-Hand Processing worked out
    {"type": "tool", ...}         an on-device lookup or action that ran
    {"type": "text", "text": ...} answer text, streamed
    {"type": "done", "engine": ...}

Where the answer comes from:
    local   the LAC answers from device data directly; nothing leaves the device
    cloud   the Akira server answers with its larger brains (Pro, Ultra, ...) and the internet
    both    the LAC gathers context, the Akira server reasons over it
Every answer comes from Akira's own brains (trained from scratch in akira/native),
the LAC's device tools, or the internet (akira/web.py). No third-party AI is used.
"""
import base64
import os
import re
import threading
import uuid

import numpy as np

from . import brains, protocol, web
from .fhp import Listener
from .lac import LocalAccessCenter
from .native import arc as arc_mod
from .native import arc_vision as vision_mod
from .native import cast as cast_mod
from .native import live as live_mod
from .native.engine import NativeBrains
from .native.router import Router

NATIVE_SYSTEM = "You are Akira, a helpful, friendly AI assistant. Answer clearly and briefly."


class Latest:
    """Loads the best trained checkpoint of a family, and reloads it when training saves a newer one."""

    def __init__(self, loader, family, models_dir):
        self.loader, self.family, self.dir = loader, family, models_dir
        self.value, self.stamp = None, None

    def get(self):
        paths = [os.path.join(self.dir, f"{self.family}-{t}.pt") for t in ("lite", "general", "max")]
        stamp = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)
        if stamp != self.stamp:
            try:
                self.value = self.loader(self.dir)
            except Exception:   # a checkpoint mid-write; keep the previous one
                return self.value
            self.stamp = stamp
        return self.value


class Akira:
    def __init__(self, device=None, models_dir="models"):
        self.lac = LocalAccessCenter(device)
        self.native = NativeBrains(models_dir)
        self.router = Router.load_default(models_dir)
        self.arc = Latest(arc_mod.load_best, "arc", models_dir)
        self.vision = Latest(vision_mod.load_best, "arc-vision", models_dir)
        self.cast = Latest(cast_mod.load_best, "cast", models_dir)
        self.live = Latest(live_mod.load_best, "live", models_dir)
        self.listeners = {}
        self._lock = threading.Lock()

    # --- First-Hand Processing --------------------------------------------
    def listen(self, session, partial):
        """Called repeatedly while the user types or speaks. Returns what is understood so far."""
        with self._lock:
            listener = self.listeners.get(session)
            if listener is None:
                if len(self.listeners) > 500:   # forget abandoned sessions
                    self.listeners.pop(next(iter(self.listeners)))
                listener = self.listeners[session] = Listener(self.lac, self.router)
        return listener.update(partial)

    def status(self):
        return {
            "web": os.environ.get("AKIRA_WEB", "on") != "off",
            "router": {"trained": self.router is not None, "metrics": self.router.metrics if self.router else None},
            "native_brains": {**{b: self.native.get(b).info() for b in self.native.available()},
                              **{m.info["brain"]: m.info for m in (self.arc.get(), self.vision.get(), self.cast.get(), self.live.get()) if m}},
            "brains": [b.to_dict() for b in brains.BRAINS.values()],
        }

    # --- Live: speech to text --------------------------------------------------
    def transcribe(self, pcm16):
        """16 kHz mono 16-bit PCM bytes -> text, with Akira's own speech brain. None if Live isn't trained."""
        live = self.live.get()
        if live is None:
            return None
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768
        return live.transcribe(audio)

    # --- answering ----------------------------------------------------------
    def respond(self, messages, images=None, modality="text", family="auto", tier="auto",
                session=None, stop=None, prefer="auto"):
        """messages: [{"role": "user"|"assistant", "content": str}], ending with the user's turn.
        prefer is accepted for compatibility and ignored: every brain is Akira's own."""
        text = messages[-1]["content"] if messages else ""
        history = messages[:-1]
        req = protocol.Request(text=text, images=images or [], modality=modality, family=family, tier=tier,
                               router=self.router)
        route = protocol.route(req)
        yield {"type": "route", **route.to_dict()}

        listener = self.listeners.pop(session, None) if session else None
        if listener is None and route.local_intents:
            listener = Listener(self.lac, self.router)
        prefetched = {}
        if listener is not None:
            done = listener.finish(text)
            prefetched = done["results"]
            if done["understood"]:
                yield {"type": "understood", "notes": done["understood"], "sources": sorted(prefetched)}

        brain = route.brain
        if route.placement == "local":
            answered = False
            for ev in self.local_answer(text, route.local_intents, prefetched):
                answered = True
                yield ev
            if answered:
                yield {"type": "done", "engine": "device"}
                return

        # Arc draws with its own image model; cloud brains write text only
        if brain.family == "arc" and protocol.ARC_MAKE.search(text) and not images:
            arc = self.arc.get()
            subject = arc_mod.subject_of(text) if arc else None
            if subject is not None:
                name = arc_mod.CLASSES[subject]
                yield {"type": "text", "text": f"Here are four takes on a {name}."}
                for png in arc.draw(subject, n=4):
                    yield {"type": "image", "media_type": "image/png", "data": base64.b64encode(png).decode()}
                yield {"type": "done", "engine": f"native:{arc.brain}"}
                return
            if arc:
                yield {"type": "text", "text": "I can draw these so far: " + ", ".join(arc_mod.CLASSES)
                                               + ". Ask for one of them, like \"draw a horse\"."}
                yield {"type": "done", "engine": f"native:{arc.brain}"}
                return

        # Cast reads the shared screen with Akira's own screen reader, before any brain answers
        if modality == "screen" and images:
            cast = self.cast.get()
            screen_text = None
            if cast is not None:
                try:
                    lines = cast.read(base64.b64decode(images[0]["data"]))
                except Exception:
                    lines = []
                screen_text = "\n".join(line["text"] for line in lines)
                yield {"type": "tool", "name": "read_screen", "input": {"lines": len(lines)}, "where": "device"}
                if screen_text:
                    prefetched = {**prefetched, "text_read_from_screen": screen_text[:6000]}
            yield from self._native_screen_answer(text, screen_text, brain, history, stop)
            return

        # Arc looks at a photo with its own eyes
        if images:
            vision = self.vision.get()
            if vision is not None:
                try:
                    seen = vision.see(base64.b64decode(images[0]["data"]))
                except Exception:
                    seen = None
                if seen:
                    best, p = seen[0]
                    others = ", ".join(f"{n} ({q:.0%})" for n, q in seen[1:])
                    msg = (f"That looks like **{best}** ({p:.0%} sure)." if p >= 0.4 else
                           f"I'm not sure. My best guess is **{best}** ({p:.0%}).") + (f" It could also be {others}." if others else "")
                    yield {"type": "tool", "name": "look", "input": {}, "where": "device"}
                    yield {"type": "text", "text": msg}
                    yield {"type": "done", "engine": f"native:{vision.brain}"}
                    return

        # device facts first, then the internet, then Akira's language brain
        if route.local_intents:
            events = list(self.local_answer(text, route.local_intents, prefetched))
            if not any(e["type"] == "text" for e in events):
                sources = [i for i in route.local_intents if i not in ("calculator", "clock", "settings")] or route.local_intents
                events.append({"type": "text", "text": f"I can't see your {', '.join(sources)} from here. On your own device, "
                                                       "Akira answers this without using the cloud. Here, I can help you "
                                                       "draft a reply or plan instead."})
            yield from events
            yield {"type": "done", "engine": "device"}
            return
        if brain.family == "arc" and protocol.ARC_MAKE.search(text):
            yield {"type": "text", "text": "Image creation needs Arc's image model, which isn't trained yet. "
                                           "I can describe the image in words instead."}
            yield {"type": "done", "engine": "device"}
            return
        # the internet: news, prices and anything current, plus factual questions a small brain would guess at
        if (os.environ.get("AKIRA_WEB", "on") != "off" and brain.family in ("core", "una") and not images
                and (route.web or (web.QUESTION.match(text) and brain.family == "core" and not web.PERSONAL.search(text)))):
            yield {"type": "tool", "name": "web_search", "input": {"query": web.clean_query(text)}, "where": "internet"}
            try:
                found, sources = web.answer(text)
            except Exception:
                found, sources = None, []
            if found:
                yield {"type": "text", "text": found}
                yield {"type": "done", "engine": "web", "sources": sources}
                return
            if route.web:
                yield {"type": "text", "text": "I couldn't reach the internet just now. Try again in a minute."}
                yield {"type": "done", "engine": "web"}
                return
        engine = self.native.best_for(brain.family, brain.rank)
        if engine is None:
            yield {"type": "text", "text": "None of Akira's language brains are trained yet. "
                                           "Train them with scripts/train_queue.py or the Colab notebook."}
            yield {"type": "done", "engine": "none"}
            return
        convo = [{"role": "system", "content": NATIVE_SYSTEM}] + history[-6:] + [{"role": "user", "content": text}]
        for piece in engine.stream_chat(convo, max_new_tokens=220, stop=stop):
            yield {"type": "text", "text": piece}
        yield {"type": "done", "engine": f"native:{engine.brain}"}

    def _native_screen_answer(self, question, screen_text, brain, history, stop):
        """Answer about the user's screen: show what Cast read, then let Core answer from it."""
        if screen_text is None:
            yield {"type": "text", "text": "Cast's screen reader isn't trained yet, so I can't read your screen. "
                                           "Try again once its training finishes."}
            yield {"type": "done", "engine": "none"}
            return
        if not screen_text.strip():
            yield {"type": "text", "text": "I can see your screen but can't make out any text on it."}
            yield {"type": "done", "engine": "native:cast"}
            return
        shown = [ln for ln in screen_text.splitlines() if len(ln.strip()) > 2][:15]
        yield {"type": "text", "text": "**On your screen I can read:**\n\n" + "\n".join(f"> {ln}" for ln in shown) + "\n\n"}
        engine = self.native.best_for("core", brain.rank)
        if engine is not None and question.strip():
            convo = [{"role": "system", "content": NATIVE_SYSTEM + "\nThe user is sharing their screen. Text on it:\n"
                      + screen_text[:1500]}] + history[-4:] + [{"role": "user", "content": question}]
            for piece in engine.stream_chat(convo, max_new_tokens=160, stop=stop):
                yield {"type": "text", "text": piece}
        yield {"type": "done", "engine": "native:cast" + (f"+{engine.brain}" if engine else "")}

    # --- on-device answers (LAC) ---------------------------------------------
    def local_answer(self, text, intents, prefetched):
        """Answer straight from device data when the request is a lookup or an action."""
        t = text.lower()
        out = []

        if "settings" in intents:
            change = self._setting_change(t)
            if change:
                name, value = change
                self.lac.change_setting(name, value)
                out.append({"type": "tool", "name": "change_setting", "input": {"name": name, "value": value}, "where": "device"})
                shown = ("on" if value else "off") if isinstance(value, bool) else value
                out.append({"type": "text", "text": f"Done. {name.replace('_', ' ').capitalize()} is now {shown}."})
                return out
        if "calculator" in intents:
            expr = self._expression(text)
            if expr:
                try:
                    r = self.lac.calculator(expr)
                    out.append({"type": "tool", "name": "calculator", "input": {"expression": expr}, "where": "device"})
                    v = r["result"]
                    if isinstance(v, float) and v.is_integer():
                        v = int(v)
                    out.append({"type": "text", "text": f"{expr} = **{v:,}**" if isinstance(v, (int, float)) else str(v)})
                    return out
                except Exception:
                    pass
        if "clock" in intents:
            c = self.lac.clock()
            return [{"type": "tool", "name": "clock", "input": {}, "where": "device"},
                    {"type": "text", "text": f"It's {c['local']}."}]

        lines = []
        msgs = prefetched.get("messages")
        if "messages" in intents and msgs is None:
            msgs = self.lac.search_messages(query=text)
        if msgs:
            out.append({"type": "tool", "name": "search_messages", "input": {}, "where": "device"})
            by = {m["from"] for m in msgs[:3]}
            if len(by) == 1:   # one person's thread: show it in the order it was sent
                thread = sorted(msgs[:3], key=lambda m: m["id"])
                lines.append(f"**{thread[0]['from']}** sent you this on {thread[0]['at']}:\n\n"
                             + "\n>\n".join(f"> {m['text']}" for m in thread))
            else:
                lines.append("\n".join(f"- **{m['from']}**, {m['at']}: {m['text']}" for m in msgs[:3]))
        cal = prefetched.get("calendar") if "relations" not in intents else None
        if "calendar" in intents and cal is None and "relations" not in intents:
            cal = self.lac.calendar(when=text)
        if cal:
            out.append({"type": "tool", "name": "calendar", "input": {}, "where": "device"})
            lines.append("**Calendar**\n" + "\n".join(
                f"- {e['title']}: {e['start']} to {e['end']}" + (f" with {', '.join(e['with'])}" if e["with"] else "")
                for e in cal[:5]))
        rel = prefetched.get("related")
        if "relations" in intents and rel is None:
            rel = self.lac.related(text)
        if rel:
            out.append({"type": "tool", "name": "related", "input": {}, "where": "device"})
            items = [r for r in rel if r["kind"] in ("file", "decision", "meeting", "task")][:5] or rel[:5]
            lines.append("**Related**\n" + "\n".join(f"- {r['kind'].capitalize()}: {r['title']}" for r in items))
        files = prefetched.get("files")
        if "files" in intents and files is None and not rel:
            files = self.lac.search_files(text)
        if files and not rel:
            out.append({"type": "tool", "name": "search_files", "input": {}, "where": "device"})
            lines.append("**Files**\n" + "\n".join(f"- {f['name']} ({f['path']}, {f['modified']}): {f['summary']}" for f in files[:5]))
        photos = prefetched.get("photos")
        if "photos" in intents and photos is None:
            photos = self.lac.search_photos(text)
        if photos:
            out.append({"type": "tool", "name": "search_photos", "input": {}, "where": "device"})
            lines.append("**Photos**\n" + "\n".join(f"- {p['name']}, {p['taken']}: {', '.join(p['tags'])}" for p in photos))
        if "contacts" in intents:
            people = [c for name in re.findall(r"[A-Z][a-z]+", text) for c in self.lac.search_contacts(name)]
            if people:
                out.append({"type": "tool", "name": "search_contacts", "input": {}, "where": "device"})
                lines.append("\n".join(f"**{c['name']}**: {c['phone']}" for c in {c["id"]: c for c in people}.values()))
        if lines:
            out.append({"type": "text", "text": "\n\n".join(lines)})
        return out

    @staticmethod
    def _setting_change(t):
        names = {"wi-fi": "wifi", "wifi": "wifi", "bluetooth": "bluetooth", "dark mode": "dark_mode",
                 "do not disturb": "do_not_disturb", "dnd": "do_not_disturb", "brightness": "brightness", "volume": "volume"}
        for phrase, name in names.items():
            if phrase in t:
                if name in ("brightness", "volume"):
                    n = re.search(r"(\d{1,3})", t)
                    if n:
                        return name, max(0, min(100, int(n.group(1))))
                    if re.search(r"\b(up|louder|higher|brighter|increase|raise)\b", t):
                        return name, 80
                    if re.search(r"\b(down|quieter|lower|dimmer|decrease|reduce)\b", t):
                        return name, 30
                    return None
                if re.search(r"\b(off|disable|deactivate|stop)\b", t):
                    return name, False
                if re.search(r"\b(on|enable|activate|start)\b", t):
                    return name, True
        if re.search(r"\bmute\b", t):
            return "volume", 0
        return None

    @staticmethod
    def _expression(text):
        t = re.sub(r"(?i)^\s*(akira[, ]+)?(what('s| is)|calculate|how much is|compute)\s+", "", text.strip()).rstrip("?. ")
        t = re.sub(r"(?i)\btimes\b|\bx\b", "*", t)
        t = re.sub(r"(?i)\bdivided by\b", "/", t)
        t = re.sub(r"(?i)\bplus\b", "+", t)
        t = re.sub(r"(?i)\bminus\b", "-", t)
        t = re.sub(r"(?i)\bsquare root of\s+([\d.]+)", r"sqrt(\1)", t)
        t = re.sub(r"(?i)\btip of ([\d.]+)% on \$?([\d.]+)", r"\1% of \2", t)
        t = t.replace("$", "")
        return t if re.search(r"\d", t) else None


def new_session_id():
    return uuid.uuid4().hex
