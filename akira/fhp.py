"""First-Hand Processing: start the work while the user is still talking.

A Listener receives the request as it grows ("Akira, what was that message",
"... Thomas sent me", "... about the movies", "... the other day"). Each
update re-reads the partial text, works out what has become clear (a source,
a person, a topic, a time) and runs the matching on-device lookups right away,
caching each one. When the user finishes, the answer starts from what has
already been found instead of from nothing.

    listener = Listener(lac)
    listener.update("what was that message Thomas")      # Messages relevant, sender narrowed
    listener.update("what was that message Thomas sent me about the movies the other day")
    listener.finish()  ->  {"understood": [...], "results": {...}}
"""
import re
import time

from . import protocol
from .relations import STOP, words

TIME_PHRASES = re.compile(r"\b(today|yesterday|the other day|this week|last week|recently|this month|last month|"
                          r"tonight|tomorrow|this morning|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b", re.I)
ABOUT = re.compile(r"\babout (?:the |my |our )?([\w' -]+?)(?=\s+(?:the other day|yesterday|today|last|this|recently|from|on)\b|[?.!,]|$)", re.I)
EXTRA_STOP = STOP | {"message", "messages", "sent", "text", "texted", "said", "say", "remember", "was", "akira",
                     "thing", "um", "uh", "like", "hey", "know", "whats", "that"}


class Listener:
    def __init__(self, lac, router=None):
        self.lac = lac
        self.router = router
        self.text = ""
        self.understood = []      # ordered notes: what Akira has worked out so far
        self.results = {}         # lookup key -> result
        self.started = time.time()
        self._notes = set()

    def _note(self, key, message):
        if key not in self._notes:
            self._notes.add(key)
            self.understood.append({"at_ms": int((time.time() - self.started) * 1000), "note": message})

    def _lookup(self, key, fn, *args, **kw):
        if key in self.results:
            return self.results[key]
        try:
            out = fn(*args, **kw)
        except Exception as e:   # a half-finished phrase can make a lookup fail; try again on the next update
            return {"error": str(e)}
        self.results[key] = out
        return out

    def _people(self, text):
        found = []
        for c in self.lac.device.contacts():
            names = [c["name"].lower()] + [p for p in c["name"].lower().split() if len(p) > 2]
            if any(re.search(r"\b" + re.escape(n) + r"\b", text.lower()) for n in names):
                found.append(c["name"])
        return found

    def update(self, partial):
        """Feed the request as it is so far. Returns a snapshot of what is understood."""
        self.text = partial
        t = partial.strip()
        if not t:
            return self.snapshot()
        intents = {name for name, pat in protocol.LOCAL.items() if pat.search(t)}
        if self.router is not None and len(t.split()) >= 3:
            intents |= set(self.router.predict(t)["intents"])

        people = self._people(t)
        when = TIME_PHRASES.search(t)
        about = ABOUT.search(t)
        topic = about.group(1).strip() if about else ""

        for name in sorted(intents):
            self._note("intent:" + name, f"{name.capitalize()} is relevant")
        for p in people:
            self._note("person:" + p, f"narrowed to {p}")
        if topic:
            self._note("topic:" + topic, f"looking for '{topic}'")
        if when:
            self._note("when:" + when.group(0).lower(), f"time: {when.group(0).lower()}")

        sender = people[0] if people else ""
        when_s = when.group(0) if when else ""
        query = topic or " ".join(w for w in words(t) if w not in EXTRA_STOP and not any(w in p.lower() for p in people))

        if "messages" in intents:
            self._lookup(f"messages|{sender}|{query}|{when_s}", self.lac.search_messages, query=query, sender=sender, when=when_s)
        if "contacts" in intents and people:
            for p in people:
                self._lookup(f"contacts|{p}", self.lac.search_contacts, p)
        if "calendar" in intents:
            self._lookup(f"calendar|{query}|{when_s}", self.lac.calendar, query=query, when=when_s)
        if "files" in intents and query:
            self._lookup(f"files|{query}", self.lac.search_files, query)
        if "photos" in intents and query:
            self._lookup(f"photos|{query}", self.lac.search_photos, query)
        if "relations" in intents and query:
            self._lookup(f"related|{query}", self.lac.related, query)
        if "settings" in intents:
            self._lookup("settings", self.lac.get_settings)
        return self.snapshot()

    def snapshot(self):
        return {"text": self.text, "understood": list(self.understood),
                "lookups": list(self.results.keys())}

    def finish(self, final_text=None):
        if final_text is not None and final_text != self.text:
            self.update(final_text)
        # keep only the most specific lookup per source: later keys were made with more of the sentence
        latest = {}
        for key, val in self.results.items():
            latest[key.split("|")[0]] = (key, val)
        return {"understood": self.understood,
                "results": {src: val for src, (key, val) in latest.items() if val not in ([], {}, None)}}
