"""The device data Akira's Local Access Center can reach.

A real app supplies its own DeviceData, wired to the phone's or computer's
messages, contacts, calendar, files, photos and settings (with the user's
permission). DemoDevice below is sample data so every feature can be tried
out of the box. Nothing in it is real.
"""
from datetime import datetime, timedelta

from .relations import RelationGraph


class DeviceData:
    """Override these in your app. Each returns plain dicts.

    The base class has no data: that is the right choice for a public website,
    where Akira's server can't see a visitor's messages, calendar or files.
    """
    personal = False       # True when messages, contacts, calendar, files and photos are real

    def messages(self): return []
    def contacts(self): return []
    def events(self): return []
    def files(self): return []
    def photos(self): return []
    def settings(self): return {}
    def set_setting(self, name, value): raise KeyError(name)
    def graph(self): return RelationGraph()


class DemoDevice(DeviceData):
    personal = True

    def __init__(self, now=None):
        self.now = now or datetime.now()
        d = lambda days, h=12, m=0: (self.now - timedelta(days=days)).replace(hour=h, minute=m, second=0, microsecond=0)
        self._contacts = [
            {"id": "p_thomas", "name": "Thomas Reed", "phone": "+1 555 0142", "relation": "friend"},
            {"id": "p_maya", "name": "Maya Chen", "phone": "+1 555 0177", "relation": "coworker", "company": "Riverbook"},
            {"id": "p_jonas", "name": "Jonas Patel", "phone": "+1 555 0109", "relation": "coworker", "company": "Riverbook"},
            {"id": "p_mom", "name": "Mom", "phone": "+1 555 0100", "relation": "family"},
        ]
        self._messages = [
            {"id": "m1", "from": "Thomas Reed", "at": d(2, 19, 4), "text": "Hey! Want to catch the new Dune movie Friday? The 8:15 showing at the Alamo has seats left. I can grab tickets."},
            {"id": "m2", "from": "Thomas Reed", "at": d(2, 19, 6), "text": "Also the movies after are on me if you drive."},
            {"id": "m3", "from": "Maya Chen", "at": d(1, 9, 30), "text": "Notes from yesterday's keyboard review are in the Riverbook drive. We went with the split layout."},
            {"id": "m4", "from": "Mom", "at": d(0, 8, 12), "text": "Don't forget Grandma's birthday dinner on Sunday at 6."},
            {"id": "m5", "from": "Jonas Patel", "at": d(5, 16, 40), "text": "Can you send me the Riverbook battery test results before the keyboard meeting?"},
        ]
        self._events = [
            {"id": "e1", "title": "Riverbook keyboard review", "start": d(2, 14), "end": d(2, 15), "with": ["Maya Chen", "Jonas Patel"]},
            {"id": "e2", "title": "Grandma's birthday dinner", "start": d(-3, 18), "end": d(-3, 20), "with": ["Mom"]},
            {"id": "e3", "title": "Dentist", "start": d(-1, 10, 30), "end": d(-1, 11), "with": []},
        ]
        self._files = [
            {"id": "f1", "name": "Riverbook keyboard review notes.docx", "path": "Riverbook/Meetings/", "modified": d(2, 15, 20), "summary": "Split vs. unified layout comparison; team chose split layout."},
            {"id": "f2", "name": "Riverbook split layout spec v2.pdf", "path": "Riverbook/Specs/", "modified": d(1, 11), "summary": "Key dimensions and switch choice for the split keyboard."},
            {"id": "f3", "name": "Battery test results.xlsx", "path": "Riverbook/Testing/", "modified": d(6, 17), "summary": "Runtime under load for three cell options."},
            {"id": "f4", "name": "Tax return 2025.pdf", "path": "Personal/", "modified": d(40), "summary": "Filed tax return."},
        ]
        self._photos = [
            {"id": "ph1", "name": "IMG_2041.jpg", "taken": d(9), "tags": ["beach", "sunset", "Maya Chen"]},
            {"id": "ph2", "name": "IMG_2107.jpg", "taken": d(2, 15), "tags": ["whiteboard", "keyboard", "sketch"]},
            {"id": "ph3", "name": "IMG_2110.jpg", "taken": d(1, 20), "tags": ["dog", "park"]},
        ]
        self._settings = {"wifi": True, "bluetooth": True, "do_not_disturb": False, "dark_mode": False, "brightness": 70, "volume": 40}

    def messages(self): return self._messages
    def contacts(self): return self._contacts
    def events(self): return self._events
    def files(self): return self._files
    def photos(self): return self._photos
    def settings(self): return self._settings

    def set_setting(self, name, value):
        if name not in self._settings:
            raise KeyError(name)
        self._settings[name] = value
        return self._settings[name]

    def graph(self):
        g = RelationGraph()
        for c in self._contacts:
            g.add(c["id"], "person", c["name"], relation=c["relation"], company=c.get("company", ""))
        for m in self._messages:
            g.add(m["id"], "message", m["text"][:60], text=m["text"], at=m["at"].isoformat())
        for e in self._events:
            g.add(e["id"], "meeting", e["title"], at=e["start"].isoformat())
        for f in self._files:
            g.add(f["id"], "file", f["name"], path=f["path"], summary=f["summary"])
        g.add("proj_riverbook", "project", "Riverbook", summary="Hardware keyboard product")
        g.add("dec_layout", "decision", "Riverbook keyboard layout: go with split layout", at=self._events[0]["end"].isoformat())
        g.add("task_spec", "task", "Update split layout spec", owner="Maya Chen")
        people = {c["name"]: c["id"] for c in self._contacts}
        for m in self._messages:
            g.link(m["id"], "sent_by", people[m["from"]])
        for e in self._events:
            for name in e["with"]:
                g.link(e["id"], "attended_by", people[name])
        g.link("e1", "part_of", "proj_riverbook")
        g.link("dec_layout", "decided_at", "e1")
        g.link("dec_layout", "part_of", "proj_riverbook")
        g.link("f1", "records", "e1")
        g.link("f1", "documents", "dec_layout")
        g.link("f2", "implements", "dec_layout")
        g.link("f3", "part_of", "proj_riverbook")
        g.link("m3", "mentions", "dec_layout")
        g.link("m5", "mentions", "f3")
        g.link("task_spec", "follows_from", "dec_layout")
        g.link("task_spec", "edits", "f2")
        return g
