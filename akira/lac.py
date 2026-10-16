"""LAC, the Local Access Center.

Everything Akira can do on the device itself: search messages, contacts,
calendar, files and photos, follow relations, change settings, do maths and
tell the time. These run without the cloud, so they work offline and keep
personal data on the device unless a cloud brain needs it to answer.

Each tool is plain Python. The same tools are offered to cloud brains (as
tool definitions) and called directly when Akira answers locally.
"""
import ast
import math
import operator
import re
from datetime import datetime, timedelta

from .device import DemoDevice, DeviceData
from .relations import words

TIME_WORDS = [
    (r"\btoday\b", 0, 1), (r"\byesterday\b", 1, 2), (r"\bthe other day\b", 1, 5),
    (r"\bthis week\b", 0, 7), (r"\blast week\b", 7, 14), (r"\brecently\b", 0, 7),
    (r"\bthis month\b", 0, 31), (r"\blast month\b", 31, 62),
]


def time_window(text, now):
    """Turn phrases like 'the other day' into a (start, end) range, or None."""
    t = text.lower()
    for pat, back_start, back_end in TIME_WORDS:
        if re.search(pat, t):
            return (now - timedelta(days=back_end)).replace(hour=0, minute=0), now - timedelta(days=back_start) + timedelta(days=1)
    return None


def _in_window(when, window):
    return window is None or window[0] <= when <= window[1]


# --- safe calculator -------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "log", "log10", "log2", "exp", "floor", "ceil")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_NAMES = {"pi": math.pi, "e": math.e}


def calculate(expression):
    expr = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", expr)   # "15% of 80"
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, ast.Name) and n.id in _NAMES:
            return _NAMES[n.id]
        raise ValueError("unsupported expression")

    value = ev(ast.parse(expr, mode="eval"))
    return round(value, 10) if isinstance(value, float) else value


class LocalAccessCenter:
    def __init__(self, device: DeviceData = None):
        self.device = device or DemoDevice()
        self.graph = self.device.graph()
        self.now = getattr(self.device, "now", None) or datetime.now()

    # --- tools ------------------------------------------------------------
    def search_messages(self, query="", sender="", when=""):
        window = time_window(when or query, self.now)
        q = set(words(query))
        out = []
        for m in self.device.messages():
            if sender and sender.lower() not in m["from"].lower():
                continue
            if not _in_window(m["at"], window):
                continue
            score = len(q & set(words(m["text"] + " " + m["from"]))) if q else 1
            if score or not q:
                out.append((score, m))
        out.sort(key=lambda sm: (-sm[0], -sm[1]["at"].timestamp()))
        return [{**m, "at": m["at"].strftime("%a %b %d, %I:%M %p")} for _, m in out[:5]]

    def search_contacts(self, name):
        n = name.lower()
        return [c for c in self.device.contacts() if n in c["name"].lower() or any(n == p for p in words(c["name"]))]

    def calendar(self, query="", when=""):
        window = time_window(when or query, self.now)
        q = set(words(query))
        out = []
        for e in self.device.events():
            if window and not (_in_window(e["start"], window)):
                continue
            if q and not q & set(words(e["title"] + " " + " ".join(e["with"]))):
                continue
            out.append({**e, "start": e["start"].strftime("%a %b %d, %I:%M %p"), "end": e["end"].strftime("%I:%M %p")})
        return out

    def search_files(self, query):
        q = set(words(query))
        scored = [(len(q & set(words(f["name"] + " " + f["path"] + " " + f["summary"]))), f) for f in self.device.files()]
        return [{**f, "modified": f["modified"].strftime("%b %d")} for s, f in sorted(scored, key=lambda x: -x[0]) if s][:5]

    def search_photos(self, query):
        q = set(words(query))
        hits = [p for p in self.device.photos() if q & set(words(" ".join(p["tags"]) + " " + p["name"]))]
        return [{**p, "taken": p["taken"].strftime("%b %d")} for p in hits]

    def related(self, query, kinds=None):
        return self.graph.related(query, want=set(kinds) if kinds else None)

    def get_settings(self):
        return dict(self.device.settings())

    def change_setting(self, name, value):
        return {name: self.device.set_setting(name, value)}

    def calculator(self, expression):
        return {"expression": expression, "result": calculate(expression)}

    def clock(self):
        n = datetime.now()
        return {"local": n.strftime("%A %B %d %Y, %I:%M %p"), "iso": n.isoformat(timespec="seconds")}

    # --- tool definitions for cloud brains ---------------------------------
    TOOL_SPECS = [
        ("search_messages", "Search the user's text messages on their device. Returns up to 5 messages with sender, time and text.",
         {"query": {"type": "string", "description": "words to look for"},
          "sender": {"type": "string", "description": "part of the sender's name"},
          "when": {"type": "string", "description": "a time phrase such as 'yesterday' or 'last week'"}}, []),
        ("search_contacts", "Look up people in the user's contacts by name.",
         {"name": {"type": "string"}}, ["name"]),
        ("calendar", "List calendar events, optionally filtered by words and a time phrase.",
         {"query": {"type": "string"}, "when": {"type": "string"}}, []),
        ("search_files", "Search the user's local files by name, folder and summary.",
         {"query": {"type": "string"}}, ["query"]),
        ("search_photos", "Search the user's photos by tag, for example 'beach' or a person's name.",
         {"query": {"type": "string"}}, ["query"]),
        ("related", "Follow connections between people, messages, meetings, projects, files, decisions and tasks. "
                    "Use for questions like 'documents related to the decision we made after that meeting'. Returns ranked items with the path that connects them.",
         {"query": {"type": "string"}, "kinds": {"type": "array", "items": {"type": "string",
          "enum": ["person", "message", "meeting", "project", "file", "decision", "task"]}}}, ["query"]),
        ("get_settings", "Read the device settings (wifi, bluetooth, do_not_disturb, dark_mode, brightness, volume).", {}, []),
        ("change_setting", "Change one device setting.",
         {"name": {"type": "string", "enum": ["wifi", "bluetooth", "do_not_disturb", "dark_mode", "brightness", "volume"]},
          "value": {"type": ["boolean", "integer"]}}, ["name", "value"]),
        ("calculator", "Evaluate a maths expression exactly, e.g. '15% of 80' or '(3.5^2) * pi'.",
         {"expression": {"type": "string"}}, ["expression"]),
        ("clock", "Current local date and time on the device.", {}, []),
    ]

    def tool_definitions(self):
        return [{"name": n, "description": d, "eager_input_streaming": True,
                 "input_schema": {"type": "object", "properties": p, "required": r}}
                for n, d, p, r in self.TOOL_SPECS]

    def run_tool(self, name, args):
        spec = next((s for s in self.TOOL_SPECS if s[0] == name), None)
        if spec is None:
            raise ValueError(f"unknown tool {name}")
        if not isinstance(args, dict):
            raise ValueError("tool input must be an object")
        missing = [k for k in spec[3] if k not in args]
        if missing:
            raise ValueError(f"missing {', '.join(missing)}")
        args = {k: v for k, v in args.items() if k in spec[2]}
        return getattr(self, name)(**args)
