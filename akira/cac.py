"""CAC, the Cloud Access Center.

Runs a brain on a large cloud model when the job needs more knowledge,
reasoning or context than the on-device brains have. The cloud brain can call
back into the LAC's tools (messages, calendar, files, relations, ...), and each
call runs here on the device, so personal data only leaves the device when the
cloud brain asks for a specific piece of it.

Needs ANTHROPIC_API_KEY (or another Anthropic credential). Without one,
CloudAccessCenter.available is False and Akira answers with its native brains.
"""
import base64
import json
import os

from . import brains

SYSTEM = """You are Akira, a personal AI assistant. You are one assistant: never mention internal brain names (Core, Arc, Una, Live, Cast), tiers, models or routing.
You run partly on the user's device. Tools that start with search_, calendar, related, settings, calculator and clock read or change the user's device; use them whenever the answer depends on the user's own messages, people, schedule, files, photos or settings, and never invent personal details.
You can search the web (web_search) and read pages (web_fetch). Use them for anything current, specific or that you are unsure of, and cite what you used as Markdown links.
Answer directly first, then add detail when it helps. Use Markdown in text replies."""

FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}   # models that accept server-side refusal fallback
MAX_TOOL_ROUNDS = 6


def web_tools(model):
    """Anthropic's server-side web search and fetch. The newest variants (with result filtering) need Opus or
    Sonnet; other models get the basic ones. AKIRA_WEB=off turns the internet off."""
    if os.environ.get("AKIRA_WEB", "on") == "off":
        return []
    newest = model.startswith(("claude-opus", "claude-sonnet"))
    return [{"type": "web_search_20260209" if newest else "web_search_20250305", "name": "web_search", "max_uses": 5},
            {"type": "web_fetch_20260209" if newest else "web_fetch_20250910", "name": "web_fetch", "max_uses": 3}]


class CloudAccessCenter:
    def __init__(self, lac):
        self.lac = lac
        self.client = None
        try:
            import anthropic
            if os.environ.get("AKIRA_CLOUD", "on") != "off":
                self.client = anthropic.Anthropic()
        except Exception:   # no SDK or no credential: native brains only
            self.client = None

    @property
    def available(self):
        if self.client is None:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    or os.environ.get("AKIRA_CLOUD") == "on")

    def _content(self, text, images):
        blocks = []
        for img in images or []:
            data, media = img["data"], img.get("media_type", "image/jpeg")
            if isinstance(data, bytes):
                data = base64.b64encode(data).decode()
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        blocks.append({"type": "text", "text": text or "(see image)"})
        return blocks

    def _valid(self, name, args):
        spec = next((s for s in self.lac.TOOL_SPECS if s[0] == name), None)
        return spec is not None and isinstance(args, dict) and all(k in args for k in spec[3])

    def stream(self, brain: brains.Brain, history, text, images=None, prefetched=None, events=None, stop=None):
        """Run one turn. Yields ("text", str) and ("tool", {...}) events; returns at end of turn.

        history: earlier [{"role", "content"}] turns as plain text.
        prefetched: LAC results First-Hand Processing already gathered while the user was talking.
        """
        persona = brains.PERSONAS.get(brain.family, "")
        system = SYSTEM + ("\n\n" + persona if persona else "")
        user_text = text
        if prefetched:
            user_text += ("\n\n<on_device_context>\nAkira already looked these up on the device while the user "
                          "was talking; use them if relevant:\n" + json.dumps(prefetched, default=str)[:12000]
                          + "\n</on_device_context>")
        messages = [{"role": m["role"], "content": m["content"]} for m in history if m.get("content")]
        messages.append({"role": "user", "content": self._content(user_text, images)})

        params = {
            "model": brain.model,
            "max_tokens": 64000,
            "system": system,
            "tools": self.lac.tool_definitions() + web_tools(brain.model),
            "output_config": {"effort": brain.effort},
        }
        use_fallback = brain.model in FALLBACK_MODELS
        for _ in range(MAX_TOOL_ROUNDS):
            if use_fallback:
                ctx = self.client.beta.messages.stream(messages=messages, betas=["server-side-fallback-2026-07-01"],
                                                       fallbacks="default", **params)
            else:
                ctx = self.client.messages.stream(messages=messages, **params)
            with ctx as s:
                for event in s:
                    if stop is not None and stop.is_set():
                        return
                    if event.type == "text":
                        yield ("text", event.text)
                response = s.get_final_message()

            if response.stop_reason == "refusal":
                yield ("text", "\n\nI can't help with that one.")
                return
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue
            for b in response.content:
                if b.type == "server_tool_use":    # web search / fetch, run by the cloud
                    yield ("tool", {"name": b.name, "input": b.input, "where": "internet"})
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                return
            if response.stop_reason == "max_tokens":
                yield ("text", "\n\n(The answer was cut short.)")
                return
            results = []
            for b in tool_uses:
                if not self._valid(b.name, b.input):
                    results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                    "content": json.dumps({"INVALID_JSON": json.dumps(b.input, default=str)})})
                    continue
                try:
                    out = self.lac.run_tool(b.name, b.input)
                    yield ("tool", {"name": b.name, "input": b.input, "where": "device"})
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out, default=str)[:20000]})
                except Exception as e:  # tool errors go back to the model, which can recover
                    results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True, "content": str(e)})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": results})
        yield ("text", "\n\n(Stopped after several lookups.)")
