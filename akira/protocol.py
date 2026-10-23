"""Alternate Protocol: the orchestration layer.

For every request it decides, in a few milliseconds and without a cloud call:

  1. which intelligence:   Core, Arc, Una, Live or Cast
  2. which tier:           Standard/Pro/Ultra (Core) or Lite/General/Max (specialists)
  3. where it runs:        local (LAC, on the device), cloud (CAC: the Akira server's larger brains
                           and the internet) or both

The user never picks. They just talk to Akira.

Two deciders work together. The router brain (akira/native/router.py, trained
from scratch) reads the request and predicts all three; when it is confident
its call wins. The hand-written rules below decide everything else and are
the fallback when no router has been trained.
"""
import re
from dataclasses import dataclass, field

from . import brains
from .web import NEEDS_WEB

CODE = re.compile(r"```|\b(def|class|function|const|let|var|import|return|async|await)\b|=>|\{\s*\}|;\s*$|"
                  r"\b(bug|debug|stack ?trace|traceback|exception|compile|compiler|refactor|regex|sql|python|javascript|"
                  r"typescript|java|rust|golang|c\+\+|html|css|react|api endpoint|unit test|function|code|script|syntax error)\b",
                  re.I | re.M)
ARC_MAKE = re.compile(r"\b(draw|paint|sketch|illustrat\w*|render|generate (an? )?(image|picture|photo|logo|icon)|"
                      r"make (this|it|the (photo|image|picture))( look)?|cinematic|logo|poster|thumbnail|wallpaper|"
                      r"edit (this|the) (photo|image|picture)|remove the background|upscale|colou?r grade)\b", re.I)
ARC_LOOK = re.compile(r"\b(what('s| is) in (this|the) (image|photo|picture)|describe (this|the) (image|photo|picture)|"
                      r"what do you see|identify|read (this|the) (sign|label|screenshot))\b", re.I)
REASONING = re.compile(r"\b(prove|analy[sz]e|analysis|strategy|compare|trade-?offs?|plan|step by step|architecture|"
                       r"design|research|in depth|in-depth|detailed|thorough|optimi[sz]e|why does|explain why|"
                       r"evaluate|critique|derive|business plan|essay|report)\b", re.I)

LOCAL = {
    # "said"/"wrote" only about the user's own conversations: "what did Maya say", not "who wrote Hamlet"
    "messages": re.compile(r"\b(message|messages|text(ed)?|sms|sent me|chat with)\b|"
                           r"\b(said|wrote|say|write)\b.*\b(to me|to us|me back)\b|\bdid \w+ (say|write)\b", re.I),
    "contacts": re.compile(r"\b(contact|phone number|number for|call|email address)\b", re.I),
    "calendar": re.compile(r"\b(calendar|meeting|schedule|appointment|event|dinner|lunch|what('s| is) on|free on|busy)\b", re.I),
    "files": re.compile(r"\b(file|files|document|documents|doc|docs|spreadsheet|pdf|notes|folder|spec)\b", re.I),
    "photos": re.compile(r"\b(my photos?|my pictures?|photos? (of|from)|pictures? (of|from)|camera roll|gallery)\b", re.I),
    "settings": re.compile(r"\b(wi-?fi|bluetooth|brightness|volume|dark mode|do not disturb|dnd)\b", re.I),
    "calculator": re.compile(r"(\d\s*[-+*/^×÷%]\s*\(?\d)|\b\d+(\.\d+)?\s*% of\b|\b(calculate|sqrt|square root)\b|"
                            r"\b\d+(\.\d+)?\s+(times|divided by|plus|minus|x)\s+\d", re.I),
    "clock": re.compile(r"\b(what time is it|what('s| is) the time|what('s| is) (the date|today's date)|what day is it)\b", re.I),
    "relations": re.compile(r"\b(related to|connected to|linked to|relating to|associated with|that meeting|the decision)\b", re.I),
}
LOOKUP = re.compile(r"^\s*(akira[, ]+)?(what|what's|show|find|when|where|who|turn|set|switch|enable|disable|"
                    r"open|calculate|how much is|look up|search|list|did)\b", re.I)


@dataclass
class Request:
    text: str = ""
    images: list = field(default_factory=list)      # base64 or bytes, already checked by the caller
    modality: str = "text"                           # text | audio | screen
    tier: str = "auto"                              # auto | 0 | 1 | 2 | a tier name
    family: str = "auto"                            # auto | core | arc | una | live | cast
    cloud_available: bool = True
    router: object = None                            # a trained akira.native.router.Router, if any


@dataclass
class Route:
    brain: brains.Brain
    placement: str               # local | cloud | both
    local_intents: list
    reasons: list
    web: bool = False            # the answer needs the internet (news, prices, "search for", a link, ...)

    def to_dict(self):
        return {"brain": self.brain.to_dict(), "placement": self.placement,
                "local_intents": self.local_intents, "reasons": self.reasons, "web": self.web}


def pick_family(req: Request, reasons):
    if req.family in brains.FAMILIES:
        reasons.append(f"family set by caller: {req.family}")
        return req.family
    if req.modality in ("screen", "video"):
        reasons.append("shared screen + voice -> Cast")
        return "cast"
    if req.modality == "audio":
        reasons.append("live voice -> Live")
        return "live"
    if CODE.search(req.text):
        reasons.append("code detected -> Una")
        return "una"
    if ARC_MAKE.search(req.text):
        reasons.append("visual creation or editing -> Arc")
        return "arc"
    if req.images and (ARC_LOOK.search(req.text) or not req.text.strip()):
        reasons.append("image to analyse -> Arc")
        return "arc"
    reasons.append("general request -> Core")
    return "core"


def pick_rank(req: Request, family, reasons):
    tiers = brains.FAMILIES[family]["tiers"]
    if req.tier != "auto":
        rank = tiers.index(req.tier) if req.tier in tiers else int(req.tier)
        reasons.append(f"tier set by caller: {tiers[rank]}")
        return rank
    score = 0
    n = len(req.text.split())
    score += 0 if n < 25 else 1 if n < 120 else 2
    score += min(3, len(REASONING.findall(req.text)) * 1.5)
    score += 1 if req.text.count("?") > 1 else 0
    score += 1 if req.images else 0
    if family == "una" and n > 60:
        score += 1
    if family in ("live", "cast"):
        score = min(score, 2)   # keep live conversation fast
    rank = 0 if score < 1.5 else 1 if score < 3.5 else 2
    reasons.append(f"complexity {score:g} -> {tiers[rank]}")
    return rank


ROUTER_FAMILY_CONFIDENCE = 0.8
ROUTER_RANK_CONFIDENCE = 0.6


# asking Akira to make something up: a story, poem, song or ideas (Core writes it, nothing to look up)
CREATIVE = re.compile(r"\b(write|compose|make up|create|come up with|tell|give|invent|dream up)\b.{0,40}\b("
                      r"story|stories|poem|poems|poetry|song|lyrics|haiku|limerick|sonnet|rap|verse|fairy ?tale|fable|"
                      r"tale|play|script for an? (\w+ )?(play|movie|film|scene|skit)|screenplay|skit|monologue|joke|riddle|"
                      r"bedtime story|ideas?|names?|slogans?|taglines?|plot)\b|\bbrainstorm\b|\bonce upon a time\b",
                      re.I)


def is_creative(text):
    plain = re.sub(r"(?i)\bscripts? for an? (\w+ )?(play|movie|film|scene|skit)\b", "", text)   # a play's script isn't code
    return bool(CREATIVE.search(text)) and not CODE.search(plain)


def route(req: Request) -> Route:
    reasons = []
    learned = None
    if is_creative(req.text) and not req.images and req.family == "auto" and req.modality == "text":
        rank = pick_rank(req, "core", reasons) if req.tier == "auto" else None
        brain = brains.get("core", req.tier if req.tier != "auto" else brains.tier_for_rank("core", rank))
        reasons.append("creative writing -> Core writes it")
        return Route(brain, "cloud" if req.cloud_available else "local", [], reasons, False)
    if req.router is not None and req.text.strip() and req.modality == "text":
        learned = req.router.predict(req.text)

    if (learned and req.family == "auto" and learned["family_confidence"] >= ROUTER_FAMILY_CONFIDENCE
            and not (req.images and learned["family"] == "core" and ARC_LOOK.search(req.text))):
        family = learned["family"]
        reasons.append(f"router brain: {family} ({learned['family_confidence']:.0%} sure)")
    else:
        family = pick_family(req, reasons)

    if (learned and req.tier == "auto" and family == learned["family"]
            and learned["rank_confidence"] >= ROUTER_RANK_CONFIDENCE):
        rank = learned["rank"]
        reasons.append(f"router brain: {brains.FAMILIES[family]['tiers'][rank]} ({learned['rank_confidence']:.0%} sure)")
    else:
        rank = pick_rank(req, family, reasons)
    brain = brains.get(family, brains.tier_for_rank(family, rank))

    intents = [name for name, pat in LOCAL.items() if pat.search(req.text)]
    if learned:
        intents += [i for i in learned["intents"] if i not in intents]
    if "photos" in intents and family == "arc":
        intents.remove("photos")
    if intents:
        reasons.append("needs on-device data: " + ", ".join(intents))

    if not req.cloud_available:
        placement = "local"
        reasons.append("cloud unavailable -> local only")
    elif intents and family == "core" and rank == 0 and LOOKUP.search(req.text) and not REASONING.search(req.text):
        placement = "local"
        reasons.append("simple on-device task -> answer locally, nothing leaves the device")
    elif intents:
        placement = "both"
        reasons.append("LAC gathers local context, the Akira server reasons over it")
    else:
        placement = "cloud"
    needs_web = bool(NEEDS_WEB.search(req.text)) and family in ("core", "una")
    if needs_web:
        reasons.append("needs current information -> search the internet")
    return Route(brain, placement, intents, reasons, needs_web)
