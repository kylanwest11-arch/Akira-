"""The brains behind Akira.

Akira is one assistant. Underneath, different "brains" handle different jobs:

    Core   Standard -> Pro -> Ultra      general intelligence
    Arc    Lite -> General -> Max        images and visual work
    Una    Lite -> General -> Max        code
    Live   Lite -> General -> Max        real-time audio
    Cast   Lite -> General -> Max        real-time audio + screen viewing

Each brain names the backend model that runs it. Backends are swappable: point
a brain at a different model in BRAIN_MODELS (or the AKIRA_MODEL_<ID> env var)
without touching the rest of Akira.
"""
import os
from dataclasses import dataclass, field

FAMILIES = {
    "core": {"name": "Core", "tiers": ["standard", "pro", "ultra"], "job": "general intelligence"},
    "arc": {"name": "Arc", "tiers": ["lite", "general", "max"], "job": "images and visual work"},
    "una": {"name": "Una", "tiers": ["lite", "general", "max"], "job": "code"},
    "live": {"name": "Live", "tiers": ["lite", "general", "max"], "job": "real-time audio"},
    "cast": {"name": "Cast", "tiers": ["lite", "general", "max"], "job": "real-time audio and screen viewing"},
}

# Tier rank 0/1/2 maps onto the same three cloud model sizes for every family.
CLOUD_BY_RANK = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
EFFORT_BY_RANK = ["low", "medium", "high"]

# Things a brain can take in or produce. "image_out" has no backend yet.
CAPABILITIES = {
    "core": {"text", "image_in", "tools"},
    "arc": {"text", "image_in", "image_out", "tools"},
    "una": {"text", "image_in", "tools"},
    "live": {"text", "audio_in", "audio_out", "tools"},
    "cast": {"text", "audio_in", "audio_out", "image_in", "screen_in", "tools"},
}

PERSONAS = {
    "core": "",
    "arc": "You are working as Arc, Akira's visual intelligence. Describe and analyse images precisely. "
           "When asked to create or edit an image, write a detailed visual brief: subject, composition, "
           "lighting, lens, colour grade and mood.",
    "una": "You are working as Una, Akira's coding intelligence. Write correct, idiomatic code, explain "
           "why bugs happen before fixing them, and keep explanations tight.",
    "live": "You are speaking out loud in a live voice conversation. Answer in short, natural spoken "
            "sentences. No Markdown, no lists, no code blocks.",
    "cast": "You are in a live voice conversation and can see the user's screen. The attached image is their "
            "screen right now, and the on-device context may include the text Akira read from it. Refer to "
            "specific things on the screen. Answer in short, natural spoken sentences.",
}


@dataclass
class Brain:
    family: str
    tier: str
    model: str
    effort: str
    capabilities: set = field(default_factory=set)

    @property
    def id(self):
        return f"{self.family}-{self.tier}"

    @property
    def label(self):
        return f"{FAMILIES[self.family]['name']} {self.tier.capitalize()}"

    @property
    def rank(self):
        return FAMILIES[self.family]["tiers"].index(self.tier)

    def to_dict(self):
        return {"id": self.id, "label": self.label, "family": self.family, "tier": self.tier,
                "job": FAMILIES[self.family]["job"], "model": self.model,
                "capabilities": sorted(self.capabilities)}


def _build():
    brains = {}
    for fam, spec in FAMILIES.items():
        for rank, tier in enumerate(spec["tiers"]):
            bid = f"{fam}-{tier}"
            model = os.environ.get("AKIRA_MODEL_" + bid.upper().replace("-", "_"), CLOUD_BY_RANK[rank])
            brains[bid] = Brain(fam, tier, model, EFFORT_BY_RANK[rank], set(CAPABILITIES[fam]))
    return brains


BRAINS = _build()


def get(family, tier):
    return BRAINS[f"{family}-{tier}"]


def tier_for_rank(family, rank):
    return FAMILIES[family]["tiers"][max(0, min(2, rank))]
