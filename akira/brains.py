"""The brains behind Akira.

Akira is one assistant. Underneath, different "brains" handle different jobs:

    Core   Standard -> Pro -> Ultra      general intelligence
    Arc    Lite -> General -> Max        images and visual work
    Una    Lite -> General -> Max        code
    Live   Lite -> General -> Max        real-time audio
    Cast   Lite -> General -> Max        real-time audio + screen viewing

Every brain is one of Akira's own models, trained from scratch (akira/native):
the brain "core-pro" runs models/core-pro.pt. A brain that isn't trained yet
falls back to the nearest trained one.
"""
from dataclasses import dataclass, field

FAMILIES = {
    "core": {"name": "Core", "tiers": ["standard", "pro", "ultra"], "job": "general intelligence"},
    "arc": {"name": "Arc", "tiers": ["lite", "general", "max"], "job": "images and visual work"},
    "una": {"name": "Una", "tiers": ["lite", "general", "max"], "job": "code"},
    "live": {"name": "Live", "tiers": ["lite", "general", "max"], "job": "real-time audio"},
    "cast": {"name": "Cast", "tiers": ["lite", "general", "max"], "job": "real-time audio and screen viewing"},
}

# Things a brain can take in or produce. "image_out" has no backend yet.
CAPABILITIES = {
    "core": {"text", "image_in", "tools"},
    "arc": {"text", "image_in", "image_out", "tools"},
    "una": {"text", "image_in", "tools"},
    "live": {"text", "audio_in", "audio_out", "tools"},
    "cast": {"text", "audio_in", "audio_out", "image_in", "screen_in", "tools"},
}



@dataclass
class Brain:
    family: str
    tier: str
    model: str          # the checkpoint file that runs this brain
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
            brains[bid] = Brain(fam, tier, f"{bid}.pt", set(CAPABILITIES[fam]))
    return brains


BRAINS = _build()


def get(family, tier):
    return BRAINS[f"{family}-{tier}"]


def tier_for_rank(family, rank):
    return FAMILIES[family]["tiers"][max(0, min(2, rank))]
