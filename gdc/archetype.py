"""
Archetype — "who the agent is."

Per the conceptual architecture: a stable, slow-forming disposition chosen at
instantiation and updated only slowly through fruit-tested experience. It
biases which strategies feel natural and which entities excite first. It does
not change during a single run.

This is a first, deliberately modest implementation: an Archetype narrows and
orders the strategy options the Executive is willing to consider for a goal.
Deeper effects (edge-maps, cross-domain trust) belong to the community layer
and are not built yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .entities import GoalType


@dataclass
class Archetype:
    """A named, stable cognitive disposition.

    `preferred_strategies` maps goal type -> ordered list of strategy names
    (matching `Strategy.name` in executive.STRATEGY_LIBRARY) that this
    archetype favours for that goal type. Strategies not listed are still
    available, just never preferred first. An empty mapping means "no bias" —
    the Executive's default ordering is used as-is.
    """
    name: str
    description: str
    preferred_strategies: dict[GoalType, list[str]] = field(default_factory=dict)
    domain: str = "general"


CHAVER = Archetype(
    name="Chaver",
    description=(
        "A companion mind: reaches for understanding before action, "
        "prefers to gather and connect before committing to a conclusion."
    ),
    preferred_strategies={
        GoalType.SOLVE: ["recall-and-hypothesise", "compare-and-decide"],
        GoalType.LEARN: ["explore-and-generalise"],
        GoalType.PLAN: ["recall-and-sequence"],
    },
    domain="general",
)
