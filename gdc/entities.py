"""
Core data primitives for the Goal-Driven Cognitive Architecture.

These are deliberately plain dataclasses. The architecture — not the model —
supplies structure, so the structure has to be explicit and inspectable.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------
# Executive layer
# --------------------------------------------------------------------------
class GoalType(str, Enum):
    LEARN = "learn"
    SOLVE = "solve"
    PLAN = "plan"
    PERFORM = "perform"
    EXPLORE = "explore"
    BRAINSTORM = "brainstorm"
    RECALL = "recall"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    SATISFIED = "satisfied"
    ABANDONED = "abandoned"
    SUSPENDED = "suspended"


@dataclass
class Goal:
    """The current intention. 'Why cognition occurs.'"""
    description: str
    goal_type: GoalType
    success_criteria: list[str] = field(default_factory=list)
    urgency: float = 0.5          # 0..1, feeds Attention
    priority: float = 0.5         # 0..1, executive scheduling
    status: GoalStatus = GoalStatus.ACTIVE
    id: str = field(default_factory=lambda: _new_id("goal"))
    parent_id: Optional[str] = None


@dataclass
class Strategy:
    """'How the goal will be pursued.' Determines what Attention retrieves
    and which memory-interaction operations run this cycle."""
    name: str
    description: str
    # ordered memory-interaction operations this strategy prefers
    operations: list[str] = field(default_factory=list)
    # extra text used to focus retrieval (biases the attention query)
    retrieval_focus: str = ""


# --------------------------------------------------------------------------
# Long-term memory
# --------------------------------------------------------------------------
class MemoryLevel(str, Enum):
    TOPIC = "topic"
    SUBJECT = "subject"
    CONCEPT = "concept"
    ENTITY = "entity"


@dataclass
class MemoryNode:
    """
    A node in distributed cortical memory. The same class spans every level
    (Topic -> Subject -> Concept -> Entity) because 'each level is a network
    composed of the level beneath it.'

    An Entity is not a word; `associations` carries the multimodal experience
    (appearance, movement, sound, motor interaction, prior episodes...).
    """
    name: str
    level: MemoryLevel
    text: str = ""                              # canonical text used for embedding
    associations: dict[str, Any] = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)   # ids one level up
    children: list[str] = field(default_factory=list)  # ids one level down
    salience: float = 0.5                       # intrinsic importance 0..1
    episodic: bool = False                      # True if hippocampus-consolidated
    created: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: _new_id("node"))
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def embed_text(self) -> str:
        assoc = " ".join(f"{k}:{v}" for k, v in self.associations.items())
        return f"{self.name}. {self.text} {assoc}".strip()


# --------------------------------------------------------------------------
# Working memory
# --------------------------------------------------------------------------
class WMKind(str, Enum):
    GOAL = "goal"
    STRATEGY = "strategy"
    ENTITY_REF = "entity_ref"     # pointer into LTM
    CONCEPT_REF = "concept_ref"   # pointer into LTM
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    PLAN = "plan"
    OBSERVATION = "observation"
    DECISION = "decision"


@dataclass
class WMItem:
    """
    A temporary reference held in the workspace. Working memory stores
    references/pointers into long-term memory, plus derived transient objects
    (hypotheses, plans, observations) produced during thought.
    """
    kind: WMKind
    content: str                                # human-readable summary
    ref: Optional[str] = None                   # id of a MemoryNode, if a pointer
    activation: float = 1.0                     # decays each cycle, boosted by attention
    source: str = "attention"                   # what put it here
    created_cycle: int = 0
    last_touched_cycle: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _new_id("wm"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d
