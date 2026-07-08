"""
Attention — a routing mechanism.

It builds a query from the current goal + strategy + already-active workspace
contents, scores long-term memory against it, and loads the winners into working
memory as references. Score blends relevance, novelty, importance and urgency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import Embedder
from .entities import Goal, Strategy, WMItem, WMKind, MemoryLevel
from .long_term_memory import LongTermMemory
from .working_memory import WorkingMemory


@dataclass
class AttentionWeights:
    relevance: float = 0.55
    novelty: float = 0.15
    importance: float = 0.20
    urgency: float = 0.10


class Attention:
    def __init__(self, embedder: Embedder, weights: AttentionWeights | None = None):
        self.embedder = embedder
        self.w = weights or AttentionWeights()

    def _query_embedding(self, goal: Goal, strategy: Strategy, wm: WorkingMemory) -> np.ndarray:
        parts = [goal.description, strategy.retrieval_focus]
        # let the current live workspace bias what we look for next
        parts += [i.content for i in wm.by_kind(WMKind.HYPOTHESIS)]
        parts += [i.content for i in wm.by_kind(WMKind.OBSERVATION)]
        return self.embedder.encode(" ".join(p for p in parts if p))

    def allocate(self, goal: Goal, strategy: Strategy, ltm: LongTermMemory,
                 wm: WorkingMemory, cycle: int, k: int = 4) -> list[tuple[str, float]]:
        """Select memory and load references into working memory. Returns the
        (node_name, score) list actually admitted, for tracing."""
        q = self._query_embedding(goal, strategy, wm)
        candidates = ltm.recall(q, k=k * 3)

        already = {i.ref for i in wm.refs()}
        scored = []
        for node, relevance in candidates:
            novelty = 1.0 if node.id not in already else 0.2
            importance = node.salience
            urgency = goal.urgency
            score = (self.w.relevance * relevance
                     + self.w.novelty * novelty
                     + self.w.importance * importance
                     + self.w.urgency * urgency)
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        admitted: list[tuple[str, float]] = []
        for node, score in scored[:k]:
            kind = WMKind.CONCEPT_REF if node.level in (
                MemoryLevel.CONCEPT, MemoryLevel.SUBJECT, MemoryLevel.TOPIC
            ) else WMKind.ENTITY_REF
            wm.add(WMItem(kind=kind, content=f"{node.name}: {node.text}",
                          ref=node.id, activation=min(1.0, 0.6 + 0.4 * score),
                          source="attention"), cycle)
            admitted.append((node.name, round(score, 3)))
        return admitted
