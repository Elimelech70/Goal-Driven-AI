"""
Long-term memory: distributed cortical store (Topic -> Subject -> Concept -> Entity).

The store keeps nodes and their embeddings. The `recall` method is the cortical
retrieval; `consolidate` plays the hippocampal role of binding transient working
-memory contents into new, indexed, stable nodes.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .embeddings import Embedder, cosine
from .entities import MemoryNode, MemoryLevel


class LongTermMemory:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.nodes: dict[str, MemoryNode] = {}

    # -- construction -------------------------------------------------------
    def add(self, node: MemoryNode) -> MemoryNode:
        if node.embedding is None:
            node.embedding = self.embedder.encode(node.embed_text())
        self.nodes[node.id] = node
        return node

    def link(self, parent_id: str, child_id: str) -> None:
        """Wire a level to the level beneath it."""
        parent, child = self.nodes[parent_id], self.nodes[child_id]
        if child_id not in parent.children:
            parent.children.append(child_id)
        if parent_id not in child.parents:
            child.parents.append(parent_id)

    # -- retrieval (cortical recall) ---------------------------------------
    def recall(self, query_emb: np.ndarray, k: int = 6,
               level: Optional[MemoryLevel] = None) -> list[tuple[MemoryNode, float]]:
        scored: list[tuple[MemoryNode, float]] = []
        for node in self.nodes.values():
            if level is not None and node.level != level:
                continue
            if node.embedding is None:
                continue
            scored.append((node, cosine(query_emb, node.embedding)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def neighbors(self, node_id: str, k: int = 4) -> list[tuple[MemoryNode, float]]:
        """Associate: nearest nodes to a given node (excluding itself)."""
        node = self.nodes[node_id]
        if node.embedding is None:
            return []
        out = [(n, cosine(node.embedding, n.embedding))
               for n in self.nodes.values()
               if n.id != node_id and n.embedding is not None]
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:k]

    def children_of(self, node_id: str) -> list[MemoryNode]:
        return [self.nodes[c] for c in self.nodes[node_id].children if c in self.nodes]

    # -- hippocampal consolidation -----------------------------------------
    def consolidate(self, name: str, text: str, level: MemoryLevel,
                    parents: Optional[list[str]] = None,
                    associations: Optional[dict] = None,
                    salience: float = 0.6) -> MemoryNode:
        """Bind a transient insight into a new, indexed, episodic node."""
        node = MemoryNode(name=name, level=level, text=text,
                          associations=associations or {},
                          salience=salience, episodic=True)
        self.add(node)
        for p in (parents or []):
            if p in self.nodes:
                self.link(p, node.id)
        return node

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.level.value] = counts.get(n.level.value, 0) + 1
        return ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
