"""
Working memory — the prefrontal-cortex-equivalent substrate.

This is the piece the architecture is really about: a *writable, bounded,
persistent* workspace that holds the current goal, strategy, active entity
references, and candidate actions/hypotheses during reasoning cycles.

Key properties:
  * Bounded capacity (a la 7 +/- 2). Attention decides what stays.
  * Activation decays each cycle; attention and use re-boost it.
  * It holds *references* into long-term memory plus transient derived items.
  * It is serialisable, so a file becomes a *snapshot* of a live structure
    rather than the substrate itself.
"""
from __future__ import annotations

import json
from typing import Optional

from .entities import WMItem, WMKind


# kinds that must never be evicted while a goal is active
_PROTECTED = {WMKind.GOAL, WMKind.STRATEGY}


class WorkingMemory:
    def __init__(self, capacity: int = 9, decay: float = 0.15,
                 activation_floor: float = 0.05):
        self.capacity = capacity
        self.decay = decay
        self.activation_floor = activation_floor
        self.items: dict[str, WMItem] = {}

    # -- mutation -----------------------------------------------------------
    def add(self, item: WMItem, cycle: int) -> WMItem:
        item.created_cycle = cycle
        item.last_touched_cycle = cycle
        # dedupe references so the same node isn't loaded twice
        if item.ref is not None:
            for existing in self.items.values():
                if existing.ref == item.ref and existing.kind == item.kind:
                    self.touch(existing.id, cycle, boost=0.4)
                    return existing
        self.items[item.id] = item
        self._enforce_capacity()
        return item

    def touch(self, item_id: str, cycle: int, boost: float = 0.3) -> None:
        it = self.items.get(item_id)
        if it:
            it.activation = min(1.0, it.activation + boost)
            it.last_touched_cycle = cycle

    def set_goal(self, text: str, cycle: int) -> None:
        self._replace_singleton(WMKind.GOAL, text, cycle)

    def set_strategy(self, text: str, cycle: int) -> None:
        self._replace_singleton(WMKind.STRATEGY, text, cycle)

    def _replace_singleton(self, kind: WMKind, text: str, cycle: int) -> None:
        for i in [k for k, v in self.items.items() if v.kind == kind]:
            del self.items[i]
        self.add(WMItem(kind=kind, content=text, activation=1.0, source="executive"),
                 cycle)

    # -- dynamics -----------------------------------------------------------
    def decay_all(self) -> None:
        for it in self.items.values():
            if it.kind in _PROTECTED:
                continue
            it.activation = max(0.0, it.activation - self.decay)
        # drop anything that has faded below the floor
        for i in [k for k, v in self.items.items()
                  if v.kind not in _PROTECTED and v.activation < self.activation_floor]:
            del self.items[i]

    def _enforce_capacity(self) -> None:
        if len(self.items) <= self.capacity:
            return
        evictable = [(v.activation, k) for k, v in self.items.items()
                     if v.kind not in _PROTECTED]
        evictable.sort()  # lowest activation first
        while len(self.items) > self.capacity and evictable:
            _, victim = evictable.pop(0)
            self.items.pop(victim, None)

    # -- inspection ---------------------------------------------------------
    def by_kind(self, kind: WMKind) -> list[WMItem]:
        return [v for v in self.items.values() if v.kind == kind]

    def refs(self) -> list[WMItem]:
        return [v for v in self.items.values()
                if v.kind in (WMKind.ENTITY_REF, WMKind.CONCEPT_REF)]

    def active_text(self, top: int = 8) -> str:
        ordered = sorted(self.items.values(), key=lambda x: x.activation, reverse=True)
        return "\n".join(f"- [{i.kind.value} a={i.activation:.2f}] {i.content}"
                         for i in ordered[:top])

    def render(self) -> str:
        ordered = sorted(self.items.values(), key=lambda x: x.activation, reverse=True)
        lines = [f"  {i.kind.value:11s} a={i.activation:0.2f}  {i.content}"
                 for i in ordered]
        return "\n".join(lines) if lines else "  (empty)"

    # -- persistence (snapshot, not substrate) -----------------------------
    def snapshot(self) -> list[dict]:
        return [i.to_dict() for i in self.items.values()]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.snapshot(), f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.items = {}
        for d in data:
            d = dict(d)
            d["kind"] = WMKind(d["kind"])
            item = WMItem(**{k: v for k, v in d.items() if k in WMItem.__dataclass_fields__})
            self.items[item.id] = item
