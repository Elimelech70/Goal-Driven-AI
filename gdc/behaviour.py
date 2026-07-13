"""
Behaviour — "the manner of the walk."

Per the conceptual architecture: behaviour sits sideways to the cognitive
loop, not inside it — it colours attention and evaluation simultaneously
rather than being a stage of its own. Unlike Archetype, it is expected to be
retuned over time through fruit-tested experience (occasional, not rare).

Each trait is a continuous 0..1 knob with a documented, concrete effect on
the loop — no trait exists that doesn't visibly change behaviour:

  discipline  - raises the bar (evidence needed) before abandoning a
                strategy. High discipline = stubborn; low = quick to pivot.
  methodical  - requires a strategy to run for a minimum number of cycles
                before it may be abandoned or force-acted on early.
  creativity  - shifts Attention's novelty weight upward, biasing retrieval
                toward less-obvious material over pure relevance.
  sensitivity - cautious (1.0) reacts to weak signal (lower evidence
                thresholds trigger a strategy change); bold (0.0) requires
                stronger disconfirmation before rerouting.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Behaviour:
    name: str = "balanced"
    discipline: float = 0.5
    methodical: float = 0.5
    creativity: float = 0.5
    sensitivity: float = 0.5

    def min_cycles_before_pivot(self) -> int:
        """Methodical behaviour insists on walking the strategy's full
        operation sequence at least this many times before it can be
        abandoned early."""
        return 1 + round(self.methodical * 2)  # 1..3 cycles

    def refs_threshold(self) -> int:
        """How much retrieved material is needed before a hypothesis is
        allowed to trigger a strategy change. Cautious/undisciplined agents
        need less; bold/disciplined ones need more."""
        base = 2.0
        base -= self.sensitivity          # cautious lowers the bar
        base += self.discipline           # disciplined raises the bar
        return max(1, round(base))

    def novelty_bias(self) -> float:
        """Additive bump to Attention's novelty weight."""
        return (self.creativity - 0.5) * 0.3


CALM = Behaviour(name="calm", discipline=0.6, methodical=0.6,
                  creativity=0.3, sensitivity=0.4)
