"""
The cognitive agent — the loop that ties every component together.

    Executive -> Goal -> Strategy -> Attention -> Working Memory
       -> Memory Interaction -> Thought -> Evaluation
       -> Continue | Change Strategy | Learn | Act

Nothing here is token-driven. The LLM is only reached inside specific memory
operations. The control flow is code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .archetype import Archetype
from .attention import Attention
from .behaviour import Behaviour
from .embeddings import Embedder
from .entities import Goal, GoalType, MemoryLevel, WMKind
from .executive import Control, Executive
from .llm import LLMBackend
from .long_term_memory import LongTermMemory
from .memory_interaction import MemoryInteraction
from .working_memory import WorkingMemory


@dataclass
class RunResult:
    goal: str
    decision: str | None
    cycles: int
    trace: list[str] = field(default_factory=list)


class CognitiveAgent:
    def __init__(self, ltm: LongTermMemory, llm: LLMBackend, embedder: Embedder,
                 wm_capacity: int = 9, verbose: bool = True, name: str = "agent",
                 archetype: Archetype | None = None, behaviour: Behaviour | None = None):
        self.name = name
        self.archetype = archetype
        self.behaviour = behaviour or Behaviour()
        self.ltm = ltm
        self.llm = llm
        self.embedder = embedder
        self.executive = Executive(llm=llm, archetype=archetype, behaviour=self.behaviour)
        self.attention = Attention(embedder, behaviour=self.behaviour)
        self.interaction = MemoryInteraction(ltm, llm, embedder)
        self.wm = WorkingMemory(capacity=wm_capacity)
        self.verbose = verbose
        self.trace: list[str] = []

    def _log(self, line: str) -> None:
        self.trace.append(line)
        if self.verbose:
            print(line)

    def run(self, goal: Goal, max_cycles: int = 6) -> RunResult:
        if max_cycles < 1:
            raise ValueError(f"max_cycles must be >= 1, got {max_cycles}")
        self.executive.push_goal(goal)
        archetype_note = f" [{self.archetype.name}]" if self.archetype else ""
        self._log(f"\n=== {self.name}{archetype_note} — GOAL: {goal.description}  "
                  f"({goal.goal_type.value}) ===")
        self._log(f"    success criteria: {goal.success_criteria}")

        strategy_attempt = 0
        strategy = self.executive.choose_strategy(goal, strategy_attempt)
        strategy_started_cycle = 0
        self.wm.set_goal(goal.description, 0)
        self.wm.set_strategy(strategy.name, 0)

        decision: str | None = None

        for cycle in range(max_cycles):
            self._log(f"\n--- cycle {cycle} | strategy: {strategy.name} "
                      f"({', '.join(strategy.operations)}) ---")

            # 1. ATTENTION: route knowledge into the workspace
            admitted = self.attention.allocate(goal, strategy, self.ltm, self.wm,
                                                cycle, k=4)
            self._log("  attention admitted: "
                      + (", ".join(f"{n}({s})" for n, s in admitted) or "nothing new"))

            # 2. MEMORY INTERACTION == THOUGHT: run the strategy's operations
            for op in strategy.operations:
                outputs = self.interaction.run_operation(op, self.wm, goal, cycle)
                for o in outputs:
                    self._log(f"  [{op}] {o}")

            # 3. show the live workspace
            self._log("  working memory:")
            self._log(self.wm.render())

            # 4. EVALUATION: executive decides what happens next
            cycles_in_strategy = cycle - strategy_started_cycle + 1
            ev = self.executive.evaluate(goal, self.wm, cycle, max_cycles,
                                          cycles_in_strategy)
            self._log(f"  evaluation: {ev.control.value} "
                      f"(progress~{ev.progress:.2f}) — {ev.reason}")

            if ev.control == Control.ACT:
                existing_decision = self.wm.by_kind(WMKind.DECISION)
                existing_facts = self.wm.by_kind(WMKind.FACT)
                if existing_decision:
                    decision = existing_decision[0].content
                elif goal.goal_type == GoalType.LEARN and existing_facts:
                    # a LEARN goal resolves into a generalisation, not an action
                    decision = existing_facts[0].content
                else:
                    decision = self.interaction.decide(self.wm, goal, cycle)
                    self._log(f"  [decide] {decision}")
                self._consolidate(goal, decision, cycle)
                break

            if ev.control == Control.DONE:
                break

            if ev.control == Control.CHANGE_STRATEGY:
                strategy_attempt += 1
                strategy = self.executive.choose_strategy(goal, strategy_attempt)
                strategy_started_cycle = cycle + 1
                self.wm.set_strategy(strategy.name, cycle)

            if ev.control == Control.LEARN:
                self._consolidate(goal, None, cycle)

            # 5. CONTINUE: activation decays before the next cycle
            self.wm.decay_all()

        self._log(f"\n=== RESULT: {decision or '(no decision)'} ===")
        return RunResult(goal=goal.description, decision=decision,
                         cycles=cycle + 1, trace=self.trace)

    def _consolidate(self, goal: Goal, decision: str | None, cycle: int) -> None:
        """Hippocampal write-back: bind the episode into long-term memory."""
        hyps = self.wm.by_kind(WMKind.HYPOTHESIS)
        text = decision or (hyps[0].content if hyps else goal.description)
        node = self.ltm.consolidate(
            name=f"episode: {goal.description[:40]}",
            text=text, level=MemoryLevel.ENTITY, salience=0.7,
            associations={"goal_type": goal.goal_type.value, "cycle": cycle})
        self._log(f"  consolidated episode -> LTM node {node.id}")
