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

from .attention import Attention
from .embeddings import Embedder
from .entities import Goal, MemoryLevel, WMKind
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
                 wm_capacity: int = 9, verbose: bool = True):
        self.ltm = ltm
        self.llm = llm
        self.embedder = embedder
        self.executive = Executive(llm=llm)
        self.attention = Attention(embedder)
        self.interaction = MemoryInteraction(ltm, llm, embedder)
        self.wm = WorkingMemory(capacity=wm_capacity)
        self.verbose = verbose
        self.trace: list[str] = []

    def _log(self, line: str) -> None:
        self.trace.append(line)
        if self.verbose:
            print(line)

    def run(self, goal: Goal, max_cycles: int = 6) -> RunResult:
        self.executive.push_goal(goal)
        self._log(f"\n=== GOAL: {goal.description}  ({goal.goal_type.value}) ===")
        self._log(f"    success criteria: {goal.success_criteria}")

        strategy_attempt = 0
        strategy = self.executive.choose_strategy(goal, strategy_attempt)
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
            ev = self.executive.evaluate(goal, self.wm, cycle, max_cycles)
            self._log(f"  evaluation: {ev.control.value} "
                      f"(progress~{ev.progress:.2f}) — {ev.reason}")

            if ev.control == Control.ACT:
                if not self.wm.by_kind(WMKind.DECISION):
                    decision = self.interaction.decide(self.wm, goal, cycle)
                    self._log(f"  [decide] {decision}")
                else:
                    decision = self.wm.by_kind(WMKind.DECISION)[0].content
                self._consolidate(goal, decision, cycle)
                break

            if ev.control == Control.DONE:
                break

            if ev.control == Control.CHANGE_STRATEGY:
                strategy_attempt += 1
                strategy = self.executive.choose_strategy(goal, strategy_attempt)
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
