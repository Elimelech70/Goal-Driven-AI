"""
Cognitive agent — current architecture loop.

External/Internal Input -> Sensory Transduction -> Entity Formation -> LTM
Goal/Intention -> Executive/PFC -> Attention -> Working Memory -> Memory
Interaction -> Branches/Hypotheses/Decision -> Evaluation -> Action/Learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .attention import Attention
from .embeddings import Embedder
from .entities import Goal, WMKind
from .executive import Control, Executive
from .llm import LLMBackend
from .long_term_memory import LongTermMemory
from .memory_interaction import MemoryInteraction
from .sensory import SensoryCortex
from .working_memory import WorkingMemory


@dataclass
class RunResult:
    goal: str
    decision: str | None
    cycles: int
    trace: list[str] = field(default_factory=list)


class CognitiveAgent:
    def __init__(self, ltm: LongTermMemory, llm: LLMBackend, embedder: Embedder,
                 wm_capacity: int = 12, verbose: bool = True):
        self.ltm = ltm
        self.llm = llm
        self.embedder = embedder
        self.executive = Executive(llm=llm)
        self.attention = Attention(embedder)
        self.interaction = MemoryInteraction(ltm, llm, embedder)
        self.sensory = SensoryCortex()
        self.wm = WorkingMemory(capacity=wm_capacity)
        self.verbose = verbose
        self.trace: list[str] = []

    def _log(self, line: str) -> None:
        self.trace.append(line)
        if self.verbose:
            print(line)

    def admit_input(self, text: str, cycle: int = 0) -> None:
        node = self.sensory.admit_language(text, self.ltm, self.wm, cycle)
        self._log(f"  sensory admitted -> {node.name}/{node.level.value}")

    def run(self, goal: Goal, max_cycles: int = 6, input_text: str | None = None) -> RunResult:
        self.executive.push_goal(goal)
        self._log(f"\n=== GOAL: {goal.description}  ({goal.goal_type.value}) ===")
        self._log(f"    success criteria: {goal.success_criteria}")

        if input_text:
            self.admit_input(input_text, cycle=0)

        strategy_attempt = 0
        strategy = self.executive.choose_strategy(goal, strategy_attempt)
        self.wm.set_goal(goal.description, 0)
        self.wm.set_intention(goal.description, 0)
        self.wm.set_strategy(strategy.name, 0)

        decision: str | None = None

        for cycle in range(max_cycles):
            self._log(f"\n--- cycle {cycle} | strategy: {strategy.name} "
                      f"({', '.join(strategy.operations)}) ---")

            admitted = self.attention.allocate(goal, strategy, self.ltm, self.wm, cycle, k=4)
            self._log("  attention admitted: "
                      + (", ".join(f"{n}({s})" for n, s in admitted) or "nothing new"))

            for op in strategy.operations:
                outputs = self.interaction.run_operation(op, self.wm, goal, cycle)
                for o in outputs:
                    self._log(f"  [{op}] {o}")

            self._log("  working memory:")
            self._log(self.wm.render())

            ev = self.executive.evaluate(goal, self.wm, cycle, max_cycles)
            self._log(f"  evaluation: {ev.control.value} "
                      f"(progress~{ev.progress:.2f}) — {ev.reason}")

            if ev.control == Control.ACT:
                decisions = self.wm.by_kind(WMKind.DECISION)
                pending = self.wm.get_slot("pending_action")
                if decisions:
                    decision = decisions[0].content
                elif pending:
                    decision = pending.content
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
                self.wm.set_strategy(strategy.name, cycle)

            if ev.control == Control.LEARN:
                self._consolidate(goal, None, cycle)

            self.wm.decay_all()

        self._log(f"\n=== RESULT: {decision or '(no decision)'} ===")
        return RunResult(goal=goal.description, decision=decision, cycles=cycle + 1,
                         trace=self.trace)

    def _consolidate(self, goal: Goal, decision: str | None, cycle: int) -> None:
        hyps = self.wm.by_kind(WMKind.HYPOTHESIS)
        leading = self.wm.get_slot("leading_hypothesis")
        text = decision or (leading.content if leading else (hyps[0].content if hyps else goal.description))
        parents = [i.ref for i in self.wm.refs() if i.ref]
        node = self.ltm.consolidate_episode(
            goal_text=goal.description,
            wm_text=self.wm.active_text(top=12),
            outcome=text,
            parents=parents)
        node.associations.update({"goal_type": goal.goal_type.value, "cycle": cycle})
        self._log(f"  consolidated episode -> LTM node {node.id}")
