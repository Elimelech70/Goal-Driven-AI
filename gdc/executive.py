"""
Executive (PFC-inspired) and Evaluation.

The Executive owns goals, chooses a strategy for the current goal, and after each
thought cycle evaluates progress and issues a control decision:

    CONTINUE | CHANGE_STRATEGY | LEARN | ACT | DONE

Evaluation is what makes the loop autonomous and self-correcting.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .archetype import Archetype
from .behaviour import Behaviour
from .entities import Goal, GoalType, Strategy, WMKind
from .llm import LLMBackend
from .working_memory import WorkingMemory


class Control(str, Enum):
    CONTINUE = "continue"
    CHANGE_STRATEGY = "change_strategy"
    LEARN = "learn"
    ACT = "act"
    DONE = "done"


@dataclass
class Evaluation:
    control: Control
    reason: str
    progress: float  # 0..1 rough progress estimate


# A small strategy library keyed by goal type. Ordered lists are the operations
# the strategy prefers each cycle; the executive can rotate strategies.
STRATEGY_LIBRARY: dict[GoalType, list[Strategy]] = {
    GoalType.SOLVE: [
        Strategy("recall-and-hypothesise",
                 "Retrieve relevant knowledge, associate, then hypothesise.",
                 operations=["associate", "hypothesise"],
                 retrieval_focus="relevant concepts, rules, prior cases"),
        Strategy("compare-and-decide",
                 "Weigh retrieved items against each other and commit.",
                 operations=["compare", "predict", "decide"],
                 retrieval_focus="decision criteria, risks, trade-offs"),
    ],
    GoalType.LEARN: [
        Strategy("explore-and-generalise",
                 "Gather examples and abstract the shared concept.",
                 operations=["associate", "generalise"],
                 retrieval_focus="examples, definitions, related concepts"),
    ],
    GoalType.PLAN: [
        Strategy("recall-and-sequence",
                 "Retrieve steps and order them.",
                 operations=["associate", "hypothesise", "decide"],
                 retrieval_focus="steps, dependencies, resources"),
    ],
}


class Executive:
    def __init__(self, llm: LLMBackend | None = None, use_llm_judge: bool = False,
                 archetype: Archetype | None = None, behaviour: Behaviour | None = None):
        self.llm = llm
        self.use_llm_judge = use_llm_judge
        self.archetype = archetype
        self.behaviour = behaviour or Behaviour()
        self.goal_stack: list[Goal] = []

    def push_goal(self, goal: Goal) -> None:
        self.goal_stack.append(goal)

    def current_goal(self) -> Goal | None:
        return self.goal_stack[-1] if self.goal_stack else None

    def choose_strategy(self, goal: Goal, attempt: int = 0) -> Strategy:
        options = STRATEGY_LIBRARY.get(goal.goal_type)
        if not options:
            return Strategy("default", "Retrieve, associate, hypothesise, decide.",
                            operations=["associate", "hypothesise", "decide"],
                            retrieval_focus=goal.description)
        options = self._archetype_ordering(goal, options)
        return options[min(attempt, len(options) - 1)]

    def _archetype_ordering(self, goal: Goal, options: list[Strategy]) -> list[Strategy]:
        """Archetype biases which strategy is reached for first, without
        removing any option — it narrows preference, not possibility."""
        preferred = (self.archetype.preferred_strategies.get(goal.goal_type)
                     if self.archetype else None)
        if not preferred:
            return options
        by_name = {s.name: s for s in options}
        ordered = [by_name[n] for n in preferred if n in by_name]
        ordered += [s for s in options if s.name not in preferred]
        return ordered

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, goal: Goal, wm: WorkingMemory, cycle: int,
                 max_cycles: int, cycles_in_strategy: int = 1) -> Evaluation:
        has_decision = bool(wm.by_kind(WMKind.DECISION))
        n_hyp = len(wm.by_kind(WMKind.HYPOTHESIS))
        n_facts = len(wm.by_kind(WMKind.FACT))
        n_refs = len(wm.refs())

        # crude progress heuristic: knowledge in + a hypothesis/fact formed + decided
        progress = min(1.0, 0.2 * min(n_refs, 3) + 0.3 * min(n_hyp + n_facts, 1)
                       + (0.5 if has_decision else 0.0))

        # methodical behaviour insists on walking the current strategy for a
        # minimum number of cycles before it may be pivoted away from early
        ready_to_pivot = cycles_in_strategy >= self.behaviour.min_cycles_before_pivot()
        refs_needed = self.behaviour.refs_threshold()

        if has_decision:
            return Evaluation(Control.ACT, "A concrete decision is in working memory.",
                              progress)
        # LEARN goals conclude with a generalisation, not a decision — recognise
        # that as a resolution once enough knowledge has been drawn in.
        if goal.goal_type == GoalType.LEARN and n_facts >= 1 and n_refs >= refs_needed \
                and ready_to_pivot:
            return Evaluation(Control.ACT, "A generalisation has been formed.",
                              progress)
        if cycle >= max_cycles - 1:
            return Evaluation(Control.ACT if (n_hyp or n_facts) else Control.DONE,
                              "Cycle budget exhausted; forcing resolution.", progress)
        # if we have knowledge and a hypothesis but no decision, switch strategy
        if n_hyp >= 1 and n_refs >= refs_needed and ready_to_pivot:
            return Evaluation(Control.CHANGE_STRATEGY,
                              "Hypothesis formed; switch to compare-and-decide.",
                              progress)
        return Evaluation(Control.CONTINUE, "Insufficient material; keep gathering.",
                          progress)
