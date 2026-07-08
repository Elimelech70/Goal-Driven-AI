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
    def __init__(self, llm: LLMBackend | None = None, use_llm_judge: bool = False):
        self.llm = llm
        self.use_llm_judge = use_llm_judge
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
        return options[min(attempt, len(options) - 1)]

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, goal: Goal, wm: WorkingMemory, cycle: int,
                 max_cycles: int) -> Evaluation:
        has_decision = bool(wm.by_kind(WMKind.DECISION))
        n_hyp = len(wm.by_kind(WMKind.HYPOTHESIS))
        n_refs = len(wm.refs())

        # crude progress heuristic: knowledge in + a hypothesis formed + decided
        progress = min(1.0, 0.25 * min(n_refs, 2) / 1.0 * 0.5
                       + 0.3 * min(n_hyp, 1) + (0.5 if has_decision else 0.0))
        progress = min(1.0, 0.2 * min(n_refs, 3) + 0.3 * min(n_hyp, 1)
                       + (0.5 if has_decision else 0.0))

        if has_decision:
            return Evaluation(Control.ACT, "A concrete decision is in working memory.",
                              progress)
        if cycle >= max_cycles - 1:
            return Evaluation(Control.ACT if n_hyp else Control.DONE,
                              "Cycle budget exhausted; forcing resolution.", progress)
        # if we have knowledge and a hypothesis but no decision, switch strategy
        if n_hyp >= 1 and n_refs >= 2:
            return Evaluation(Control.CHANGE_STRATEGY,
                              "Hypothesis formed; switch to compare-and-decide.",
                              progress)
        return Evaluation(Control.CONTINUE, "Insufficient material; keep gathering.",
                          progress)
