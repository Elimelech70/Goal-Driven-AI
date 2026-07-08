"""
Memory Interaction Engine — 'thought' emerges here.

Thought is not a module. It is continuous memory interaction occurring within
working memory, under attention, in pursuit of the goal. Each operation reads
the workspace and writes new items (facts, hypotheses, plans, decisions) back
into it. Some operations are pure code (associate, compare, merge); some call
the LLM as a *tool* (hypothesise, generalise, predict, decide).
"""
from __future__ import annotations

from .embeddings import Embedder, cosine
from .entities import Goal, WMItem, WMKind
from .llm import LLMBackend
from .long_term_memory import LongTermMemory
from .working_memory import WorkingMemory


class MemoryInteraction:
    def __init__(self, ltm: LongTermMemory, llm: LLMBackend, embedder: Embedder):
        self.ltm = ltm
        self.llm = llm
        self.embedder = embedder

    # -- pure-code operations ----------------------------------------------
    def associate(self, wm: WorkingMemory, cycle: int) -> list[str]:
        """Pull neighbours of the most active reference into the workspace."""
        refs = sorted(wm.refs(), key=lambda x: x.activation, reverse=True)
        notes = []
        if not refs:
            return notes
        top = refs[0]
        for node, sim in self.ltm.neighbors(top.ref, k=2):
            wm.add(WMItem(kind=WMKind.CONCEPT_REF,
                          content=f"{node.name}: {node.text}", ref=node.id,
                          activation=0.55, source="associate"), cycle)
            notes.append(f"associate {top.content.split(':')[0]} -> {node.name} ({sim:.2f})")
        return notes

    def compare(self, wm: WorkingMemory, cycle: int) -> list[str]:
        refs = sorted(wm.refs(), key=lambda x: x.activation, reverse=True)[:2]
        if len(refs) < 2:
            return []
        a = self.ltm.nodes[refs[0].ref]
        b = self.ltm.nodes[refs[1].ref]
        sim = cosine(a.embedding, b.embedding)
        rel = "reinforce each other" if sim > 0.4 else "point in different directions"
        obs = f"{a.name} and {b.name} {rel} (sim={sim:.2f})"
        wm.add(WMItem(kind=WMKind.OBSERVATION, content=obs, activation=0.7,
                      source="compare"), cycle)
        return [obs]

    # -- LLM-assisted operations (LLM as tool) ------------------------------
    def hypothesise(self, wm: WorkingMemory, goal: Goal, cycle: int) -> list[str]:
        prompt = (f"Goal: {goal.description}\n\nActive working memory:\n"
                  f"{wm.active_text()}\n\n"
                  "Form ONE concrete hypothesis that moves toward the goal. "
                  "Answer in a single sentence beginning 'Hypothesis:'.")
        text = self.llm.generate(prompt, system=_SYS, max_tokens=120)
        wm.add(WMItem(kind=WMKind.HYPOTHESIS, content=text.strip(),
                      activation=0.85, source="hypothesise"), cycle)
        return [text.strip()]

    def generalise(self, wm: WorkingMemory, goal: Goal, cycle: int) -> list[str]:
        prompt = (f"From these active items:\n{wm.active_text()}\n\n"
                  "State the shared underlying concept in one sentence "
                  "beginning 'Generalisation:'.")
        text = self.llm.generate(prompt, system=_SYS, max_tokens=100)
        wm.add(WMItem(kind=WMKind.FACT, content=text.strip(), activation=0.7,
                      source="generalise"), cycle)
        return [text.strip()]

    def predict(self, wm: WorkingMemory, goal: Goal, cycle: int) -> list[str]:
        prompt = (f"Goal: {goal.description}\nWorking memory:\n{wm.active_text()}\n\n"
                  "Predict the most likely near-term outcome in one sentence "
                  "beginning 'Prediction:'.")
        text = self.llm.generate(prompt, system=_SYS, max_tokens=100)
        wm.add(WMItem(kind=WMKind.OBSERVATION, content=text.strip(),
                      activation=0.7, source="predict"), cycle)
        return [text.strip()]

    def decide(self, wm: WorkingMemory, goal: Goal, cycle: int) -> str:
        prompt = (f"Goal: {goal.description}\nWorking memory:\n{wm.active_text()}\n\n"
                  "Commit to a single concrete decision/action now. "
                  "Answer in one sentence beginning 'Decision:'.")
        text = self.llm.generate(prompt, system=_SYS, max_tokens=120).strip()
        wm.add(WMItem(kind=WMKind.DECISION, content=text, activation=1.0,
                      source="decide"), cycle)
        return text

    # -- dispatcher ---------------------------------------------------------
    def run_operation(self, op: str, wm: WorkingMemory, goal: Goal,
                      cycle: int) -> list[str]:
        table = {
            "associate": lambda: self.associate(wm, cycle),
            "compare": lambda: self.compare(wm, cycle),
            "hypothesise": lambda: self.hypothesise(wm, goal, cycle),
            "generalise": lambda: self.generalise(wm, goal, cycle),
            "predict": lambda: self.predict(wm, goal, cycle),
            "decide": lambda: [self.decide(wm, goal, cycle)],
        }
        fn = table.get(op)
        return fn() if fn else [f"(unknown op: {op})"]


_SYS = ("You are the reasoning tool inside a goal-driven cognitive architecture. "
        "You do not control the process; you answer the narrow operation asked, "
        "briefly and concretely.")
