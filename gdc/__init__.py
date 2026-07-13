"""
gdc — Goal-Driven Cognitive Architecture, reference implementation.

Public API surface used by run_demo.py and downstream code, plus the
`build_llm` / `build_embedder` auto-detecting factories described in the
README (sentence-transformers -> ollama -> stub for embeddings; ollama ->
stub for the LLM).
"""
from __future__ import annotations

from typing import Optional

from .agent import CognitiveAgent, RunResult
from .archetype import Archetype, CHAVER
from .attention import Attention, AttentionWeights
from .behaviour import CALM, Behaviour
from .config import AgentConfig, BackendConfig, Config, load_config
from .embeddings import (Embedder, OllamaEmbedder, SentenceTransformerEmbedder,
                          StubEmbedder, cosine)
from .entities import (Goal, GoalStatus, GoalType, MemoryLevel, MemoryNode,
                        Strategy, WMItem, WMKind)
from .executive import Control, Evaluation, Executive
from .llm import LLMBackend, OllamaBackend, StubBackend
from .long_term_memory import LongTermMemory
from .memory_interaction import MemoryInteraction
from .seed_memory import build_trading_memory
from .working_memory import WorkingMemory

__all__ = [
    "CognitiveAgent", "RunResult",
    "Archetype", "CHAVER",
    "Attention", "AttentionWeights",
    "CALM", "Behaviour",
    "AgentConfig", "BackendConfig", "Config", "load_config",
    "Embedder", "OllamaEmbedder", "SentenceTransformerEmbedder", "StubEmbedder", "cosine",
    "Goal", "GoalStatus", "GoalType", "MemoryLevel", "MemoryNode", "Strategy", "WMItem", "WMKind",
    "Control", "Evaluation", "Executive",
    "LLMBackend", "OllamaBackend", "StubBackend",
    "LongTermMemory",
    "MemoryInteraction",
    "build_trading_memory",
    "WorkingMemory",
    "build_llm", "build_embedder",
]


def build_embedder(backend: str = "auto", model: Optional[str] = None,
                    host: Optional[str] = None) -> Embedder:
    """sentence-transformers -> ollama -> stub."""
    if backend in ("auto", "sentence-transformers", "st"):
        try:
            emb = SentenceTransformerEmbedder(**({"model_name": model} if model else {}))
            return emb
        except Exception:
            if backend != "auto":
                raise
    if backend in ("auto", "ollama"):
        kwargs = {}
        if model:
            kwargs["model"] = model
        if host:
            kwargs["host"] = host
        emb = OllamaEmbedder(**kwargs)
        if emb.available():
            return emb
        if backend != "auto":
            raise RuntimeError("Ollama embedder requested but server unavailable")
    return StubEmbedder()


def build_llm(backend: str = "auto", model: Optional[str] = None,
              host: Optional[str] = None) -> LLMBackend:
    """ollama -> stub."""
    if backend in ("auto", "ollama"):
        kwargs = {}
        if model:
            kwargs["model"] = model
        if host:
            kwargs["host"] = host
        llm = OllamaBackend(**kwargs)
        if llm.available():
            return llm
        if backend != "auto":
            raise RuntimeError("Ollama LLM requested but server unavailable")
    return StubBackend()
