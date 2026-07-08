"""
Loads config.json (project root) into a Config object. Every field has a
default, so a missing file or missing key is not an error — CLI flags in
run_demo.py still take precedence over whatever this returns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


@dataclass
class BackendConfig:
    backend: str = "auto"          # auto | ollama | stub (llm) / auto | ollama | sentence-transformers | stub (embedder)
    model: Optional[str] = None
    host: str = "http://localhost:11434"


@dataclass
class AgentConfig:
    wm_capacity: int = 9
    max_cycles: int = 6


@dataclass
class Config:
    llm: BackendConfig = field(default_factory=lambda: BackendConfig(
        model="qwen2.5:7b-instruct-q4_K_M"))
    embedder: BackendConfig = field(default_factory=lambda: BackendConfig(
        model="nomic-embed-text"))
    agent: AgentConfig = field(default_factory=AgentConfig)


def load_config(path: Optional[str] = None) -> Config:
    cfg = Config()
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        return cfg
    with open(p) as f:
        raw = json.load(f)
    if "llm" in raw:
        cfg.llm = BackendConfig(**{**cfg.llm.__dict__, **raw["llm"]})
    if "embedder" in raw:
        cfg.embedder = BackendConfig(**{**cfg.embedder.__dict__, **raw["embedder"]})
    if "agent" in raw:
        cfg.agent = AgentConfig(**{**cfg.agent.__dict__, **raw["agent"]})
    return cfg
