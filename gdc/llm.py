"""
LLM backends.

Per the architecture, the LLM is an *optional knowledge/tool provider*, not the
central reasoning component. The cognitive loop runs with a StubBackend so the
whole machine is inspectable with no GPU. Point it at Ollama to get real
generation on your laptop GPU.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Optional


class LLMBackend:
    """Minimal text-in / text-out contract. Everything above this line is code."""
    name = "base"

    def generate(self, prompt: str, system: str = "", max_tokens: int = 256,
                 temperature: float = 0.4) -> str:
        raise NotImplementedError

    def available(self) -> bool:
        return True


class OllamaBackend(LLMBackend):
    """
    Talks to a local Ollama server (http://localhost:11434).

    Recommended models for an average laptop NVIDIA GPU:
        6 GB VRAM (e.g. RTX 3060 laptop): qwen2.5:7b-instruct-q4_K_M
        8 GB VRAM (e.g. RTX 4060/4070):   qwen2.5:7b-instruct  or llama3.1:8b
    Ollama offloads to the GPU automatically.
    """
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b-instruct-q4_K_M",
                 host: str = "http://localhost:11434", timeout: int = 120):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status != 200:
                    return False
                tags = json.loads(r.read().decode("utf-8")).get("models", [])
                return any(m.get("name", "").split(":")[0] == self.model.split(":")[0]
                           for m in tags)
        except Exception:
            return False

    def generate(self, prompt: str, system: str = "", max_tokens: int = 256,
                 temperature: float = 0.4) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read().decode("utf-8"))
        return body.get("response", "").strip()


class StubBackend(LLMBackend):
    """
    Deterministic, dependency-free stand-in. Produces plausible, inspectable
    text so the architecture can be watched end-to-end without a model. It is
    intentionally shallow — it proves the *control flow*, not the intelligence.
    """
    name = "stub"

    def generate(self, prompt: str, system: str = "", max_tokens: int = 256,
                 temperature: float = 0.4) -> str:
        p = prompt.lower()
        # Key off the distinctive instruction phrase (near the end of each prompt),
        # in priority order, so the operation — not the workspace dump — decides.
        if "concrete decision" in p or "beginning 'decision:'" in p:
            return ("Decision: enter a half-size long; stop just below the nearest "
                    "support; reassess if RSI crosses 75.")
        if "beginning 'generalisation:'" in p:
            return ("Generalisation: momentum trades are safe only when paired with a "
                    "defined stop and reduced size — risk control is the governing factor.")
        if "beginning 'prediction:'" in p:
            return ("Prediction: continuation is likely short-term, but reversal risk "
                    "rises as RSI approaches overbought.")
        if "beginning 'hypothesis:'" in p:
            return ("Hypothesis: momentum plus RSI 68 favours a cautious long, "
                    "contingent on a defined stop below support.")
        if "satisfy" in p or "criteria" in p:
            return "PARTIAL: criteria partly met; a concrete sized decision is still missing."
        return "Observation noted; continue reasoning."
