"""
Embedding backends for memory retrieval.

Three options, same contract:
  * SentenceTransformerEmbedder - self-contained, runs on your GPU or CPU.
  * OllamaEmbedder             - reuses the Ollama server (nomic-embed-text).
  * StubEmbedder               - deterministic hashed bag-of-words, no downloads.

The StubEmbedder lets the whole cognitive loop run offline. It is crude but
gives stable, meaningful-enough cosine similarity for the demo.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from typing import Optional

import numpy as np


class Embedder:
    dim = 0

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def available(self) -> bool:
        return True


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SentenceTransformerEmbedder(Embedder):
    """all-MiniLM-L6-v2 is ~80 MB and fast on any laptop GPU."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text, normalize_embeddings=True),
                          dtype=np.float32)


class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "nomic-embed-text",
                 host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.dim = 768

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                if r.status != 200:
                    return False
                tags = json.loads(r.read().decode("utf-8")).get("models", [])
                return any(m.get("name", "").split(":")[0] == self.model.split(":")[0]
                           for m in tags)
        except Exception:
            return False

    def encode(self, text: str) -> np.ndarray:
        payload = json.dumps({"model": self.model, "prompt": text}).encode()
        req = urllib.request.Request(f"{self.host}/api/embeddings", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            v = json.loads(r.read())["embedding"]
        return np.asarray(v, dtype=np.float32)


class StubEmbedder(Embedder):
    """
    Deterministic hashed bag-of-words. Each token is hashed into a fixed-width
    vector; token vectors are averaged. Similar wording -> similar vectors.
    No network, no model, reproducible.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tok(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _hash_vec(self, token: str) -> np.ndarray:
        h = hashlib.sha1(token.encode()).digest()
        seed = int.from_bytes(h[:4], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        toks = self._tok(text)
        if not toks:
            return np.zeros(self.dim, dtype=np.float32)
        v = np.mean([self._hash_vec(t) for t in toks], axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v
