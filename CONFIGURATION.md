# Configuration Reference

Complete record of how this `ai-brain-like` install is configured: the
project's own `config.json`, the Python environment, and the Ollama
service/models it talks to. See `USAGE.md` for how to run things,
`README.md` for the architecture.

---

## 1. Project configuration — `config.json`

Read by `gdc/config.py::load_config()` at startup in `run_demo.py`. CLI
flags (`--backend`, `--model`, `--max-cycles`) override these values;
`--config <path>` loads a different file entirely. Any field missing from
the file falls back to the default shown.

```json
{
  "llm": {
    "backend": "auto",
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "host": "http://localhost:11434"
  },
  "embedder": {
    "backend": "auto",
    "model": "nomic-embed-text",
    "host": "http://localhost:11434"
  },
  "agent": {
    "wm_capacity": 9,
    "max_cycles": 6
  }
}
```

### Field reference

| Key | Default | Meaning |
|---|---|---|
| `llm.backend` | `auto` | `auto` \| `ollama` \| `stub`. `auto` uses Ollama if the model tag below is actually pulled and the server responds, else falls back to `StubBackend`. |
| `llm.model` | `qwen2.5:7b-instruct-q4_K_M` | Ollama model tag used for generation (hypothesize/predict/decide/generalise operations). |
| `llm.host` | `http://localhost:11434` | Ollama server URL for generation. |
| `embedder.backend` | `auto` | `auto` \| `ollama` \| `sentence-transformers` \| `stub`. `auto` tries sentence-transformers first, then Ollama, then `StubEmbedder`. |
| `embedder.model` | `nomic-embed-text` | Ollama embedding model tag (only used if the sentence-transformers path isn't available/installed). |
| `embedder.host` | `http://localhost:11434` | Ollama server URL for embeddings. |
| `agent.wm_capacity` | `9` | Max items held in working memory (the bounded "PFC register") before lowest-activation items are evicted. Goal and strategy slots are protected from eviction. |
| `agent.max_cycles` | `6` | Cycle budget per goal before the executive forces a resolution. |

### Backend resolution order (implemented in `gdc/__init__.py`)

- **Embedder**: `sentence-transformers` (if installed) → `ollama` (if server reachable **and** `embedder.model` is present in `ollama list`) → `StubEmbedder` (deterministic hashed bag-of-words, no network).
- **LLM**: `ollama` (if server reachable **and** `llm.model` is present in `ollama list`) → `StubBackend` (canned deterministic responses keyed on prompt phrasing).

Explicitly requesting `ollama` (rather than `auto`) raises `RuntimeError` instead of silently falling back if the server/model isn't available.

---

## 2. Python environment

| Item | Value |
|---|---|
| Interpreter | Python 3.12.3 |
| Location | `/home/craig/ai-brain-like/.venv` |
| Installed packages | `numpy 2.5.0` only |

`sentence-transformers` / `torch` from `requirements.txt` are **not**
installed in this venv — embeddings run through Ollama's `nomic-embed-text`
instead. Install `requirements.txt` in full only if you want local
GPU-accelerated embeddings without going through Ollama:
```bash
.venv/bin/pip install -r requirements.txt
```

---

## 3. Ollama service

| Item | Value |
|---|---|
| Version | 0.15.2 |
| Service manager | systemd (`/etc/systemd/system/ollama.service`) |
| Status | `active`, `enabled` (starts automatically on boot) |
| Runs as | system user `ollama` |
| API endpoint | `http://localhost:11434` |
| Model storage | `/usr/share/ollama/.ollama/models` (default — no `OLLAMA_MODELS` override set) |

### Installed models

| Model | Size | Role in this project |
|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | 4.7 GB | LLM — reasoning tool called from memory-interaction operations |
| `nomic-embed-text:latest` | 274 MB | Embedder — memory recall / attention similarity scoring |

Total on-disk footprint: ~4.7 GB combined (`du -sh /usr/share/ollama/.ollama/models`).

### GPU

| Item | Value |
|---|---|
| Device | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6141 MiB (~6 GB) |

`qwen2.5:7b-instruct-q4_K_M` was chosen specifically to fit this VRAM
budget per the README's sizing table (6 GB → Q4_K_M quantized 7B). Ollama
offloads to the GPU automatically; no manual CUDA configuration was needed.

### Managing the service

```bash
systemctl status ollama          # check state
sudo systemctl restart ollama    # restart
ollama list                      # installed models
ollama pull <tag>                # add a model
ollama rm <tag>                  # remove a model
```

---

## 4. How it fits together

```
run_demo.py
  -> gdc.config.load_config()        reads config.json above
  -> gdc.build_embedder(...)         resolves to OllamaEmbedder(nomic-embed-text)
  -> gdc.build_llm(...)              resolves to OllamaBackend(qwen2.5:7b-instruct-q4_K_M)
  -> both talk HTTP to               http://localhost:11434  (systemd-managed ollama serve)
```

Changing which model/host is used only requires editing `config.json` (or
passing `--model` / `--config`) — no code changes needed, since
`gdc/__init__.py`'s `build_llm`/`build_embedder` factories take model/host
as parameters.
