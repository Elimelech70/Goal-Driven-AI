# Usage

This is the Goal-Driven Cognitive Architecture reference implementation,
set up in this directory with a Python venv and local Ollama models. See
`README.md` for the architecture explanation; this file covers running it.

## Quick start

```bash
cd /home/craig/ai-brain-like
.venv/bin/python run_demo.py
```

That runs two goals (a SOLVE and a LEARN) through the full cognitive loop —
attention routing, working memory, strategy execution, evaluation, and
consolidation back into long-term memory — and prints the trace.

## Backends

Two independent pieces are pluggable: the **embedder** (memory recall) and
the **LLM** (reasoning tool, called from inside memory operations, not
driving the loop itself). Both fall back to offline stubs automatically.

Currently installed:

| Component | Backend | Model |
|---|---|---|
| Embedder | Ollama | `nomic-embed-text` (274 MB) |
| LLM | Ollama | `qwen2.5:7b-instruct-q4_K_M` (4.7 GB, fits 6 GB VRAM) |

Both were pulled with `ollama pull <name>` and are served by the local
`ollama serve` daemon at `http://localhost:11434`.

### Force the offline stub (no GPU, no model, watch control flow only)

```bash
.venv/bin/python run_demo.py --backend stub
```

This forces the LLM to the deterministic `StubBackend`. The embedder still
auto-detects — to force it offline too, edit `config.json`'s
`embedder.backend` to `"stub"` (see below).

### Pin a specific model

```bash
.venv/bin/python run_demo.py --model qwen2.5:14b-instruct-q4_K_M
```

## Configuration

`config.json` (project root) holds the defaults `run_demo.py` reads on
startup. CLI flags (`--backend`, `--model`, `--max-cycles`) override it;
`--config <path>` points at a different file entirely.

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

- `backend`: `auto` | `ollama` | `stub` for the LLM; `auto` | `ollama` |
  `sentence-transformers` | `stub` for the embedder. `auto` degrades
  gracefully — for `ollama` it checks the model is actually pulled (not
  just that the server is up) before using it, otherwise it falls back to
  the stub.
- `host`: point at a remote Ollama server if you're not running it locally.
- `agent.wm_capacity`: size of the working-memory workspace (default 9,
  matching the "PFC register" the architecture describes).
- `agent.max_cycles`: cycle budget per goal before the executive forces a
  resolution.

To run against a different Ollama host or a differently-sized model without
touching the checked-in `config.json`, copy it and pass `--config`:

```bash
cp config.json config.local.json
# edit config.local.json
.venv/bin/python run_demo.py --config config.local.json
```

## Using it as a library

```python
from gdc import (CognitiveAgent, Goal, GoalType, LongTermMemory,
                  build_trading_memory, build_llm, build_embedder, load_config)

cfg = load_config()
embedder = build_embedder(cfg.embedder.backend, cfg.embedder.model, cfg.embedder.host)
llm = build_llm(cfg.llm.backend, cfg.llm.model, cfg.llm.host)

ltm = LongTermMemory(embedder)
build_trading_memory(ltm)          # swap in your own domain here

agent = CognitiveAgent(ltm, llm, embedder, wm_capacity=cfg.agent.wm_capacity)
result = agent.run(Goal(
    description="Decide whether to open a long on TICK given momentum up, RSI 68",
    goal_type=GoalType.SOLVE,
    success_criteria=["A concrete, risk-bounded position decision"],
    urgency=0.7,
), max_cycles=cfg.agent.max_cycles)

print(result.decision)
agent.wm.save("working_memory_snapshot.json")   # persist the PFC substrate
```

## Adding / changing Ollama models

```bash
ollama pull <model-tag>     # e.g. qwen2.5:14b-instruct-q4_K_M for 12GB+ VRAM
ollama list                 # confirm it's present
```

Then update `config.json`'s `llm.model` (or `embedder.model`) to the new
tag, or pass `--model` on the command line for the LLM.

## Troubleshooting

- **`[backends]` line shows `StubEmbedder`/`stub` when you expected
  Ollama** — check `ollama list` shows the model, and `ollama serve` is
  running (`ps aux | grep ollama`). `auto` mode silently falls back to
  stubs rather than erroring, by design.
- **Explicit `--backend ollama` raises `RuntimeError`** — this means
  Ollama's HTTP API is reachable but the specific model tag isn't pulled;
  run `ollama pull <tag>`.
- **Slow generation** — the qwen2.5 7B model at Q4 quantization is sized
  for 6GB+ VRAM; check `nvidia-smi` while running to confirm it's on GPU,
  not falling back to CPU.

## Files

```
ai-brain-like/
  README.md              architecture explanation
  USAGE.md               this file
  config.json            backend/model defaults for run_demo.py
  run_demo.py            entry point
  requirements.txt
  .venv/                 Python virtualenv (numpy only; add
                          sentence-transformers if you want local embeddings
                          instead of Ollama's)
  gdc/
    __init__.py           public API + build_llm/build_embedder factories
    config.py             config.json loader
    entities.py           Goal, Strategy, MemoryNode, WMItem primitives
    llm.py                LLM backends: Ollama (GPU) + offline Stub
    embeddings.py         SentenceTransformer / Ollama / Stub embedders
    long_term_memory.py   distributed hierarchical store + recall + consolidation
    working_memory.py     bounded, decaying, writable, serialisable workspace
    attention.py           relevance/novelty/importance/urgency routing
    memory_interaction.py cognitive operations; LLM called as a tool
    executive.py           goals, strategy library, evaluation & control
    agent.py                the full cognitive loop
    seed_memory.py         example trading knowledge (Topic->Subject->Concept->Entity)
```
