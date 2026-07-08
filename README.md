# Goal-Driven Cognitive Architecture — reference implementation

A runnable Python implementation of the goal-driven (not token-driven) cognitive
loop from `Goal_Driven_Cognitive_Architecture_Revised.md`. It runs on a laptop
with an average NVIDIA GPU, and it also runs with **zero model** using offline
stubs so you can watch the control flow first.

The governing principle is kept literal: *the architecture supplies the
structure; the LLM is an optional tool provider.* The control flow is code. The
model is only reached inside specific memory-interaction operations.

> "God is not a God of confusion but of order." — 1 Corinthians 14:33
> (structure imposed on a structureless substrate; cf. Proverbs 16:9 — the mind
> plans the way, but the steps are established by a governing will)

---

## How the code maps to your document

| Document stage            | Code                                                        |
|---------------------------|-------------------------------------------------------------|
| Executive (PFC)           | `executive.py` — goal stack, strategy choice, evaluation    |
| Goal                      | `entities.py::Goal` (type, success criteria, urgency)       |
| Strategy                  | `entities.py::Strategy` + `executive.py::STRATEGY_LIBRARY`  |
| Attention (routing)       | `attention.py` — relevance/novelty/importance/urgency score |
| Long-term memory          | `long_term_memory.py` — Topic→Subject→Concept→Entity + recall |
| Entity (multimodal)       | `entities.py::MemoryNode.associations`                      |
| **Working memory (PFC substrate)** | `working_memory.py` — bounded, decaying, writable, serialisable |
| Memory Interaction        | `memory_interaction.py` — associate/compare/hypothesise/... |
| Thought (emergent)        | the interaction loop inside `agent.py::run`, not a module   |
| Evaluation                | `executive.py::evaluate` → CONTINUE/CHANGE_STRATEGY/LEARN/ACT |
| Hippocampal consolidation | `long_term_memory.py::consolidate`, called by `agent._consolidate` |

**Working memory is the part your revision is really about.** It is a bounded
workspace (`capacity`, default 9) holding the current goal, strategy, active
*references* into long-term memory, and transient derived items (hypotheses,
observations, decisions). Activation decays each cycle; attention and use
re-boost it; the lowest-activation items are evicted (goal and strategy are
protected). `save()`/`load()` make a file a *snapshot* of the live structure —
not the substrate itself. That is the bridge from your file-based context toward
genuine working memory.

---

## Run it now (no GPU, no model)

```bash
pip install numpy
python run_demo.py --backend stub
```

You'll see the full loop: attention routing with scores, the workspace after
each cycle, strategy switching, the decision, and the episode consolidated back
into long-term memory (which then resurfaces in the second goal's attention).

---

## Run it on your laptop GPU

Two pieces: an **embedder** (for memory recall) and an **LLM** (the reasoning
tool). Both are pluggable and both fall back to stubs if unavailable.

### 1. Embeddings (self-contained)
```bash
pip install -r requirements.txt   # pulls sentence-transformers + torch
```
`all-MiniLM-L6-v2` is ~80 MB and uses your GPU automatically if CUDA torch is
installed. To force CUDA torch on Windows/Linux:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. LLM via Ollama (easiest GPU path)
Install Ollama from https://ollama.com, then pull a model sized to your VRAM:

| Laptop GPU (VRAM)          | Recommended model                       | Command |
|----------------------------|-----------------------------------------|---------|
| RTX 3050/3060 (6 GB)       | `qwen2.5:7b-instruct-q4_K_M` (~4.7 GB)  | `ollama pull qwen2.5:7b-instruct-q4_K_M` |
| RTX 4060 / 4070 (8 GB)     | `qwen2.5:7b-instruct` or `llama3.1:8b`  | `ollama pull qwen2.5:7b-instruct` |
| 12 GB+                     | `qwen2.5:14b-instruct-q4_K_M`           | `ollama pull qwen2.5:14b-instruct-q4_K_M` |

Ollama offloads to the GPU automatically. Then:

```bash
ollama serve                      # if not already running as a service
python run_demo.py                # auto-detects Ollama
# or pin the model:
python run_demo.py --model qwen2.5:7b-instruct-q4_K_M
```

If you want embeddings through Ollama too: `ollama pull nomic-embed-text`
(the code will use it when sentence-transformers isn't installed).

---

## Use it in your own code

```python
from gdc import (CognitiveAgent, Goal, GoalType, LongTermMemory,
                 build_trading_memory, build_llm, build_embedder)

embedder = build_embedder("auto")     # sentence-transformers -> ollama -> stub
llm      = build_llm("auto")          # ollama -> stub

ltm = LongTermMemory(embedder)
build_trading_memory(ltm)             # swap in your own domain here

agent = CognitiveAgent(ltm, llm, embedder)
result = agent.run(Goal(
    description="Decide whether to open a long on TICK given momentum up, RSI 68",
    goal_type=GoalType.SOLVE,
    success_criteria=["A concrete, risk-bounded position decision"],
    urgency=0.7,
), max_cycles=6)

print(result.decision)
agent.wm.save("working_memory_snapshot.json")   # persist the PFC substrate
```

---

## Files

```
goal_driven_cognition/
  run_demo.py            entry point (auto-detects backends)
  requirements.txt
  gdc/
    entities.py          Goal, Strategy, MemoryNode, WMItem primitives
    llm.py               LLM backends: Ollama (GPU) + offline Stub
    embeddings.py        SentenceTransformer / Ollama / Stub embedders
    long_term_memory.py  distributed hierarchical store + recall + consolidation
    working_memory.py    bounded, decaying, writable, serialisable workspace
    attention.py         relevance/novelty/importance/urgency routing
    memory_interaction.py cognitive operations; LLM called as a tool
    executive.py         goals, strategy library, evaluation & control
    agent.py             the full cognitive loop
    seed_memory.py       example trading knowledge (Topic→Subject→Concept→Entity)
```

## Where to extend next (your open target)

- **Attention** currently scores relevance/novelty/importance/urgency with fixed
  weights. Making those weights themselves goal-conditioned is the natural next
  step toward genuine executive control of attention.
- **Working memory** eviction is activation-based. Adding *typed slots* (e.g. a
  reserved slot for the leading hypothesis and one for the pending action) would
  more closely mirror the prefrontal register you described.
- **Evaluation** uses a heuristic. Set `Executive(use_llm_judge=True)` hooks are
  in place to let the model judge success-criteria satisfaction directly.
```
