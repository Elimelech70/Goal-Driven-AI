"""
Demo runner.

    python run_demo.py

It auto-detects backends:
  * If Ollama is running with a chat model + nomic-embed-text, it uses your GPU.
  * Otherwise it falls back to offline stubs so you can watch the control flow.

Set an explicit model:
    python run_demo.py --model qwen2.5:7b-instruct-q4_K_M
"""
from __future__ import annotations

import argparse

from gdc import (CognitiveAgent, Goal, GoalType, LongTermMemory,
                 build_trading_memory, build_llm, build_embedder, load_config)


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg.llm.model, help="Ollama model tag")
    ap.add_argument("--backend", default=cfg.llm.backend,
                    choices=["auto", "ollama", "stub"])
    ap.add_argument("--config", default=None, help="path to config.json")
    ap.add_argument("--max-cycles", type=int, default=cfg.agent.max_cycles)
    args = ap.parse_args()

    if args.config:
        cfg = load_config(args.config)

    embedder = build_embedder(cfg.embedder.backend, cfg.embedder.model, cfg.embedder.host)
    llm = build_llm(args.backend, args.model, cfg.llm.host)
    print(f"[backends] embedder={type(embedder).__name__}  llm={llm.name}")

    ltm = LongTermMemory(embedder)
    build_trading_memory(ltm)
    print(f"[memory] seeded LTM -> {ltm.summary()}")

    agent = CognitiveAgent(ltm, llm, embedder, wm_capacity=cfg.agent.wm_capacity, verbose=True)

    # --- a SOLVE goal: exercises the full act/decide path ------------------
    solve = Goal(
        description=("Decide whether to open a long position on TICK given "
                     "momentum is up and RSI is 68"),
        goal_type=GoalType.SOLVE,
        success_criteria=["A concrete, risk-bounded position decision"],
        urgency=0.7,
    )
    agent.run(solve, max_cycles=args.max_cycles)

    # --- a LEARN goal on a fresh workspace ---------------------------------
    agent2 = CognitiveAgent(ltm, llm, embedder, wm_capacity=cfg.agent.wm_capacity, verbose=True)
    learn = Goal(
        description="Learn what governs whether a momentum trade is safe",
        goal_type=GoalType.LEARN,
        success_criteria=["A generalisation naming the governing factor"],
        urgency=0.4,
    )
    agent2.run(learn, max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()
