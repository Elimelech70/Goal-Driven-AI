"""
Seed a small distributed memory: Topic -> Subject -> Concept -> Entity.

Domain: trading, so the demo maps onto the Catalyst world. Swap this out for any
domain; the architecture is content-agnostic.
"""
from __future__ import annotations

from .entities import MemoryNode, MemoryLevel
from .long_term_memory import LongTermMemory


def build_trading_memory(ltm: LongTermMemory) -> None:
    def node(name, level, text, salience=0.5, assoc=None):
        return ltm.add(MemoryNode(name=name, level=level, text=text,
                                   salience=salience, associations=assoc or {}))

    # Topic
    trading = node("Trading", MemoryLevel.TOPIC,
                   "Buying and selling financial instruments to manage risk and return.",
                   0.6)

    # Subjects
    ta = node("Technical Analysis", MemoryLevel.SUBJECT,
              "Reading price and volume to infer probable near-term movement.", 0.6)
    risk = node("Risk Management", MemoryLevel.SUBJECT,
                "Controlling downside so no single position can be ruinous.", 0.8)
    ltm.link(trading.id, ta.id)
    ltm.link(trading.id, risk.id)

    # Concepts under Technical Analysis
    momentum = node("Momentum", MemoryLevel.CONCEPT,
                    "Tendency of a trend to persist; rising price with rising volume.",
                    0.6)
    rsi = node("RSI", MemoryLevel.CONCEPT,
               "Relative Strength Index; >70 suggests overbought, <30 oversold.", 0.6)
    support = node("Support and Resistance", MemoryLevel.CONCEPT,
                   "Price levels where reversals have repeatedly occurred.", 0.55)
    for c in (momentum, rsi, support):
        ltm.link(ta.id, c.id)

    # Concepts under Risk Management
    sizing = node("Position Sizing", MemoryLevel.CONCEPT,
                  "Choosing trade size so a loss is a small fraction of capital.", 0.8)
    stop = node("Stop Loss", MemoryLevel.CONCEPT,
                "A predefined exit that caps the loss on a position.", 0.75)
    for c in (sizing, stop):
        ltm.link(risk.id, c.id)

    # Entities (concrete facts / episodes / rules)
    facts = [
        ("Overbought reversal risk", rsi.id,
         "When RSI exceeds 70 during a strong uptrend, reversal risk rises but "
         "trends can persist while overbought."),
        ("Momentum-with-stop rule", momentum.id,
         "Ride momentum only with a stop below the most recent support."),
        ("Half-size on uncertainty", sizing.id,
         "When signals conflict, take half the normal position size."),
        ("Support below entry", support.id,
         "A nearby support level gives a logical place to set the stop."),
    ]
    for name, parent, text in facts:
        e = node(name, MemoryLevel.ENTITY, text, 0.6)
        ltm.link(parent, e.id)
