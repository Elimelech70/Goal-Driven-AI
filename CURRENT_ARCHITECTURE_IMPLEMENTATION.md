# Current Architecture Implementation Alignment

This package has been updated to conform to `AI_ARCHITECTURE_CURRENT_BASELINE.md`.

## Architecture reflected in code

```text
External / Internal Input
        ↓
Sensory Transduction              gdc/sensory.py
        ↓
Entity Formation                  EntityProfile, MemoryNode(level=ENTITY)
        ↓
Long-Term Entity Memory           LongTermMemory nodes + relationships
        ↓
Goal / Intention                  Goal + WorkingMemory live_intention slot
        ↓
Executive / PFC Controller        gdc/executive.py
        ↓
Attention Engine                  gdc/attention.py
        ↓
Working Memory                    gdc/working_memory.py
        ↓
Memory Interaction Operations     gdc/memory_interaction.py
        ↓
Hypotheses / Branches / Decisions ThoughtBranch + WM slots/items
        ↓
Evaluation                        Executive.evaluate()
        ↓
Action OR Strategy Change OR Learning
        ↓
Hippocampal Consolidation         LongTermMemory.consolidate_episode()
        ↓
Long-Term Memory Update
```

## Key conformance updates

1. **Working memory as substrate**
   - Added protected typed slots: active goal, live intention, current strategy, leading hypothesis, pending action, unresolved conflict.
   - Working memory still stores active items/references, but executive state is now explicit and inspectable.

2. **Intention-conditioned attention**
   - Attention queries include live intention and current working-memory context.
   - Attention policy is now adjusted by goal type and strategy.

3. **Relationship metadata and relationship strength**
   - Added `MemoryRelationship` with relation type, strength, metadata, evidence, and relationship embedding text.
   - LTM retrieval now considers relationship context with active memory references.

4. **Entity → Concept → Subject → Topic/Domain layering**
   - Existing `MemoryLevel` hierarchy remains, but links now carry relationship semantics.

5. **Branch thinking and leading hypothesis**
   - Branch comparison now writes the leading branch into a protected working-memory slot.
   - Hypotheses can be explicitly selected as the leading hypothesis before decision.

6. **Evaluation loop closer to baseline**
   - Evaluation considers references, branches, hypotheses, leading hypothesis, conflict, and pending action.

## Still intentionally minimal

This remains a runnable architecture skeleton. The next serious work is persistence, tests, richer sensory encoders, and better branch scoring.
