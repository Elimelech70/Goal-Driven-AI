# Goal-Driven AI Implementation Guide — Updated Baseline

## Purpose

Translate the current architecture into runnable software components.

## Architecture

```text
External / Internal Input
  ↓
Sensory Transduction
  ↓
Entity Formation
  ↓
Long-Term Entity Memory
  ↓
Goal / Intention
  ↓
Executive / PFC Controller
  ↓
Attention Engine
  ↓
Working Memory Substrate
  ↓
Memory Interaction Operations
  ↓
Hypotheses / Branches / Plans / Decisions
  ↓
Evaluation
  ↓
Action OR Strategy Change OR Learning
  ↓
Hippocampal Consolidation
  ↓
Long-Term Memory Update
```

## Core Objects

- `Goal`: description, type, priority, urgency, context tags, success criteria.
- `Strategy`: operation sequence, retrieval focus, abstraction level, attention policy.
- `EntityProfile`: sensory features, actions, locations, relations, emotions, episodes, metadata.
- `MemoryNode`: entity/concept/subject/topic/domain node with embedding and graph links.
- `MemoryRelationship`: typed relationship with strength, metadata, evidence, and embedding text.
- `WMItem`: live working-memory item or pointer.
- `ThoughtBranch`: divergent candidate path.

## Working Memory

Working memory is the live substrate, not a file and not merely a prompt.

Protected slots:

```text
active_goal
live_intention
current_strategy
leading_hypothesis
pending_action
unresolved_conflict
```

Active items:

```text
entity_ref
concept_ref
relationship_ref
hypothesis
branch
observation
decision
sensory_entity
```

## Attention Engine

Inputs:

- goal / live intention
- strategy retrieval focus
- abstraction level
- current hypotheses and observations
- vector relevance
- novelty
- salience / importance
- urgency
- relationship strength to currently active memory

Output:

- ranked memory nodes admitted into working memory as references.

## Long-Term Memory

Long-term memory is graph + vector memory:

```text
Entity → Concept → Subject → Topic → Domain
```

Relationships are first-class objects rather than simple strings.

## LLM Bootstrap Layer

Early stage:

- entity extraction
- summarisation
- concept explanation
- hypothesis wording
- prediction wording
- decision phrasing

Later stage:

- advisory/tool role only
- more behaviour handled by graph/vector/entity memory and executive control

## Cognitive Loop

```python
while active_goal:
    strategy = executive.choose_strategy(goal)
    attention.allocate(goal, strategy, long_term_memory, working_memory)
    memory_interaction.run(strategy.operations)
    evaluation = executive.evaluate(goal, working_memory)

    if evaluation == ACT:
        act_or_decide()
        consolidate_episode()
    elif evaluation == CHANGE_STRATEGY:
        executive.choose_next_strategy()
    elif evaluation == LEARN:
        consolidate_episode()
    else:
        working_memory.decay_all()
```

## Development Roadmap

### Phase 1 — Done
- Package structure
- Embeddings / stub embeddings
- Entity memory
- Vector recall

### Phase 2 — Done
- Working memory
- Activation and decay
- Attention scoring

### Phase 3 — Done
- Goal hierarchy basics
- Executive strategy choice
- Live intention slot

### Phase 4 — Started / Improved
- Relationship graph
- Relationship metadata
- Relationship strength in recall

### Phase 5 — Started / Improved
- Hypothesis engine
- Branch thinking
- Leading hypothesis slot

### Phase 6 — Started
- Autonomous loop
- Evaluation-based strategy switching
- Action/decision path

### Phase 7 — Started
- Hippocampal consolidation
- Episode write-back

### Phase 8 — Next
- Disk-backed persistent memory
- Tests
- Multi-goal scheduling
- Better conflict handling
- Rich sensory encoders
