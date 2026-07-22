# RExPolicy Roadmap

The roadmap intentionally separates the next runnable experiment from the final
robot-experience system. Features do not move into an earlier phase merely
because they are useful in the final design.

## Phase 0 — migrated baseline

Status: **present in this repository**.

- Newton batched simulator and task metrics.
- Frozen-capable GR00T VLM + current Flow-DiT inference path.
- 19D EEF + Linker L10 execution contract.
- Runtime tests and compact scene assets.

No flywheel collector or trainer has been implemented yet.

## Phase 1 — minimum flywheel

Status: **next implementation; not started**.

Prove one hypothesis on one Reach task: policy rollouts filtered by Newton reward
can improve K=1 held-out Flow-DiT performance without a PPO teacher or new human
demonstrations. The exact scope is in
[minimum_flywheel.md](minimum_flywheel.md).

## Phase 2 — visual-language grounding

- Two visually distinguishable targets in the same scene.
- Instructions choose the target.
- A small set of equivalent paraphrases.
- Counterfactual evaluation: changing only the instruction changes behavior.

## Phase 3 — task and behavior expansion

- Reach to pregrasp, grasp, lift, transport, and place curricula.
- A small number of deliberately different reward profiles.
- Simple per-task and per-behavior replay balancing.
- Physics, camera, object, and scene randomization.

## Phase 4 — autonomous task authoring

- GPT-generated versioned task specifications and instruction batches.
- Diverse reward-profile generation rather than one global shaping reward.
- Deterministic success and safety predicates.
- Declarative reward compilation and automated validation.
- Quality-diversity archive and reward-gap discovery.

## Final system

The intended long-term design, including reward diversity, instruction
generation, replayable experience storage, evaluation gates, and model
promotion, is preserved in [final_flywheel.md](final_flywheel.md).

