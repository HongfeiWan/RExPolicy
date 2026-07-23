# RExPolicy Roadmap

The roadmap separates the runnable minimum flywheel from the intended
multi-task, multi-reward system. The current implementation deliberately proves
the smallest useful self-improvement loop before adding automatic task
authoring.

## Phase 0 — migrated baseline

Status: **complete**.

- Newton batched simulator, GPU observations, action chunks, and deterministic
  reset.
- Frozen-capable GR00T VLM plus the original Flow-DiT inference path.
- Validated 19D EEF + Linker L10 action representation.
- Required compact assets and focused runtime tests.

## Phase 1 — minimum Reach flywheel

Status: **implemented; node3 two-GPU and soak acceptance remain runtime gates**.

The fixed `reach_green_cap/v1` task establishes whether the existing Flow-DiT
can improve using its own Newton-scored chunks. Its instruction asks the open
right hand to approach a safe pre-grasp point beside the green bottle cap
without touching or moving the bottle. Only EEF XYZ is effective; orientation,
hand, and arm targets remain fixed.

The implementation provides:

- same-state K-way action sampling, within-state advantages, and success-first
  continuation;
- all-success retention and successful-path back-indexing;
- FP32 DiT/AdamW updates with BF16 autocast and DDP-global weight
  normalization;
- guaranteed once-per-generation coverage of every newly selected sample;
- metadata-only Success Archive shards and deterministic historical
  round-robin replay;
- atomic full-state checkpoint/resume, run logs, heartbeat, GPU monitoring, and
  stop-at-generation-boundary control;
- sealed K=1 held-out evaluation and non-regression gating.

No PPO, GAE, critic, full-action teacher, compact replacement policy, or human
pretraining dataset is required. See [minimum_flywheel.md](minimum_flywheel.md)
for the learning contract and [bootstrap_ddp.md](bootstrap_ddp.md) for the
node3 validation protocol.

## Phase 2 — visual-language grounding

- Add two visually distinguishable targets in the same scene.
- Make the instruction, not simulator metadata exposed to the policy, select
  the target.
- Add equivalent paraphrases and counterfactual target instructions.
- Test whether the frozen VLM preserves identity and pose before introducing an
  adapter or limited unfreezing.

## Phase 3 — task and behavior expansion

- Extend the curriculum from Reach to pre-grasp, grasp, lift, transport, and
  place.
- Add a deliberately diverse set of reward profiles for the same canonical
  success predicate.
- Balance replay by behavior cell, reward profile, and initial-state group
  while keeping every valid success eligible.
- Add physics, camera, object, and scene randomization only after fixed-task
  replay remains deterministic.

## Phase 4 — autonomous task authoring

- Ask GPT to generate versioned task specifications, reward-profile
  populations, and instruction batches.
- Compile generated rewards into deterministic simulator metrics and validate
  reachability, visibility, reward direction, success, and safety.
- Separate collection/training instructions from held-out promotion and sealed
  audit instructions.
- Add quality-diversity indexing and reward-gap discovery without deleting the
  underlying Success Archive.

## Final system

The stable end-state decisions—unified success semantics, intentionally
non-uniform rewards, replay instead of stored video, and held-out model
promotion—are preserved in [final_flywheel.md](final_flywheel.md).
