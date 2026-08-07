# Minimum Flywheel Specification

Canonical roadmap position: **Phase 1**.

Status: **the learning contract is implemented and active; canonical node3
runtime acceptance remains open**. Phase order and completion are governed by
[README.md](README.md); this document does not define a separate route.

## Hypothesis

Can a frozen GR00T VLM plus its existing Flow-DiT improve on a simple Newton
task using only policy-sampled action chunks and simulator feedback?

The minimum experiment answers this with one versioned task and reward profile.
It does not claim general manipulation or general language understanding.

## Architecture

```text
reach_green_cap/v1 reset recipe
                 │
                 ▼
deterministic reset + selected-action replay
                 │
                 ▼
K Newton worlds at the identical root state
                 │
        image + instruction + proprioception
                 │
        ┌────────┴────────┐
        ▼                 ▼
 frozen GR00T VLM     state encoder
        └────────┬────────┘
                 ▼
         FP32 Flow-DiT + K noises
                 │
                 ▼
       K projected effective chunks
                 │
                 ▼
    short-prefix execution and task metrics
                 │
        ┌────────┴────────┐
        ▼                 ▼
 pre-success advantage    every valid success
 bootstrap                + successful root path
        └────────┬────────┘
                 ▼
 current samples + Success Archive round-robin
                 │
                 ▼
 globally weighted Flow-Matching through DDP
                 │
                 ▼
 atomic checkpoint → fixed K=1 evaluation gate
```

This is chunk-level selection, not whole-episode elite filtering. Same-state
alternatives supply the counterfactual baseline, so no PPO likelihood ratio,
critic, GAE, or full-action state teacher is needed.

## `reach_green_cap/v1` contract

The environment keeps `task_mode="bottle_transfer"` as its backward-compatible
default. The flywheel explicitly selects `task_mode="reach_green_cap"`,
`control_mode="pd_eef_pose_abs"`, and task ID `reach_green_cap/v1`.

The instruction is:

> move the open right hand to the safe pre-grasp position beside the green
> bottle cap without moving the bottle

The authoritative target is the settled bottle center plus
`(0.180, 0.080, 0.050)m` in world coordinates. It is transformed to the right
arm base frame with:

```text
p_goal_base = R_world_baseᵀ × (p_goal_world - t_world_base)
```

Task semantics are fixed:

- an integer reset recipe translates the settled bottle deterministically by
  at most `0.010m` on each XY axis; all K branches receive the same offset;
- success requires EEF-to-goal distance `≤ 0.040m` for two consecutive control
  steps;
- any hand–bottle contact during a physics frame is a failure;
- bottle displacement from the settled pose `> 0.010m` is a failure;
- dense reward is
  `clip((previous_distance - current_distance) / 0.030, -1, 1)`;
- success overrides the dense reward with `+1`; failure overrides it with `-1`.

Only `eef_9d[:3]` is effective. EEF rotation and every hand/arm target are
projected to the reset-state hold target and masked out of Flow-Matching loss.
Collection, continuation, episode JSONL, Success Archive replay, and training
all use the projected action actually executed by Newton, never the raw decoded
action.

The production reachability oracle uses 10Hz control, 60Hz simulation, 16
substeps per frame, and 60 settle frames. Its absolute-goal controller must:

- succeed within eight control steps;
- keep maximum bottle displacement below `0.005m`;
- produce no hand–bottle contact;
- reproduce success step and per-step metrics after reset/replay.

Same-state comparison covers joint/body position and velocity, control targets,
object/TCP/goal data, task counters, contacts, and both RGB observations.
Floating values use an absolute tolerance of `1e-5`; integer, Boolean, and
image tensors must match exactly.

## Collection and selection

For each decision:

1. Reset every candidate world with the episode recipe and replay the exact
   selected effective-action history.
2. Reject the decision if the full state fingerprint differs across worlds.
3. Form the frozen VLM/state conditions and sample K chunks with independent
   diffusion noise.
4. Project every action to the task-effective controls, execute only the
   configured prefix, and measure reward, success, failure, and termination.
5. Compute the median score inside the same-state comparison pool.
6. If no branch succeeds, retain positive advantages up to the configured
   fraction, falling back to the best continuation if needed.
7. If any branch succeeds, retain **all** successful branches with positive
   weights; success cannot be displaced by a higher-scoring failure.
8. Continue one root—best valid success first, otherwise best score—and repeat
   until success, failure, truncation, or the control-step limit.

Every candidate has a stable ID containing generation, rank, episode, decision,
and world. Directly successful chunks receive `direct_branch`; if the chosen
root episode succeeds, every chosen chunk on that root receives
`selected_success_path`. Both roles can apply to the same sample.

## Coverage and globally weighted training

The update order on each rank is:

1. current-generation successes without replacement;
2. other current-generation selected samples without replacement;
3. historical successes chosen by the archive cursor;
4. seeded replacement samples only after the coverage pass.

The optimizer-step count grows automatically until every current-generation
sample is exposed at least once. Ranks with fewer samples receive zero-weight
dummy slots so every rank performs the same number of backward calls.

For each complete gradient-accumulation window, let `W` be the all-reduced sum
of real sample weights. Each rank backpropagates:

```text
world_size × Σ_local(weight_i × loss_i) / W
```

DDP averaging therefore yields the true global weighted mean even with
batch size one, unequal rank-local counts, and gradient accumulation. A
non-finite or zero global weight is a hard error.

The GR00T VLM and all non-DiT modules are frozen in BF16. Trainable
`action_head.model` parameters, their gradients, and AdamW moments are FP32.
Sampling and training forward passes use BF16 autocast without GradScaler.
Adam state may be offloaded to ordinary CPU memory during collection; the VLM
may be offloaded while DiT updates.

A layer-balanced parameter probe records gradient norm, update RMS, maximum
absolute update, changed fraction, and cross-rank maximum difference. A
non-finite/zero gradient, non-finite/no-op update, or parameter divergence
aborts the generation before checkpoint commit.

## Replay and Success Archive

Episode shards contain reset recipes, version and fingerprints, instructions,
all executed candidate prefixes, comparative metrics, selection flags, and the
chosen root path. `data_generation=N` is recorded separately from
`sampling_policy_generation=N-1`, because generation N data is collected before
the update that creates policy N. The per-rank Success Archive is an append-only set of
immutable JSONL references from a stable success sample ID to its source
episode record.

The Success Archive is metadata-only: it does not copy images, video, frozen
features, object histories, or action tensors already present in the episode
recipe. Historical replay loads the referenced recipe, reconstructs the root
with selected effective actions, reruns the candidate prefix, renders current
observations, and recomputes frozen VLM features.

New successful references become replay-eligible only after their generation
commits. Historical selection is deterministic round-robin, four per rank and
generation by default. There is no hard top-k deletion, so every retained
success remains reachable over time. Duplicate IDs, missing source records,
fingerprint mismatch, or early replay termination are hard errors.

The repository contains a separate immutable QD index and explicit
quality-balanced Success Archive planner. The minimum flywheel intentionally
does not consume or checkpoint that external state; its production replay
behavior remains deterministic historical round-robin.

## Checkpoint and evaluation gate

At a generation boundary, an atomic checkpoint captures FP32 DiT, CPU AdamW
state, optimizer/generation counters, all rank RNG states, archive shards and
cursors, manifests, and checksums. Resume is strict and ignores incomplete
`.partial-*` directories. Operational details are in
[bootstrap_ddp.md](bootstrap_ddp.md).

The held-out suite contains 8 sealed reset recipes × 4 fixed diffusion seeds.
It runs K=1 with no same-state branch selection, never enters training or either
archive, and reports success, safety failure, invalid action, final distance,
return, and bottle displacement.

Against `last-good`, reject and stop when:

- the candidate contains any invalid action;
- safety failures increase;
- success decreases by more than `1/32`;
- success does not increase and median final distance worsens by more than
  `0.005m`.

The rejected candidate and diagnostics remain inspectable, but its optimizer
state is not used to continue training.

## Deliberate non-goals

These are non-goals of the active Phase 1 execution path, not claims that no
supporting control-plane or offline artifact exists elsewhere in the
repository.

The minimum loop does not add:

- PPO, DPPO, GAE, residual RL, a critic, or a state-policy teacher;
- activated provider-generated runtime tasks, instruction batches, or reward
  populations;
- live MAP-Elites/Pareto/QD sampling in the trainer or learned behavior
  embeddings;
- a persistent multi-root beam;
- VLM fine-tuning, tactile memory, or recurrent policy state;
- a compact diffusion head replacing GR00T's existing Flow-DiT.

Those capabilities may be added only after the two-GPU engineering smoke and
the staged 10→20 generation K=1-gated soak are stable.
