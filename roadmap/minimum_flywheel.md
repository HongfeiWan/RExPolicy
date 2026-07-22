# Minimum Flywheel Specification

## Question to answer

Can a frozen GR00T VLM plus the existing trainable Flow-DiT improve on a simple
Newton Reach task using only its own sampled rollouts and simulator reward?

This phase validates the learning loop. It does not attempt to validate general
language understanding, large-scale task generation, or complete manipulation.

## Minimal architecture

```text
One Reach task
      ↓
Newton parallel environments
      ↓
image + one instruction + proprioception
      ↓
frozen GR00T VLM + trainable current Flow-DiT
      ↓
independent diffusion-noise rollouts
      ↓
dense distance progress + deterministic success
      ↓
retain the top 20% of actual executed trajectories
      ↓
ordinary Flow-Matching training
      ↓
fixed K=1 held-out evaluation
      ↓
promote only when progress improves
```

## Task constraints

- Open only EEF XYZ translation.
- Keep orientation and Linker L10 targets fixed.
- Randomize the target inside a small reachable volume.
- Use one target object and one instruction.
- Use short execution prefixes and replan frequently.
- Train only on action windows built from actions actually executed by Newton.

The initial dense signal is distance reduction:

```text
r_t = distance_t - distance_t_plus_1
```

Success is a separate threshold held for a fixed number of control steps.

## Collection and storage

The first collector may use independent parallel episodes instead of a branch
tree. Multiple worlds start from the same or grouped reset recipe and differ by
the Flow-DiT sampling noise.

Each episode stores only:

```text
task_id
initial state or deterministic reset recipe
simulator and diffusion RNG seeds
instruction
actual executed actions
per-step distance and reward
success and termination
policy version
```

Video and per-frame object-state logs are not required. Re-enter Newton with the
initial recipe and executed actions to render observations later.

## First training rule

Rank trajectories by task progress and retain the top 20 percent. Append them to
a growing elite replay buffer and use the existing unweighted Flow-Matching loss.
Filtering the dataset is sufficient for the first experiment; per-sample
advantage weighting can follow only after this loop works.

Freeze the VLM. Update the DiT and action-side state/action projection layers at
a conservative learning rate.

## Evaluation

Reserve at least 64 reset recipes that never enter training. Evaluate each with
four fixed diffusion seeds, using K=1 without privileged candidate selection.

Track only:

- normalized distance progress;
- success rate;
- invalid-action rate.

The loop is viable when the top sampled trajectories consistently outperform the
median and successive promoted models improve K=1 held-out progress. A full
Reach success is not required in the first generation.

## Explicit non-goals

Do not add these to Phase 1:

- PPO, DPPO, residual RL, or a state-policy teacher;
- GPT-generated runtime rewards or instruction populations;
- reward populations, MAP-Elites, Pareto archives, or behavior embeddings;
- per-chunk beam search or recursive state branching;
- multi-task replay balancing and automatic curricula;
- tactile memory, recurrent policies, or VLM fine-tuning;
- a new compact diffusion head replacing the current DiT.

After the engineering loop works with the retained GR00T checkpoint, a separate
ablation can reset the action side to test strict zero-demonstration bootstrap.
Do not mix that scientific question into the first pipeline validation.

