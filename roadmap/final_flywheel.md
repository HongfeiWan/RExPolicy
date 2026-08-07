# Final Data-Flywheel Design

This document preserves the intended end state. Several control-plane and
derived-view components now exist, but the architecture below remains
aspirational: active training is still the single-task Reach flywheel with
historical round-robin replay.

This is the target reached after the ordered phases in the canonical
[roadmap](README.md); it is not a second implementation or release route.

## Stable architectural decisions

- Use the pretrained GR00T VLM as the initial visual-language representation and
  freeze it during early policy improvement.
- Retain GR00T's current Flow-DiT action head and update it with simulator-selected
  experience; DiT is already the velocity model inside a diffusion policy.
- Keep trainable DiT parameters, gradients, and optimizer moments in FP32 while
  using BF16 autocast for memory-efficient forward computation.
- Do not require a full-action state PPO teacher.
- Generate actions with the policy itself, evaluate them in Newton, and train on
  successful or relatively advantageous executed action chunks.
- Store replay recipes, effective actions, seeds, metrics, and versions rather
  than rollout video or per-frame bottle trajectories. Reconstruct state and
  images by replaying the episode in the simulator.
- Keep task completion and safety semantics stable while allowing multiple,
  deliberately different shaping rewards for the same task.
- Derive short-horizon advantages by comparing chunks executed from the same
  simulator state; keep full-episode success and termination as long-horizon
  guards.
- Use positive/top-fraction advantage filtering only to bootstrap success. Once
  a chunk or trajectory succeeds, never hard-top-k it away: place it in a
  persistent Success Archive with nonzero training eligibility.
- Normalize the weighted objective over the complete DDP accumulation window
  and expose every newly selected sample before replacement replay.
- Do not require PPO, a critic, GAE, or a full-action state teacher.

## Final architecture

```text
                       GPT task authoring
         canonical goal + instruction sets + reward profiles
                + success/safety specs + test cases
                                │
                                ▼
                  task compiler and validators
          asset / reachability / visibility / reward tests
                                │
                                ▼
                    versioned task registry
                                │
                                ▼
                   task and curriculum manager
                                │
                                ▼
                    Newton GPU environments
              initial state + images + proprioception
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        frozen GR00T VLM                  state encoder
                └───────────────┬───────────────┘
                                ▼
                      trainable Flow-DiT θ_k
                                │
                    K independent noise samples
                                ▼
                      K candidate action chunks
                                │
                                ▼
                   same-state Newton rollouts
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
       shaping rewards    success/safety oracle   raw behavior metrics
             └──────────────────┼──────────────────┘
                                ▼
               within-state baseline and advantages
                  ┌──────────────┼──────────────┐
                  ▼                             ▼
       pre-success advantage bootstrap       all valid successes
                  │                             │
                  │                metadata-only Success Archive
                  │             recipe references + nonzero eligibility
                  └──────────────┼──────────────┘
                                 ▼
              behavior / reward-profile / state-balanced sampler
                                │
                                ▼
              advantage-weighted Flow-Matching learner
                                │
                                ▼
                      candidate model θ_k+1
                                │
                                ▼
                     held-out evaluation gate
                     K=1 policy and K>1 search
                         │                 │
                       pass               fail
                         │                 │
                    promote actor       keep θ_k
                         │
                         └──── next generation
```

## Reward diversity

The final system does not collapse every task into one shaping reward. For a
fixed task goal, GPT may propose a population of reward profiles emphasizing
different valid solutions: speed, smoothness, path length, contact force,
clearance, energy, approach direction, grasp region, regrasp, or physics
robustness.

The invariant layer contains only:

- the canonical task goal;
- deterministic success predicates;
- failure and safety constraints.

The deterministic success predicate has priority over shaping reward. A
successful trajectory with a lower shaping score remains a success and cannot
be removed by a top-fraction reward filter. Shaping rewards distinguish
nonterminal progress and quality among valid successes; they do not redefine
task completion.

Advantages are normalized within the same task, initial-state group, and reward
profile. Reward magnitudes from different profiles are never compared directly.

Newton now records reward-free Event Ledgers that can be deterministically
rescored, compiled into exact success behavior descriptors, and organized in
immutable quality-diversity indexes. A deterministic Success Archive planning
path balances behavior, reward, and initial-state strata, but the active Phase
1 trainer does not yet consume or checkpoint that state and remains
round-robin.

Process-reward artifacts remain explicitly `shadow_only` with
`oracle_authority=none`. They cannot alter canonical success or safety,
lifecycle admission, activation, or model promotion.

The policy may eventually receive a natural-language style request or continuous
behavior condition. With the condition hidden, Flow-DiT should sample the
mixture of successful modes; with it present, behavior should be controllable.

## Instruction generation

Instructions are generated from the same canonical task specification as the
reward and success predicates. They are versioned and split before use into:

- collection/training instructions;
- promotion instructions;
- sealed audit instructions.

Instructions must not leak simulator coordinates, internal entity names, reward
thresholds, or state unavailable to the policy. Equivalent instructions should
produce equivalent behavior, while counterfactual target instructions should
change the selected object or goal.

TaskSpec v2 now enforces separate train, promotion, and sealed-audit instruction
commitments. Automatic authoring can produce a provider-attested, statically
valid quarantined candidate; it does not admit the candidate into a registry or
make it available to training.

## Local and global selection

Short same-state branches provide low-variance chunk-level comparisons. Complete
episodes provide terminal success, stability, and long-horizon quality. Local
progress cannot replace the full episode predicate because a locally attractive
action may make the eventual grasp or placement worse.

At each decision, K chunks start from one reconstructed root state. A practical
bootstrap can obtain that state by deterministic reset plus replay of the
selected action history; native simulator snapshots may replace replay later as
an optimization. Before success is available, subtract a within-group median
baseline, select positive advantages up to a configured fraction, and apply
normalized advantage weights to the per-sample Flow-Matching loss. This
accelerates discovery of a workable behavior. These comparative advantages do
not imply a PPO objective and require neither a critic nor GAE.

As soon as a decision contains valid successes, the hard top-fraction rule no
longer applies to them. Every successful executed chunk receives a positive
training weight. When the selected root later completes the task, all executed
chunks on that successful path are indexed as one successful trajectory. Both
direct successes and successful paths enter the persistent Success Archive;
quality scores may prioritize them but can never reduce their replay
eligibility to zero.

The root continuation is chosen separately from archive admission:
deterministic success takes priority, then branch score. It is therefore valid
to continue only the best success online while retaining every other successful
branch for learning. The selected prefix is appended to the episode history and
branching repeats until success, failure, truncation, or the episode horizon
fires. A future system may retain multiple roots, but the Phase 1
receding-horizon implementation keeps one.

The archive stores actual executed actions. Predicted but unexecuted chunk tails
are never labeled with the return of the executed prefix.

Selection quality and behavior coverage are separate quantities. A top-20%
filter deliberately removes probability mass and can still collapse onto one
reward-favored solution, especially when K is small; it is retained only for
the pre-success bootstrap pool. Keeping all successes is necessary but still
not sufficient: the most common behavior, reward profile, or easy initial state
can dominate an unbalanced replay stream. Mature training therefore samples the
Success Archive by behavior cell, reward profile, and initial-state group,
using quotas or inverse-frequency weighting while maintaining a nonzero floor
for every valid success. Diffusion sampling can represent a multimodal success
set, but archive admission alone does not create that balance automatically.

PPO remains an optional later ablation, not a prerequisite. Its entropy term can
encourage stochasticity but does not guarantee distinct successful modes, and
its likelihood-ratio machinery is substantially more invasive for Flow-DiT than
the same-state advantage-weighted regression used here.

## Relationship to the Phase 1 bootstrap

The minimum implementation includes same-state chunk comparison, `all_success`
selection, success-first single-root continuation, successful-path
back-indexing, a metadata-only cross-generation Success Archive, globally
weighted FP32/DDP Flow-Matching, atomic resume, and a sealed K=1 non-regression
gate. Every current-generation selected sample is scheduled at least once, and
historical successes return through a deterministic round-robin cursor. Phase 1
uses the single versioned `reach_green_cap/v1` reward profile.

The repository also provides reward-free ledgers, immutable reward/process/QD
views, provider-attested sandboxed authoring to `quarantined_static`, audited
lifecycle transitions, and an explicit quality-balanced Success Archive plan.
These extend the Phase 1 contracts without changing its active learner. The
only runtime bridge remains Reach, activation has not been operationally
exercised for an authored task, and QD state is not part of trainer checkpoints
or online sampling.

## Evaluation and promotion

Candidate promotion must separately measure:

- K=1 raw-policy task progress and success;
- K>1 policy-plus-search performance;
- invalid actions and safety violations;
- regressions on learned task families;
- behavior coverage and mode retention when diversity training is active.

This separation prevents simulator lookahead from hiding a DiT that has not
internalized the selected behavior.

The minimum gate already evaluates 8 sealed reset recipes with 4 fixed
diffusion seeds each at K=1. It rejects invalid actions, increased safety
failures, a success drop greater than `1/32`, or a median final-distance
regression greater than `0.005m` when success does not improve. Mature
multi-task promotion extends this fixed-suite principle with family-specific
regression and diversity gates.

## Later capability gates

Before admitting tasks that require finer perception or memory, test whether the
frozen VLM representation preserves the target identity, pose, and instruction
distinctions. If not, add a trainable adapter, limited VLM unfreezing, depth,
contact sensing, observation history, or policy memory only when the task proves
that one of them is necessary.
