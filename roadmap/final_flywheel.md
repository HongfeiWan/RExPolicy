# Final Data-Flywheel Design

This document preserves the intended end state. It is a roadmap, not the current
implementation scope.

## Stable architectural decisions

- Use the pretrained GR00T VLM as the initial visual-language representation and
  freeze it during early policy improvement.
- Retain GR00T's current Flow-DiT action head and update it with simulator-selected
  experience; DiT is already the velocity model inside a diffusion policy.
- Do not require a full-action state PPO teacher.
- Generate actions with the policy itself, evaluate them in Newton, and train on
  successful or relatively advantageous executed actions.
- Store replay recipes, actions, seeds, metrics, and versions rather than rollout
  video. Reconstruct images by replaying the episode in the simulator.
- Keep task completion and safety semantics stable while allowing multiple,
  deliberately different shaping rewards for the same task.

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
                  replayable experience archive
              elite + historical + failure metadata
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

Advantages are normalized within the same task, initial-state group, and reward
profile. Reward magnitudes from different profiles are never compared directly.

Newton records raw physical metrics so old trajectories can be rescored under
new profiles. A later quality-diversity archive may retain the best successful
trajectories in behavior cells rather than keeping a single global top-k.

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

## Local and global selection

Short same-state branches provide low-variance chunk-level comparisons. Complete
episodes provide terminal success, stability, and long-horizon quality. Local
progress cannot replace the full episode predicate because a locally attractive
action may make the eventual grasp or placement worse.

The archive stores actual executed actions. Predicted but unexecuted chunk tails
are never labeled with the return of the executed prefix.

## Evaluation and promotion

Candidate promotion must separately measure:

- K=1 raw-policy task progress and success;
- K>1 policy-plus-search performance;
- invalid actions and safety violations;
- regressions on learned task families;
- behavior coverage and mode retention when diversity training is active.

This separation prevents simulator lookahead from hiding a DiT that has not
internalized the selected behavior.

## Later capability gates

Before admitting tasks that require finer perception or memory, test whether the
frozen VLM representation preserves the target identity, pose, and instruction
distinctions. If not, add a trainable adapter, limited VLM unfreezing, depth,
contact sensing, observation history, or policy memory only when the task proves
that one of them is necessary.

