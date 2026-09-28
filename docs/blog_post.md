# Building a reproducible Walker2d PPO pipeline

The interesting part of a reinforcement learning project is rarely the first
successful rollout. It is the system around the rollout. A training script can
produce a moving robot while leaving the reader unable to answer basic
questions: which seed was used, which reward was optimized, whether a reset was
counted as a terminal transition, and whether the checkpoint can be evaluated
again. This project treats those questions as product requirements.

The target is MuJoCo Walker2d-v5. The agent controls six continuous torques and
receives a state observation from a simulated biped. The policy is a Gaussian
actor with a scalar critic. Proximal Policy Optimization updates both heads
using clipped likelihood ratios. Generalized Advantage Estimation reduces the
variance of the policy gradient while preserving a configurable bias and
variance tradeoff.

## Start with boundaries

The repository is divided into five boundaries. Configuration loads YAML into
typed dataclasses. Environment construction owns Gymnasium and MuJoCo. Models
contain only neural network definitions. The PPO package collects rollouts,
computes advantages, and performs updates. Evaluation owns metrics and artifact
serialization. This structure keeps the reward function testable without
starting a simulator and keeps model shape errors visible in a fast test.

The first useful test is not a long run. It is a short run that proves the
environment accepts the model action shape, the rollout has the expected time
dimension, GAE returns finite values, and the resulting checkpoint can be loaded.
The project runs this check with a small vectorized MuJoCo environment. The
same code path then scales to the configured eight workers.

## Reward design

The baseline reward combines forward velocity, a healthy bonus, control cost,
and a penalty for an unhealthy state. The second variant bounds the velocity
incentive and reduces the control cost coefficient. Keeping both functions
pure and deterministic makes the ablation interpretable. The trainer does not
silently replace one reward with another.

Reward shaping is also a warning. A high return is only meaningful relative to
the exact reward definition. This repository therefore reports return together
with distance, speed, fall rate, and an action-energy proxy. A policy that
earns more by producing unnecessarily large actions should not look equivalent
to a policy that walks efficiently.

## Reproducibility is a feature

The seed reaches Python, NumPy, PyTorch, action spaces, and environment resets.
The checkpoint stores model weights, optimizer state, step count, and the
configuration object. Training logs are local JSONL by default. This avoids a
runtime dependency on a tracking service and allows a run to be inspected
offline. A W&B adapter can be added around the same records when a user has
credentials and wants a public comparison dashboard.

The default command is deliberately plain:

```sh
make train config=configs/reward_v1.yaml seed=42
```

The command creates a checkpoint and a local training log. Evaluation is a
separate command so training and measurement are not accidentally conflated.
The evaluator uses fixed episode seeds and writes a JSON metrics record.

## Robustness

The domain-randomization wrapper perturbs body masses by plus or minus twenty
percent and floor friction by plus or minus thirty percent. The training
environment samples a new setting at reset. Held-out settings must remain
separate from training settings. Otherwise a robustness table can overstate
generalization by evaluating on values the policy already saw.

The same principle applies to baselines. A random policy establishes a lower
bound. A Stable-Baselines3 PPO policy provides an external implementation
reference. The project PPO policy is the primary result. Every row in the
experiment table must aggregate multiple seeds and retain the exact config.

## What the current evidence says

The current repository has passed unit tests, static checks, and a real
Walker2d integration run. A bounded policy checkpoint also evaluates through
the fixed-episode evaluator. These are integration results, not evidence that
the policy has converged or that one reward is superior.

The long experiment matrix remains a separate measurement phase. It should run
five seeds for each primary configuration, three seeds per learning-rate value,
and report mean plus or minus standard deviation. The code makes that phase
repeatable without embedding generated checkpoints in Git.

## A small system is easier to trust

The project intentionally avoids hidden telemetry and runtime network calls.
There is no user data. The demo is local and checkpoint-based. Docker and CI
exercise installation, tests, and linting. The implementation is not a claim
that simulated locomotion is safe in the physical world. It is a compact,
inspectable foundation for studying PPO and reproducible experimentation.
