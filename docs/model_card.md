# Model card

## Intended use

This policy is intended for research and education in simulated bipedal
locomotion. It is not intended for physical robots or safety-critical control.

## Method

The model is a PyTorch actor-critic trained with clipped PPO and GAE on
Gymnasium `Walker2d-v5`. Actions come from a diagonal Gaussian policy.

## Limitations

The current repository contains an integration checkpoint, not a validated
five-seed result. Simulation performance does not establish real-world safety,
transfer, or robustness. The model can fail under unseen dynamics.
