# Design

The architecture separates configuration, environments, models, PPO optimization,
and evaluation. Each boundary remains testable without a long training run.

Configuration is YAML plus typed dataclasses. The actor-critic uses orthogonally
initialized shared MLP features with Gaussian actions and a scalar value head.
Rollouts are collected before GAE computes advantages. PPO then performs clipped
minibatch updates and writes a self-contained PyTorch checkpoint.

MuJoCo remains an optional runtime boundary for unit tests. Environment imports
are local so reward, observation, model, and GAE tests run without a display.
