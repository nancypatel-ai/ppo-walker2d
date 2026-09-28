"""Generalized Advantage Estimation."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute advantages and returns for a time-major rollout."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for step in reversed(range(rewards.shape[0])):
        next_value = last_value if step == rewards.shape[0] - 1 else values[step + 1]
        continuation = 1.0 - terminated[step]
        delta = rewards[step] + gamma * next_value * continuation - values[step]
        running = delta + gamma * gae_lambda * continuation * running
        advantages[step] = running
    return advantages, advantages + values
