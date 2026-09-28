"""Rollout collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ppo_walker2d.models.actor_critic import ActorCritic


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    values: torch.Tensor
    last_observation: np.ndarray


def collect_rollout(
    environment: Any,
    policy: ActorCritic,
    observation: np.ndarray,
    steps: int,
    device: torch.device,
) -> Rollout:
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminated: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    current = observation
    for _ in range(steps):
        tensor = torch.as_tensor(current, dtype=torch.float32, device=device)
        with torch.no_grad():
            distribution = policy.distribution(tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(-1)
            _, value = policy(tensor)
        next_observation, reward, done, truncated, _ = environment.step(
            action.cpu().numpy()
        )
        observations.append(tensor)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(torch.as_tensor(reward, dtype=torch.float32, device=device))
        terminated.append(
            torch.as_tensor(done | truncated, dtype=torch.float32, device=device)
        )
        values.append(value)
        current = next_observation
        if np.any(done | truncated):
            current, _ = environment.reset()
    return Rollout(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        log_probabilities=torch.stack(log_probs),
        rewards=torch.stack(rewards),
        terminated=torch.stack(terminated),
        values=torch.stack(values),
        last_observation=current,
    )
