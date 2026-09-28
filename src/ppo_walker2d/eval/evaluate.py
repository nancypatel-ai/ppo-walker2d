"""Policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ppo_walker2d.eval.metrics import EvaluationMetrics
from ppo_walker2d.models.actor_critic import ActorCritic


def evaluate(
    environment: Any, policy: ActorCritic, episodes: int = 20, seed: int = 0
) -> EvaluationMetrics:
    returns: list[float] = []
    distances: list[float] = []
    speeds: list[float] = []
    energies: list[float] = []
    falls = 0
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        energy = 0.0
        steps = 0
        done = False
        while not done:
            tensor = torch.as_tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                action, _ = policy(tensor)
            action_array = action.numpy()
            observation, reward, terminated, truncated, info = environment.step(
                action_array
            )
            episode_return += float(reward)
            energy += float(np.square(action_array).sum())
            steps += 1
            done = bool(terminated or truncated)
        returns.append(episode_return)
        distances.append(float(info.get("x_position", 0.0)))
        speeds.append(distances[-1] / max(steps * 0.002, 1e-8))
        energies.append(energy)
        falls += int(bool(terminated and not info.get("healthy", True)))
    return EvaluationMetrics(
        mean_return=float(np.mean(returns)),
        mean_distance=float(np.mean(distances)),
        mean_speed=float(np.mean(speeds)),
        fall_rate=falls / episodes,
        energy_proxy=float(np.mean(energies)),
        episodes=episodes,
    )
