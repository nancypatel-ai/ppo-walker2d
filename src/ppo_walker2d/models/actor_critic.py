"""Actor-critic network definitions."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(input_size: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layer = nn.Linear(previous, size)
        nn.init.orthogonal_(layer.weight, gain=2**0.5)
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.Tanh()))
        previous = size
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Gaussian actor and scalar critic with shared feature extraction."""

    def __init__(
        self, observation_size: int, action_size: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.features = _mlp(observation_size, hidden_sizes)
        feature_size = hidden_sizes[-1]
        self.actor = nn.Linear(feature_size, action_size)
        self.critic = nn.Linear(feature_size, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def distribution(self, observations: torch.Tensor) -> torch.distributions.Normal:
        mean, _ = self(observations)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        log_probability = distribution.log_prob(actions).sum(-1)
        entropy = distribution.entropy().sum(-1)
        _, value = self(observations)
        return log_probability, entropy, value
