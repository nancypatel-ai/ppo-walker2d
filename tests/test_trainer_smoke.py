"""Trainer smoke tests."""

import torch

from ppo_walker2d.config import ProjectConfig, TrainingConfig
from ppo_walker2d.ppo.gae import compute_gae


def test_gae_returns_finite_values() -> None:
    rewards = torch.ones(4)
    values = torch.zeros(4)
    advantages, returns = compute_gae(
        rewards, values, torch.zeros(4), torch.zeros(()), 0.99, 0.95
    )
    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages, returns)


def test_training_defaults_are_reproducible() -> None:
    config = ProjectConfig(training=TrainingConfig())
    assert config.training.gamma == 0.99
    assert config.seed == 42
