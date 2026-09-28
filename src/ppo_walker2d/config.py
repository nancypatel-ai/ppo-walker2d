"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class TrainingConfig:
    total_steps: int = 100_000
    num_envs: int = 8
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    checkpoint_interval: int = 50_000
    device: str = "auto"


@dataclass
class ProjectConfig:
    name: str = "base"
    environment: str = "Walker2d-v5"
    seed: int = 42
    reward: str = "reward_v1"
    domain_randomization: bool = False
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML config and apply defaults for omitted fields."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    training = TrainingConfig(**raw.pop("training", {}))
    return ProjectConfig(training=training, **raw)


def seed_everything(seed: int) -> None:
    """Seed all local random number generators used by the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
