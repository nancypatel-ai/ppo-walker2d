"""PPO optimization and checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ppo_walker2d.config import ProjectConfig, resolve_device
from ppo_walker2d.logging import JsonlLogger
from ppo_walker2d.models.actor_critic import ActorCritic
from ppo_walker2d.ppo.gae import compute_gae
from ppo_walker2d.ppo.rollout import collect_rollout


def update_policy(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Any,
    last_value: torch.Tensor,
    config: ProjectConfig,
) -> dict[str, float]:
    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.values,
        rollout.terminated,
        last_value,
        config.training.gamma,
        config.training.gae_lambda,
    )
    observations = rollout.observations.reshape(-1, rollout.observations.shape[-1])
    actions = rollout.actions.reshape(-1, rollout.actions.shape[-1])
    old_log_probs = rollout.log_probabilities.reshape(-1)
    returns = returns.flatten()
    advantages = advantages.flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    total = observations.shape[0]
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    for _ in range(config.training.update_epochs):
        indices = torch.randperm(total, device=observations.device)
        for batch in indices.split(config.training.minibatch_size):
            log_probs, entropy, values = policy.evaluate_actions(
                observations[batch], actions[batch]
            )
            ratio = (log_probs - old_log_probs[batch]).exp()
            unclipped = ratio * advantages[batch]
            clipped = ratio.clamp(
                1.0 - config.training.clip_coef, 1.0 + config.training.clip_coef
            ) * advantages[batch]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[batch] - values).pow(2).mean()
            loss = (
                policy_loss
                + config.training.value_coef * value_loss
                - config.training.entropy_coef * entropy.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), config.training.max_grad_norm
            )
            optimizer.step()
            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy.mean().detach())
            metrics["approx_kl"] += float(
                (old_log_probs[batch] - log_probs).mean().detach()
            )
    return metrics


def save_checkpoint(
    path: str | Path,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    steps: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "steps": steps,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def train(config: ProjectConfig, environment: Any) -> Path:
    """Train a policy and return the best available checkpoint path."""
    device = resolve_device(config.training.device)
    observation, _ = environment.reset(seed=config.seed)
    policy = ActorCritic(
        int(observation.shape[-1]),
        int(environment.action_space.shape[-1]),
        config.training.hidden_sizes,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.training.learning_rate)
    total_steps = 0
    checkpoint = Path("checkpoints") / f"{config.name}__seed{config.seed}.pt"
    logger = JsonlLogger(Path("outputs") / f"{config.name}__seed{config.seed}.jsonl")
    while total_steps < config.training.total_steps:
        rollout = collect_rollout(
            environment,
            policy,
            observation,
            config.training.rollout_steps,
            device,
        )
        observation = rollout.last_observation
        with torch.no_grad():
            _, last_value = policy(
                torch.as_tensor(observation, dtype=torch.float32, device=device)
            )
        metrics = update_policy(policy, optimizer, rollout, last_value, config)
        total_steps += config.training.rollout_steps
        logger.write({"step": total_steps, **metrics})
        if total_steps >= config.training.total_steps:
            save_checkpoint(checkpoint, policy, optimizer, config, total_steps)
    logger.close()
    return checkpoint
