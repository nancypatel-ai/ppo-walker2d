"""Record a checkpoint rollout as a local video."""

import argparse

import gymnasium as gym
import torch

from ppo_walker2d.config import load_config
from ppo_walker2d.models.actor_critic import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="assets/rollout")
    args = parser.parse_args()
    config = load_config(args.config)
    environment = gym.make(config.environment, render_mode="rgb_array")
    assert environment.observation_space.shape is not None
    assert environment.action_space.shape is not None
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = ActorCritic(
        environment.observation_space.shape[0],
        environment.action_space.shape[0],
        config.training.hidden_sizes,
    )
    policy.load_state_dict(checkpoint["model"])
    recorded = gym.wrappers.RecordVideo(
        environment, args.output, episode_trigger=lambda _: True
    )
    observation, _ = recorded.reset(seed=config.seed)
    for _ in range(1000):
        with torch.no_grad():
            action, _ = policy(torch.as_tensor(observation, dtype=torch.float32))
        observation, _, terminated, truncated, _ = recorded.step(action.numpy())
        if terminated or truncated:
            break
    recorded.close()


if __name__ == "__main__":
    main()
