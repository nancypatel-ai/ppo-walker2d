"""Evaluation entry point."""

import argparse

import torch

from ppo_walker2d.config import load_config
from ppo_walker2d.envs.wrappers import make_env
from ppo_walker2d.eval.evaluate import evaluate
from ppo_walker2d.models.actor_critic import ActorCritic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    environment = make_env(config.environment, config.seed, config.reward)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = ActorCritic(
        environment.observation_space.shape[0],
        environment.action_space.shape[0],
        config.training.hidden_sizes,
    )
    policy.load_state_dict(checkpoint["model"])
    metrics = evaluate(environment, policy)
    metrics.save("checkpoints/metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
