"""Evaluate random and optional Stable-Baselines3 PPO baselines."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from ppo_walker2d.config import load_config, seed_everything
from ppo_walker2d.envs.wrappers import make_env
from ppo_walker2d.logging import WandbLogger


def evaluate_random(environment: Any, episodes: int, seed: int) -> dict[str, float]:
    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            observation, reward, terminated, truncated, _ = environment.step(
                environment.action_space.sample()
            )
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


def evaluate_sb3(
    model: Any, environment: Any, episodes: int, seed: int
) -> dict[str, float]:
    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = environment.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sb3", action="store_true")
    parser.add_argument("--total-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.total_steps is not None:
        config.training.total_steps = args.total_steps
    seed_everything(args.seed)
    environment = make_env(config.environment, args.seed)
    result = evaluate_random(environment, args.episodes, args.seed)
    environment.close()
    logger = WandbLogger(
        os.getenv("WANDB_PROJECT"), f"random__seed{args.seed}"
    )
    logger.write(result)
    logger.close()
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/random_baseline.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(result)
    if args.sb3:
        try:
            from stable_baselines3 import PPO
        except ImportError as error:
            raise SystemExit("Install the baselines extra first.") from error
        sb3_environment = gym.make(config.environment)
        sb3_environment.reset(seed=args.seed)
        model = PPO(
            "MlpPolicy",
            sb3_environment,
            n_steps=128,
            batch_size=64,
            seed=args.seed,
            verbose=0,
        )
        model.learn(total_timesteps=config.training.total_steps)
        sb3_result = evaluate_sb3(model, sb3_environment, args.episodes, args.seed)
        sb3_environment.close()
        logger = WandbLogger(
            os.getenv("WANDB_PROJECT"), f"sb3__seed{args.seed}"
        )
        logger.write(sb3_result)
        logger.close()
        print({"sb3": sb3_result})
    else:
        print("Install the baselines extra to run the Stable-Baselines3 comparison.")


if __name__ == "__main__":
    main()
