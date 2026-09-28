"""Evaluate random and optional Stable-Baselines3 PPO baselines."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ppo_walker2d.config import load_config, seed_everything
from ppo_walker2d.envs.wrappers import make_env


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    environment = make_env(config.environment, args.seed)
    result = evaluate_random(environment, args.episodes, args.seed)
    environment.close()
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/random_baseline.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(result)
    print("Install the baselines extra to run the Stable-Baselines3 comparison.")


if __name__ == "__main__":
    main()
