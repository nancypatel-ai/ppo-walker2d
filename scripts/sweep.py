"""Run the declared local experiment matrix."""

import argparse
import json
from pathlib import Path

from ppo_walker2d.config import load_config, seed_everything
from ppo_walker2d.envs.wrappers import make_vector_env
from ppo_walker2d.ppo.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--total-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.total_steps is not None:
        config.training.total_steps = args.total_steps
    Path("outputs").mkdir(exist_ok=True)
    result_path = Path("outputs") / "sweep_manifest.jsonl"
    with result_path.open("a", encoding="utf-8") as handle:
        for seed in args.seeds:
            seed_everything(seed)
            config.seed = seed
            environment = make_vector_env(
                config.environment,
                config.training.num_envs,
                seed,
                config.reward,
                config.domain_randomization,
            )
            checkpoint = train(config, environment)
            environment.close()
            handle.write(
                json.dumps(
                    {"config": config.name, "seed": seed, "checkpoint": str(checkpoint)}
                )
                + "\n"
            )
            print(checkpoint)

        if config.name == "base":
            for learning_rate in (1e-4, 3e-4, 1e-3):
                print(f"S1 learning_rate={learning_rate}")


if __name__ == "__main__":
    main()
