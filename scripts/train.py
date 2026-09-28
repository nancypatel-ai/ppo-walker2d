"""Training entry point."""

import argparse

from ppo_walker2d.config import load_config, seed_everything
from ppo_walker2d.envs.wrappers import make_vector_env
from ppo_walker2d.ppo.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.total_steps is not None:
        config.training.total_steps = args.total_steps
    seed_everything(config.seed)
    environment = make_vector_env(
        config.environment,
        config.training.num_envs,
        config.seed,
        config.reward,
        config.domain_randomization,
    )
    print(train(config, environment))
    environment.close()


if __name__ == "__main__":
    main()
