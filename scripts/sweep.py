"""Small local learning-rate sweep entry point."""

import argparse

from ppo_walker2d.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    for learning_rate in (1e-4, 3e-4, 1e-3):
        print(f"{config.name}: learning_rate={learning_rate}")


if __name__ == "__main__":
    main()
