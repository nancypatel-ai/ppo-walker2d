# ppo-walker2d

A reproducible PPO training pipeline for MuJoCo Walker2d.

This repository turns CS690K HW1 into a tested, config-driven PPO implementation.

The trainer uses Gymnasium, MuJoCo, and PyTorch. It supports deterministic seeds,
GAE, configurable reward shaping, domain randomization, checkpoints, and evaluation.

## Status

Phase 0 is complete. Training results will be added after the implementation and experiment phases.

## Quickstart

```sh
make test
make lint
make train config=configs/reward_v1.yaml seed=42
```

Run the test and quality checks with `make test` and `make lint`. Use
`make eval checkpoint=checkpoints/reward_v1__seed42.pt` after training.

## Experiments

The requested five-seed experiment matrix is defined in `docs/results.md`. Long
training runs are intentionally not checked into Git. Results must be generated
on the target machine and reported as mean plus or minus standard deviation.

## Privacy

The runtime does not collect telemetry or user data. W&B logging is not enabled
by default. Training works offline after dependencies are installed.

## License

MIT
