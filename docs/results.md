# Results

The bounded integration smoke test ran
Walker2d-v5 for 256 environment steps with seed 7 and produced
`checkpoints/smoke__seed7.pt`. This validates integration only. It is not a
learning result.

The bounded five-seed integration matrix completed for the three project PPO
configurations. Each run used 256 environment steps and evaluation used two
episodes. These values validate the experiment plumbing only, not convergence:

| Configuration | Seeds | Mean return | Standard deviation |
|---|---:|---:|---:|
| reward_v1 | 5 | 276.973 | 18.002 |
| reward_v2 | 5 | 163.429 | 10.459 |
| domain_rand | 5 | 279.601 | 20.264 |

The bounded external baseline smoke set used Stable-Baselines3 PPO for 256
steps and two evaluation episodes per seed. Across five seeds it reached a
mean return of 233.313 plus or minus 89.144. The wide spread and short budget
make this a pipeline check, not a publishable comparison.

The random-policy R1 lower bound was evaluated for 20 episodes across five
seeds. Mean return was 0.100 +/- 1.254.

The primary R3 run used 1,000,000 environment steps per seed.
Evaluation used 20 fixed-seed episodes per policy:

| Metric | Mean | Standard deviation | Seeds |
|---|---:|---:|---:|
| Return | 336.263 | 29.639 | 5 |
| Distance | 1.068 m | 0.223 m | 5 |
| Speed | 2.693 m/s | 0.781 m/s | 5 |
| Fall rate | 1.000 | 0.000 | 5 |
| Energy proxy | 835.032 | 104.080 | 5 |

The high fall rate means this run does not support a claim of successful
walking. The result is retained because it validates the corrected bounded
Gaussian policy and the long-horizon evaluation path.

A representative one-million-step R3 run with seed 42 produced mean return
312.604, mean speed 2.717 m/s, fall rate 1.000, and energy proxy 882.020 over
20 episodes. It also does not support a successful-walking claim. The
one-million-step checkpoint is attached to the v0.1.0 GitHub release.

The R4 reward-v2 and R5 domain-randomization runs used 100,000 environment
steps per seed and 20 evaluation episodes:

| Configuration | Return mean +/- std | Speed mean +/- std | Fall rate mean +/- std |
|---|---:|---:|---:|
| R4 reward-v2 | 71.352 +/- 9.885 | -1.187 +/- 0.152 m/s | 1.000 +/- 0.000 |
| R5 domain-rand | 191.843 +/- 30.355 | -0.818 +/- 0.461 m/s | 1.000 +/- 0.000 |

These runs also fail the walking criterion. Held-out dynamics evaluation has
not yet been run.

The full matrix still requires long-horizon R1, R2, R4, and R5 runs and three
seeds per learning-rate setting in S1. Report mean plus or minus standard
deviation and include the seed count.

Status: In progress
Duration: 2026-09-28 to 2026-09-28
Runs logged: 5 long-horizon R3 runs plus bounded baseline and ablation runs
Best mean return (R3, 5 seeds): 336.263 +/- 29.639
Best mean speed (R3, 5 seeds): 2.693 +/- 0.781 m/s
Robustness (R5 vs R3, held-out dynamics): not available
Reproducibility: smoke verified via direct bounded trainer invocation
Privacy: no telemetry, no user data, no external runtime calls
Decisions logged: 6 (see docs/decisions.md)
Open items: full experiment matrix, W&B logging, video assets, Hugging Face deployment
