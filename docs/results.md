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

The R2 Stable-Baselines3 PPO baseline used 100,000 training steps and 20
evaluation episodes for each of five seeds. Mean return was 332.125 +/- 75.971.
This external implementation outperformed the current project policy at the
same bounded budget, which is an actionable baseline result.

The primary R3 run used 1,000,000 environment steps per seed.
Evaluation used 20 fixed-seed episodes per policy:

| Metric | Mean | Standard deviation | Seeds |
|---|---:|---:|---:|
| Return | 563.007 | 44.308 | 5 |
| Distance | 2.365 m | 0.114 m | 5 |
| Speed | 4.566 m/s | 0.719 m/s | 5 |
| Fall rate | 1.000 | 0.000 | 5 |
| Energy proxy | 1126.413 | 118.890 | 5 |

The high fall rate means this run does not support a claim of successful
walking. The result is retained because it validates the corrected bounded
Gaussian policy and the long-horizon evaluation path.

A representative pre-correction seed-42 checkpoint produced mean return
312.604, but it is retained only as a historical release artifact. The
corrected five-seed result above is the authoritative R3 measurement.

The R4 reward-v2 and R5 domain-randomization runs used 100,000 environment
steps per seed and 20 evaluation episodes:

| Configuration | Return mean +/- std | Speed mean +/- std | Fall rate mean +/- std |
|---|---:|---:|---:|
| R4 reward-v2 | 102.872 +/- 20.617 | -0.206 +/- 0.408 m/s | 1.000 +/- 0.000 |
| R5 domain-rand | 205.739 +/- 16.334 | -0.358 +/- 0.267 m/s | 1.000 +/- 0.000 |

These corrected runs also fail the walking criterion. Held-out R5 evaluation
on the corrected checkpoints produced 238.511 +/- 59.708 for low mass and
friction, 147.220 +/- 48.948 for nominal dynamics, and 118.962 +/- 14.053 for
high mass and friction. Relative to the corrected R3 mean return, the nominal
held-out delta was -415.787.

The corrected S1 learning-rate sweep used three rates, three seeds per rate,
and 100,000 environment steps per run. Evaluation used five episodes per
checkpoint:

| Learning rate | Return mean +/- std | Seeds |
|---:|---:|---:|
| 0.0001 | 294.227 +/- 16.441 | 3 |
| 0.0003 | 248.995 +/- 12.985 | 3 |
| 0.0010 | 232.963 +/- 39.693 | 3 |

All S1 policies had fall rate 1.000. The sweep therefore selects no validated
walking hyperparameter.

The requested matrix is represented by measured R1 through R5 runs and S1.
The corrected R3 five-seed rerun uses the vector-reset fix from 2026-09-29.
Budgets differ by configuration: R3 used 1,000,000 steps per seed, while R2,
R4, R5, and S1 used 100,000 steps per run. Results do not support a claim of
successful walking.

Status: In progress
Duration: 2026-09-28 to 2026-09-28
Runs logged: 34 local runs across R1 through R5 and S1; W&B runs: 0
Best mean return (R3, 5 seeds): 563.007 +/- 44.308
Best mean speed (R3, 5 seeds): 4.566 +/- 0.719 m/s
Robustness (R5 vs R3, held-out dynamics): -415.787 nominal return delta
Reproducibility: smoke verified via direct bounded trainer invocation
Privacy: no telemetry, no user data, no external runtime calls
Decisions logged: 6 (see docs/decisions.md)
Open items: public W&B runs, Hugging Face deployment, real policy demo video, successful walking policy
