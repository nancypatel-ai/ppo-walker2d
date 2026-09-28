# Results

No multi-seed experiments have run yet. The bounded integration smoke test ran
Walker2d-v5 for 256 environment steps with seed 7 and produced
`checkpoints/smoke__seed7.pt`. This validates integration only. It is not a
learning result.

The full matrix requires five seeds for each R1 through R5 run and three seeds
for each learning-rate setting in S1. Report mean plus or minus standard
deviation and include the seed count.

Status: In progress
Duration: 2026-09-28 to 2026-09-28
Runs logged: 0
Best mean return (R3, 5 seeds): not available
Best mean speed (R3, 5 seeds): not available
Robustness (R5 vs R3, held-out dynamics): not available
Reproducibility: smoke verified via direct bounded trainer invocation
Privacy: no telemetry, no user data, no external runtime calls
Decisions logged: 3 (see docs/decisions.md)
Open items: full experiment matrix, W&B logging, video assets, Hugging Face deployment
