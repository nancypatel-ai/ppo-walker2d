# Decisions

## 2026-09-28: Start with a minimal scaffold

The project begins with the requested directory contract, packaging metadata, CI, and a smoke test. Implementation details remain isolated until the scaffold passes local quality checks. This keeps the first commit reviewable and makes later failures attributable to implementation changes.

Alternatives considered: implementing the trainer before validating packaging and CI. Rejected because the project requires reproducible setup before experiments.

## 2026-09-28: Keep W&B optional

The core trainer does not require W&B credentials or network access. This keeps
local execution private and reproducible. A future experiment runner can add
W&B as an optional integration after credentials are supplied.

## 2026-09-28: Use a bounded smoke test before long runs

The smoke test uses 256 steps and seed 7. It validates MuJoCo integration and
checkpoint serialization without presenting it as evidence of learning.
