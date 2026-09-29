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

## 2026-09-28: Keep external integrations optional

W&B, Stable-Baselines3, and Gradio are optional dependency groups. The core
trainer remains offline and installable without external service credentials.
This preserves the privacy requirement while keeping comparison and demo paths
available to a user who opts into them.

## 2026-09-28: Use offscreen MuJoCo rendering in the recorder

The recorder uses MuJoCo's renderer rather than MoviePy or a hosted service.
This keeps video generation local. macOS CoreGraphics may require an active
graphics session; Linux CI should use an EGL-capable runtime. The training and
evaluation paths do not depend on rendering.

## 2026-09-28: Make W&B and Spaces adapters opt-in

The code exposes an optional W&B logger and a root Spaces entry point, but the
default trainer remains offline. This avoids silently transmitting metrics.
Publishing a public W&B project and Space still requires account credentials.
