# Decisions

## 2026-09-28: Start with a minimal scaffold

The project begins with the requested directory contract, packaging metadata, CI, and a smoke test. Implementation details remain isolated until the scaffold passes local quality checks. This keeps the first commit reviewable and makes later failures attributable to implementation changes.

Alternatives considered: implementing the trainer before validating packaging and CI. Rejected because the project requires reproducible setup before experiments.
