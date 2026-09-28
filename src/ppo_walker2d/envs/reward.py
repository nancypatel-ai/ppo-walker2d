"""Deterministic reward shaping functions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardTerms:
    forward_velocity: float
    healthy_bonus: float
    control_cost: float
    alive_penalty: float

    @property
    def total(self) -> float:
        return (
            self.forward_velocity
            + self.healthy_bonus
            - self.control_cost
            - self.alive_penalty
        )


def reward_v1(
    forward_velocity: float,
    control_cost: float,
    is_healthy: bool,
) -> float:
    """Baseline shaped reward used by the primary experiment."""
    return RewardTerms(
        forward_velocity=forward_velocity,
        healthy_bonus=1.0 if is_healthy else 0.0,
        control_cost=control_cost,
        alive_penalty=0.0 if is_healthy else 1.0,
    ).total


def reward_v2(
    forward_velocity: float,
    control_cost: float,
    is_healthy: bool,
) -> float:
    """More conservative shaping with bounded forward incentive."""
    bounded_velocity = max(-1.0, min(1.0, forward_velocity))
    return RewardTerms(
        forward_velocity=bounded_velocity,
        healthy_bonus=0.5 if is_healthy else 0.0,
        control_cost=0.5 * control_cost,
        alive_penalty=1.0 if not is_healthy else 0.0,
    ).total
