"""Reward tests."""

from ppo_walker2d.envs.reward import reward_v1, reward_v2


def test_reward_v1_is_deterministic() -> None:
    assert reward_v1(2.0, 0.5, True) == 2.5
    assert reward_v1(2.0, 0.5, True) == reward_v1(2.0, 0.5, True)


def test_reward_v2_bounds_forward_term() -> None:
    assert reward_v2(10.0, 0.0, True) == 1.5
