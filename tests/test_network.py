"""Network tests."""

import torch

from ppo_walker2d.models.actor_critic import ActorCritic


def test_actor_critic_shapes() -> None:
    policy = ActorCritic(17, 6, (32, 32))
    actions, values = policy(torch.zeros(4, 17))
    assert actions.shape == (4, 6)
    assert values.shape == (4,)
    log_probability, entropy, evaluated_values = policy.evaluate_actions(
        torch.zeros(4, 17), torch.zeros(4, 6)
    )
    assert log_probability.shape == (4,)
    assert entropy.shape == (4,)
    assert evaluated_values.shape == (4,)
