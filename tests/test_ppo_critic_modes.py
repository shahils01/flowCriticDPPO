import torch

from ppo.algorithms.ppo.algorithm.PPO import Critic, PPO
from ppo.algorithms.ppo.flow_gae import ValueFlowCritic


def test_ppo_uses_flow_critic_by_default():
    model = PPO(obs_shape=3, action_dim=2, n_embd=8, num_quants=5)

    values = model.get_values(torch.zeros(4, 3))

    assert isinstance(model.critic, ValueFlowCritic)
    assert values.shape == (4, 5)


def test_ppo_can_use_legacy_critic_by_flag():
    model = PPO(
        obs_shape=3,
        action_dim=2,
        n_embd=8,
        num_quants=5,
        critic_type="legacy",
    )

    values = model.get_values(torch.zeros(4, 3))

    assert isinstance(model.critic, Critic)
    assert values.shape == (4, 5)
