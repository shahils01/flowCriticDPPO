from types import SimpleNamespace

import torch

from ppo.algorithms.ppo.algorithm.PPO import Critic, PPO
from ppo.algorithms.ppo.algorithm.ppo_policy import PPO_Policy
from ppo.algorithms.ppo.flow_gae import FloQValueCritic, FlowFieldValueCritic, ValueFlowCritic


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def _policy_args(lr=3e-4, critic_lr=1e-4):
    return SimpleNamespace(
        algorithm_name="ppo",
        lr=lr,
        critic_lr=critic_lr,
        opti_eps=1e-5,
        weight_decay=0.0,
        use_policy_active_masks=True,
        n_embd=8,
        critic_type="flow_field",
        num_flow_steps=2,
        flow_integrator="euler",
        flow_particle_scale=0.05,
        flow_max_particle_scale=2.0,
        flow_max_velocity=5.0,
        flow_time_embed_dim=8,
    )


def test_ppo_uses_direct_transport_critic_by_default():
    model = PPO(obs_shape=3, action_dim=2, n_embd=8, num_quants=5)

    values = model.get_values(torch.zeros(4, 3))

    assert isinstance(model.critic, ValueFlowCritic)
    assert values.shape == (4, 5)


def test_ppo_can_use_flow_field_critic_by_flag():
    model = PPO(
        obs_shape=3,
        action_dim=2,
        n_embd=8,
        num_quants=5,
        critic_type="flow_field",
        num_flow_steps=3,
        flow_integrator="rk4",
    )

    values = model.get_values(torch.zeros(4, 3))

    assert isinstance(model.critic, FlowFieldValueCritic)
    assert model.critic.num_flow_steps == 3
    assert model.critic.integrator == "rk4"
    assert values.shape == (4, 5)


def test_ppo_can_use_floq_critic_by_flag():
    model = PPO(
        obs_shape=3,
        action_dim=2,
        n_embd=8,
        num_quants=5,
        critic_type="floq",
        num_flow_steps=3,
        flow_particle_scale=0.1,
        flow_time_embed_dim=8,
    )

    values = model.get_values(torch.zeros(4, 3))
    flow_loss = model.critic_flow_matching_loss(torch.zeros(4, 3), torch.ones(4, 5))

    assert isinstance(model.critic, FloQValueCritic)
    assert not hasattr(model.critic, "base_head")
    assert torch.allclose(model.critic_particles, torch.tensor([-0.08, -0.04, 0.0, 0.04, 0.08]))
    assert values.shape == (4, 5)
    assert flow_loss.shape == (4, 5)


def test_policy_optimizer_uses_separate_critic_lr():
    policy = PPO_Policy(
        _policy_args(lr=3e-4, critic_lr=1e-4),
        obs_space=Box((3,)),
        act_space=Discrete(2),
        num_quants=5,
    )

    assert policy.optimizer.param_groups[0]["lr"] == 3e-4
    assert policy.optimizer.param_groups[1]["lr"] == 1e-4

    policy.lr_decay(episode=5, episodes=10)

    assert policy.optimizer.param_groups[0]["lr"] == 1.5e-4
    assert policy.optimizer.param_groups[1]["lr"] == 5e-5


def test_ppo_keeps_flow_as_direct_alias():
    model = PPO(obs_shape=3, action_dim=2, n_embd=8, num_quants=5, critic_type="flow")

    assert model.critic_type == "direct"
    assert isinstance(model.critic, ValueFlowCritic)


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
