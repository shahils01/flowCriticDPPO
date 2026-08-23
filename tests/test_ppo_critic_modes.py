from types import SimpleNamespace

import torch

from ppo.algorithms.ppo.algorithm.PPO import Actor, PPO
from ppo.algorithms.ppo.algorithm.ppo_policy import PPO_Policy
from ppo.algorithms.ppo.value_critics import (
    FlowMatchingValueCritic,
    QuantileValueCritic,
    ScalarValueCritic,
)


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def _policy_args(value_method="scalar", lr=3e-4, critic_lr=1e-4):
    return SimpleNamespace(
        lr=lr,
        critic_lr=critic_lr,
        opti_eps=1e-5,
        weight_decay=0.0,
        use_policy_active_masks=True,
        n_embd=8,
        value_method=value_method,
        num_value_particles=5,
        num_flow_steps=2,
        flow_integrator="euler",
        flow_particle_scale=0.1,
        flow_max_velocity=5.0,
        flow_time_embed_dim=8,
        flow_entropy_eps=1e-6,
    )


def test_scalar_flag_selects_scalar_critic():
    model = PPO(
        obs_shape=3,
        action_dim=2,
        n_embd=8,
        value_method="scalar",
    )
    values = model.get_values(torch.zeros(4, 3))
    assert isinstance(model.actor, Actor)
    assert isinstance(model.critic, ScalarValueCritic)
    assert model.target_critic is None
    assert values.shape == (4, 1)


def test_wasserstein_flag_selects_inverse_cdf_critic():
    model = PPO(
        obs_shape=3,
        action_dim=2,
        n_embd=8,
        value_method="wasserstein",
        num_value_particles=5,
    )
    values = model.get_values(torch.zeros(4, 3))
    assert isinstance(model.critic, QuantileValueCritic)
    assert isinstance(model.target_critic, QuantileValueCritic)
    assert values.shape == (4, 5)
    assert torch.all(values[..., 1:] >= values[..., :-1])


def test_flow_flag_selects_only_flow_matching_critic():
    model = PPO(
        obs_shape=3,
        action_dim=2,
        n_embd=8,
        value_method="flow",
        num_value_particles=5,
        num_flow_steps=3,
        flow_particle_scale=0.1,
        flow_time_embed_dim=8,
    )
    states = torch.zeros(4, 3)
    values = model.get_values(states)
    loss = model.critic_flow_matching_loss(states, torch.ones(4, 5))
    assert isinstance(model.critic, FlowMatchingValueCritic)
    assert isinstance(model.target_critic, FlowMatchingValueCritic)
    assert values.shape == (4, 5)
    assert loss.shape == (4, 5)
    assert torch.allclose(
        model.flow_base_particles,
        torch.tensor([-0.08, -0.04, 0.0, 0.04, 0.08]),
    )


def test_distributional_target_critic_is_frozen_and_polyak_updated():
    model = PPO(
        obs_shape=3,
        action_dim=2,
        n_embd=8,
        value_method="wasserstein",
        num_value_particles=5,
    )
    assert all(not parameter.requires_grad for parameter in model.target_critic.parameters())
    source = next(model.critic.parameters())
    target = next(model.target_critic.parameters())
    old_target = target.detach().clone()
    with torch.no_grad():
        source.add_(1.0)
    model.update_target_critic(0.25)
    assert torch.allclose(target, old_target + 0.25)


def test_policy_optimizer_uses_separate_actor_and_critic_rates():
    policy = PPO_Policy(
        _policy_args("flow", lr=3e-4, critic_lr=1e-4),
        obs_space=Box((3,)),
        act_space=Discrete(2),
    )
    assert policy.optimizer.param_groups[0]["lr"] == 3e-4
    assert policy.optimizer.param_groups[1]["lr"] == 1e-4
    policy.lr_decay(episode=5, episodes=10)
    assert policy.optimizer.param_groups[0]["lr"] == 1.5e-4
    assert policy.optimizer.param_groups[1]["lr"] == 5e-5


def test_invalid_value_method_is_rejected():
    try:
        PPO(obs_shape=3, action_dim=2, n_embd=8, value_method="ambiguous")
    except ValueError as error:
        assert "value_method" in str(error)
    else:
        raise AssertionError("invalid value method was accepted")
