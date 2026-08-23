"""Shared PPO actor with one of three explicitly selected value methods."""

import copy
import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal

from ppo.algorithms.ppo.flow_gae import terminal_map_entropy_length
from ppo.algorithms.ppo.value_critics import (
    FlowMatchingValueCritic,
    QuantileValueCritic,
    ScalarValueCritic,
    make_uniform_particles,
)
from ppo.algorithms.utils.util import check, init


VALUE_METHODS = {"scalar", "wasserstein", "flow"}


def _init_layer(layer, gain=0.01):
    return init(
        layer,
        nn.init.orthogonal_,
        lambda bias: nn.init.constant_(bias, 0),
        gain=gain,
    )


def _to_device(value, tensor_kwargs):
    if isinstance(value, dict):
        return {key: item.to(**tensor_kwargs) for key, item in value.items()}
    return value.to(**tensor_kwargs)


class Actor(nn.Module):
    """Categorical or diagonal-Gaussian PPO actor."""

    def __init__(self, obs_dim, action_dim, hidden_dim, action_type="Discrete"):
        super().__init__()
        self.action_dim = action_dim
        self.action_type = action_type
        if action_type != "Discrete":
            # A single state-independent diagonal covariance is shared by all
            # three methods so only the value estimator changes between runs.
            self.log_std = nn.Parameter(
                torch.full((action_dim,), math.log(0.35), dtype=torch.float32)
            )

        self.net = nn.Sequential(
            nn.LayerNorm(obs_dim),
            _init_layer(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            _init_layer(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2.0)),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            _init_layer(nn.Linear(hidden_dim, action_dim)),
        )

    def forward(self, observations):
        return self.net(observations)

    def distribution(self, observations):
        outputs = self(observations)
        if self.action_type == "Discrete":
            return Categorical(logits=outputs)
        std = self.log_std.exp().expand_as(outputs)
        return Independent(Normal(outputs, std), 1)


class PPO(nn.Module):
    """PPO network with scalar, inverse-CDF, or flow-matching value critic."""

    def __init__(
        self,
        obs_shape,
        action_dim,
        n_embd,
        device=torch.device("cpu"),
        action_type="Discrete",
        value_method="scalar",
        num_value_particles=16,
        num_flow_steps=16,
        flow_integrator="euler",
        flow_particle_scale=0.25,
        flow_max_velocity=15.0,
        flow_time_embed_dim=64,
        flow_entropy_eps=1e-6,
    ):
        super().__init__()
        if value_method not in VALUE_METHODS:
            raise ValueError(
                f"value_method must be one of {sorted(VALUE_METHODS)}, got {value_method}"
            )
        if value_method != "scalar" and num_value_particles < 2:
            raise ValueError("Distributional value methods require at least two particles")
        if value_method == "flow" and flow_particle_scale <= 0.0:
            raise ValueError("flow_particle_scale must be positive")
        if value_method == "flow" and flow_entropy_eps <= 0.0:
            raise ValueError("flow_entropy_eps must be positive")

        self.action_dim = action_dim
        self.action_type = action_type
        self.device = device
        self.value_method = value_method
        self.num_value_particles = 1 if value_method == "scalar" else num_value_particles
        self.flow_entropy_eps = flow_entropy_eps
        self.tensor_kwargs = dict(dtype=torch.float32, device=device)

        self.actor = Actor(obs_shape, action_dim, n_embd, action_type)
        if value_method == "scalar":
            self.critic = ScalarValueCritic(obs_shape, n_embd)
            self.target_critic = None
        elif value_method == "wasserstein":
            self.critic = QuantileValueCritic(
                obs_shape, n_embd, self.num_value_particles
            )
            self.target_critic = copy.deepcopy(self.critic)
        else:
            self.critic = FlowMatchingValueCritic(
                state_dim=obs_shape,
                hidden_dim=n_embd,
                num_flow_steps=num_flow_steps,
                integrator=flow_integrator,
                max_velocity=flow_max_velocity,
                time_embed_dim=flow_time_embed_dim,
            )
            self.target_critic = copy.deepcopy(self.critic)
            self.register_buffer(
                "flow_base_particles",
                make_uniform_particles(
                    self.num_value_particles,
                    device=device,
                    scale=flow_particle_scale,
                ),
            )

        if self.target_critic is not None:
            for parameter in self.target_critic.parameters():
                parameter.requires_grad_(False)
        self.to(device)

    def critic_values(self, observations, use_target=False):
        critic = self.target_critic if use_target else self.critic
        if critic is None:
            raise RuntimeError("The scalar method does not use a target critic")
        if self.value_method == "flow":
            return critic(observations, self.flow_base_particles)
        return critic(observations)

    def critic_values_and_entropy_length(self, observations, use_target=False):
        if self.value_method != "flow":
            raise RuntimeError("Entropy length is only defined for the flow critic")
        critic = self.target_critic if use_target else self.critic
        with torch.enable_grad():
            particle_shape = (
                *observations.shape[:-1],
                self.flow_base_particles.numel(),
            )
            base = self.flow_base_particles.expand(particle_shape)
            base = base.clone().detach().requires_grad_(True)
            values = critic(observations, base)
            entropy_length = terminal_map_entropy_length(
                values,
                base,
                eps=self.flow_entropy_eps,
                create_graph=False,
            )
        return values, entropy_length

    @torch.no_grad()
    def update_target_critic(self, tau):
        if self.target_critic is None:
            return
        if not 0.0 < tau <= 1.0:
            raise ValueError("target critic tau must lie in (0, 1]")
        for target_parameter, source_parameter in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_parameter.lerp_(source_parameter, tau)

    def critic_flow_matching_loss(self, observations, target_particles):
        if self.value_method != "flow":
            raise RuntimeError("Flow-matching loss requires value_method='flow'")
        return self.critic.flow_matching_loss(
            observations, self.flow_base_particles, target_particles
        )

    def evaluate_actions(self, observations, actions, compute_values=True):
        observations = _to_device(check(observations), self.tensor_kwargs)
        actions = check(actions).to(**self.tensor_kwargs)
        distribution = self.actor.distribution(observations)
        if self.action_type == "Discrete":
            actions = actions.long().squeeze(-1)
        action_log_probs = distribution.log_prob(actions).unsqueeze(-1)
        entropy = distribution.entropy().unsqueeze(-1)
        values = self.critic_values(observations) if compute_values else None
        return action_log_probs, values, entropy

    def get_actions(self, observations):
        observations = _to_device(check(observations), self.tensor_kwargs)
        values = self.critic_values(observations)
        distribution = self.actor.distribution(observations)
        actions = distribution.sample()
        if self.action_type == "Discrete":
            actions = actions.unsqueeze(-1)
        action_log_probs = distribution.log_prob(
            actions.squeeze(-1) if self.action_type == "Discrete" else actions
        ).unsqueeze(-1)
        return actions, action_log_probs, values

    def act(self, observations, deterministic=False):
        observations = _to_device(check(observations), self.tensor_kwargs)
        distribution = self.actor.distribution(observations)
        if deterministic:
            if self.action_type == "Discrete":
                actions = distribution.probs.argmax(dim=-1)
            else:
                actions = distribution.base_dist.loc
        else:
            actions = distribution.sample()
        return actions.unsqueeze(-1) if self.action_type == "Discrete" else actions

    def get_values(self, observations, use_target=False):
        observations = _to_device(check(observations), self.tensor_kwargs)
        return self.critic_values(observations, use_target=use_target)

    def get_values_and_entropy_length(self, observations, use_target=False):
        observations = _to_device(check(observations), self.tensor_kwargs)
        return self.critic_values_and_entropy_length(
            observations, use_target=use_target
        )
