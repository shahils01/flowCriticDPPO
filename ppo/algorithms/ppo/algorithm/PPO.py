import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from ppo.algorithms.utils.util import check, init
from ppo.algorithms.utils.transformer_act import discrete_autoregreesive_act, discrete_decentralized_act
from ppo.algorithms.utils.transformer_act import discrete_parallel_act
from ppo.algorithms.utils.transformer_act import continuous_autoregreesive_act
from ppo.algorithms.utils.transformer_act import continuous_parallel_act
from ppo.algorithms.utils.transformer_act import continuous_moe_act, continuous_moe_eval
from ppo.algorithms.ppo.flow_gae import (
    FloQValueCritic,
    FlowFieldValueCritic,
    ValueFlowCritic,
    make_standard_normal_particles,
    make_uniform_particles,
    terminal_map_entropy_autograd,
)


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

def _to_tpdv(x, tpdv):
    if isinstance(x, dict):
        return {k: v.to(**tpdv) for k, v in x.items()}
    return x.to(**tpdv)

class Critic(nn.Module):

    def __init__(self, obs_shape, n_embd, device, num_quants):
        super(Critic, self).__init__()

        self.obs_shape = obs_shape
        self.n_embd = n_embd

        self.head_ = nn.ModuleList()
        for n in range(1):
            critic = nn.Sequential(nn.LayerNorm(obs_shape),
                                init_(nn.Linear(obs_shape, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, num_quants)))

            self.head_.append(critic)

    def forward(self, obs):
        # obs: (batch, 1, obs_dim)                
        v_loc = self.head_[0](obs)
        return v_loc


class Actor(nn.Module):

    def __init__(self, obs_shape, action_dim, n_embd, device, action_type='Discrete'):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
                        
        print('action_dim = ', action_dim)
        print('obs_shape = ', obs_shape)
        
        self.mlp_ = nn.ModuleList()
        for n in range(1):
            actor = nn.Sequential(nn.LayerNorm(obs_shape),
                                init_(nn.Linear(obs_shape, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                init_(nn.Linear(n_embd, action_dim)))

            self.mlp_.append(actor)

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, obs):
        logit = self.mlp_[0](obs)
        return logit


class FlowActor(nn.Module):
    """
    One-step conditional flow policy for continuous actions.

    The policy samples eps ~ N(0, I) and takes one reverse Euler step
    a = eps - v_theta(eps, t=1, obs). For PPO updates, it returns the
    FPO-style log-ratio proxy -L_CFM(a | obs), where the conditional
    flow-matching target is eps - a at x_t = t eps + (1 - t) a.
    """

    def __init__(
        self,
        obs_shape,
        action_dim,
        n_embd,
        max_velocity=5.0,
        base_std=0.35,
        loss_samples=8,
        output_scale=0.25,
    ):
        super(FlowActor, self).__init__()

        self.action_dim = action_dim
        self.max_velocity = max_velocity
        self.base_std = base_std
        self.loss_samples = loss_samples
        self.output_scale = output_scale

        self.net = nn.Sequential(
            nn.LayerNorm(obs_shape + action_dim + 1),
            init_(nn.Linear(obs_shape + action_dim + 1, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, action_dim)),
        )

    def velocity(self, obs, x_t, t):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.shape[:-1] != x_t.shape[:-1]:
            t = t.expand(*x_t.shape[:-1], 1)

        h = torch.cat([obs, x_t, t], dim=-1)
        v = self.output_scale * self.net(h)
        if self.max_velocity is not None and self.max_velocity > 0:
            v = self.max_velocity * torch.tanh(v / self.max_velocity)
        return v

    def sample(self, obs, deterministic=False):
        eps = torch.zeros(*obs.shape[:-1], self.action_dim, device=obs.device, dtype=obs.dtype)
        if not deterministic:
            eps = self.base_std * torch.randn_like(eps)
        t1 = torch.ones(*obs.shape[:-1], 1, device=obs.device, dtype=obs.dtype)
        return eps - self.velocity(obs, eps, t1)

    def flow_matching_loss(self, obs, action, loss_samples=None):
        loss_samples = self.loss_samples if loss_samples is None else loss_samples

        if loss_samples == 1:
            eps = self.base_std * torch.randn_like(action)
            t = torch.rand(*action.shape[:-1], 1, device=action.device, dtype=action.dtype)
            obs_expanded = obs
            action_expanded = action
        else:
            sample_shape = (loss_samples, *action.shape)
            eps = self.base_std * torch.randn(sample_shape, device=action.device, dtype=action.dtype)
            t = torch.rand(
                loss_samples,
                *action.shape[:-1],
                1,
                device=action.device,
                dtype=action.dtype,
            )
            obs_expanded = obs.unsqueeze(0).expand(loss_samples, *obs.shape)
            action_expanded = action.unsqueeze(0).expand(loss_samples, *action.shape)

        x_t = t * eps + (1.0 - t) * action_expanded
        target_velocity = eps - action_expanded
        velocity = self.velocity(obs_expanded, x_t, t)
        loss = 0.5 * (velocity - target_velocity).pow(2).mean(dim=-1, keepdim=True)
        if loss_samples == 1:
            return loss
        return loss.mean(dim=0)

    def flow_log_prob_proxy(self, obs, action, loss_samples=None):
        return -self.flow_matching_loss(obs, action, loss_samples=loss_samples)

    def action_log_prob_proxy_from_noise(self, obs, action, eps):
        t = torch.rand(*action.shape[:-1], 1, device=action.device, dtype=action.dtype)
        x_t = t * eps + (1.0 - t) * action
        target_velocity = eps - action
        velocity = self.velocity(obs, x_t, t)
        return -0.5 * (velocity - target_velocity).pow(2).mean(dim=-1, keepdim=True)


class PPO(nn.Module):

    def __init__(
        self,
        obs_shape,
        action_dim,
        n_embd,
        device=torch.device("cpu"),
        action_type='Discrete',
        num_quants=1,
        critic_type="direct",
        num_flow_steps=8,
        flow_integrator="euler",
        flow_particle_scale=0.05,
        flow_max_particle_scale=2.0,
        flow_max_velocity=5.0,
        flow_time_embed_dim=64,
        policy_type="gaussian",
        flow_policy_max_velocity=5.0,
        flow_policy_base_std=0.35,
        flow_policy_loss_samples=8,
        flow_policy_output_scale=0.25,
    ):
        super(PPO, self).__init__()

        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.policy_type = policy_type
        self.device = device
        self.n_embd = n_embd
        self.obs_shape = obs_shape
        self.num_quants = num_quants
        self.critic_type = "direct" if critic_type == "flow" else critic_type
   
        # Actor-Critic Networks
        if self.critic_type == "direct":
            self.critic = ValueFlowCritic(obs_shape, hidden_dim=n_embd)
            self.register_buffer("critic_particles", make_standard_normal_particles(num_quants, device))
        elif self.critic_type == "flow_field":
            self.critic = FlowFieldValueCritic(
                obs_shape,
                hidden_dim=n_embd,
                num_flow_steps=num_flow_steps,
                integrator=flow_integrator,
                particle_scale=flow_particle_scale,
                max_particle_scale=flow_max_particle_scale,
                max_velocity=flow_max_velocity,
            )
            self.register_buffer("critic_particles", make_standard_normal_particles(num_quants, device))
        elif self.critic_type == "floq":
            self.critic = FloQValueCritic(
                obs_shape,
                hidden_dim=n_embd,
                num_flow_steps=num_flow_steps,
                integrator=flow_integrator,
                max_velocity=flow_max_velocity,
                time_embed_dim=flow_time_embed_dim,
            )
            self.register_buffer(
                "critic_particles",
                flow_particle_scale
                * make_uniform_particles(num_quants, device, low=-1.0, high=1.0),
            )
        elif self.critic_type == "legacy":
            self.critic = Critic(obs_shape, n_embd, device, num_quants)
            self.register_buffer("critic_particles", make_standard_normal_particles(num_quants, device))
        else:
            raise ValueError(f"Unknown critic_type: {critic_type}")
        if self.action_type == "Discrete" and self.policy_type != "gaussian":
            raise ValueError("Flow policies are only supported for continuous action spaces")
        if self.policy_type == "gaussian":
            self.actor = Actor(obs_shape, action_dim, n_embd, device, self.action_type)
        elif self.policy_type == "flow":
            self.actor = FlowActor(
                obs_shape,
                action_dim,
                n_embd,
                max_velocity=flow_policy_max_velocity,
                base_std=flow_policy_base_std,
                loss_samples=flow_policy_loss_samples,
                output_scale=flow_policy_output_scale,
            )
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        # self.value_entropy_weight = torch.nn.Parameter(torch.ones(1)/2)
   
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete' and self.policy_type == "gaussian":
            self.actor.zero_std(self.device)

    def critic_values(self, obs):
        if self.critic_type in {"direct", "flow_field", "floq"}:
            return self.critic(obs, self.critic_particles)
        return self.critic(obs)

    def critic_values_and_entropy(self, obs, create_graph=False):
        if self.critic_type not in {"direct", "flow_field", "floq"}:
            values = self.critic_values(obs)
            entropy = torch.zeros_like(values[..., :1])
            return values, entropy

        with torch.enable_grad():
            particle_shape = (*obs.shape[:-1], self.critic_particles.numel())
            z = self.critic_particles.expand(particle_shape).clone().detach()
            z = z.requires_grad_(True)
            values = self.critic(obs, z)
            entropy = terminal_map_entropy_autograd(
                values,
                z,
                create_graph=create_graph,
            )
        return values, entropy

    def critic_flow_matching_loss(self, obs, returns):
        if self.critic_type != "floq":
            raise RuntimeError("critic_flow_matching_loss is only available for critic_type='floq'")
        return self.critic.flow_matching_loss(obs, self.critic_particles, returns)

    def forward(self, obs, action, gate_entropy=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)
        obs = _to_tpdv(check(obs), self.tpdv)
        action = check(action).to(**self.tpdv)
        batch_size = np.shape(obs)[0]

        v_loc = self.critic_values(obs)

        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)
        elif self.policy_type == "gaussian":
            action_log, entropy = continuous_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)
        else:
            action_log = self.actor.flow_log_prob_proxy(obs, action)
            entropy = torch.zeros_like(action_log)

        return action_log, v_loc, entropy, gate_entropy

    def get_actions(self, obs):
        obs = _to_tpdv(check(obs), self.tpdv)
        batch_size = np.shape(obs)[0]

        v_loc = self.critic_values(obs)
        
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_decentralized_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)
        elif self.policy_type == "gaussian":
            output_action, output_action_log = continuous_autoregreesive_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)
        else:
            output_action = self.actor.sample(obs)
            output_action_log = self.actor.flow_log_prob_proxy(obs, output_action)

        return output_action, output_action_log, v_loc

    def get_values(self, obs):
        obs = _to_tpdv(check(obs), self.tpdv)
        v_tot = self.critic_values(obs)
        return v_tot

    def get_values_and_entropy(self, obs):
        obs = _to_tpdv(check(obs), self.tpdv)
        return self.critic_values_and_entropy(obs, create_graph=False)
