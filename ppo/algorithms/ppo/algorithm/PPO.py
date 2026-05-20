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
    ):
        super(PPO, self).__init__()

        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
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
        self.actor = Actor(obs_shape, action_dim, n_embd, device, self.action_type)

        # self.value_entropy_weight = torch.nn.Parameter(torch.ones(1)/2)
   
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.actor.zero_std(self.device)

    def critic_values(self, obs):
        if self.critic_type in {"direct", "flow_field", "floq"}:
            return self.critic(obs, self.critic_particles)
        return self.critic(obs)

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
        else:
            action_log, entropy = continuous_parallel_act(self.actor, obs, action, batch_size, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy, gate_entropy

    def get_actions(self, obs):
        obs = _to_tpdv(check(obs), self.tpdv)
        batch_size = np.shape(obs)[0]

        v_loc = self.critic_values(obs)
        
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_decentralized_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.actor, obs, batch_size, self.action_dim, self.tpdv)

        return output_action, output_action_log, v_loc

    def get_values(self, obs):
        obs = _to_tpdv(check(obs), self.tpdv)
        v_tot = self.critic_values(obs)
        return v_tot
