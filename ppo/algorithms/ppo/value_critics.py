"""Value critics used by the three supported PPO variants."""

import math

import torch
import torch.nn as nn

from ppo.algorithms.utils.util import init


def _init_layer(layer, gain=1.0):
    return init(
        layer,
        nn.init.orthogonal_,
        lambda bias: nn.init.constant_(bias, 0),
        gain=gain,
    )


def _critic_mlp(obs_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.LayerNorm(obs_dim),
        _init_layer(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2.0)),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        _init_layer(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2.0)),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        _init_layer(nn.Linear(hidden_dim, output_dim), gain=1.0),
    )


class ScalarValueCritic(nn.Module):
    """Conventional scalar state-value critic for baseline PPO."""

    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = _critic_mlp(obs_dim, hidden_dim, 1)

    def forward(self, observations):
        return self.net(observations)


class QuantileValueCritic(nn.Module):
    """Fixed-grid inverse-CDF critic for Wasserstein-directional DPPO."""

    def __init__(self, obs_dim, hidden_dim, num_quantiles):
        super().__init__()
        if num_quantiles < 2:
            raise ValueError("QuantileValueCritic requires at least two quantiles")
        self.num_quantiles = num_quantiles
        self.net = _critic_mlp(obs_dim, hidden_dim, num_quantiles)

    def forward(self, observations):
        # Quantile regression does not prevent crossing. Sorting makes the
        # returned tensor a valid empirical inverse CDF and remains
        # differentiable almost everywhere.
        return torch.sort(self.net(observations), dim=-1).values


def make_uniform_particles(num_particles, device=None, scale=1.0):
    """Return deterministic midpoint particles from Uniform[-scale, scale]."""
    if num_particles < 2:
        raise ValueError("A flow critic requires at least two particles")
    device = torch.device("cpu") if device is None else device
    q = (
        torch.arange(num_particles, dtype=torch.float32, device=device) + 0.5
    ) / num_particles
    return scale * (2.0 * q - 1.0)


class FlowVectorField(nn.Module):
    """State-conditioned one-dimensional velocity field."""

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        max_velocity=15.0,
        time_embed_dim=64,
    ):
        super().__init__()
        if time_embed_dim <= 0 or time_embed_dim % 2 != 0:
            raise ValueError("time_embed_dim must be a positive even integer")
        if max_velocity is not None and max_velocity <= 0.0:
            raise ValueError("max_velocity must be positive or None")

        self.state_dim = state_dim
        self.max_velocity = max_velocity
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                time_embed_dim // 2,
                dtype=torch.float32,
            )
        )
        self.register_buffer("time_frequencies", frequencies)

        self.net = nn.Sequential(
            nn.Linear(state_dim + 1 + time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, particles, flow_time):
        flow_time = torch.as_tensor(
            flow_time, dtype=particles.dtype, device=particles.device
        ).expand_as(particles)
        frequencies = self.time_frequencies.to(
            dtype=particles.dtype, device=particles.device
        )
        angles = 2.0 * math.pi * flow_time * frequencies
        time_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        velocity = self.net(torch.cat([states, particles, time_features], dim=-1))
        if self.max_velocity is not None:
            velocity = self.max_velocity * torch.tanh(velocity / self.max_velocity)
        return velocity


class FlowMatchingValueCritic(nn.Module):
    """Flow-matching critic used by Flow-Geometric DPPO."""

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        num_flow_steps=16,
        integrator="euler",
        max_velocity=15.0,
        time_embed_dim=64,
    ):
        super().__init__()
        if num_flow_steps < 1:
            raise ValueError("num_flow_steps must be positive")
        if integrator not in {"euler", "rk4"}:
            raise ValueError("integrator must be 'euler' or 'rk4'")

        self.state_dim = state_dim
        self.num_flow_steps = num_flow_steps
        self.integrator = integrator
        self.vector_field = FlowVectorField(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            max_velocity=max_velocity,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, states, base_particles):
        states, particles = _align_states_and_particles(
            states, base_particles, self.state_dim
        )
        dt = 1.0 / self.num_flow_steps
        for step in range(self.num_flow_steps):
            flow_time = step * dt
            if self.integrator == "euler":
                particles = particles + dt * self.vector_field(
                    states, particles, flow_time
                )
            else:
                particles = self._rk4_step(states, particles, flow_time, dt)
        return particles.squeeze(-1)

    def flow_matching_loss(self, states, base_particles, target_particles):
        """Conditional flow-matching loss with a monotone empirical coupling."""
        states, source = _align_states_and_particles(
            states, base_particles, self.state_dim
        )
        targets = torch.as_tensor(
            target_particles, dtype=source.dtype, device=source.device
        )
        if targets.ndim != source.ndim - 1:
            raise ValueError(
                "target_particles must have shape (..., num_particles)"
            )
        if targets.shape[-1] != source.shape[-2]:
            raise ValueError("source and target particle counts must match")

        targets = torch.sort(targets, dim=-1).values.unsqueeze(-1)
        flow_time = torch.rand_like(source)
        interpolant = (1.0 - flow_time) * source + flow_time * targets
        target_velocity = targets - source
        predicted_velocity = self.vector_field(states, interpolant, flow_time)
        return (predicted_velocity - target_velocity).pow(2).squeeze(-1)

    def _rk4_step(self, states, particles, flow_time, dt):
        k1 = self.vector_field(states, particles, flow_time)
        k2 = self.vector_field(states, particles + 0.5 * dt * k1, flow_time + 0.5 * dt)
        k3 = self.vector_field(states, particles + 0.5 * dt * k2, flow_time + 0.5 * dt)
        k4 = self.vector_field(states, particles + dt * k3, flow_time + dt)
        return particles + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _align_states_and_particles(states, particles, state_dim):
    if states.shape[-1] != state_dim:
        raise ValueError(
            f"Expected state dimension {state_dim}, received {states.shape[-1]}"
        )
    particles = torch.as_tensor(
        particles, dtype=states.dtype, device=states.device
    )
    if particles.ndim == 1:
        particles = particles.expand(*states.shape[:-1], particles.numel())
    elif particles.shape[:-1] != states.shape[:-1]:
        particles = particles.expand(*states.shape[:-1], particles.shape[-1])

    particles = particles.unsqueeze(-1)
    states = states.unsqueeze(-2).expand(
        *states.shape[:-1], particles.shape[-2], state_dim
    )
    return states, particles
