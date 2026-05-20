import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_standard_normal_particles(num_particles, device=None):
    if device is None:
        device = torch.device("cpu")
    if num_particles == 1:
        return torch.zeros(1, device=device)
    q = (torch.arange(num_particles, dtype=torch.float32, device=device) + 0.5) / num_particles
    return math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)


def make_uniform_particles(num_particles, device=None, low=-1.0, high=1.0):
    if device is None:
        device = torch.device("cpu")
    if num_particles == 1:
        return torch.zeros(1, device=device)
    q = (torch.arange(num_particles, dtype=torch.float32, device=device) + 0.5) / num_particles
    return low + (high - low) * q


class ValueFlowCritic(nn.Module):
    """
    Practical state-conditioned transport map T_phi(s, z).

    The module maps Gaussian particles z and state features s to return particles.
    It is intentionally independent from the current PPO critic so it can be
    integrated after the spectral estimator is tested.
    """

    def __init__(self, state_dim, hidden_dim=128, z_dim=1, num_layers=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.state_dim = state_dim
        self.z_dim = z_dim

        layers = []
        input_dim = state_dim + z_dim
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states, z):
        """
        Args:
            states: Tensor with shape (..., state_dim).
            z: Tensor with shape (num_particles,), (..., num_particles), or
                (..., num_particles, z_dim).

        Returns:
            Return particles with shape (..., num_particles).
        """
        states, z = _align_states_and_particles(states, z, self.state_dim, self.z_dim)
        features = torch.cat([states, z], dim=-1)
        return self.net(features).squeeze(-1)

    def _align_states_and_particles(self, states, z):
        return _align_states_and_particles(states, z, self.state_dim, self.z_dim)


def _init_last_linear(module, weight_std=0.0):
    for layer in reversed(module):
        if isinstance(layer, nn.Linear):
            if weight_std == 0.0:
                nn.init.zeros_(layer.weight)
            else:
                nn.init.normal_(layer.weight, mean=0.0, std=weight_std)
            nn.init.zeros_(layer.bias)
            return


def _inverse_softplus(value):
    value = torch.as_tensor(value, dtype=torch.float32)
    return torch.log(torch.expm1(value).clamp_min(1e-12))


class FlowVectorField(nn.Module):
    """State-conditioned vector field v_phi(s, x_tau, tau)."""

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        z_dim=1,
        num_layers=2,
        max_velocity=5.0,
        zero_init_output=True,
        time_embed_dim=0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if time_embed_dim < 0 or time_embed_dim % 2 != 0:
            raise ValueError("time_embed_dim must be a non-negative even integer")

        self.state_dim = state_dim
        self.z_dim = z_dim
        self.max_velocity = max_velocity
        self.time_embed_dim = time_embed_dim
        if time_embed_dim > 0:
            frequencies = torch.exp(
                torch.linspace(
                    math.log(1.0),
                    math.log(1000.0),
                    time_embed_dim // 2,
                    dtype=torch.float32,
                )
            )
            self.register_buffer("time_frequencies", frequencies)

        layers = []
        time_dim = time_embed_dim if time_embed_dim > 0 else 1
        input_dim = state_dim + z_dim + time_dim
        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, z_dim))
        self.net = nn.Sequential(*layers)
        if zero_init_output:
            _init_last_linear(self.net)

    def forward(self, states, x, tau):
        tau = torch.as_tensor(tau, dtype=x.dtype, device=x.device)
        tau = tau.expand_as(x[..., :1])
        if self.time_embed_dim > 0:
            freqs = self.time_frequencies.to(dtype=x.dtype, device=x.device)
            angles = 2.0 * math.pi * tau * freqs
            tau_features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        else:
            tau_features = tau
        features = torch.cat([states, x, tau_features], dim=-1)
        velocity = self.net(features)
        if self.max_velocity is not None:
            velocity = self.max_velocity * torch.tanh(velocity / self.max_velocity)
        return velocity


class FlowFieldValueCritic(nn.Module):
    """
    Continuous-time value flow critic.

    Particles evolve from z_0 ~ N(0, 1) through:
        d x_tau / d tau = v_phi(s, x_tau, tau), tau in [0, 1].
    """

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        z_dim=1,
        num_layers=2,
        num_flow_steps=8,
        integrator="euler",
        particle_scale=0.05,
        learn_particle_scale=True,
        max_particle_scale=2.0,
        max_velocity=5.0,
    ):
        super().__init__()
        if num_flow_steps < 1:
            raise ValueError("num_flow_steps must be at least 1")
        if integrator not in {"euler", "rk4"}:
            raise ValueError("integrator must be either 'euler' or 'rk4'")
        if particle_scale <= 0.0:
            raise ValueError("particle_scale must be positive")
        if max_particle_scale is not None and max_particle_scale <= 0.0:
            raise ValueError("max_particle_scale must be positive or None")

        self.state_dim = state_dim
        self.z_dim = z_dim
        self.num_flow_steps = num_flow_steps
        self.integrator = integrator
        self.learn_particle_scale = learn_particle_scale
        self.max_particle_scale = max_particle_scale
        self.vector_field = FlowVectorField(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            num_layers=num_layers,
            max_velocity=max_velocity,
        )
        self.base_head = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        _init_last_linear(self.base_head)

        if learn_particle_scale:
            self.raw_particle_scale = nn.Parameter(_inverse_softplus(particle_scale))
        else:
            self.register_buffer(
                "fixed_particle_scale", torch.tensor(float(particle_scale))
            )

    def forward(self, states, z):
        base_value = self.base_head(states)
        states, x = _align_states_and_particles(states, z, self.state_dim, self.z_dim)
        dt = 1.0 / self.num_flow_steps

        for step in range(self.num_flow_steps):
            tau = step * dt
            if self.integrator == "euler":
                x = x + dt * self.vector_field(states, x, tau)
            else:
                x = self._rk4_step(states, x, tau, dt)

        scale = self._particle_scale(dtype=x.dtype, device=x.device)
        return base_value + scale * x.squeeze(-1)

    def _particle_scale(self, dtype, device):
        if self.learn_particle_scale:
            scale = F.softplus(self.raw_particle_scale).to(dtype=dtype, device=device)
            if self.max_particle_scale is not None:
                scale = scale.clamp(max=self.max_particle_scale)
            return scale
        return self.fixed_particle_scale.to(dtype=dtype, device=device)

    def _rk4_step(self, states, x, tau, dt):
        k1 = self.vector_field(states, x, tau)
        k2 = self.vector_field(states, x + 0.5 * dt * k1, tau + 0.5 * dt)
        k3 = self.vector_field(states, x + 0.5 * dt * k2, tau + 0.5 * dt)
        k4 = self.vector_field(states, x + dt * k3, tau + dt)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class FloQValueCritic(nn.Module):
    """
    FloQ-style value critic.

    This critic has no monolithic/base value head. It maps uniform initial
    particles z_0 to value particles by integrating a state-conditioned velocity
    field, and exposes a linear flow-matching loss for supervised value targets.
    """

    def __init__(
        self,
        state_dim,
        hidden_dim=128,
        z_dim=1,
        num_layers=2,
        num_flow_steps=8,
        integrator="euler",
        max_velocity=5.0,
        time_embed_dim=64,
    ):
        super().__init__()
        if num_flow_steps < 1:
            raise ValueError("num_flow_steps must be at least 1")
        if integrator not in {"euler", "rk4"}:
            raise ValueError("integrator must be either 'euler' or 'rk4'")

        self.state_dim = state_dim
        self.z_dim = z_dim
        self.num_flow_steps = num_flow_steps
        self.integrator = integrator
        self.vector_field = FlowVectorField(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            num_layers=num_layers,
            max_velocity=max_velocity,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, states, z):
        states, x = _align_states_and_particles(states, z, self.state_dim, self.z_dim)
        dt = 1.0 / self.num_flow_steps

        for step in range(self.num_flow_steps):
            tau = step * dt
            if self.integrator == "euler":
                x = x + dt * self.vector_field(states, x, tau)
            else:
                x = self._rk4_step(states, x, tau, dt)

        return x.squeeze(-1)

    def flow_matching_loss(self, states, z0, targets):
        states, x0 = _align_states_and_particles(states, z0, self.state_dim, self.z_dim)
        targets = torch.as_tensor(targets, dtype=x0.dtype, device=x0.device)
        if targets.dim() >= 3 and targets.shape[-1] == self.z_dim:
            targets = targets.squeeze(-1)
        targets = targets.unsqueeze(-1)
        targets = targets.expand_as(x0)

        t = torch.rand_like(x0)
        x_t = (1.0 - t) * x0 + t * targets
        target_velocity = targets - x0
        pred_velocity = self.vector_field(states, x_t, t)
        return (pred_velocity - target_velocity).pow(2).squeeze(-1)

    def _rk4_step(self, states, x, tau, dt):
        k1 = self.vector_field(states, x, tau)
        k2 = self.vector_field(states, x + 0.5 * dt * k1, tau + 0.5 * dt)
        k3 = self.vector_field(states, x + 0.5 * dt * k2, tau + 0.5 * dt)
        k4 = self.vector_field(states, x + dt * k3, tau + dt)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _align_states_and_particles(states, z, state_dim, z_dim):
    if states.shape[-1] != state_dim:
        raise ValueError(f"Expected states last dim {state_dim}, got {states.shape[-1]}")

    batch_shape = states.shape[:-1]

    if z.dim() == 0:
        raise ValueError("z must contain a particle dimension")

    if z.dim() >= 2 and z.shape[-1] == z_dim:
        num_particles = z.shape[-2]
        z_batch_shape = z.shape[:-2]
    else:
        num_particles = z.shape[-1]
        z_batch_shape = z.shape[:-1]
        z = z.unsqueeze(-1)
        if z_dim != 1:
            z = z.expand(*z.shape[:-1], z_dim)

    if not _is_broadcastable_to(z_batch_shape, batch_shape):
        raise ValueError(
            "z batch shape must be broadcastable to states batch shape: "
            f"got {tuple(z_batch_shape)} and {tuple(batch_shape)}"
        )

    z = z.expand(*batch_shape, num_particles, z_dim)

    states = states.unsqueeze(-2).expand(*batch_shape, z.shape[-2], state_dim)
    return states, z


def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _is_broadcastable_to(source_shape, target_shape):
    try:
        return torch.broadcast_shapes(tuple(source_shape), tuple(target_shape)) == tuple(
            target_shape
        )
    except RuntimeError:
        return False


def quantile_weight(q, mode="uniform", alpha=0.1, eta=1.0, normalize_smooth=True):
    """
    Compute quantile weights omega(q).

    Tail modes use CVaR-style weights: 1 / alpha on the selected tail and 0
    elsewhere. Smooth modes are normalized over the particle dimension so their
    empirical mean weight is 1.
    """
    if not torch.is_tensor(q):
        q = torch.as_tensor(q, dtype=torch.float32)

    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha must be in (0, 1]")

    normalized_mode = mode.lower().replace("-", "_")

    if normalized_mode in {"uniform", "mean", "all", "one", "ones"}:
        return torch.ones_like(q)
    if normalized_mode in {"lower_tail", "lower", "cvar_lower"}:
        return (q <= alpha).to(dtype=q.dtype) / alpha
    if normalized_mode in {"upper_tail", "upper", "cvar_upper"}:
        return (q >= 1.0 - alpha).to(dtype=q.dtype) / alpha
    if normalized_mode in {"smooth_lower_tail", "smooth_lower", "exp_lower"}:
        weights = torch.exp(-eta * q)
    elif normalized_mode in {"smooth_upper_tail", "smooth_upper", "exp_upper"}:
        weights = torch.exp(eta * q)
    else:
        raise ValueError(f"Unknown quantile weight mode: {mode}")

    if normalize_smooth:
        weights = weights / weights.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights


def spectral_td_residual(
    rewards,
    current_particles,
    next_particles,
    z,
    gamma,
    masks=None,
    mode="uniform",
    alpha=0.1,
    eta=1.0,
    weights=None,
):
    """
    Compute delta_t^omega from return particles.

    Args:
        rewards: Tensor broadcastable to (..., 1) or (..., num_particles).
        current_particles: T_phi(s_t, z), shape (..., num_particles).
        next_particles: T_phi_target(s_{t+1}, z), shape (..., num_particles).
        z: Gaussian particles used to compute q = Phi(z).
        gamma: Discount factor.
        masks: Optional continuation mask broadcastable to rewards.
        mode, alpha, eta: Parameters for quantile_weight.
        weights: Optional precomputed omega(Phi(z)).

    Returns:
        Spectral TD residual with shape (..., 1).
    """
    current_particles = torch.as_tensor(current_particles)
    next_particles = torch.as_tensor(
        next_particles, dtype=current_particles.dtype, device=current_particles.device
    )
    rewards = torch.as_tensor(
        rewards, dtype=current_particles.dtype, device=current_particles.device
    )

    if masks is None:
        masks = torch.ones_like(rewards)
    else:
        masks = torch.as_tensor(
            masks, dtype=current_particles.dtype, device=current_particles.device
        )

    if weights is None:
        z = torch.as_tensor(z, dtype=current_particles.dtype, device=current_particles.device)
        q = normal_cdf(z)
        weights = quantile_weight(q, mode=mode, alpha=alpha, eta=eta)
    else:
        weights = torch.as_tensor(
            weights, dtype=current_particles.dtype, device=current_particles.device
        )

    td_particles = rewards + gamma * masks * next_particles - current_particles
    return (weights * td_particles).mean(dim=-1, keepdim=True)


def terminal_map_entropy(value_particles, z, eps=1e-6):
    """
    Estimate entropy of the terminal one-dimensional value-flow distribution.

    For ordered base particles z_0 and terminal particles T(s, z_0), use the
    change-of-variables approximation:
        H[T(s, Z)] ~= H[Z] + E_z log |dT(s, z) / dz|.

    The derivative is estimated with finite differences over adjacent particles.
    """
    value_particles = torch.as_tensor(value_particles)
    z = torch.as_tensor(z, dtype=value_particles.dtype, device=value_particles.device)
    if z.dim() != 1:
        raise ValueError("z must be a one-dimensional particle grid")
    if z.numel() < 2:
        return torch.zeros_like(value_particles[..., :1])
    if value_particles.shape[-1] != z.numel():
        raise ValueError(
            "value_particles last dimension must match z: "
            f"got {value_particles.shape[-1]} and {z.numel()}"
        )

    dz = z[1:] - z[:-1]
    if torch.any(dz == 0):
        raise ValueError("z particles must be distinct")

    slopes = (value_particles[..., 1:] - value_particles[..., :-1]) / dz
    log_abs_jacobian = slopes.abs().clamp_min(eps).log().mean(dim=-1, keepdim=True)

    base_width = (z[-1] - z[0]).abs() + dz.abs().mean()
    base_entropy = base_width.clamp_min(eps).log()
    return base_entropy + log_abs_jacobian


def compute_flow_gae(
    rewards,
    current_particles,
    next_particles,
    z,
    gamma,
    gae_lambda,
    masks=None,
    mode="uniform",
    alpha=0.1,
    eta=1.0,
    weights=None,
    entropy_beta=0.0,
    entropy_delta_mode="bellman",
    entropy_eps=1e-6,
):
    """
    Compute spectral GAE by backward recursion:
        A_t = delta_t^omega + gamma * lambda * mask_t * A_{t+1}

    Tensor convention is time-major. Particle tensors have shape
    (T, ..., num_particles); rewards and masks have shape (T, ..., 1) or a
    broadcastable equivalent.
    """
    if rewards.shape[0] != current_particles.shape[0]:
        raise ValueError("rewards and current_particles must share the time dimension")

    if masks is None:
        masks = torch.ones_like(rewards)

    deltas = spectral_td_residual(
        rewards=rewards,
        current_particles=current_particles,
        next_particles=next_particles,
        z=z,
        gamma=gamma,
        masks=masks,
        mode=mode,
        alpha=alpha,
        eta=eta,
        weights=weights,
    )
    if entropy_beta != 0.0:
        current_entropy = terminal_map_entropy(current_particles, z, eps=entropy_eps)
        next_entropy = terminal_map_entropy(next_particles, z, eps=entropy_eps)
        if entropy_delta_mode == "bellman":
            entropy_delta = gamma * masks * next_entropy - current_entropy
        elif entropy_delta_mode == "difference":
            entropy_delta = masks * (next_entropy - current_entropy)
        else:
            raise ValueError(
                "entropy_delta_mode must be either 'bellman' or 'difference'"
            )
        deltas = deltas + entropy_beta * entropy_delta

    advantages = torch.zeros_like(deltas)
    gae = torch.zeros_like(deltas[-1])
    for step in reversed(range(deltas.shape[0])):
        gae = deltas[step] + gamma * gae_lambda * masks[step] * gae
        advantages[step] = gae

    return advantages
