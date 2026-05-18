import math

import torch
import torch.nn as nn


def make_standard_normal_particles(num_particles, device=None):
    if device is None:
        device = torch.device("cpu")
    if num_particles == 1:
        return torch.zeros(1, device=device)
    q = (torch.arange(num_particles, dtype=torch.float32, device=device) + 0.5) / num_particles
    return math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)


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
        states, z = self._align_states_and_particles(states, z)
        features = torch.cat([states, z], dim=-1)
        return self.net(features).squeeze(-1)

    def _align_states_and_particles(self, states, z):
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected states last dim {self.state_dim}, got {states.shape[-1]}"
            )

        batch_shape = states.shape[:-1]

        if z.dim() == 1:
            z = z.view(*([1] * len(batch_shape)), z.shape[0], 1)
            z = z.expand(*batch_shape, z.shape[-2], self.z_dim)
        elif (
            z.dim() >= 2
            and z.shape[-1] == self.z_dim
            and not _is_broadcastable_to(z.shape[:-1], batch_shape)
        ):
            z = z.expand(*batch_shape, z.shape[-2], self.z_dim)
        else:
            z = z.unsqueeze(-1)
            z = z.expand(*batch_shape, z.shape[-2], self.z_dim)

        states = states.unsqueeze(-2).expand(*batch_shape, z.shape[-2], self.state_dim)
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

    advantages = torch.zeros_like(deltas)
    gae = torch.zeros_like(deltas[-1])
    for step in reversed(range(deltas.shape[0])):
        gae = deltas[step] + gamma * gae_lambda * masks[step] * gae
        advantages[step] = gae

    return advantages
