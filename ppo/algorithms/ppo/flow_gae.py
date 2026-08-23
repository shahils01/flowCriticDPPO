"""Flow-oriented discrepancy and Flow-GAE estimators."""

import torch


def quantile_midpoints(num_particles, device=None, dtype=torch.float32):
    if num_particles < 1:
        raise ValueError("num_particles must be positive")
    return (
        torch.arange(num_particles, device=device, dtype=dtype) + 0.5
    ) / num_particles


def spectral_weights(
    num_particles,
    mode="uniform",
    alpha=0.1,
    eta=1.0,
    device=None,
    dtype=torch.float32,
):
    """Evaluate and empirically normalize the spectral preference weights."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    q = quantile_midpoints(num_particles, device=device, dtype=dtype)
    mode = mode.lower().replace("-", "_")

    if mode == "uniform":
        weights = torch.ones_like(q)
    elif mode == "lower_tail":
        weights = (q <= alpha).to(dtype)
    elif mode == "upper_tail":
        weights = (q >= 1.0 - alpha).to(dtype)
    elif mode == "smooth_lower_tail":
        weights = torch.exp(-eta * q)
    elif mode == "smooth_upper_tail":
        weights = torch.exp(eta * q)
    else:
        raise ValueError(f"Unknown spectral weight mode: {mode}")

    mean_weight = weights.mean()
    if mean_weight <= 0:
        raise ValueError(
            f"alpha={alpha} selects no particles from a {num_particles}-particle grid"
        )
    return weights / mean_weight


def terminal_map_entropy_length(
    value_particles,
    base_particles,
    eps=1e-6,
    max_log_length=30.0,
    create_graph=False,
):
    """Estimate exp(H) from the Jacobian of a pointwise one-dimensional flow."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if value_particles.shape != base_particles.shape:
        raise ValueError("value_particles and base_particles must have identical shapes")
    if not base_particles.requires_grad or not value_particles.requires_grad:
        raise ValueError("entropy-length estimation requires a differentiable base grid")

    # The value flow acts on particles independently. Consequently, the
    # gradient of the sum gives the diagonal Jacobian in one autograd call.
    jacobian_diag = torch.autograd.grad(
        value_particles.sum(),
        base_particles,
        retain_graph=create_graph,
        create_graph=create_graph,
    )[0]
    spacing = base_particles[..., 1:] - base_particles[..., :-1]
    base_width = (
        (base_particles[..., -1:] - base_particles[..., :1]).abs()
        + spacing.abs().mean(dim=-1, keepdim=True)
    )
    entropy = base_width.clamp_min(eps).log()
    entropy = entropy + jacobian_diag.abs().clamp_min(eps).log().mean(
        dim=-1, keepdim=True
    )
    return torch.exp(entropy.clamp(max=max_log_length))


def flow_td_residual(
    rewards,
    current_particles,
    target_next_particles,
    gamma,
    masks,
    weight_mode="uniform",
    alpha=0.1,
    eta=1.0,
    entropy_beta=0.0,
    current_entropy_length=None,
    target_next_entropy_length=None,
):
    """Return the sorted-particle Flow-TD residual and Bellman targets."""
    current = torch.sort(torch.as_tensor(current_particles), dim=-1).values
    target_next = torch.sort(
        torch.as_tensor(
            target_next_particles, dtype=current.dtype, device=current.device
        ),
        dim=-1,
    ).values
    rewards = torch.as_tensor(rewards, dtype=current.dtype, device=current.device)
    masks = torch.as_tensor(masks, dtype=current.dtype, device=current.device)
    bellman_targets = rewards + gamma * masks * target_next

    weights = spectral_weights(
        current.shape[-1],
        mode=weight_mode,
        alpha=alpha,
        eta=eta,
        device=current.device,
        dtype=current.dtype,
    )
    residual = (weights * (bellman_targets - current)).mean(
        dim=-1, keepdim=True
    )

    if entropy_beta != 0.0:
        if current_entropy_length is None or target_next_entropy_length is None:
            raise ValueError(
                "entropy lengths are required when flow_entropy_beta is nonzero"
            )
        current_length = torch.as_tensor(
            current_entropy_length, dtype=current.dtype, device=current.device
        )
        next_length = torch.as_tensor(
            target_next_entropy_length, dtype=current.dtype, device=current.device
        )
        residual = residual + entropy_beta * (
            gamma * masks * next_length - current_length
        )

    return residual, bellman_targets


def generalized_advantage_estimate(deltas, gamma, gae_lambda, masks):
    """Unnormalized backward GAE recursion shared by all three methods."""
    deltas = torch.as_tensor(deltas)
    masks = torch.as_tensor(masks, dtype=deltas.dtype, device=deltas.device)
    advantages = torch.zeros_like(deltas)
    accumulator = torch.zeros_like(deltas[-1])
    for step in reversed(range(deltas.shape[0])):
        accumulator = (
            deltas[step]
            + gamma * gae_lambda * masks[step] * accumulator
        )
        advantages[step] = accumulator
    return advantages


def compute_flow_gae(
    rewards,
    current_particles,
    target_next_particles,
    gamma,
    gae_lambda,
    masks,
    weight_mode="uniform",
    alpha=0.1,
    eta=1.0,
    entropy_beta=0.0,
    current_entropy_length=None,
    target_next_entropy_length=None,
    trace_masks=None,
):
    deltas, bellman_targets = flow_td_residual(
        rewards=rewards,
        current_particles=current_particles,
        target_next_particles=target_next_particles,
        gamma=gamma,
        masks=masks,
        weight_mode=weight_mode,
        alpha=alpha,
        eta=eta,
        entropy_beta=entropy_beta,
        current_entropy_length=current_entropy_length,
        target_next_entropy_length=target_next_entropy_length,
    )
    advantages = generalized_advantage_estimate(
        deltas,
        gamma=gamma,
        gae_lambda=gae_lambda,
        masks=masks if trace_masks is None else trace_masks,
    )
    return advantages, bellman_targets
