"""Inverse-CDF Wasserstein-directional advantage estimator."""

import torch

from ppo.algorithms.ppo.flow_gae import generalized_advantage_estimate


def inverse_cdf_entropy(quantiles, eps=1e-6):
    """Spacing estimator of differential entropy from ordered quantiles."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    quantiles = torch.sort(torch.as_tensor(quantiles), dim=-1).values
    if quantiles.shape[-1] < 2:
        raise ValueError("inverse-CDF entropy requires at least two quantiles")
    dq = 1.0 / quantiles.shape[-1]
    slopes = (quantiles[..., 1:] - quantiles[..., :-1]) / dq
    return slopes.clamp_min(eps).log().mean(dim=-1, keepdim=True)


def wasserstein_directional_td_residual(
    rewards,
    current_quantiles,
    target_next_quantiles,
    gamma,
    masks,
    entropy_beta=1.0,
    entropy_eps=1e-6,
):
    """Signed inverse-CDF displacement with the previous entropy correction."""
    current = torch.sort(torch.as_tensor(current_quantiles), dim=-1).values
    target_next = torch.sort(
        torch.as_tensor(
            target_next_quantiles, dtype=current.dtype, device=current.device
        ),
        dim=-1,
    ).values
    rewards = torch.as_tensor(rewards, dtype=current.dtype, device=current.device)
    masks = torch.as_tensor(masks, dtype=current.dtype, device=current.device)
    bellman_targets = rewards + gamma * masks * target_next

    residual = (bellman_targets - current).mean(dim=-1, keepdim=True)
    if entropy_beta != 0.0:
        residual = residual + entropy_beta * (
            inverse_cdf_entropy(bellman_targets, eps=entropy_eps)
            - inverse_cdf_entropy(current, eps=entropy_eps)
        )
    return residual, bellman_targets


def compute_wasserstein_gae(
    rewards,
    current_quantiles,
    target_next_quantiles,
    gamma,
    gae_lambda,
    masks,
    entropy_beta=1.0,
    entropy_eps=1e-6,
    trace_masks=None,
):
    deltas, bellman_targets = wasserstein_directional_td_residual(
        rewards=rewards,
        current_quantiles=current_quantiles,
        target_next_quantiles=target_next_quantiles,
        gamma=gamma,
        masks=masks,
        entropy_beta=entropy_beta,
        entropy_eps=entropy_eps,
    )
    advantages = generalized_advantage_estimate(
        deltas,
        gamma=gamma,
        gae_lambda=gae_lambda,
        masks=masks if trace_masks is None else trace_masks,
    )
    return advantages, bellman_targets
