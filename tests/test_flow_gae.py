import torch

from ppo.algorithms.ppo.flow_gae import (
    compute_flow_gae,
    normal_cdf,
    quantile_weight,
    spectral_td_residual,
)


def test_uniform_weight_matches_mean_td_residual():
    rewards = torch.tensor([[1.0], [0.5]])
    current = torch.tensor([[0.2, 0.4, 0.6], [1.0, 1.5, 2.0]])
    next_values = torch.tensor([[0.5, 0.7, 0.9], [0.0, 0.2, 0.4]])
    z = torch.tensor([-1.0, 0.0, 1.0])
    gamma = 0.9

    residual = spectral_td_residual(
        rewards=rewards,
        current_particles=current,
        next_particles=next_values,
        z=z,
        gamma=gamma,
        mode="uniform",
    )
    expected = (rewards + gamma * next_values - current).mean(dim=-1, keepdim=True)

    assert torch.allclose(residual, expected)


def test_lower_tail_weighting_only_uses_q_below_alpha():
    q = torch.tensor([0.05, 0.20, 0.50, 0.90])
    weights = quantile_weight(q, mode="lower_tail", alpha=0.25)
    expected = torch.tensor([4.0, 4.0, 0.0, 0.0])

    assert torch.equal(weights, expected)


def test_upper_tail_weighting_only_uses_q_above_one_minus_alpha():
    q = torch.tensor([0.05, 0.20, 0.75, 0.95])
    weights = quantile_weight(q, mode="upper_tail", alpha=0.25)
    expected = torch.tensor([0.0, 0.0, 4.0, 4.0])

    assert torch.equal(weights, expected)


def test_tail_weighting_in_spectral_residual_uses_normal_quantiles():
    z = torch.tensor([-2.0, 0.0, 2.0])
    q = normal_cdf(z)
    rewards = torch.tensor([[0.0]])
    current = torch.tensor([[0.0, 0.0, 0.0]])
    next_values = torch.tensor([[1.0, 10.0, 100.0]])

    lower = spectral_td_residual(
        rewards=rewards,
        current_particles=current,
        next_particles=next_values,
        z=z,
        gamma=1.0,
        mode="lower_tail",
        alpha=0.1,
    )

    expected_weights = (q <= 0.1).float() / 0.1
    expected = (expected_weights * next_values).mean(dim=-1, keepdim=True)

    assert torch.allclose(lower, expected)


def test_compute_flow_gae_matches_deterministic_recursion():
    rewards = torch.tensor([[[0.0]], [[0.0]], [[0.0]]])
    current = torch.zeros(3, 1, 2)
    next_values = torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]]])
    z = torch.tensor([-1.0, 1.0])
    gamma = 0.5
    gae_lambda = 0.5

    advantages = compute_flow_gae(
        rewards=rewards,
        current_particles=current,
        next_particles=next_values,
        z=z,
        gamma=gamma,
        gae_lambda=gae_lambda,
        mode="uniform",
    )

    deltas = torch.tensor([[[0.5]], [[1.0]], [[1.5]]])
    expected = torch.empty_like(deltas)
    expected[2] = deltas[2]
    expected[1] = deltas[1] + gamma * gae_lambda * expected[2]
    expected[0] = deltas[0] + gamma * gae_lambda * expected[1]

    assert torch.allclose(advantages, expected)
