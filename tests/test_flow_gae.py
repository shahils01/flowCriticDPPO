import torch

from ppo.algorithms.ppo.flow_gae import (
    compute_flow_gae,
    flow_td_residual,
    generalized_advantage_estimate,
    spectral_weights,
    terminal_map_entropy_length,
)
from ppo.algorithms.ppo.value_critics import (
    FlowMatchingValueCritic,
    make_uniform_particles,
)
from ppo.algorithms.ppo.wasserstein_gae import (
    inverse_cdf_entropy,
    wasserstein_directional_td_residual,
)


def test_spectral_weights_are_empirically_normalized():
    for mode in (
        "uniform",
        "lower_tail",
        "upper_tail",
        "smooth_lower_tail",
        "smooth_upper_tail",
    ):
        weights = spectral_weights(16, mode=mode, alpha=0.1)
        assert torch.allclose(weights.mean(), torch.tensor(1.0))


def test_flow_residual_is_invariant_to_particle_permutations():
    current = torch.tensor([[3.0, 1.0, 2.0]])
    target_next = torch.tensor([[6.0, 4.0, 5.0]])
    residual, targets = flow_td_residual(
        rewards=torch.tensor([[1.0]]),
        current_particles=current,
        target_next_particles=target_next,
        gamma=0.5,
        masks=torch.ones(1, 1),
    )

    assert torch.allclose(targets, torch.tensor([[3.0, 3.5, 4.0]]))
    assert torch.allclose(residual, (targets - current.sort(dim=-1).values).mean(-1, keepdim=True))


def test_uniform_flow_residual_recovers_mean_td():
    rewards = torch.tensor([[1.0], [0.5]])
    current = torch.tensor([[0.2, 0.4, 0.6], [1.0, 1.5, 2.0]])
    target_next = torch.tensor([[0.5, 0.7, 0.9], [0.0, 0.2, 0.4]])
    residual, _ = flow_td_residual(
        rewards,
        current,
        target_next,
        gamma=0.9,
        masks=torch.ones_like(rewards),
    )
    expected = (rewards + 0.9 * target_next - current).mean(-1, keepdim=True)
    assert torch.allclose(residual, expected)


def test_nonuniform_flow_residual_depends_on_distribution_shape():
    current = torch.tensor([[4.0, 6.0]])
    target_next = torch.tensor([[0.0, 10.0]])
    uniform, _ = flow_td_residual(
        torch.zeros(1, 1), current, target_next, 1.0, torch.ones(1, 1)
    )
    lower_tail, _ = flow_td_residual(
        torch.zeros(1, 1),
        current,
        target_next,
        1.0,
        torch.ones(1, 1),
        weight_mode="lower_tail",
        alpha=0.5,
    )
    assert torch.allclose(uniform, torch.zeros_like(uniform))
    assert lower_tail.item() < 0.0


def test_entropy_length_matches_linear_uniform_flow():
    base = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    base = base.expand(2, 4).clone().requires_grad_(True)
    scale = torch.tensor([[3.0], [0.5]])
    values = scale * base
    entropy_length = terminal_map_entropy_length(values, base)
    assert torch.allclose(entropy_length, 2.0 * scale)


def test_flow_gae_uses_entropy_length_bellman_residual():
    current = torch.tensor([[[1.0, 1.0]]])
    target_next = current.clone()
    advantages, _ = compute_flow_gae(
        rewards=torch.zeros(1, 1, 1),
        current_particles=current,
        target_next_particles=target_next,
        gamma=0.5,
        gae_lambda=0.0,
        masks=torch.ones(1, 1, 1),
        entropy_beta=2.0,
        current_entropy_length=torch.tensor([[[3.0]]]),
        target_next_entropy_length=torch.tensor([[[4.0]]]),
    )
    expected = -0.5 + 2.0 * (0.5 * 4.0 - 3.0)
    assert torch.allclose(advantages, torch.tensor([[[expected]]]))


def test_generalized_advantage_estimate_is_not_internally_normalized():
    deltas = torch.tensor([[[0.5]], [[1.0]], [[1.5]]])
    advantages = generalized_advantage_estimate(
        deltas, gamma=0.5, gae_lambda=0.5, masks=torch.ones_like(deltas)
    )
    expected = torch.tensor([[[0.84375]], [[1.375]], [[1.5]]])
    assert torch.allclose(advantages, expected)


def test_flow_matching_critic_uses_particlewise_targets():
    critic = FlowMatchingValueCritic(
        state_dim=3,
        hidden_dim=8,
        num_flow_steps=2,
        time_embed_dim=8,
    )
    states = torch.zeros(2, 3)
    base = 0.1 * make_uniform_particles(4)
    targets = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    values = critic(states, base)
    loss = critic.flow_matching_loss(states, base, targets)
    assert values.shape == (2, 4)
    assert loss.shape == (2, 4)
    assert torch.isfinite(loss).all()


def test_wasserstein_residual_uses_ordered_inverse_cdfs():
    current = torch.tensor([[2.0, 0.0, 1.0]])
    target = torch.tensor([[4.0, 3.0, 2.0]])
    residual, bellman = wasserstein_directional_td_residual(
        rewards=torch.zeros(1, 1),
        current_quantiles=current,
        target_next_quantiles=target,
        gamma=1.0,
        masks=torch.ones(1, 1),
        entropy_beta=0.0,
    )
    assert torch.allclose(bellman, torch.tensor([[2.0, 3.0, 4.0]]))
    assert torch.allclose(residual, torch.tensor([[2.0]]))


def test_inverse_cdf_entropy_tracks_scale():
    quantiles = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    scaled = 2.0 * quantiles
    difference = inverse_cdf_entropy(scaled) - inverse_cdf_entropy(quantiles)
    assert torch.allclose(difference, torch.log(torch.tensor(2.0)).reshape(1, 1))
