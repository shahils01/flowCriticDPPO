import torch

from ppo.algorithms.ppo.flow_gae import (
    FloQValueCritic,
    FlowFieldValueCritic,
    ValueFlowCritic,
    compute_flow_gae,
    make_uniform_particles,
    normal_cdf,
    terminal_map_entropy,
    terminal_map_entropy_autograd,
    terminal_map_entropy_from_transport,
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


def test_value_flow_critic_accepts_shared_and_batched_particles():
    critic = ValueFlowCritic(state_dim=3, hidden_dim=8)
    states = torch.zeros(4, 3)

    shared_z = torch.linspace(-1.0, 1.0, 4).unsqueeze(-1)
    shared_particles = critic(states, shared_z)

    batched_z = torch.linspace(-1.0, 1.0, 4).repeat(4, 1)
    batched_particles = critic(states, batched_z)

    assert shared_particles.shape == (4, 4)
    assert batched_particles.shape == (4, 4)


def test_flow_field_critic_starts_with_small_particle_scale():
    critic = FlowFieldValueCritic(
        state_dim=3,
        hidden_dim=8,
        particle_scale=0.05,
        max_velocity=5.0,
    )
    states = torch.zeros(2, 3)
    z = torch.tensor([-1.0, 0.0, 1.0])

    particles = critic(states, z)
    expected = 0.05 * z.expand(2, 3)

    assert torch.allclose(particles, expected, atol=1e-6)


def test_floq_critic_uses_uniform_particles_without_base_head():
    critic = FloQValueCritic(
        state_dim=3,
        hidden_dim=8,
        num_flow_steps=2,
        max_velocity=5.0,
        time_embed_dim=8,
    )
    states = torch.zeros(2, 3)
    z = 0.1 * make_uniform_particles(3)

    particles = critic(states, z)
    loss = critic.flow_matching_loss(states, z, torch.ones(2, 3))

    assert not hasattr(critic, "base_head")
    assert particles.shape == (2, 3)
    assert loss.shape == (2, 3)


def test_terminal_map_entropy_matches_linear_uniform_flow():
    z = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    scale = 3.0
    values = scale * z.expand(2, 4)

    entropy = terminal_map_entropy(values, z)

    expected = torch.full((2, 1), torch.log(torch.tensor(2.0 * scale)))
    assert torch.allclose(entropy, expected)


def test_terminal_map_entropy_autograd_matches_linear_uniform_flow():
    z = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    z = z.expand(2, 4).clone().requires_grad_(True)
    scale = torch.tensor([[3.0], [0.5]])
    values = scale * z

    entropy = terminal_map_entropy_autograd(values, z)

    expected = torch.log(2.0 * scale)
    assert torch.allclose(entropy, expected)


def test_terminal_map_entropy_from_transport_uses_per_sample_jacobian():
    states = torch.tensor([[3.0], [0.5]])
    z = torch.tensor([-0.75, -0.25, 0.25, 0.75])

    def transport_map(states, z):
        return states * z

    entropy = terminal_map_entropy_from_transport(transport_map, states, z)

    expected = torch.log(2.0 * states)
    assert torch.allclose(entropy, expected)


def test_compute_flow_gae_can_add_bellman_entropy_delta():
    z = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    rewards = torch.zeros(1, 1, 1)
    current = z.reshape(1, 1, 4)
    next_values = (2.0 * z).reshape(1, 1, 4)

    advantages = compute_flow_gae(
        rewards=rewards,
        current_particles=current,
        next_particles=next_values,
        z=z,
        gamma=1.0,
        gae_lambda=1.0,
        mode="uniform",
        entropy_beta=0.5,
    )

    expected_drift = (next_values - current).mean(dim=-1, keepdim=True)
    expected_entropy_delta = torch.log(torch.tensor(4.0)) - torch.log(torch.tensor(2.0))
    expected = expected_drift + 0.5 * expected_entropy_delta

    assert torch.allclose(advantages, expected)


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
