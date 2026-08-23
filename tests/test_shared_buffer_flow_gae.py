from types import SimpleNamespace

import numpy as np

from ppo.utils.shared_buffer import SharedReplayBuffer


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def _make_args(value_method="scalar", num_particles=4):
    return SimpleNamespace(
        episode_length=3,
        n_rollout_threads=1,
        gamma=0.5,
        gae_lambda=0.5,
        use_valuenorm=False,
        value_method=value_method,
        num_value_particles=num_particles,
        flow_weight_mode="uniform",
        flow_alpha=0.1,
        flow_eta=1.0,
        flow_entropy_beta=0.0,
        wasserstein_entropy_beta=0.0,
        distribution_entropy_eps=1e-6,
    )


def _make_buffer(value_method="scalar", num_particles=4):
    return SharedReplayBuffer(
        args=_make_args(value_method, num_particles),
        obs_space=Box((2,)),
        act_space=Discrete(2),
        env_name="test",
    )


def test_scalar_method_uses_classical_gae_and_scalar_targets():
    buffer = _make_buffer("scalar")
    buffer.rewards[:] = 0.0
    buffer.value_preds[:] = 0.0
    next_values = np.zeros((3, 1, 1), dtype=np.float32)
    next_values[-1] = 3.0
    buffer.compute_returns(next_values=next_values)

    expected = np.array([0.09375, 0.375, 1.5], dtype=np.float32).reshape(
        3, 1, 1
    )
    np.testing.assert_allclose(buffer.advantages, expected)
    np.testing.assert_allclose(buffer.value_targets, expected)
    assert np.isfinite(buffer.advantages).all()


def test_wasserstein_method_uses_target_critic_bellman_particles():
    buffer = _make_buffer("wasserstein", num_particles=2)
    buffer.rewards[:] = 0.0
    buffer.value_preds[:] = 0.0
    target_next = np.array(
        [
            [[-1.0, 1.0]],
            [[0.0, 2.0]],
            [[1.0, 3.0]],
        ],
        dtype=np.float32,
    )
    buffer.compute_returns(
        next_values=target_next,
    )

    expected_targets = 0.5 * target_next
    np.testing.assert_allclose(buffer.value_targets, expected_targets)
    deltas = expected_targets.mean(axis=-1, keepdims=True)
    expected_advantages = np.empty_like(deltas)
    expected_advantages[2] = deltas[2]
    expected_advantages[1] = deltas[1] + 0.25 * expected_advantages[2]
    expected_advantages[0] = deltas[0] + 0.25 * expected_advantages[1]
    np.testing.assert_allclose(buffer.advantages, expected_advantages)


def test_flow_method_uses_particlewise_bellman_targets():
    buffer = _make_buffer("flow", num_particles=2)
    buffer.rewards[:] = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(
        3, 1, 1
    )
    buffer.value_preds[:] = 0.0
    target_next = np.array(
        [
            [[-1.0, 1.0]],
            [[0.0, 2.0]],
            [[1.0, 3.0]],
        ],
        dtype=np.float32,
    )
    buffer.compute_returns(
        next_values=target_next,
    )
    expected_targets = buffer.rewards + 0.5 * target_next
    np.testing.assert_allclose(buffer.value_targets, expected_targets)
    assert not np.allclose(
        buffer.value_targets[0, ..., 0], buffer.value_targets[0, ..., 1]
    )


def test_flow_entropy_length_is_scaled_to_return_units():
    args = _make_args("flow", num_particles=2)
    args.flow_entropy_beta = 1.0
    buffer = SharedReplayBuffer(args, Box((2,)), Discrete(2), "test")
    buffer.rewards[:] = 0.0
    buffer.value_preds[:] = 0.0
    buffer.value_entropy_lengths[:] = 2.0
    buffer.has_value_entropy_lengths = True
    target_lengths = np.full((3, 1, 1), 4.0, dtype=np.float32)
    target_next = np.zeros((3, 1, 2), dtype=np.float32)
    buffer.compute_returns(
        next_values=target_next,
        target_next_entropy_lengths=target_lengths,
    )
    # Each TD entropy term is gamma * 4 - 2 = 0, so every advantage is zero.
    np.testing.assert_allclose(buffer.advantages, 0.0)


def test_time_limit_bootstraps_without_crossing_the_reset_trace():
    buffer = _make_buffer("scalar")
    buffer.rewards[:] = 0.0
    buffer.value_preds[:] = 0.0
    buffer.masks[1] = 0.0
    buffer.bootstrap_masks[0] = 1.0
    next_values = np.zeros((3, 1, 1), dtype=np.float32)
    next_values[0] = 2.0

    buffer.compute_returns(next_values=next_values)

    # The timeout transition gets gamma V(s_terminal)=1, but its GAE trace
    # cannot include the first transition from the reset episode.
    np.testing.assert_allclose(buffer.advantages[0], 1.0)


def test_returns_require_next_value_predictions():
    buffer = _make_buffer("flow", num_particles=2)
    try:
        buffer.compute_returns()
    except ValueError as error:
        assert "next_values" in str(error)
    else:
        raise AssertionError("missing target predictions were accepted")
