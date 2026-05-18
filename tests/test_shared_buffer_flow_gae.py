from types import SimpleNamespace

import numpy as np

from ppo.utils.shared_buffer import SharedReplayBuffer


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def _make_args(critic_type="flow", use_legacy_scalar_gae=False, num_quants=1):
    return SimpleNamespace(
        episode_length=3,
        n_rollout_threads=1,
        hidden_size=1,
        gamma=0.5,
        gae_lambda=0.5,
        use_gae=True,
        critic_type=critic_type,
        use_legacy_scalar_gae=use_legacy_scalar_gae,
        use_popart=False,
        use_valuenorm=False,
        use_proper_time_limits=False,
        algorithm_name="ppo",
        num_quants=num_quants,
        flow_weight_mode="uniform",
        flow_alpha=0.1,
        flow_eta=1.0,
        dgae_epsilon=1.0,
        use_value_entropy=False,
        true_integration=False,
    )


def _make_buffer(critic_type="flow", use_legacy_scalar_gae=False, num_quants=1):
    return SharedReplayBuffer(
        args=_make_args(
            critic_type=critic_type,
            use_legacy_scalar_gae=use_legacy_scalar_gae,
            num_quants=num_quants,
        ),
        obs_space=Box((2,)),
        act_space=Discrete(2),
        env_name="test",
    )


def test_flow_critic_compute_returns_uses_flow_gae_and_bellman_targets():
    buffer = _make_buffer()
    buffer.rewards[:] = np.array([[[[0.0]]], [[[0.0]]], [[[0.0]]]], dtype=np.float32)
    buffer.value_preds[:-1] = 0.0

    buffer.compute_returns(next_value=np.array([[3.0]], dtype=np.float32))

    deltas = np.array([0.0, 0.0, 1.5], dtype=np.float32)
    expected = np.empty((3, 1, 1, 1), dtype=np.float32)
    expected[2] = deltas[2]
    expected[1] = deltas[1] + 0.5 * 0.5 * expected[2]
    expected[0] = deltas[0] + 0.5 * 0.5 * expected[1]

    np.testing.assert_allclose(buffer.advantages, expected)
    np.testing.assert_allclose(buffer.returns[:-1], deltas.reshape(3, 1, 1, 1))


def test_legacy_critic_compute_returns_uses_old_scalar_path():
    buffer = _make_buffer(critic_type="legacy")
    buffer.rewards[:] = np.array([[[[0.0]]], [[[0.0]]], [[[0.0]]]], dtype=np.float32)
    buffer.value_preds[:-1] = 0.0

    buffer.compute_returns(next_value=np.array([[3.0]], dtype=np.float32))

    expected_raw_gae = np.array([0.09375, 0.375, 1.5], dtype=np.float32)
    expected_returns = expected_raw_gae.reshape(3, 1, 1, 1)

    np.testing.assert_allclose(buffer.returns[:-1], expected_returns)
