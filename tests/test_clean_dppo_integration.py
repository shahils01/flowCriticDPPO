import numpy as np
import pytest
import torch

from ppo.algorithms.ppo.algorithm.ppo_policy import PPO_Policy
from ppo.algorithms.ppo.ppo_trainer import PPOTrainer
from ppo.config import get_config
from ppo.utils.shared_buffer import SharedReplayBuffer


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


def _args(value_method):
    return get_config().parse_args(
        [
            "--value_method",
            value_method,
            "--episode_length",
            "4",
            "--n_rollout_threads",
            "2",
            "--n_embd",
            "8",
            "--ppo_epoch",
            "1",
            "--num_mini_batch",
            "2",
            "--num_value_particles",
            "4",
            "--num_flow_steps",
            "2",
            "--flow_time_embed_dim",
            "8",
            "--use_valuenorm",
            "false",
        ]
    )


@pytest.mark.parametrize("value_method", ["scalar", "wasserstein", "flow"])
def test_one_complete_ppo_update_for_each_value_method(value_method):
    torch.manual_seed(3)
    np.random.seed(3)
    args = _args(value_method)
    obs_space = Box((3,))
    act_space = Box((2,))
    policy = PPO_Policy(args, obs_space, act_space)
    trainer = PPOTrainer(args, policy)
    buffer = SharedReplayBuffer(args, obs_space, act_space, "test")

    observations = np.random.randn(args.n_rollout_threads, 3).astype(np.float32)
    buffer.set_step_obs(0, observations)
    for _ in range(args.episode_length):
        values, actions, log_probs = policy.get_actions(
            observations, masks=None
        )
        next_observations = np.random.randn(
            args.n_rollout_threads, 3
        ).astype(np.float32)
        buffer.insert(
            next_observations,
            actions.detach().numpy(),
            log_probs.detach().numpy(),
            values.detach().numpy(),
            np.random.randn(args.n_rollout_threads, 1).astype(np.float32),
            np.ones((args.n_rollout_threads, 1), dtype=np.float32),
            active_masks=np.ones(
                (args.n_rollout_threads, 1), dtype=np.float32
            ),
            transition_next_obs=next_observations,
            bootstrap_masks=np.ones(
                (args.n_rollout_threads, 1), dtype=np.float32
            ),
        )
        observations = next_observations

    flat_next_obs = buffer.get_all_transition_next_obs()
    if value_method == "scalar":
        next_values = policy.get_values(flat_next_obs, masks=None)
    else:
        next_values = policy.get_target_values(flat_next_obs)
    buffer.compute_returns(
        buffer.reshape_target_predictions(next_values.detach().numpy())
    )
    train_info = trainer.train(buffer)

    assert all(np.isfinite(value) for value in train_info.values())


def test_discrete_actor_uses_all_categories_and_joint_log_probability():
    args = _args("scalar")
    policy = PPO_Policy(args, Box((3,)), Discrete(4))
    observations = np.zeros((32, 3), dtype=np.float32)
    values, actions, log_probs = policy.get_actions(observations, masks=None)
    evaluated_values, evaluated_log_probs, entropy = policy.evaluate_actions(
        observations, actions.detach().numpy(), masks=None
    )

    assert actions.shape == (32, 1)
    assert actions.min() >= 0 and actions.max() < 4
    assert log_probs.shape == (32, 1)
    assert evaluated_values.shape == values.shape
    assert evaluated_log_probs.shape == (32, 1)
    assert entropy.ndim == 0


def test_cli_exposes_exactly_three_methods_and_num_quants_alias():
    parser = get_config()
    action = next(
        action for action in parser._actions if action.dest == "value_method"
    )
    assert action.choices == ["scalar", "wasserstein", "flow"]
    assert parser.parse_args(["--num_quants", "7"]).num_value_particles == 7
    assert parser.parse_args(["--use_wandb"]).use_wandb is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--critic_type", "floq"])
