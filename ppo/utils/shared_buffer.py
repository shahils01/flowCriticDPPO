"""Single-agent rollout buffer for the three supported value methods."""

import numpy as np
import torch

from ppo.algorithms.ppo.flow_gae import (
    compute_flow_gae,
    generalized_advantage_estimate,
)
from ppo.algorithms.ppo.wasserstein_gae import compute_wasserstein_gae
from ppo.utils.util import get_shape_from_act_space, get_shape_from_obs_space


class SharedReplayBuffer:
    """Store one fixed-length vectorized rollout and construct critic targets."""

    def __init__(self, args, obs_space, act_space, env_name):
        del env_name  # Kept in the signature for runner compatibility.
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.value_method = args.value_method
        self.num_value_particles = (
            1 if self.value_method == "scalar" else args.num_value_particles
        )
        if self.value_method != "scalar" and self.num_value_particles < 2:
            raise ValueError("Distributional value methods require at least two particles")

        self.flow_weight_mode = args.flow_weight_mode
        self.flow_alpha = args.flow_alpha
        self.flow_eta = args.flow_eta
        self.flow_entropy_beta = args.flow_entropy_beta
        self.wasserstein_entropy_beta = args.wasserstein_entropy_beta
        self.distribution_entropy_eps = args.distribution_entropy_eps

        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape, dict) or len(obs_shape) != 1:
            raise ValueError(
                f"Expected a vector observation space, received {obs_shape}"
            )
        action_shape = get_shape_from_act_space(act_space)
        rollout_shape = (self.episode_length, self.n_rollout_threads)

        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )
        self.transition_next_obs = np.zeros(
            (*rollout_shape, *obs_shape), dtype=np.float32
        )
        self.actions = np.zeros(
            (*rollout_shape, action_shape), dtype=np.float32
        )
        self.action_log_probs = np.zeros(
            (*rollout_shape, 1), dtype=np.float32
        )
        self.value_preds = np.zeros(
            (*rollout_shape, self.num_value_particles), dtype=np.float32
        )
        self.value_targets = np.zeros_like(self.value_preds)
        self.advantages = np.zeros((*rollout_shape, 1), dtype=np.float32)
        self.rewards = np.zeros((*rollout_shape, 1), dtype=np.float32)
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1),
            dtype=np.float32,
        )
        self.bootstrap_masks = np.ones((*rollout_shape, 1), dtype=np.float32)
        self.active_masks = np.ones((*rollout_shape, 1), dtype=np.float32)
        self.value_entropy_lengths = np.zeros(
            (*rollout_shape, 1), dtype=np.float32
        )
        self.has_value_entropy_lengths = False
        self.step = 0

    def insert(
        self,
        obs,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        active_masks=None,
        value_entropy_length=None,
        transition_next_obs=None,
        bootstrap_masks=None,
    ):
        """Insert a vectorized environment step."""
        transition_next_obs = obs if transition_next_obs is None else transition_next_obs
        bootstrap_masks = masks if bootstrap_masks is None else bootstrap_masks

        self.obs[self.step + 1] = np.asarray(obs, dtype=np.float32)
        self.transition_next_obs[self.step] = np.asarray(
            transition_next_obs, dtype=np.float32
        )
        self.actions[self.step] = np.asarray(actions, dtype=np.float32)
        self.action_log_probs[self.step] = np.asarray(
            action_log_probs, dtype=np.float32
        )
        self.value_preds[self.step] = np.asarray(value_preds, dtype=np.float32)
        self.rewards[self.step] = np.asarray(rewards, dtype=np.float32)
        self.masks[self.step + 1] = np.asarray(masks, dtype=np.float32)
        self.bootstrap_masks[self.step] = np.asarray(
            bootstrap_masks, dtype=np.float32
        )
        if active_masks is not None:
            self.active_masks[self.step] = np.asarray(
                active_masks, dtype=np.float32
            )
        if value_entropy_length is not None:
            self.value_entropy_lengths[self.step] = np.asarray(
                value_entropy_length, dtype=np.float32
            )
            self.has_value_entropy_lengths = True
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.has_value_entropy_lengths = False

    def compute_returns(
        self,
        next_values=None,
        value_normalizer=None,
        target_next_entropy_lengths=None,
    ):
        """Compute advantages and particlewise Bellman targets."""
        if next_values is None:
            raise ValueError("next_values are required to construct Bellman targets")
        next_values = np.asarray(next_values, dtype=np.float32)
        if next_values.shape != self.value_preds.shape:
            if next_values.size != self.value_preds.size:
                raise ValueError(
                    "next_values must have shape "
                    f"{self.value_preds.shape}, got {next_values.shape}"
                )
            next_values = next_values.reshape(self.value_preds.shape)

        current_values = self._denormalize(self.value_preds, value_normalizer)
        next_values = self._denormalize(next_values, value_normalizer)
        rewards = torch.as_tensor(self.rewards, dtype=torch.float32)
        bootstrap_masks = torch.as_tensor(
            self.bootstrap_masks, dtype=torch.float32
        )
        trace_masks = torch.as_tensor(self.masks[1:], dtype=torch.float32)

        current = torch.as_tensor(current_values, dtype=torch.float32)
        following = torch.as_tensor(next_values, dtype=torch.float32)
        if self.value_method == "scalar":
            deltas = rewards + self.gamma * bootstrap_masks * following - current
            advantages = generalized_advantage_estimate(
                deltas, self.gamma, self.gae_lambda, trace_masks
            )
            value_targets = advantages + current
        elif self.value_method == "wasserstein":
            advantages, value_targets = compute_wasserstein_gae(
                rewards=rewards,
                current_quantiles=current,
                target_next_quantiles=following,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                masks=bootstrap_masks,
                entropy_beta=self.wasserstein_entropy_beta,
                entropy_eps=self.distribution_entropy_eps,
                trace_masks=trace_masks,
            )
        else:
            current_lengths = None
            target_lengths = None
            if self.flow_entropy_beta != 0.0:
                if not self.has_value_entropy_lengths:
                    raise ValueError(
                        "Current flow entropy lengths were not stored during rollout"
                    )
                if target_next_entropy_lengths is None:
                    raise ValueError("Target flow entropy lengths are required")
                entropy_scale = self._normalization_scale(value_normalizer)
                current_lengths = self.value_entropy_lengths * entropy_scale
                target_lengths = (
                    np.asarray(target_next_entropy_lengths, dtype=np.float32)
                    .reshape(self.value_entropy_lengths.shape)
                    * entropy_scale
                )

            advantages, value_targets = compute_flow_gae(
                rewards=rewards,
                current_particles=current,
                target_next_particles=following,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                masks=bootstrap_masks,
                weight_mode=self.flow_weight_mode,
                alpha=self.flow_alpha,
                eta=self.flow_eta,
                entropy_beta=self.flow_entropy_beta,
                current_entropy_length=current_lengths,
                target_next_entropy_length=target_lengths,
                trace_masks=trace_masks,
            )

        self.advantages[:] = advantages.detach().cpu().numpy()
        self.value_targets[:] = value_targets.detach().cpu().numpy()

    @staticmethod
    def _denormalize(values, value_normalizer):
        if value_normalizer is None:
            return np.asarray(values, dtype=np.float32)
        return value_normalizer.denormalize(values)

    @staticmethod
    def _normalization_scale(value_normalizer):
        if value_normalizer is None:
            return 1.0
        _, variance = value_normalizer.running_mean_var()
        return float(torch.sqrt(variance).reshape(-1)[0].detach().cpu())

    def feed_forward_generator_transformer(
        self, advantages, num_mini_batch=None, mini_batch_size=None
    ):
        """Yield shuffled flat mini-batches, using every rollout sample."""
        batch_size = self.episode_length * self.n_rollout_threads
        permutation = torch.randperm(batch_size).numpy()
        if mini_batch_size is None:
            if num_mini_batch is None or num_mini_batch < 1:
                raise ValueError("num_mini_batch must be positive")
            if batch_size < num_mini_batch:
                raise ValueError("The rollout batch is smaller than num_mini_batch")
            samplers = np.array_split(permutation, num_mini_batch)
        else:
            if mini_batch_size < 1:
                raise ValueError("mini_batch_size must be positive")
            samplers = [
                permutation[start : start + mini_batch_size]
                for start in range(0, batch_size, mini_batch_size)
            ]

        def flatten(values):
            return values.reshape(batch_size, *values.shape[2:])

        observations = flatten(self.obs[:-1])
        actions = flatten(self.actions)
        value_predictions = flatten(self.value_preds)
        value_targets = flatten(self.value_targets)
        masks = flatten(self.masks[:-1])
        active_masks = flatten(self.active_masks)
        action_log_probs = flatten(self.action_log_probs)
        advantages = flatten(advantages)

        for indices in samplers:
            yield (
                observations[indices],
                actions[indices],
                value_predictions[indices],
                value_targets[indices],
                masks[indices],
                active_masks[indices],
                action_log_probs[indices],
                advantages[indices],
            )

    def get_step_obs(self, step):
        return self.obs[step]

    def get_all_transition_next_obs(self):
        return self.transition_next_obs.reshape(
            -1, *self.transition_next_obs.shape[2:]
        )

    def reshape_target_predictions(self, predictions):
        return np.asarray(predictions).reshape(self.value_preds.shape)

    def reshape_target_entropy_lengths(self, entropy_lengths):
        return np.asarray(entropy_lengths).reshape(
            self.value_entropy_lengths.shape
        )

    def set_step_obs(self, step, obs):
        self.obs[step] = np.asarray(obs, dtype=np.float32)
