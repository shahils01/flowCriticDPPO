"""PPO trainer with explicit losses for the three supported value methods."""

import math
import torch
import torch.nn as nn

from ppo.algorithms.ppo.flow_gae import quantile_midpoints
from ppo.algorithms.utils.util import check
from ppo.utils.util import huber_loss, mse_loss
from ppo.utils.valuenorm import ValueNorm


class PPOTrainer:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tensor_kwargs = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.value_method = args.value_method
        self.num_value_particles = policy.num_value_particles

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.quantile_huber_kappa = args.quantile_huber_kappa
        self.target_critic_tau = args.target_critic_tau
        if not 0.0 <= args.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        if not 0.0 <= args.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if self.clip_param <= 0.0:
            raise ValueError("clip_param must be positive")
        if self.quantile_huber_kappa <= 0.0:
            raise ValueError("quantile_huber_kappa must be positive")
        if self.value_method != "scalar" and not 0.0 < self.target_critic_tau <= 1.0:
            raise ValueError("target_critic_tau must lie in (0, 1]")

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.value_normalizer = (
            ValueNorm(1, device=device) if self._use_valuenorm else None
        )

    def _normalize_targets(self, targets):
        if self.value_normalizer is None:
            return targets
        return self.value_normalizer.normalize(targets)

    @staticmethod
    def _masked_mean(loss, active_masks, enabled):
        if not enabled:
            return loss.mean()
        mask = active_masks.expand_as(loss)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def scalar_value_loss(
        self, values, old_values, value_targets, active_masks
    ):
        clipped_values = old_values + (values - old_values).clamp(
            -self.clip_param, self.clip_param
        )
        normalized_targets = self._normalize_targets(value_targets)
        clipped_error = normalized_targets - clipped_values
        original_error = normalized_targets - values

        if self._use_huber_loss:
            clipped_loss = huber_loss(clipped_error, self.huber_delta)
            original_loss = huber_loss(original_error, self.huber_delta)
        else:
            clipped_loss = mse_loss(clipped_error)
            original_loss = mse_loss(original_error)
        loss = (
            torch.maximum(original_loss, clipped_loss)
            if self._use_clipped_value_loss
            else original_loss
        )
        return self._masked_mean(loss, active_masks, self._use_value_active_masks)

    def quantile_value_loss(self, values, value_targets, active_masks):
        targets = self._normalize_targets(value_targets)
        errors = targets.unsqueeze(-2) - values.unsqueeze(-1)
        absolute_errors = errors.abs()
        kappa = self.quantile_huber_kappa
        huber = torch.where(
            absolute_errors <= kappa,
            0.5 * errors.pow(2),
            kappa * (absolute_errors - 0.5 * kappa),
        )
        quantiles = quantile_midpoints(
            values.shape[-1], device=values.device, dtype=values.dtype
        ).view(1, -1, 1)
        weights = (quantiles - (errors.detach() < 0).to(values.dtype)).abs()
        per_sample = (weights * huber / kappa).mean(
            dim=(-2, -1), keepdim=False
        ).unsqueeze(-1)
        return self._masked_mean(
            per_sample, active_masks, self._use_value_active_masks
        )

    def flow_value_loss(self, observations, value_targets, active_masks):
        targets = self._normalize_targets(value_targets)
        loss = self.policy.critic_flow_matching_loss(observations, targets)
        per_sample = loss.mean(dim=-1, keepdim=True)
        return self._masked_mean(
            per_sample, active_masks, self._use_value_active_masks
        )

    def ppo_update(self, sample):
        (
            obs_batch,
            actions_batch,
            old_values_batch,
            value_targets_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            advantage_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(
            **self.tensor_kwargs
        )
        advantage_batch = check(advantage_batch).to(**self.tensor_kwargs)
        old_values_batch = check(old_values_batch).to(**self.tensor_kwargs)
        value_targets_batch = check(value_targets_batch).to(**self.tensor_kwargs)
        active_masks_batch = check(active_masks_batch).to(**self.tensor_kwargs)

        values, action_log_probs, action_entropy = (
            self.policy.evaluate_actions(
                obs_batch, actions_batch, masks_batch, active_masks_batch
            )
        )
        importance_ratio = torch.exp(
            action_log_probs - old_action_log_probs_batch
        )
        surrogate_1 = importance_ratio * advantage_batch
        surrogate_2 = torch.clamp(
            importance_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        ) * advantage_batch
        clipped_surrogate = torch.minimum(surrogate_1, surrogate_2)
        if self._use_policy_active_masks:
            policy_loss = -(
                clipped_surrogate * active_masks_batch
            ).sum() / active_masks_batch.sum().clamp_min(1.0)
        else:
            policy_loss = -clipped_surrogate.mean()

        if self.value_method == "scalar":
            value_loss = self.scalar_value_loss(
                values,
                old_values_batch,
                value_targets_batch,
                active_masks_batch,
            )
        elif self.value_method == "wasserstein":
            value_loss = self.quantile_value_loss(
                values, value_targets_batch, active_masks_batch
            )
        else:
            value_loss = self.flow_value_loss(
                obs_batch, value_targets_batch, active_masks_batch
            )

        loss = (
            policy_loss
            - action_entropy * self.entropy_coef
            + value_loss * self.value_loss_coef
        )
        self.policy.optimizer.zero_grad()
        loss.backward()
        actor_parameters = list(self.policy.transformer.actor.parameters())
        critic_parameters = list(self.policy.transformer.critic.parameters())
        actor_grad_norm = _gradient_norm(actor_parameters)
        critic_grad_norm = _gradient_norm(critic_parameters)
        parameters = actor_parameters + critic_parameters
        if self._use_max_grad_norm:
            nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.policy.optimizer.step()
        if self.value_method != "scalar":
            self.policy.update_target_critic(self.target_critic_tau)

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            action_entropy,
            actor_grad_norm,
            importance_ratio,
        )

    def _update_value_normalizer(self, buffer):
        if self.value_normalizer is None:
            return
        targets = torch.as_tensor(
            buffer.value_targets, dtype=torch.float32, device=self.device
        )
        active = torch.as_tensor(
            buffer.active_masks, dtype=torch.bool, device=self.device
        ).expand_as(targets)
        selected = targets[active]
        if selected.numel() > 0:
            self.value_normalizer.update(selected.reshape(-1, 1))

    def train(self, buffer):
        self._update_value_normalizer(buffer)
        valid = buffer.active_masks.astype(bool)
        valid_advantages = buffer.advantages[valid]
        if valid_advantages.size == 0:
            valid_advantages = buffer.advantages.reshape(-1)
        mean = valid_advantages.mean()
        std = valid_advantages.std()
        advantages = (buffer.advantages - mean) / (std + 1e-5)

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
        }
        for _ in range(self.ppo_epoch):
            generator = buffer.feed_forward_generator_transformer(
                advantages, self.num_mini_batch
            )
            for sample in generator:
                outputs = self.ppo_update(sample)
                value_loss, critic_grad, policy_loss, entropy, actor_grad, ratio = outputs
                train_info["value_loss"] += value_loss.item()
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += entropy.item()
                train_info["actor_grad_norm"] += float(actor_grad)
                train_info["critic_grad_norm"] += float(critic_grad)
                train_info["ratio"] += ratio.mean().item()

        number_updates = self.ppo_epoch * self.num_mini_batch
        return {key: value / number_updates for key, value in train_info.items()}

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()


def _gradient_norm(parameters):
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().norm(2).item() ** 2
    return math.sqrt(squared_norm)
