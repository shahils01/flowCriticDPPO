"""Isaac Lab runner using the same clean PPO/DPPO training path."""

import time

import numpy as np
import torch

from ppo.runner.shared.base_runner import Runner


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class IsaacRunner(Runner):
    def run(self):
        self.warmup()
        started_at = time.time()
        num_updates = self.num_env_steps // (
            self.episode_length * self.n_rollout_threads
        )
        running_rewards = np.zeros(self.n_rollout_threads, dtype=np.float32)
        completed_rewards = []

        for update in range(num_updates):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(update, num_updates)

            for step in range(self.episode_length):
                values, entropy_lengths, env_actions, log_probs = self.collect(step)
                obs, rewards, terminated, truncated, infos = self.envs.step(env_actions)
                obs = _to_numpy(obs)
                rewards = _to_numpy(rewards).reshape(
                    self.n_rollout_threads, -1
                ).mean(axis=1, keepdims=True)
                terminated = _to_numpy(terminated).reshape(
                    self.n_rollout_threads, -1
                ).all(axis=1, keepdims=True)
                truncated = _to_numpy(truncated).reshape(
                    self.n_rollout_threads, -1
                ).all(axis=1, keepdims=True)
                dones = np.logical_or(terminated, truncated)
                transition_next_obs = self.transition_next_observations(
                    obs, infos, truncated
                )

                running_rewards += rewards[:, 0]
                for env_index in np.flatnonzero(dones[:, 0]):
                    completed_rewards.append(running_rewards[env_index])
                    running_rewards[env_index] = 0.0

                self.insert(
                    obs=obs,
                    transition_next_obs=transition_next_obs,
                    rewards=rewards,
                    dones=dones,
                    terminated=terminated,
                    values=values,
                    entropy_lengths=entropy_lengths,
                    actions=_to_numpy(env_actions),
                    log_probs=log_probs,
                )

            self.compute()
            train_info = self.train()
            total_steps = (
                (update + 1) * self.episode_length * self.n_rollout_threads
            )
            if update % self.save_interval == 0 or update == num_updates - 1:
                self.save(update)
            if update % self.log_interval == 0:
                train_info["average_step_rewards"] = float(self.buffer.rewards.mean())
                self.log_train(train_info, total_steps)
                elapsed = max(time.time() - started_at, 1e-6)
                fps = int(total_steps / elapsed)
                print(
                    f"{self.all_args.scenario} | {self.algorithm_name} | "
                    f"update {update + 1}/{num_updates} | steps {total_steps} | FPS {fps}"
                )
                if completed_rewards:
                    self.log_scalar(
                        "average_episode_rewards",
                        float(np.mean(completed_rewards)),
                        total_steps,
                    )
                    completed_rewards.clear()
            if self.use_eval and update % self.eval_interval == 0:
                self.eval(total_steps)
                if self.eval_envs is self.envs:
                    self.warmup()

    def warmup(self):
        self.buffer.set_step_obs(0, _to_numpy(self.envs.reset()))

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        observations = self.buffer.get_step_obs(step)
        masks = self.buffer.masks[step]
        if self.all_args.value_method == "flow" and self.all_args.flow_entropy_beta:
            values, entropy_lengths, actions, log_probs = (
                self.trainer.policy.get_actions_with_value_entropy_length(
                    observations, masks
                )
            )
            entropy_lengths = _to_numpy(entropy_lengths)
        else:
            values, actions, log_probs = self.trainer.policy.get_actions(
                observations, masks
            )
            entropy_lengths = None
        return (
            _to_numpy(values),
            entropy_lengths,
            actions,
            _to_numpy(log_probs),
        )

    def insert(
        self,
        *,
        obs,
        transition_next_obs,
        rewards,
        dones,
        terminated,
        values,
        entropy_lengths,
        actions,
        log_probs,
    ):
        trace_masks = np.ones((self.n_rollout_threads, 1), dtype=np.float32)
        trace_masks[dones[:, 0]] = 0.0
        bootstrap_masks = np.ones_like(trace_masks)
        bootstrap_masks[terminated[:, 0]] = 0.0
        self.buffer.insert(
            obs,
            actions,
            log_probs,
            values,
            rewards,
            trace_masks,
            active_masks=np.ones_like(trace_masks),
            value_entropy_length=entropy_lengths,
            transition_next_obs=transition_next_obs,
            bootstrap_masks=bootstrap_masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        self.trainer.prep_rollout()
        observations = self.eval_envs.reset()
        num_envs = self.n_rollout_threads
        masks = np.ones((num_envs, 1), dtype=np.float32)
        running_rewards = np.zeros(num_envs, dtype=np.float32)
        episode_rewards = []

        while len(episode_rewards) < self.all_args.eval_episodes:
            actions = self.trainer.policy.act(
                observations, masks, deterministic=True
            )
            observations, rewards, terminated, truncated, _ = self.eval_envs.step(
                actions
            )
            rewards = _to_numpy(rewards).reshape(num_envs, -1).mean(axis=1)
            dones = np.logical_or(
                _to_numpy(terminated), _to_numpy(truncated)
            ).reshape(num_envs, -1).all(axis=1)
            running_rewards += rewards
            masks.fill(1.0)
            masks[dones] = 0.0
            for env_index in np.flatnonzero(dones):
                episode_rewards.append(running_rewards[env_index])
                running_rewards[env_index] = 0.0
                if len(episode_rewards) >= self.all_args.eval_episodes:
                    break

        self.log_scalar(
            "eval_average_episode_rewards",
            float(np.mean(episode_rewards)),
            total_num_steps,
        )
