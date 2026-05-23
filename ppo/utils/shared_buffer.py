import torch
import numpy as np
import torch.nn.functional as F
from ppo.algorithms.ppo.flow_gae import (
    compute_flow_gae,
    make_standard_normal_particles,
    make_uniform_particles,
)
from ppo.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param obs_space: (gym.Space) observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, obs_space, act_space, env_name, use_value_entropy=True):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.env_name = env_name
        self.critic_type = getattr(args, "critic_type", "direct")
        if self.critic_type == "flow":
            self.critic_type = "direct"
        self.use_legacy_scalar_gae = (
            getattr(args, "use_legacy_scalar_gae", False)
            or self.critic_type == "legacy"
        )
        self.num_quants = args.num_quants
        self.flow_weight_mode = getattr(args, "flow_weight_mode", "uniform")
        self.flow_alpha = getattr(args, "flow_alpha", 0.1)
        self.flow_eta = getattr(args, "flow_eta", 1.0)
        self.flow_entropy_beta = getattr(args, "flow_entropy_beta", 0.0)
        self.flow_entropy_delta_mode = getattr(args, "flow_entropy_delta_mode", "bellman")
        self.flow_entropy_eps = getattr(args, "flow_entropy_eps", 1e-6)
        self.dgae_epsilon = args.dgae_epsilon
        self.use_value_entropy = args.use_value_entropy
        self.true_integration = args.true_integration
        
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_is_dict = isinstance(obs_shape, dict)

        if not self.obs_is_dict:
            if type(obs_shape[-1]) == list:
                obs_shape = obs_shape[:1]

            if env_name == 'IsaacLab' and len(obs_shape) == 1:
                obs_shape = (obs_shape[-1],)

            self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, *obs_shape), dtype=np.float32)
        else:
            self.obs = {
                k: np.zeros((self.episode_length + 1, self.n_rollout_threads, 1, *shape), dtype=np.float32)
                for k, shape in obs_shape.items()
            }
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1, self.num_quants), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.value_entropies = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.float32)
        self.has_value_entropies = False
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        act_shape = get_shape_from_act_space(act_space)
        print('act_shape after = ', act_shape)
        action_log_shape = 1 if getattr(args, "policy_type", "gaussian") == "flow" else act_shape

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, action_log_shape), dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

        self.q = np.exp(np.linspace(0, 1, self.num_quants))
        self.q = self.q[1:] - self.q[:-1]
        self.q = np.tile(self.q, (self.n_rollout_threads, 1))
        self.q = self.q[:, np.newaxis, :]

        self.gamma_normalizer = ((1/args.gamma) ** torch.arange(args.episode_length, dtype=torch.float32)).unsqueeze(1).repeat(self.n_rollout_threads,1,1)
        self.gamma_normalizer = self.gamma_normalizer.detach().cpu().numpy()

        if self.num_quants > 1:
            self.quantile_spacing = 1.0 / (self.num_quants - 1)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks, bad_masks=None, active_masks=None, value_entropy=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        if self.obs_is_dict:
            for k in self.obs.keys():
                self.obs[k][self.step + 1] = np.expand_dims(obs[k], axis=1).copy()
        else:
            self.obs[self.step + 1] = np.expand_dims(obs, axis=1).copy()
        self.actions[self.step] = np.expand_dims(actions, axis=1).copy()
        self.action_log_probs[self.step] = np.expand_dims(action_log_probs, axis=1).copy()
        self.value_preds[self.step] = np.expand_dims(value_preds, axis=1).copy()
        if value_entropy is not None:
            self.value_entropies[self.step] = np.expand_dims(value_entropy, axis=1).copy()
            self.has_value_entropies = True
        self.rewards[self.step] = np.expand_dims(rewards, axis=1).copy()
        self.masks[self.step + 1] = np.expand_dims(masks, axis=1).copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = np.expand_dims(bad_masks, axis=1).copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = np.expand_dims(active_masks, axis=1).copy()
        
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        if self.obs_is_dict:
            for k in self.obs.keys():
                self.obs[k][0] = self.obs[k][-1].copy()
        else:
            self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        self.value_entropies[0] = self.value_entropies[-1].copy()
        self.has_value_entropies = False

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None, flow_weight_mode="uniform", next_value_entropy=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        self.value_preds[-1] = np.expand_dims(next_value, axis=1).copy()
        if next_value_entropy is not None:
            self.value_entropies[-1] = np.expand_dims(next_value_entropy, axis=1).copy()
            self.has_value_entropies = True
        if self.critic_type in {"direct", "flow_field", "floq"}:
            self.compute_flow_returns(value_normalizer, flow_weight_mode)
            return

        if self.num_quants == 1 and not self.use_legacy_scalar_gae:
            self.compute_scalar_flow_gae_returns(value_normalizer)
            return

        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = self.rewards[step] + self.wasserstein_like_distance(self.gamma * value_normalizer.denormalize(
                        self.value_preds[step + 1]) * self.masks[step + 1], value_normalizer.denormalize(self.value_preds[step]), step)
                
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                if self.num_quants == 1:
                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                                self.masks[step + 1] - self.value_preds[step]
                else:
                    delta = self.wasserstein_like_distance(self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                            self.masks[step + 1], self.value_preds[step], step)

                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                self.advantages[step] = (gae - gae.mean()) / (gae.std() + 1e-8) #gae
                self.returns[step] = gae + self.value_preds[step]

    def get_denormalized_value_particles(self, value_normalizer=None):
        if self._use_popart or self._use_valuenorm:
            return value_normalizer.denormalize(self.value_preds)
        return self.value_preds

    def compute_flow_returns(self, value_normalizer=None, flow_weight_mode="uniform"):
        value_particles = self.get_denormalized_value_particles(value_normalizer)
        current_particles = value_particles[:-1]
        next_particles = value_particles[1:]
        masks = self.masks[1:]
        current_entropy = None
        next_entropy = None
        if self.flow_entropy_beta != 0.0 and not self.has_value_entropies:
            raise RuntimeError(
                "flow entropy is enabled, but no critic-side value entropies were "
                "stored in the buffer"
            )
        if self.has_value_entropies:
            current_entropy = self.value_entropies[:-1]
            next_entropy = self.value_entropies[1:]

        if self.critic_type == "floq":
            z = make_uniform_particles(self.num_quants, torch.device("cpu"))
        else:
            z = make_standard_normal_particles(self.num_quants, torch.device("cpu"))
        advantages = compute_flow_gae(
            rewards=torch.as_tensor(self.rewards, dtype=torch.float32),
            current_particles=torch.as_tensor(current_particles, dtype=torch.float32),
            next_particles=torch.as_tensor(next_particles, dtype=torch.float32),
            z=z,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            masks=torch.as_tensor(masks, dtype=torch.float32),
            mode=flow_weight_mode,
            alpha=self.flow_alpha,
            eta=self.flow_eta,
            entropy_beta=self.flow_entropy_beta,
            entropy_delta_mode=self.flow_entropy_delta_mode,
            entropy_eps=self.flow_entropy_eps,
            current_entropy=current_entropy,
            next_entropy=next_entropy,
        ).detach().cpu().numpy()

        self.advantages[:] = advantages
        self.returns[:-1] = self.rewards + self.gamma * masks * next_particles

    def compute_scalar_flow_gae_returns(self, value_normalizer=None):
        value_particles = self.get_denormalized_value_particles(value_normalizer)
        current_particles = value_particles[:-1]
        next_particles = value_particles[1:]

        z = make_standard_normal_particles(self.num_quants, torch.device("cpu"))
        advantages = compute_flow_gae(
            rewards=torch.as_tensor(self.rewards, dtype=torch.float32),
            current_particles=torch.as_tensor(current_particles, dtype=torch.float32),
            next_particles=torch.as_tensor(next_particles, dtype=torch.float32),
            z=z,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            masks=torch.as_tensor(self.masks[1:], dtype=torch.float32),
            mode="uniform",
        ).detach().cpu().numpy()

        self.advantages[:] = advantages
        self.returns[:-1] = advantages + current_particles

    def wasserstein_like_distance(self, icdf1, icdf2, step):
        """
        Compute the Wasserstein distance between each pair of ICDF functions.

        Parameters:
        icdf1 (torch.Tensor): Tensor of shape [2048, num_quantiles] representing the first set of ICDFs.
        icdf2 (torch.Tensor): Tensor of shape [2048, num_quantiles] representing the second set of ICDFs.

        Returns:
        torch.Tensor: Tensor of shape [2048, 1] representing the Wasserstein distance for each pair of ICDFs.
        """
        # Compute the Wasserstein distance
        # Wasserstein distance between two distributions is the area between their CDFs
        # For ICDFs, this can be approximated by the average absolute difference between the ICDF values
        # distances = torch.sum((1/64)*(icdf1 - icdf2), dim=1, keepdim=True)\             
        if self.use_value_entropy:
            del_icdf1 = (icdf1[:,:,1:] - icdf1[:,:,:-1])/self.quantile_spacing
            del_icdf2 = (icdf2[:,:,1:] - icdf2[:,:,:-1])/self.quantile_spacing
                        
            icdf1_mids = (icdf1[:,:,1:] + icdf1[:,:,:-1])/2
            icdf2_mids = (icdf2[:,:,1:] + icdf2[:,:,:-1])/2
            
            if self.true_integration:
                distances = np.sum(self.q*((icdf1_mids - icdf2_mids) + (self.dgae_epsilon/self.gamma**step)*(np.log(del_icdf1+1e-6)-np.log(del_icdf2+1e-6))), axis=-1, keepdims=True)
            else:
                distances = np.mean((icdf1_mids - icdf2_mids) + (self.dgae_epsilon/self.gamma**step)*(np.log(del_icdf1+1e-6)-np.log(del_icdf2+1e-6)), axis=-1, keepdims=True)
        
        else:
            distances = np.mean((icdf1 - icdf2), axis=-1, keepdims=True)
    
        return distances


    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, 1)

        if self.obs_is_dict:
            obs = {
                k: self.obs[k][:-1].reshape(-1, *self.obs[k].shape[2:])
                for k in self.obs.keys()
            }
            obs = {k: obs[k][rows, cols] for k in obs.keys()}

            next_obs = {
                k: self.obs[k][1:].reshape(-1, *self.obs[k].shape[2:])
                for k in self.obs.keys()
            }
            next_obs = {k: next_obs[k][rows, cols] for k in next_obs.keys()}
        else:
            obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
            obs = obs[rows, cols]

            next_obs = self.obs[1:].reshape(-1, *self.obs.shape[2:])
            next_obs = next_obs[rows, cols]

        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]

        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            if self.obs_is_dict:
                obs_batch = {
                    k: obs[k][indices].reshape(-1, *self.obs[k].shape[2:])
                    for k in obs.keys()
                }
                next_obs_batch = {
                    k: next_obs[k][indices].reshape(-1, *self.obs[k].shape[2:])
                    for k in next_obs.keys()
                }
            else:
                obs_batch = obs[indices].reshape(-1, *self.obs.shape[2:])
                next_obs_batch = next_obs[indices].reshape(-1, *self.obs.shape[2:])

            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])

            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch,\
                   active_masks_batch, old_action_log_probs_batch, adv_targ, next_obs_batch

    def get_step_obs(self, step):
        if self.obs_is_dict:
            return {k: np.concatenate(self.obs[k][step]) for k in self.obs.keys()}
        return np.concatenate(self.obs[step])

    def set_step_obs(self, step, obs):
        if self.obs_is_dict:
            for k in self.obs.keys():
                self.obs[k][step] = np.expand_dims(obs[k], axis=1).copy()
        else:
            self.obs[step] = np.expand_dims(obs, axis=1).copy()
