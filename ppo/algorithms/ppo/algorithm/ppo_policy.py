"""Policy wrapper for the clean three-method PPO implementation."""

import torch

from ppo.algorithms.ppo.algorithm.PPO import PPO
from ppo.algorithms.utils.util import check
from ppo.utils.util import get_shape_from_act_space, get_shape_from_obs_space


class PPO_Policy:
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = getattr(args, "critic_lr", args.lr)
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        self.value_method = args.value_method
        requested_particles = args.num_value_particles
        self.num_value_particles = (
            1 if self.value_method == "scalar" else requested_particles
        )

        self.action_type = (
            "Continuous" if act_space.__class__.__name__ == "Box" else "Discrete"
        )
        self.obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(self.obs_shape, dict):
            raise ValueError("Dictionary observations are not supported by this PPO network")
        self.obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else None
        if self.obs_dim is None:
            raise ValueError(f"Expected a vector observation space, received {self.obs_shape}")

        self.act_dim = (
            act_space.n
            if self.action_type == "Discrete"
            else get_shape_from_act_space(act_space)
        )
        self.action_output_dim = 1 if self.action_type == "Discrete" else self.act_dim
        self.tensor_kwargs = dict(dtype=torch.float32, device=device)

        self.transformer = PPO(
            self.obs_dim,
            self.act_dim,
            n_embd=args.n_embd,
            device=device,
            action_type=self.action_type,
            value_method=self.value_method,
            num_value_particles=requested_particles,
            num_flow_steps=args.num_flow_steps,
            flow_integrator=args.flow_integrator,
            flow_particle_scale=args.flow_particle_scale,
            flow_max_velocity=args.flow_max_velocity,
            flow_time_embed_dim=args.flow_time_embed_dim,
            flow_entropy_eps=args.flow_entropy_eps,
        )

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.transformer.actor.parameters(),
                    "lr": self.lr,
                    "initial_lr": self.lr,
                },
                {
                    "params": self.transformer.critic.parameters(),
                    "lr": self.critic_lr,
                    "initial_lr": self.critic_lr,
                },
            ],
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        fraction = 1.0 - episode / float(episodes)
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * fraction

    def _reshape_obs(self, observations):
        return observations.reshape(-1, *self.obs_shape)

    def get_actions(self, observations, masks):
        actions, action_log_probs, values = self.transformer.get_actions(
            self._reshape_obs(observations)
        )
        return (
            values.view(-1, self.num_value_particles),
            actions.view(-1, self.action_output_dim),
            action_log_probs.view(-1, 1),
        )

    def get_actions_with_value_entropy_length(self, observations, masks):
        if self.value_method != "flow":
            raise RuntimeError("Entropy length is only available for value_method='flow'")
        observations = self._reshape_obs(observations)
        actions, action_log_probs, values = self.transformer.get_actions(observations)
        _, entropy_length = self.transformer.get_values_and_entropy_length(observations)
        return (
            values.view(-1, self.num_value_particles),
            entropy_length.view(-1, 1),
            actions.view(-1, self.action_output_dim),
            action_log_probs.view(-1, 1),
        )

    def get_values(self, observations, masks):
        values = self.transformer.get_values(self._reshape_obs(observations))
        return values.view(-1, self.num_value_particles)

    def get_target_values(self, observations, masks=None):
        if self.value_method == "scalar":
            raise RuntimeError("Scalar PPO does not use a target critic")
        values = self.transformer.get_values(
            self._reshape_obs(observations), use_target=True
        )
        return values.view(-1, self.num_value_particles)

    def get_target_values_and_entropy_length(self, observations, masks=None):
        if self.value_method != "flow":
            raise RuntimeError("Entropy length is only available for value_method='flow'")
        values, entropy_length = self.transformer.get_values_and_entropy_length(
            self._reshape_obs(observations), use_target=True
        )
        return (
            values.view(-1, self.num_value_particles),
            entropy_length.view(-1, 1),
        )

    def evaluate_actions(self, observations, actions, masks, active_masks=None):
        observations = self._reshape_obs(observations)
        actions = actions.reshape(-1, self.action_output_dim)
        compute_values = self.value_method != "flow"
        action_log_probs, values, entropy = self.transformer.evaluate_actions(
            observations, actions, compute_values=compute_values
        )
        action_log_probs = action_log_probs.view(-1, 1)
        if values is not None:
            values = values.view(-1, self.num_value_particles)
        entropy = entropy.view(-1, 1)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy * active_masks).sum() / active_masks.sum().clamp_min(1.0)
        else:
            entropy = entropy.mean()
        return values, action_log_probs, entropy

    def critic_flow_matching_loss(self, observations, target_particles):
        observations = check(self._reshape_obs(observations)).to(**self.tensor_kwargs)
        return self.transformer.critic_flow_matching_loss(
            observations, target_particles
        )

    def update_target_critic(self, tau):
        self.transformer.update_target_critic(tau)

    def act(self, observations, masks, deterministic=True):
        actions = self.transformer.act(
            self._reshape_obs(observations), deterministic=deterministic
        )
        return actions.view(-1, self.action_output_dim)

    def save(self, save_dir, episode):
        torch.save(
            self.transformer.state_dict(),
            str(save_dir) + "/transformer_" + str(episode) + ".pt",
        )

    def restore(self, model_dir):
        state_dict = torch.load(model_dir, map_location=self.device)
        self.transformer.load_state_dict(state_dict)

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()
