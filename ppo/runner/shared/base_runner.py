import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:  # W&B is optional when --use_wandb is false.
    wandb = None
from ppo.utils.shared_buffer import SharedReplayBuffer
from ppo.algorithms.ppo.ppo_trainer import PPOTrainer as TrainAlgo
from ppo.algorithms.ppo.algorithm.ppo_policy import PPO_Policy as Policy

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner:
    """Shared setup, target construction, checkpointing, and logging."""

    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.value_method
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.use_wandb = self.all_args.use_wandb
        if self.use_wandb and wandb is None:
            raise ImportError(
                "Weights & Biases is not installed; install wandb or set --use_wandb false"
            )
        self.use_render = self.all_args.use_render
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        self.policy = Policy(
            self.all_args,
            self.envs.observation_space,
            self.envs.action_space,
            device=self.device,
        )

        if self.model_dir is not None:
            self.restore(self.model_dir)

        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.envs.observation_space,
            self.envs.action_space,
            self.all_args.env_name,
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def transition_next_observations(observations, infos, truncated):
        """Use the pre-reset terminal observation for time-limit bootstrapping."""
        result = np.array(observations, copy=True)

        truncated = np.asarray(truncated).reshape(-1).astype(bool)
        for index in np.flatnonzero(truncated):
            info = infos[index] if not isinstance(infos, dict) else infos
            terminal = None
            if isinstance(info, dict):
                terminal = info.get("terminal_observation")
                if terminal is None:
                    terminal = info.get("final_observation")
            if terminal is None:
                continue
            if torch.is_tensor(terminal):
                terminal = terminal.detach().cpu().numpy()
            terminal = np.asarray(terminal)
            if isinstance(infos, dict) and terminal.shape[:1] == (len(truncated),):
                terminal = terminal[index]
            result[index] = terminal
        return result
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        transition_next_obs = self.buffer.get_all_transition_next_obs()
        target_next_entropy_lengths = None
        if self.all_args.value_method == "scalar":
            next_values = self.trainer.policy.get_values(
                transition_next_obs, masks=None
            )
        elif (
            self.all_args.value_method == "flow"
            and self.all_args.flow_entropy_beta != 0.0
        ):
            next_values, target_lengths = (
                self.trainer.policy.get_target_values_and_entropy_length(
                    transition_next_obs
                )
            )
            target_next_entropy_lengths = (
                self.buffer.reshape_target_entropy_lengths(_t2n(target_lengths))
            )
        else:
            next_values = self.trainer.policy.get_target_values(
                transition_next_obs
            )
        next_values = self.buffer.reshape_target_predictions(_t2n(next_values))

        self.buffer.compute_returns(
            next_values,
            self.trainer.value_normalizer,
            target_next_entropy_lengths=target_next_entropy_lengths,
        )
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        print(f"Saving checkpoint at update {episode}...")
        self.policy.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.policy.restore(model_dir)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.log_scalar(k, v, total_num_steps)

    def log_scalar(self, name, value, step):
        if self.use_wandb:
            wandb.log({name: value}, step=step)
        else:
            self.writer.add_scalar(name, value, step)
