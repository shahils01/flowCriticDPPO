"""Command-line configuration for the clean PPO/DPPO implementation."""

import argparse


def _str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, received {value!r}"
    )


def _add_bool(parser, name, default, help_text):
    parser.add_argument(
        name,
        type=_str2bool,
        nargs="?",
        const=True,
        default=default,
        help=help_text + " (bare flag means true; true/false are accepted)",
    )


def get_config():
    parser = argparse.ArgumentParser(description="Clean PPO and distributional PPO")

    # Experiment and runtime.
    parser.add_argument("--algorithm_name", default="ppo", choices=["ppo"])
    parser.add_argument("--experiment_name", default="default")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--user_name", default="xxx")
    parser.add_argument("--seed", type=int, default=1)
    _add_bool(parser, "--cuda", True, "Use CUDA when available")
    _add_bool(parser, "--cuda_deterministic", True, "Use deterministic cuDNN")
    _add_bool(parser, "--use_wandb", False, "Log metrics to Weights & Biases")
    parser.add_argument("--n_training_threads", type=int, default=1)
    parser.add_argument("--n_rollout_threads", type=int, default=32)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1)
    parser.add_argument("--num_env_steps", type=int, default=10_000_000)
    parser.add_argument("--env_name", default="mujoco")
    parser.add_argument("--episode_length", type=int, default=200)

    # Shared actor and critic architecture.
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # PPO.
    parser.add_argument("--ppo_epoch", type=int, default=15)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--num_mini_batch", type=int, default=1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--huber_delta", type=float, default=10.0)
    _add_bool(parser, "--use_clipped_value_loss", True, "Clip scalar value updates")
    _add_bool(parser, "--use_max_grad_norm", True, "Clip actor/critic gradients")
    _add_bool(parser, "--use_huber_loss", True, "Use Huber scalar value loss")
    _add_bool(parser, "--use_valuenorm", True, "Normalize value targets")
    _add_bool(parser, "--use_value_active_masks", True, "Mask inactive value samples")
    _add_bool(parser, "--use_policy_active_masks", True, "Mask inactive policy samples")
    _add_bool(parser, "--use_linear_lr_decay", False, "Linearly decay learning rates")

    # Checkpointing, logging, and evaluation.
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=5)
    _add_bool(parser, "--use_eval", False, "Evaluate during training")
    parser.add_argument("--eval_interval", type=int, default=25)
    parser.add_argument("--eval_episodes", type=int, default=32)
    _add_bool(parser, "--use_render", False, "Render evaluation environments")
    parser.add_argument("--model_dir", default=None)

    # The only algorithm selector.
    parser.add_argument(
        "--value_method",
        default="scalar",
        choices=["scalar", "wasserstein", "flow"],
        help=(
            "scalar: baseline PPO; wasserstein: inverse-CDF directional DPPO; "
            "flow: flow-matching critic with Flow-Geometric GAE"
        ),
    )
    parser.add_argument(
        "--num_value_particles",
        "--num_quants",
        dest="num_value_particles",
        type=int,
        default=16,
        help="Number of quantiles/particles (ignored by scalar PPO)",
    )
    parser.add_argument(
        "--target_critic_tau",
        type=float,
        default=0.005,
        help="Polyak coefficient for distributional target critics",
    )

    # Inverse-CDF Wasserstein baseline.
    parser.add_argument("--quantile_huber_kappa", type=float, default=1.0)
    parser.add_argument("--wasserstein_entropy_beta", type=float, default=1.0)
    parser.add_argument("--distribution_entropy_eps", type=float, default=1e-6)

    # Paper-aligned flow-matching critic and Flow-GAE.
    parser.add_argument("--num_flow_steps", type=int, default=16)
    parser.add_argument(
        "--flow_integrator", choices=["euler", "rk4"], default="euler"
    )
    parser.add_argument("--flow_particle_scale", type=float, default=0.25)
    parser.add_argument("--flow_max_velocity", type=float, default=15.0)
    parser.add_argument("--flow_time_embed_dim", type=int, default=64)
    parser.add_argument(
        "--flow_weight_mode",
        choices=[
            "uniform",
            "lower_tail",
            "upper_tail",
            "smooth_lower_tail",
            "smooth_upper_tail",
        ],
        default="uniform",
    )
    parser.add_argument("--flow_alpha", type=float, default=0.1)
    parser.add_argument("--flow_eta", type=float, default=1.0)
    parser.add_argument("--flow_entropy_beta", type=float, default=0.0)
    parser.add_argument("--flow_entropy_eps", type=float, default=1e-6)

    return parser
