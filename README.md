# Clean DPPO

This branch contains one PPO actor and exactly three value methods. Select the
method with a single flag:

| Flag | Critic | Advantage |
| --- | --- | --- |
| `--value_method scalar` | Scalar MLP | Classical GAE |
| `--value_method wasserstein` | Fixed-grid inverse CDF | Wasserstein-directional GAE |
| `--value_method flow` | State-conditioned flow-matching ODE | Flow-Geometric GAE |

The actor is the same diagonal-Gaussian PPO actor in every continuous-control
experiment. Only the critic, critic loss, and advantage construction change.
Both distributional methods use a frozen Polyak target critic and the
particlewise Bellman target
`y_i = r + gamma * mask * z'_i`.

## MuJoCo examples

Run commands from the repository root.

```bash
python ppo/scripts/train/train_mujoco.py \
  --env_name mujoco \
  --scenario Humanoid-v5 \
  --value_method scalar
```

```bash
python ppo/scripts/train/train_mujoco.py \
  --env_name mujoco \
  --scenario Humanoid-v5 \
  --value_method wasserstein \
  --num_value_particles 16 \
  --wasserstein_entropy_beta 1.0
```

```bash
python ppo/scripts/train/train_mujoco.py \
  --env_name mujoco \
  --scenario Humanoid-v5 \
  --value_method flow \
  --num_value_particles 16 \
  --num_flow_steps 16 \
  --flow_integrator euler \
  --flow_particle_scale 0.25 \
  --flow_max_velocity 15.0 \
  --flow_weight_mode uniform \
  --flow_entropy_beta 0.1
```

The flow residual sorts the current and Bellman particles and evaluates

```text
delta = mean_i[w_i (y_i - x_i)]
        + beta (gamma * mask * exp(H_next) - exp(H_current)).
```

The empirical spectral weights have mean one. `uniform` weighting with
`flow_entropy_beta=0` is intentionally the risk-neutral ablation: it exactly
reduces to mean-based TD and GAE. Use a nonuniform spectral weight or a nonzero
entropy-length coefficient for a distribution-shape-aware actor update.

The Wasserstein baseline uses ordered inverse-CDF outputs and the signed
endpoint discrepancy

```text
mean_i[y_i - x_i] + beta_W (H(y) - H(x)).
```

It is a directional discrepancy, not a symmetric probability metric.

## Important options

- `--target_critic_tau`: Polyak update coefficient for both distributional critics.
- `--quantile_huber_kappa`: quantile-regression threshold for Wasserstein DPPO.
- `--flow_weight_mode`: `uniform`, either hard tail, or either smooth tail.
- `--flow_alpha` and `--flow_eta`: hard-tail mass and smooth-tail temperature.
- `--flow_entropy_beta`: entropy-length deformation coefficient from the paper.
- `--use_valuenorm false`: disable the shared scalar value normalization used
  by default. One scalar normalizer is used so distributional geometry is
  preserved under a common affine transform.

The legacy `--num_quants` spelling remains an alias for
`--num_value_particles`. Removed algorithm flags now produce a command-line
error instead of being silently ignored. In particular, `--critic_type`, flow
actor options, `--use_value_entropy`, `--true_integration`, and
`--dgae_epsilon` are no longer supported.

The cleaned Wasserstein estimator deliberately removes the old
timestep-dependent `dgae_epsilon / gamma**step` rule and the unnormalized
exponential quadrature. This makes the baseline stable and its endpoint
functional explicit, but it is not bit-for-bit equivalent to legacy runs.

## Validation

```bash
pytest -q
ruff check ppo tests
```

The test suite includes one complete PPO update for each value method, target
critic updates, particlewise Bellman targets, entropy-length scaling, discrete
and continuous actors, and time-limit bootstrapping.
