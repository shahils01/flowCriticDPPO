#!/bin/sh
env="mujoco"
scenario="Humanoid-v5"
algo="ppo"
seed=0

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seed is ${seed}"
python train/train_mujoco.py \
 --seed ${seed} \
 --env_name ${env} \
 --algorithm_name ${algo} \
 --experiment_name ${exp} \
 --scenario ${scenario} \
 --use_value_entropy \
 --true_integration \
 --critic_lr 3e-4 \
 --lr 3e-4 \
 --entropy_coef 0.001 \
 --gamma 0.99 \
 --gae_lambda 0.95 \
 --max_grad_norm 1.0 \
 --eval_episodes 2 \
 --n_training_threads 32 \
 --n_rollout_threads 256 \
 --num_mini_batch 1 \
 --episode_length 100 \
 --eval_interval 25 \
 --num_env_steps 200000000 \
 --ppo_epoch 20 \
 --clip_param 0.2 \
 --use_eval \
 --add_center_xy \
 --use_state_agent \
 --use_value_active_masks \
 --use_policy_active_masks \
 --num_quants 16 \
 --policy_type gaussian \
 --flow_policy_base_std 0.35 \
 --flow_policy_loss_samples 8 \
 --flow_policy_output_scale 0.25 \
 --critic_type floq \
 --flow_integrator euler \
 --num_flow_steps 16 \
 --flow_particle_scale 0.25 \
 --flow_time_embed_dim 64 \
 --flow_max_velocity 15.0 \
 --flow_entropy_beta 0.1 \
 --flow_entropy_delta_mode bellman \
 --flow_weight_mode uniform \
 --exploration_steps 0 \
 --use_wandb True \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university" \
#  --model_dir /home/shahils/Desktop/gitBackupRepo/flowCriticDPPO/ppo/scripts/results/mujoco/Humanoid-v5/ppo/wandb/run-20260522_185031-abhb3620/files/transformer_900.pt \
