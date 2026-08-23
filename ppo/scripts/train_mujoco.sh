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
 --critic_lr 1e-4 \
 --lr 1e-4 \
 --entropy_coef 0.01 \
 --gamma 0.99 \
 --gae_lambda 0.95 \
 --max_grad_norm 1.0 \
 --eval_episodes 2 \
 --n_training_threads 32 \
 --n_rollout_threads 64 \
 --num_mini_batch 1 \
 --episode_length 500 \
 --eval_interval 25 \
 --num_env_steps 200000000 \
 --ppo_epoch 20 \
 --clip_param 0.1 \
 --use_eval \
 --use_value_active_masks \
 --use_policy_active_masks \
 --value_method flow \
 --num_value_particles 16 \
 --flow_integrator euler \
 --num_flow_steps 16 \
 --flow_particle_scale 0.25 \
 --flow_time_embed_dim 64 \
 --flow_max_velocity 15.0 \
 --flow_entropy_beta 0.1 \
 --flow_weight_mode uniform \
 --use_wandb true \
 --wandb_name "xxx" \
 --user_name "shahil-shaik7-clemson-university"
