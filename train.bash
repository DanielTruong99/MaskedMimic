python protomotions/eval_agent.py +robot=aidin_humanoid +simulator=isaaclab +checkpoint=results/full_body_tracker/last.ckpt +motion_file=data/motions/aidin_humanoid_B1_stand_to_walk_poses.npz

python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain_aidin +robot=aidin_humanoid +simulator=isaaclab +experiment_name=full_body_tracker +opt=wandb motion_file=data/motions/aidin_humanoid_B1_stand_to_walk_poses.npz agent.config.batch_size=1024  agent.config.num_steps=24 num_envs=2048

python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain_aidin +robot=aidin_humanoid +simulator=isaaclab +experiment_name=full_body_tracker +opt=wandb motion_file=data/motions/aidin_humanoid_B1_stand_to_walk_poses.npz num_envs=4096 agent.config.batch_size=2048