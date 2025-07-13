python protomotions/eval_agent.py +robot=aidin_humanoid +simulator=isaaclab +checkpoint=results/full_body_tracker/last.ckpt +motion_file=data/motions/aidin_humanoid_B1_stand_to_walk_poses.npz

python protomotions/eval_agent.py +robot=aidin_humanoid +simulator=isaaclab +checkpoint=results/full_body_tracker_walk_turn/last.ckpt +motion_file=data/motions/aidin_humanoid/B10_walk_turn_left_45_poses.npz

python protomotions/eval_agent.py +robot=aidin_humanoid +simulator=isaaclab +checkpoint=results/full_body_tracker_combine/last.ckpt +motion_file=data/motions/aidin_humanoid/walk_dataset_test.yaml

python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain_aidin +robot=aidin_humanoid +simulator=isaaclab +experiment_name=full_body_tracker +opt=wandb motion_file=data/motions/aidin_humanoid_B1_stand_to_walk_poses.npz agent.config.batch_size=1024  agent.config.num_steps=24 num_envs=2048

python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain_aidin +robot=aidin_humanoid +simulator=isaaclab +experiment_name=full_body_tracker +opt=wandb motion_file=data/motions/aidin_humanoid_B1_stand_to_walk_poses.npz num_envs=4096 agent.config.batch_size=2048

python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain_aidin +robot=aidin_humanoid +simulator=isaaclab +experiment_name=full_body_tracker_walk_turn +opt=wandb motion_file=data/motions/aidin_humanoid/B10_w agent.config.batch_size=1024  agent.config.num_steps=24 num_envs=4096 checkpoint=results/full_body_tracker/last.ckpt

python protomotions/train_agent.py +exp=full_body_tracker/transformer_flat_terrain_aidin +robot=aidin_humanoid +simulator=isaaclab +experiment_name=full_body_tracker_combine +opt=wandb motion_file=data/motions/aidin_humanoid/walk_dataset_train.yaml agent.config.batch_size=1024  agent.config.num_steps=24 num_envs=4096 checkpoint=results/full_body_tracker/last.ckpt