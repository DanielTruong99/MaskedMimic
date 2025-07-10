python protomotions/train_agent.py \
    +exp=full_body_tracker/transformer_flat_terrain \
    +robot=aidin_humanoid \
    +simulator=isaaclab \
    +experiment_name=full_body_tracker \
    +opt=wandb \
    motion_file=data/motions/aidin_humanoid_B1_stand_to_walk_poses.npz \
    # headless=False \
    agent.config.batch_size=1024 \
    num_envs=1024 \
