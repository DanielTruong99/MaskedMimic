import numpy as np

# Replace with your actual .npz file path
file_path = 'data/motions/aidin_humanoid/B12_walk_turn_right_90_poses.npz'

data = np.load(file_path)

# Print all arrays stored in the npz file
for key in data.files:
    print(f"{key}: {data[key]}")
    print(f"Shape of {key}: {data[key].shape}")