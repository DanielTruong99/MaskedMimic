import copy
from typing import Any

import numpy as np
import torch
from easydict import EasyDict
from protomotions.envs.mimic.mimic_utils import dof_to_local
from isaac_utils.rotations import quaternion_to_matrix

from protomotions.simulator.base_simulator.config import RobotConfig
from protomotions.utils.motion_lib import MotionLib


class AidinHumanoidMotionLib(MotionLib):
    def __init__(
            self,
            motion_file,
            robot_config: RobotConfig,
            key_body_ids,
            device="cpu",
            ref_height_adjust: float = 0,
            target_frame_rate: int = 30,
            w_last: bool = True,
            create_text_embeddings: bool = False,
            local_rot_conversion: torch.Tensor = None,
            fix_motion_heights: bool = True,
            skeleton_tree: Any = None,
    ):

        super().__init__(
            motion_file=motion_file,
            robot_config=robot_config,
            key_body_ids=key_body_ids,
            device=device,
            ref_height_adjust=ref_height_adjust,
            target_frame_rate=target_frame_rate,
            w_last=w_last,
            create_text_embeddings=create_text_embeddings,
            local_rot_conversion=local_rot_conversion,
            fix_motion_heights=fix_motion_heights,
            skeleton_tree=skeleton_tree,
        )

        motions = self.state.motions
        self.register_buffer(
            "dof_pos",
            torch.cat([m.dof_pos for m in motions], dim=0).to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

    def _load_motion_file(self, motion_file):
        '''
            npz file from loco mujoco framework includes:
            - qpos: T x J, Position of the joints, including the root joint.
            - qvel: T x J, Velocity of the joints, including the root joint.
            - xpos: T x B x 3, Position of all bodies in global coordinates.
            - xquat: T x B x 4, Quaternion of all bodies in global coordinates.
            - cvel: T x B x 6, Velocity of all bodies in global coordinates, v + w.
            - site_xpos: T x S x 3, Position of all sites in global coordinates, mimic.
            - site_xquat: T x S x 4, Quaternion of all sites in global coordinates, mimic.
            - joint_names: List of joint names.
            - body_names: List of body names.
            - site_names: List of site names.
            - frequency: Frequency of the motion data.
            - njnts: Number of joints.
            - split_points: List of split points for the motion data.

            npy file from h1_walk.npy includes:
            - dof_pos: T x J, Position of the joints, including the root joint.
            - dof_vel: T x J, Velocity of the joints, including the root joint.
            - fps: Frequency of the motion data.
            - global_angular_velocity: T x B x 3, Angular velocity of all bodies in global coordinates.
            - global_root_angular_velocity: T x 3, Angular velocity of the root joint.
            - global_root_velocity: T x 3, Linear velocity of the root joint.
            - global_rotation: T x B x 4, Quaternion of all bodies in global coordinates.
            - global_rotation_mat: T x B x 3 x 3, Rotation matrix of all bodies in global coordinates.
            - global_translation: T x B x 3, Position of all bodies in global coordinates.
            - global_velocity: T x B x 6, Velocity of all bodies in global coordinates, v + w.
            - local_rotation: T x J x 3, Local rotation of the joints.
        '''

        if motion_file.endswith(".npy"):
            motion = EasyDict(torch.load(motion_file))
        elif motion_file.endswith(".npz"):
            motion_data = np.load(motion_file, allow_pickle=True)
            motion = EasyDict()
            for key in motion_data.files:
                if isinstance(motion_data[key], np.ndarray) and np.issubdtype(motion_data[key].dtype, np.number):
                    motion[key] = torch.tensor(motion_data[key])
                else:
                    motion[key] = motion_data[key]  
            
            
            motion.body_names = np.delete(motion.body_names, motion.body_names == 'world') # Remove the world body
            motion.body_names[motion.body_names == 'root'] = 'Pelvis'  # Rename root to Pelvis

            # Map member names to match the expected structure
            motion.dof_pos = motion.qpos[:, 7:]  # Exclude the root
            motion.dof_vels = motion.qvel[:, 6:]  # Exclude the root
            motion.fps = int(motion.frequency)
            motion.global_angular_velocity = motion.cvel[:, 1:, 3:]  # Exclude the root
            motion.global_root_angular_velocity = motion.cvel[:, 1, 3:] # Pelvis is the root
            motion.global_root_velocity = motion.cvel[:, 1, :3]  # Pelvis is the root
            motion.global_rotation = motion.xquat[:, 1:]  # Exclude the root
            motion.global_rotation_mat = torch.stack(
                [quaternion_to_matrix(q, w_last=True) for q in motion.global_rotation], dim=0
            )
            motion.global_translation = motion.xpos[:, 1:]  # Exclude the root
            motion.global_velocity = motion.cvel[:, 1:, :3]  # Exclude the root

        motion.local_rotation = dof_to_local(motion.dof_pos, self.robot_config.dof_offsets,
                                             self.robot_config.joint_axis, True)

        return motion

    def _load_motions(self, motion_file, target_frame_rate):
        target_frame_rate = 50  # Force target frame rate to 50 for AidinHumanoid, avoid issues with downsampling
        super()._load_motions(motion_file, target_frame_rate)

    def _compute_motion_dof_vels(self, motion):
        # We pre-compute the dof vels in fk.
        return motion.dof_vels

    def fix_motion_heights(self, motion, skeleton_tree):
        body_heights = motion.global_translation[..., 2].clone()
        min_height = body_heights.min()

        motion.global_translation[..., 2] -= min_height
        return motion

    @staticmethod
    def _slice_motion_file(motion, motion_timings):
        start, end = motion_timings
        start_frame = round(start * motion.fps)
        if end == -1:
            end_frame = motion.global_translation.shape[0]
        else:
            end_frame = int(end * motion.fps)

        assert (
                start_frame < end_frame
        ), f"Motion start frame {start_frame} >= motion end frame {end_frame}"

        sliced_motion = {}

        for key in motion.keys():
            # if is torch.Tensor
            if isinstance(motion[key], torch.Tensor):
                if motion[key].ndim < 2: continue
                if motion[key].shape[0] < end_frame: continue
                sliced_motion[key] = motion[key][start_frame:end_frame].clone()
            else:
                sliced_motion[key] = copy.deepcopy(motion[key])

        return EasyDict(sliced_motion)

    @staticmethod
    def _fix_motion_fps(motion, orig_fps, target_frame_rate, skeleton_tree):
        skip = int(np.round(orig_fps / target_frame_rate))

        downsampled_motion = {}
        for key in motion.keys():
            # if is torch.Tensor
            if isinstance(motion[key], torch.Tensor):
                downsampled_motion[key] = motion[key][::skip].clone()
            else:
                downsampled_motion[key] = copy.deepcopy(motion[key])

        downsampled_motion["fps"] = target_frame_rate
        return EasyDict(downsampled_motion)
