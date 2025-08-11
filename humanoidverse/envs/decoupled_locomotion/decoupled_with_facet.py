from isaacgym.torch_utils import *
from humanoidverse.utils.torch_utils import (
    generate_sphere_sample_params,
    apply_sphere_sample_to_segments,
    sample_3d_directions,
)

import torch
from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma_diff_force import LeggedRobotDecoupledLocomotionStanceHeightWBCForce
import numpy as np
# from typing import Optional, Dict, List, Tuples

from isaac_utils.rotations import (
    my_quat_rotate,
)
from humanoidverse.envs.env_utils.visualization import Point

from loguru import logger
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float
from isaacgym import gymtorch, gymapi, gymutil

DEBUG = False
class LeggedRobotDecoupledLocomotionWithFACET(LeggedRobotDecoupledLocomotionStanceHeightWBCForce):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)

    def _init_buffers(self):
        super()._init_buffers()
    
    # def set_is_evaluating(self, command=None):
    #     super().set_is_evaluating()

    ########################### FEET REWARDS ###########################
    
    # def _reward_tracking_upper_body_dofs(self):
    #     # Reward the difference between the waist dof pos and the reference
    #     upper_body_pos = self.simulator.dof_pos[:, self.upper_dof_indices]
    #     upper_body_dofs_error =  torch.sum(torch.square(upper_body_pos - self.ref_upper_dof_pos), dim=1)
    #     upper_body_dofs_tracking_reward =  torch.exp(-upper_body_dofs_error/(self.upper_body_tracking_sigma.squeeze(-1)))
    #     self.upper_body_dofs_tracking_reward += upper_body_dofs_tracking_reward
    #     return upper_body_dofs_tracking_reward

    ######################### Observations #########################

    