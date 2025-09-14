import os
import sys
import time

import numpy as np
import argparse
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

import pinocchio as pin
from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy

from termcolor import colored
from sim2real.utils.arm_ik.robot_arm_ik_g1_23dof import G1_29_ArmIK_NoWrists


class CompPolicy(LocoManipPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=0.25
    ):
        super().__init__(config, model_path, rl_rate, policy_action_scale)

    def policy_action(self):
        cmd_q = np.zeros(self.num_dofs)
        cmd_dq = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)
        # Get states
        robot_state_data = self.state_processor.robot_state_data
        # self.robot_state_data_shm[0] = robot_state_data
        # Apply upper body controller
        if self.upper_body_controller:
            # Control upper qpos and tau
            upper_body_qpos, _ = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )
            arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
            for i, idx in enumerate(arm_reduced_joint_indices):
                self.ref_upper_dof_pos[0, idx] = upper_body_qpos[i]
            # Zero out wrist joints
            wrist_joint_indices = [19, 20, 21, 26, 27, 28]
            for idx in wrist_joint_indices:
                self.ref_upper_dof_pos[0, idx - 15] = 0.0

        # Get policy action
        scaled_policy_action = self.rl_inference(robot_state_data)
        if self.get_ready_state:
            # 1. Set to Default Joint Position: interpolate from current dof_pos to default angles
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            # 2. No Policy Action: set to zero
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            # 3. Policy Action: apply policy action to current joint angles
            q_target = scaled_policy_action + self.default_dof_angles
        # import ipdb; ipdb.set_trace()
        # Clip q target
        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        # Send command
        cmd_q = q_target[0]
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])


    #################################
    # Compliance control functions #
    #################################

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # Use command line model_path if provided, otherwise use config model_path
    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = LocoManipPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )
    policy.run()
