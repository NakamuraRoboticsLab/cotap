import os
import sys

import casadi
import numpy as np
import pinocchio as pin

from .weighted_moving_filter import WeightedMovingFilter

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

class H1_ArmIK:
    def __init__(self, robot_config, unit_test=False, visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.unit_test = unit_test
        self.visualization = visualization

        # === Model Load ===
        urdf_path = robot_config["ASSET_FILE"]
        mesh_dir = robot_config["ASSET_ROOT"]
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)
        self.model = self.robot.model
        self.data = self.model.createData()

        # === Joints to Lock (只锁定下肢和躯干，不包含手指和手部) ===
        self.mixed_jointsToLockIDs = [
            "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
            "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
            "torso_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_ankle_joint", "right_ankle_joint"
        ]
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.zeros(self.model.nq),
        )
        self.reduced_model = self.reduced_robot.model

        # 检查 joint 是否存在
        left_joint_name = "left_elbow_joint"
        right_joint_name = "right_elbow_joint"
        left_ee_name = "left_elbow_ee"
        right_ee_name = "right_elbow_ee"
        offset = np.array([0.25, 0, 0])

        for joint_name, ee_name in [(left_joint_name, left_ee_name), (right_joint_name, right_ee_name)]:
            joint_id = self.reduced_model.getJointId(joint_name)
            print(f"{joint_name} id:", joint_id)
            if joint_id <= 0:
                raise ValueError(f"Joint {joint_name} not found in reduced_model! Available joints: {[j.name for j in self.reduced_model.joints]}")
            if not self.reduced_model.existFrame(ee_name):
                frame_SE3 = pin.SE3(np.eye(3), offset)
                self.reduced_model.addFrame(
                    pin.Frame(
                        ee_name,
                        joint_id,
                        frame_SE3,
                        pin.FrameType.OP_FRAME
                    )
                )
            else:
                print(f"Frame {ee_name} already exists!")

        for i, f in enumerate(self.reduced_model.frames):
            if f.name == "left_elbow_ee":
                print(f"Frame {f.name}: parent={f.parent}, type={f.type}, placement={f.placement}")

        self.reduced_data = self.reduced_model.createData()

        # === Casadi Symbolic Model (仅用于后续扩展) ===
        self.cmodel = pin.casadi.Model(self.reduced_model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.reduced_model.nq, 1)

        # === Filters ===
        self.init_data = np.zeros(self.reduced_model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 8)

    def solve_ik(self, target_joint_positions, current_arm_motor_q=None, current_arm_motor_dq=None):
        """
        直接返回目标关节角度，不做末端轨迹IK。
        target_joint_positions: 目标关节角度（np.array, shape=[nq]）
        current_arm_motor_q: 当前关节角度（可选）
        current_arm_motor_dq: 当前关节速度（可选）
        """
        # 可选：平滑滤波
        self.smooth_filter.add_data(target_joint_positions)
        filtered_q = self.smooth_filter.filtered_data

        # 计算关节力矩（可选，通常为零）
        if current_arm_motor_dq is not None:
            v = current_arm_motor_dq * 0.0
        else:
            v = (filtered_q - self.init_data) * 0.0

        self.init_data = filtered_q

        tau_ff = pin.rnea(self.reduced_model, self.reduced_data, filtered_q, v, np.zeros(self.reduced_model.nv))

        return filtered_q, tau_ff