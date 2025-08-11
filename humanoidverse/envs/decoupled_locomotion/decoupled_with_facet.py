from isaacgym.torch_utils import *
from humanoidverse.utils.torch_utils import (
    generate_sphere_sample_params,
    apply_sphere_sample_to_segments,
    sample_3d_directions,
)

import torch
from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma_diff_force import LeggedRobotDecoupledLocomotionStanceHeightWBCForce
import numpy as np
from typing import Optional, Dict, List, Tuple

from isaac_utils.rotations import (
    my_quat_rotate,
)
from humanoidverse.envs.env_utils.visualization import Point

from loguru import logger
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float
from isaacgym import gymtorch, gymapi, gymutil

DEBUG = False

def clamp_norm(x: torch.Tensor, min_norm: float = 0.0,
               max_norm: float = 1.0) -> torch.Tensor:
    """
    将张量范数限制在指定区间 (Clamp tensor norm to specified range)
    
    Args:
        x: 输入张量 (Input tensor)
        min_norm: 最小范数 (Minimum norm)
        max_norm: 最大范数 (Maximum norm)
    
    Returns:
        范数被限制的张量 (Tensor with clamped norm)
    """
    norm = x.norm(dim=-1, keepdim=True)
    return x * torch.clamp(norm, min_norm, max_norm) / norm.clamp_min(1e-8)


def saturate(x: torch.Tensor, a: float) -> torch.Tensor:
    """
    饱和函数，用于力的软限制 (Saturation function for soft force limits)
    
    实现软饱和以避免数值不稳定
    Implements soft saturation to avoid numerical instability
    
    Args:
        x: 输入力向量 (Input force vector)
        a: 饱和参数 (Saturation parameter)
    
    Returns:
        饱和后的力向量 (Saturated force vector)
    """
    norm = x.norm(dim=-1, keepdim=True)
    return (x / norm.clamp_min(1e-6)) * torch.log1p(norm / a) * a


class EMA:
    """
    指数移动平均滤波器 (Exponential Moving Average Filter)
    
    用于状态平滑和噪声抑制
    Used for state smoothing and noise suppression
    """
    
    def __init__(self, data: torch.Tensor, alphas: List[float]):
        """
        初始化EMA滤波器
        
        Args:
            data: 初始数据 (Initial data)
            alphas: 平滑系数列表 (List of smoothing factors)
        """
        self.ema = data.unsqueeze(1).repeat(1, len(alphas), 1)
        self.alphas = torch.tensor(alphas, device=data.device)

    def update(self, data: torch.Tensor):
        """更新EMA值 (Update EMA values)"""
        self.ema = (self.alphas.view(1, -1, 1) * data.unsqueeze(1) +
                    (1 - self.alphas.view(1, -1, 1)) * self.ema)

    def get_smoothed(self, alpha_idx: int = 1) -> torch.Tensor:
        """获取平滑后的值 (Get smoothed values)"""
        return self.ema[:, alpha_idx, :]
    
class LeggedRobotDecoupledLocomotionWithFACET(LeggedRobotDecoupledLocomotionStanceHeightWBCForce):
    # 阻抗控制指令模式常量 (Impedance control command mode constants)
    CMD_COMPLIANT = 0    # 柔顺模式 (Compliant mode)
    CMD_LINVEL = 1       # 线速度模式 (Linear velocity mode)
    CMD_POSITION = 2     # 位置模式 (Position mode)
    CMD_LARGE_FORCE = 3  # 大力干扰模式 (Large force disturbance mode)

    def __init__(self, config, device):
        self.init_done = False
        # 首先调用父类初始化 (Initialize parent class first)
        super().__init__(config, device)

        # FACET阻抗控制配置 (FACET impedance control configuration)
        self.linear_kp_range = self.config.facet_params.linear_kp_range
        self.angular_kp_range = self.config.facet_params.angular_kp_range
        self.force_saturate = self.config.facet_params.force_saturate
        self.temporal_smoothing = self.config.facet_params.temporal_smoothing
        self.virtual_mass = self.config.facet_params.virtual_mass

        # 初始化FACET阻抗控制系统 (Initialize FACET impedance control system)
        self._init_impedance_control()

        # 初始化外力应用系统 (Initialize external force application system)
        self._init_force_application()

        # logger.info(f"FACET阻抗控制环境初始化完成 (FACET impedance control "
        #             f"environment initialized) - {self.num_envs} envs")

    def _init_impedance_control(self):
        """初始化FACET阻抗控制系统 (Initialize FACET impedance system)"""
        
        # 阻抗控制参数 (Impedance control parameters)
        self.lin_kp = torch.zeros(self.num_envs, 1, device=self.device)
        self.lin_kd = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_kp = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_kd = torch.zeros(self.num_envs, 1, device=self.device)

        # 目标状态 (Target states)
        self.command_setpos_w = torch.zeros(self.num_envs, 3,
                                            device=self.device)
        self.command_setrpy_w = torch.zeros(self.num_envs, 3,
                                            device=self.device)
        self.set_linvel = torch.zeros(self.num_envs, 3, device=self.device)

        # 虚拟动力学参数 (Virtual dynamics parameters)
        self.virtual_mass_tensor = torch.ones(self.num_envs, 1, 
                                               device=self.device) * self.virtual_mass

        # 外力状态 (External force states)
        self.force_ext_w = torch.zeros(self.num_envs, 3, device=self.device)

        # 控制模式和时序 (Control mode and timing)
        self.impedance_command_mode = torch.zeros(
            self.num_envs, 1, dtype=torch.int, device=self.device)
        self.impedance_command_time = torch.zeros(
            self.num_envs, device=self.device)

        # 参考轨迹积分缓冲区 (Reference trajectory integration buffers)
        bshape = (self.num_envs, self.temporal_smoothing + 1)
        self.ref_lin_vel_w = torch.zeros(*bshape, 3, device=self.device)
        self.ref_pos_w = torch.zeros(*bshape, 3, device=self.device)
        self.ref_yaw_w = torch.zeros(*bshape, 1, device=self.device)

        # # 力干扰对象 (Force disturbance objects)
        # self.constant_force = ForceDisturbance(
        #     self.num_envs, self.device, "constant")
        # self.impulse_force = ForceDisturbance(
        #     self.num_envs, self.device, "impulse")
        # self.spring_force = ForceDisturbance(
        #     self.num_envs, self.device, "spring")

        # EMA滤波器 (EMA filters)
        # 使用模拟器的正确根速度项 (Use correct root velocity terms from simulator)
        if hasattr(self, 'base_lin_vel') and hasattr(self, 'base_ang_vel'):
            # 使用已转换到机器人本体坐标系的速度
            # Use velocities already transformed to robot body frame
            self.lin_vel_ema = EMA(self.base_lin_vel, [0.0, 0.5, 0.8])
            self.ang_vel_ema = EMA(self.base_ang_vel, [0.0, 0.5, 0.8])
        else:
            # 如果base_lin_vel和base_ang_vel还没有初始化，使用零向量
            # If base velocities are not initialized yet, use zero vectors
            dummy_data = torch.zeros(self.num_envs, 3, device=self.device)
            self.lin_vel_ema = EMA(dummy_data, [0.0, 0.5, 0.8])
            self.ang_vel_ema = EMA(dummy_data, [0.0, 0.5, 0.8])

    def _init_force_application(self):
        """初始化外力应用系统 (Initialize external force application system)"""
        # 创建力应用张量 (Create force application tensors)
        self.external_force_tensor = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.device)
        self.external_torque_tensor = torch.zeros(
            self.num_envs, self.num_bodies, 3, device=self.device)
        
    def step(self, actor_state):
        """环境步进 (Environment step)"""
        
        # 更新阻抗控制 (Update impedance control)
        self.update_impedance_control()

        # # 更新力干扰 (Update force disturbances)
        # self.update_force_disturbances()

        # 执行父类步进 (Execute parent class step)
        result = super().step(actor_state)
        
        # # 调试可视化 (Debug visualization)
        # if hasattr(self, 'simulator'):
        #     self.debug_visualization()
        
        return result

    def update_impedance_control(self):
        """更新阻抗控制 (Update impedance control)"""
        
        # 更新控制指令 (Update control commands)
        self.update_impedance_command()

        # 积分参考轨迹 (Integrate reference trajectory)
        self._integrate_reference_trajectory()

        # 更新EMA滤波器 (Update EMA filters)
        # 使用机器人本体坐标系的速度 (Use velocities in robot body frame)
        if hasattr(self, 'base_lin_vel') and hasattr(self, 'base_ang_vel'):
            self.lin_vel_ema.update(self.base_lin_vel)
            self.ang_vel_ema.update(self.base_ang_vel)

    def update_impedance_command(self):
        """更新阻抗控制指令 (Update impedance control commands)"""
        
        # 周期性采样新指令 (Periodically sample new commands)
        sample_command = ((self.impedance_command_time - 50) % 150 == 0)
        sample_command = sample_command & (
            torch.rand(self.num_envs, device=self.device) < 0.5)

        if sample_command.any():
            sample_ids = sample_command.nonzero().squeeze(-1)

            # 随机选择控制模式 (Randomly select control mode)
            # 概率分布：柔顺40%，速度50%，位置10%，大力0%
            probs = torch.tensor([0.4, 0.5, 0.1, 0.0], device=self.device)
            mode = torch.multinomial(
                probs, num_samples=len(sample_ids), replacement=True)

            # 分配不同模式 (Assign different modes)
            self.sample_command_world(sample_ids[mode == 0])
            self.sample_command_setvel(sample_ids[mode == 1])
            self.sample_command_compliant(sample_ids[mode == 2])
            self.sample_command_large(sample_ids[mode == 3])

        self.impedance_command_time += 1

    def _integrate_reference_trajectory(self):
        """积分参考轨迹 (Integrate reference trajectory)"""
        dt = self.dt

        # 计算期望位置 (Calculate desired position)
        setpos_w = torch.where(
            (self.impedance_command_mode == self.CMD_LINVEL).reshape(
                self.num_envs, 1, 1),
            self.ref_pos_w + (self.lin_kd / self.lin_kp.clamp_min(1e-6) *
                              self.set_linvel).unsqueeze(1),
            self.command_setpos_w.unsqueeze(1)
        )

        # 计算参考加速度 (Calculate reference acceleration)
        ref_acc_w = (
            self.lin_kp.reshape(self.num_envs, 1, 1) *
            (setpos_w - self.ref_pos_w) +
            self.lin_kd.reshape(self.num_envs, 1, 1) *
            (0.0 - self.ref_lin_vel_w) +
            saturate(self.force_ext_w, self.force_saturate).reshape(
                self.num_envs, 1, 3)
        ) / self.virtual_mass_tensor.unsqueeze(1)

        # 保持Z方向稳定 (Keep Z direction stable)
        ref_acc_w[..., 2] = 0.0

        # 积分速度和位置 (Integrate velocity and position)
        ref_vel_w = self.ref_lin_vel_w + ref_acc_w * dt
        ref_vel_w[..., 2] = 0.0
        ref_vel_w = clamp_norm(ref_vel_w, 0., 2.4)

        self.ref_lin_vel_w = ref_vel_w
        self.ref_pos_w.add_(self.ref_lin_vel_w * dt)

    # ================================================================
    # 指令采样方法 (Command Sampling Methods)
    # ================================================================

    def sample_command_world(self, env_ids: torch.Tensor):
        """
        采样世界位置指令 (Sample world position command)
        
        Args:
            env_ids: 环境ID (Environment IDs)
        """
        if len(env_ids) == 0:
            return

        # 采样阻抗增益 (Sample impedance gains)
        lin_kp = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.linear_kp_range)
        lin_kd = 1.8 * lin_kp.sqrt()

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = lin_kp
        self.ang_kd[env_ids] = lin_kd

        # 采样目标位置 (Sample target position)
        offset = torch.zeros(len(env_ids), 3, device=self.device)
        offset[:, 0].uniform_(0.6, 1.2)  # X方向前进 (X direction forward)
        offset[:, 1].uniform_(-0.6, 0.6)  # Y方向左右 (Y direction left/right)

        # 使用模拟器的正确根状态 (Use correct root states from simulator)
        if (hasattr(self, 'simulator') and
                hasattr(self.simulator, 'robot_root_states')):
            current_pos = self.simulator.robot_root_states[env_ids, :3]
            self.command_setpos_w[env_ids] = current_pos + offset

        # 采样目标偏航角 (Sample target yaw angle)
        target_yaw = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            -np.pi/2, np.pi/2)
        self.command_setrpy_w[env_ids, 2:3] = target_yaw

        self.set_linvel[env_ids] = 0.0
        self.impedance_command_mode[env_ids] = self.CMD_POSITION

    def sample_command_setvel(self, env_ids: torch.Tensor):
        """
        采样速度指令 (Sample velocity command)
        
        Args:
            env_ids: 环境ID (Environment IDs)
        """
        if len(env_ids) == 0:
            return

        # 采样阻抗增益 (Sample impedance gains)
        lin_kp = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.linear_kp_range)
        lin_kd = 1.8 * lin_kp.sqrt()

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = lin_kp
        self.ang_kd[env_ids] = lin_kd

        # 采样目标速度 (Sample target velocity)
        set_linvel = torch.zeros(len(env_ids), 3, device=self.device)
        set_linvel[:, 0].uniform_(0.4, 1.5)  # 前进速度 (Forward velocity)
        self.set_linvel[env_ids] = set_linvel

        # 采样目标偏航角 (Sample target yaw angle)
        target_yaw = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            -np.pi/2, np.pi/2)
        self.command_setrpy_w[env_ids, 2:3] = target_yaw

        self.impedance_command_mode[env_ids] = self.CMD_LINVEL

    def sample_command_compliant(self, env_ids: torch.Tensor):
        """
        采样柔顺指令 (Sample compliant command)
        
        在柔顺模式下，机器人具有零刚度，表现出柔顺特性
        In compliant mode, robot has zero stiffness and shows compliant behavior
        
        Args:
            env_ids: 环境ID (Environment IDs)
        """
        if len(env_ids) == 0:
            return

        lin_kp = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            *self.linear_kp_range)
        lin_kd = 1.8 * lin_kp.sqrt()

        # 柔顺模式：零刚度 (Compliant mode: zero stiffness)
        self.lin_kp[env_ids] = 0.0
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = lin_kp
        self.ang_kd[env_ids] = lin_kd

        self.set_linvel[env_ids] = 0.0
        self.impedance_command_mode[env_ids] = self.CMD_COMPLIANT

    def sample_command_large(self, env_ids: torch.Tensor):
        """
        采样大力干扰指令 (Sample large force disturbance command)
        
        用于测试机器人在大外力下的鲁棒性
        Used to test robot robustness under large external forces
        
        Args:
            env_ids: 环境ID (Environment IDs)
        """
        if len(env_ids) == 0:
            return

        # 大力模式使用高刚度 (Large force mode uses high stiffness)
        lin_kp = torch.empty(len(env_ids), 1, device=self.device).uniform_(
            24.0, 48.0)
        lin_kd = 1.8 * lin_kp.sqrt()

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = lin_kp
        self.ang_kd[env_ids] = lin_kd

        # 随机目标偏航角 (Random target yaw angle)
        target_yaw = (torch.randint(0, 2, (len(env_ids), 1),
                                    device=self.device) * np.pi)
        self.command_setrpy_w[env_ids, 2:3] = target_yaw

        # # 采样弹簧力干扰 (Sample spring force disturbance)
        # if len(env_ids) > 0:
        #     self.spring_force.force[env_ids] = (
        #         torch.randn(len(env_ids), 3, device=self.device) * 200.0)
        #     self.spring_force.duration[env_ids] = (
        #         torch.rand(len(env_ids), 1, device=self.device) * 10.0 + 5.0)
        #     self.spring_force.time[env_ids] = 0.0

        self.impedance_command_mode[env_ids] = self.CMD_LARGE_FORCE
        
    # ================================================================
    # 信息和调试方法 (Information and Debug Methods)
    # ================================================================

    # def get_impedance_info(self) -> Dict:
    #     """
    #     获取阻抗控制信息 (Get impedance control information)
        
    #     Returns:
    #         包含控制器状态信息的字典 (Dictionary containing controller state info)
    #     """
    #     info = {
    #         'impedance_mode': self.impedance_command_mode.float().mean().item(),
    #         'lin_kp_mean': self.lin_kp.mean().item(),
    #         'lin_kd_mean': self.lin_kd.mean().item(),
    #         'ang_kp_mean': self.ang_kp.mean().item(),
    #         'ang_kd_mean': self.ang_kd.mean().item(),
    #         'force_magnitude': self.force_ext_w.norm(dim=-1).mean().item(),
    #         'virtual_mass': self.virtual_mass_tensor.mean().item(),
    #     }

    #     # # 添加力干扰信息 (Add force disturbance information)
    #     # info.update(self.constant_force.get_info())
    #     # info.update(self.impulse_force.get_info())
    #     # info.update(self.spring_force.get_info())

    #     # 添加参考轨迹信息 (Add reference trajectory information)
    #     if hasattr(self, 'ref_lin_vel_w') and hasattr(self, 'ref_pos_w'):
    #         info['ref_vel_magnitude'] = self.ref_lin_vel_w[:, 0, :].norm(
    #             dim=-1).mean().item()
    #         if hasattr(self, 'root_states') and self.root_states is not None:
    #             pos_error = (self.ref_pos_w[:, 0, :] - 
    #                          self.root_states[:, :3]).norm(dim=-1)
    #             info['pos_error_magnitude'] = pos_error.mean().item()

    #     return info

    # def log_impedance_stats(self):
    #     """记录阻抗控制统计信息 (Log impedance control statistics)"""
    #     info = self.get_impedance_info()
    #     logger.info(f"阻抗控制统计 (Impedance control stats): {info}")

    # def get_reward_components(self) -> Dict[str, torch.Tensor]:
    #     """
    #     获取所有奖励组件 (Get all reward components)
        
    #     Returns:
    #         包含各个奖励组件的字典 (Dictionary containing reward components)
    #     """
    #     rewards = {}
        
    #     # 获取所有奖励函数 (Get all reward functions)
    #     rewards['pos_tracking'] = self.impedance_pos_tracking()
    #     rewards['vel_tracking'] = self.impedance_vel_tracking()
    #     rewards['force_resistance'] = self.force_resistance()
    #     rewards['mode_stability'] = self.impedance_mode_stability()
    #     rewards['energy_efficiency'] = self.impedance_energy_efficiency()
        
    #     return rewards

    # def debug_visualization(self):
    #     """
    #     调试可视化 (Debug visualization)
        
    #     在仿真环境中绘制调试信息
    #     Draw debug information in simulation environment
    #     """
    #     if (not hasattr(self, 'simulator') or
    #             not hasattr(self.simulator, 'viewer')):
    #         return
        
    #     if (not hasattr(self.simulator, 'gym') or
    #             self.simulator.viewer is None):
    #         return

    #     # 清除之前的绘制 (Clear previous drawings)
    #     self.simulator.clear_lines()

    #     # 绘制目标位置 (Draw target positions)
    #     if hasattr(self, 'command_setpos_w'):
    #         for i in range(min(self.num_envs, 10)):  # 只绘制前10个环境
    #             target_pos = self.command_setpos_w[i].cpu().numpy()
                
    #             # 获取当前末端执行器位置 (Get current end-effector positions)
    #             left_hand_pos = self.simulator._rigid_body_pos[
    #                 i, self.left_hand_link_index].cpu().numpy()
    #             right_hand_pos = self.simulator._rigid_body_pos[
    #                 i, self.right_hand_link_index].cpu().numpy()
                
    #             # 绘制目标位置球体 (Draw target position spheres)
    #             # 左手目标位置 - 绿色 (Left hand target - Green)
    #             if self.impedance_command_mode[i, 0] == self.CMD_POSITION:
    #                 left_target = target_pos + np.array([-0.3, 0.0, 0.0])
    #                 self.simulator.draw_sphere(
    #                     pos=left_target,
    #                     radius=0.05,
    #                     color=gymapi.Vec3(0.0, 1.0, 0.0),  # 绿色
    #                     env_id=i,
    #                     pos_id=i*2
    #                 )
                
    #             # 右手目标位置 - 蓝色 (Right hand target - Blue)
    #             if self.impedance_command_mode[i, 0] == self.CMD_POSITION:
    #                 right_target = target_pos + np.array([0.3, 0.0, 0.0])
    #                 self.simulator.draw_sphere(
    #                     pos=right_target,
    #                     radius=0.05,
    #                     color=gymapi.Vec3(0.0, 0.0, 1.0),  # 蓝色
    #                     env_id=i,
    #                     pos_id=i*2+1
    #                 )
                
    #             # 绘制当前位置到目标位置的连线 (Draw lines from current to target)
    #             if self.impedance_command_mode[i, 0] == self.CMD_POSITION:
    #                 # 左手连线 (Left hand line)
    #                 left_target = target_pos + np.array([-0.3, 0.0, 0.0])
    #                 self.simulator.draw_line(
    #                     start_point=gymapi.Vec3(left_hand_pos[0],
    #                                             left_hand_pos[1],
    #                                             left_hand_pos[2]),
    #                     end_point=gymapi.Vec3(left_target[0],
    #                                           left_target[1],
    #                                           left_target[2]),
    #                     color=gymapi.Vec3(0.0, 1.0, 0.0),  # 绿色线条
    #                     env_id=i
    #                 )
                    
    #                 # 右手连线 (Right hand line)
    #                 right_target = target_pos + np.array([0.3, 0.0, 0.0])
    #                 self.simulator.draw_line(
    #                     start_point=gymapi.Vec3(right_hand_pos[0],
    #                                             right_hand_pos[1],
    #                                             right_hand_pos[2]),
    #                     end_point=gymapi.Vec3(right_target[0],
    #                                           right_target[1],
    #                                           right_target[2]),
    #                     color=gymapi.Vec3(0.0, 0.0, 1.0),  # 蓝色线条
    #                     env_id=i
    #                 )
                
    #             # 绘制速度模式的速度向量 (Draw velocity vectors for velocity mode)
    #             if self.impedance_command_mode[i, 0] == self.CMD_LINVEL:
    #                 vel_scale = 0.5  # 速度向量缩放因子
    #                 vel_vec = self.set_linvel[i].cpu().numpy() * vel_scale
                    
    #                 # 左手速度向量 - 黄色 (Left hand velocity vector - Yellow)
    #                 left_vel_end = left_hand_pos + vel_vec
    #                 self.simulator.draw_line(
    #                     start_point=gymapi.Vec3(left_hand_pos[0],
    #                                             left_hand_pos[1],
    #                                             left_hand_pos[2]),
    #                     end_point=gymapi.Vec3(left_vel_end[0],
    #                                           left_vel_end[1],
    #                                           left_vel_end[2]),
    #                     color=gymapi.Vec3(1.0, 1.0, 0.0),  # 黄色
    #                     env_id=i
    #                 )
                    
    #                 # 右手速度向量 - 橙色 (Right hand velocity vector - Orange)
    #                 right_vel_end = right_hand_pos + vel_vec
    #                 self.simulator.draw_line(
    #                     start_point=gymapi.Vec3(right_hand_pos[0],
    #                                             right_hand_pos[1],
    #                                             right_hand_pos[2]),
    #                     end_point=gymapi.Vec3(right_vel_end[0],
    #                                           right_vel_end[1],
    #                                           right_vel_end[2]),
    #                     color=gymapi.Vec3(1.0, 0.5, 0.0),  # 橙色
    #                     env_id=i
    #                 )
                
    #             # 绘制外力向量 (Draw external force vectors)
    #             if torch.norm(self.force_ext_w[i]) > 0.1:
    #                 force_scale = 0.1  # 力向量缩放因子
    #                 force_vec = self.force_ext_w[i].cpu().numpy() * force_scale
                    
    #                 # 在机器人基座位置绘制力向量 - 红色 (Draw force at robot base)
    #                 base_pos = self.simulator.robot_root_states[i, :3].cpu().numpy()
    #                 force_end = base_pos + force_vec
    #                 self.simulator.draw_line(
    #                     start_point=gymapi.Vec3(base_pos[0],
    #                                             base_pos[1],
    #                                             base_pos[2] + 0.5),
    #                     end_point=gymapi.Vec3(force_end[0],
    #                                           force_end[1],
    #                                           force_end[2] + 0.5),
    #                     color=gymapi.Vec3(1.0, 0.0, 0.0),  # 红色
    #                     env_id=i
    #                 )
                    
    #                 # 在力向量端点绘制球体表示力的大小 (Draw sphere at force end)
    #                 force_magnitude = torch.norm(
    #                     self.force_ext_w[i]).cpu().numpy()
    #                 sphere_radius = min(0.02 + force_magnitude * 0.01, 0.1)
    #                 self.simulator.draw_sphere(
    #                     pos=force_end + np.array([0, 0, 0.5]),
    #                     radius=sphere_radius,
    #                     color=gymapi.Vec3(1.0, 0.0, 0.0),  # 红色
    #                     env_id=i,
    #                     pos_id=i*4+2
    #                 )
                
    #             # 绘制阻抗控制模式指示器 (Draw impedance mode indicators)
    #             mode_pos = (self.simulator.robot_root_states[i, :3].cpu().numpy() +
    #                         np.array([0, 0, 1.0]))
    #             mode_color = gymapi.Vec3(1.0, 1.0, 1.0)  # 默认白色
                
    #             # 根据模式设置不同颜色 (Set different colors based on mode)
    #             if self.impedance_command_mode[i, 0] == self.CMD_COMPLIANT:
    #                 mode_color = gymapi.Vec3(0.8, 0.8, 0.8)  # 灰色-柔顺模式
    #             elif self.impedance_command_mode[i, 0] == self.CMD_LINVEL:
    #                 mode_color = gymapi.Vec3(1.0, 1.0, 0.0)  # 黄色-速度模式
    #             elif self.impedance_command_mode[i, 0] == self.CMD_POSITION:
    #                 mode_color = gymapi.Vec3(0.0, 1.0, 0.0)  # 绿色-位置模式
    #             elif self.impedance_command_mode[i, 0] == self.CMD_LARGE_FORCE:
    #                 mode_color = gymapi.Vec3(1.0, 0.0, 1.0)  # 紫红色-大力模式
                
    #             # 绘制模式指示器球体 (Draw mode indicator sphere)
    #             self.simulator.draw_sphere(
    #                 pos=mode_pos,
    #                 radius=0.03,
    #                 color=mode_color,
    #                 env_id=i,
    #                 pos_id=i*4+3
    #             )

    # def set_impedance_mode(self, env_ids: torch.Tensor, mode: int):
    #     """
    #     手动设置阻抗控制模式 (Manually set impedance control mode)
        
    #     Args:
    #         env_ids: 环境ID (Environment IDs)
    #         mode: 控制模式 (Control mode)
    #     """
    #     if len(env_ids) == 0:
    #         return

    #     if mode == self.CMD_COMPLIANT:
    #         self.sample_command_compliant(env_ids)
    #     elif mode == self.CMD_LINVEL:
    #         self.sample_command_setvel(env_ids)
    #     elif mode == self.CMD_POSITION:
    #         self.sample_command_world(env_ids)
    #     elif mode == self.CMD_LARGE_FORCE:
    #         self.sample_command_large(env_ids)
    #     else:
    #         logger.warning(f"未知的阻抗控制模式: {mode}")

    # def get_impedance_mode_name(self, mode: int) -> str:
    #     """
    #     获取阻抗控制模式名称 (Get impedance control mode name)
        
    #     Args:
    #         mode: 控制模式 (Control mode)
            
    #     Returns:
    #         模式名称 (Mode name)
    #     """
    #     mode_names = {
    #         self.CMD_COMPLIANT: "柔顺模式 (Compliant)",
    #         self.CMD_LINVEL: "速度跟踪 (Linear Velocity)",
    #         self.CMD_POSITION: "位置跟踪 (Position)",
    #         self.CMD_LARGE_FORCE: "大力干扰 (Large Force)"
    #     }
    #     return mode_names.get(mode, f"未知模式 (Unknown): {mode}")

    # def _reset_impedance_control(self, env_ids):
    #     """重置阻抗控制状态 (Reset impedance control states)"""
        
    #     # 重置到默认控制模式 (Reset to default control mode)
    #     self.sample_command_world(env_ids)

    #     # 重置力干扰 (Reset force disturbances)
    #     self.constant_force.reset(env_ids)
    #     self.impulse_force.reset(env_ids)
    #     self.spring_force.reset(env_ids)

    #     # 重置参考轨迹 (Reset reference trajectories)
    #     if hasattr(self, 'root_states') and self.root_states is not None:
    #         current_pos = self.root_states[env_ids, :3]
    #         current_yaw = torch.atan2(
    #             self.root_states[env_ids, 4], self.root_states[env_ids, 3])

    #         self.ref_pos_w[env_ids] = current_pos.unsqueeze(1)
    #         self.ref_lin_vel_w[env_ids] = 0.0
    #         self.ref_yaw_w[env_ids] = current_yaw.unsqueeze(1).unsqueeze(1)

    #     # 重置外力 (Reset external forces)
    #     self.force_ext_w[env_ids] = 0.0
    #     if hasattr(self, 'external_force_tensor'):
    #         self.external_force_tensor[env_ids] = 0.0
    #         self.external_torque_tensor[env_ids] = 0.0

    # def _init_buffers(self):
    #     super()._init_buffers()
    
    # def set_is_evaluating(self, command=None):
    #     super().set_is_evaluating()

    ########################### FEET REWARDS ###########################

    def _reward_impedance_pos_tracking(self):
        """
        阻抗位置跟踪奖励 (Impedance position tracking reward)
        
        基于位置误差计算奖励，误差越小奖励越大
        Calculate reward based on position error, smaller error gives higher reward
        
        Returns:
            位置跟踪奖励 (Position tracking reward)
        """
        if (not hasattr(self, 'command_setpos_w') or
                not hasattr(self, 'simulator') or
                not hasattr(self.simulator, 'robot_root_states')):
            return torch.zeros(self.num_envs, device=self.device)

        # 计算位置误差 (Calculate position error)
        # 使用模拟器的正确根状态 (Use correct root states from simulator)
        current_pos = self.simulator.robot_root_states[:, :3]
        pos_error = (current_pos - self.command_setpos_w).norm(dim=-1)
        return torch.exp(-pos_error / 0.5)

    def _reward_impedance_vel_tracking(self):
        """
        阻抗速度跟踪奖励 (Impedance velocity tracking reward)
        
        基于速度误差计算奖励
        Calculate reward based on velocity error
        
        Returns:
            速度跟踪奖励 (Velocity tracking reward)
        """
        if (not hasattr(self, 'set_linvel') or
                not hasattr(self, 'simulator') or
                not hasattr(self.simulator, 'robot_root_states')):
            return torch.zeros(self.num_envs, device=self.device)

        # 计算速度误差 (Calculate velocity error)
        # 使用模拟器的正确根状态速度 (Use correct root state velocities from simulator)
        current_vel = self.simulator.robot_root_states[:, 7:10]
        vel_error = (current_vel - self.set_linvel).norm(dim=-1)
        return torch.exp(-vel_error / 1.0)

    def _reward_force_resistance(self):
        """
        力干扰抵抗奖励 (Force disturbance resistance reward)
        
        基于外力大小给予奖励，鼓励机器人抵抗外力干扰
        Reward based on external force magnitude, encouraging force resistance
        
        Returns:
            力抵抗奖励 (Force resistance reward)
        """
        if not hasattr(self, 'force_ext_w'):
            return torch.zeros(self.num_envs, device=self.device)

        force_magnitude = self.force_ext_w.norm(dim=-1)
        return torch.exp(-force_magnitude / 100.0)

    def _reward_impedance_mode_stability(self):
        """
        阻抗模式稳定性奖励 (Impedance mode stability reward)
        
        基于当前控制模式给予不同的稳定性奖励
        Give different stability rewards based on current control mode
        
        Returns:
            模式稳定性奖励 (Mode stability reward)
        """
        stability = torch.ones(self.num_envs, device=self.device)

        # 柔顺模式稳定性调整 (Compliant mode stability adjustment)
        compliant_mask = (self.impedance_command_mode ==
                          self.CMD_COMPLIANT).squeeze(-1)
        stability[compliant_mask] *= 0.8

        # 大力模式稳定性调整 (Large force mode stability adjustment)
        large_force_mask = (self.impedance_command_mode ==
                            self.CMD_LARGE_FORCE).squeeze(-1)
        stability[large_force_mask] *= 1.2

        return stability

    def _reward_impedance_energy_efficiency(self):
        """
        阻抗控制能效奖励 (Impedance control energy efficiency reward)
        
        鼓励能效的控制策略
        Encourage energy-efficient control strategies
        
        Returns:
            能效奖励 (Energy efficiency reward)
        """
        if not hasattr(self, 'lin_kp') or not hasattr(self, 'force_ext_w'):
            return torch.zeros(self.num_envs, device=self.device)

        # 基于刚度和外力的能效计算 (Energy efficiency based on stiffness and external force)
        energy_cost = self.lin_kp.squeeze(-1) * self.force_ext_w.norm(dim=-1)
        return torch.exp(-energy_cost / 500.0)
    
    ######################### Observations #########################

    