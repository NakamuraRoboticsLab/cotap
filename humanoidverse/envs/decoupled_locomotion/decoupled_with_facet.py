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

DEBUG = False

def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x

def clamp_along(x: torch.Tensor, axis: torch.Tensor, min: float, max: float):
    projection = (x * axis).sum(dim=-1, keepdim=True)
    return x - projection * axis + projection.clamp(min, max) * axis


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
    Exponential Moving Average.
    
    Args:
        x: The tensor to compute the EMA of.
        gammas: The decay rates. Can be a single float or a list of floats.
    
    Example:
        >>> ema = EMA(x, gammas=[0.9, 0.99])
        >>> ema.update(x)
        >>> ema.ema
    """
    def __init__(self, x: torch.Tensor, gammas):
        self.gammas = torch.tensor(gammas, device=x.device)
        shape = (x.shape[0], len(self.gammas), *x.shape[1:])
        self.sum = torch.zeros(shape, device=x.device)
        shape = (x.shape[0], len(self.gammas), 1)
        self.cnt = torch.zeros(shape, device=x.device)

    def reset(self, env_ids: torch.Tensor):
        self.sum[env_ids] = 0.0
        self.cnt[env_ids] = 0.0
        
    def update(self, x: torch.Tensor):
        self.sum.mul_(self.gammas.unsqueeze(-1)).add_(x.unsqueeze(1))
        self.cnt.mul_(self.gammas.unsqueeze(-1)).add_(1.0)
        self.ema = self.sum / self.cnt
        return self.ema
    
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
        self.max_acc_xy = self.config.facet_params.max_acc_xy
        self.max_vel_xy = self.config.facet_params.max_vel_xy

        # 代理目标时间步 (Surrogate target time steps)
        self.surr_steps = [16, 24, 32]  # 可配置的多时间步 (Configurable)
        # self.surr_steps = [8, 16, 24]
        
        # 确保temporal_smoothing足够大以支持surr_steps (Ensure large enough)
        max_surr_step = max(self.surr_steps) if self.surr_steps else 32
        if self.temporal_smoothing < max_surr_step:
            self.temporal_smoothing = max_surr_step

        self.max_acc_xyz = self.max_acc_xy + (0.,)
        self.max_vel_xyz = self.max_vel_xy + (0.,)

        # 初始化FACET阻抗控制系统 (Initialize FACET impedance control system)
        self._init_impedance_control()

        # logger.info(f"FACET阻抗控制环境初始化完成 (FACET impedance control "
        #             f"environment initialized) - {self.num_envs} envs")

        self.pos_err_r = torch.zeros(self.num_envs, 1, device=self.device)

    def _init_impedance_control(self):
        """初始化FACET阻抗控制系统 (Initialize FACET impedance system)"""
        
        # 阻抗控制参数 (Impedance control parameters)
        self.lin_kp = torch.zeros(self.num_envs, 1, device=self.device)
        self.lin_kd = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_kp = torch.zeros(self.num_envs, 1, device=self.device)
        self.ang_kd = torch.zeros(self.num_envs, 1, device=self.device)

        # 目标状态 (Target states)
        self.command_setpos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_setrpy_w = torch.zeros(self.num_envs, 3, device=self.device)
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
        self.ref_yaw_vel_w = torch.zeros(*bshape, 1, device=self.device)
        self.ref_lin_acc_w = torch.zeros(*bshape, 3, device=self.device)
        
        # 代理位置目标 (Surrogate position target)
        self.surrogate_pos_target = torch.zeros(self.num_envs, len(self.surr_steps), 3, device=self.device)
        # 代理速度目标 (Surrogate velocity target)
        self.surrogate_lin_vel_target = torch.zeros(self.num_envs, len(self.surr_steps), 3, device=self.device)
        
        self.surrogate_yaw_target = torch.zeros(self.num_envs, len(self.surr_steps), 1, device=self.device)
        self.surrogate_yaw_vel_target = torch.zeros(self.num_envs, len(self.surr_steps), 1, device=self.device)

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
        
    def step(self, actor_state):
        """环境步进 (Environment step)"""
        
        # 更新阻抗控制 (Update impedance control)
        self.update_impedance_control()

        # 执行父类步进 (Execute parent class step)
        result = super().step(actor_state)

        # print("self.pos_err_r:", self.pos_err_r)
        
        return result

    def update_impedance_control(self):
        """更新阻抗控制 (Update impedance control)"""
        
        # 更新控制指令 (Update control commands)
        self.update_impedance_command()

        # 滚动更新参考轨迹缓冲区 (Rolling update of reference trajectory buffer)
        # 将历史数据向前滚动，为新数据腾出空间 (Roll historical data forward)
        self.ref_lin_vel_w[:, :-1] = self.ref_lin_vel_w[:, :-1].roll(1, dims=1)
        self.ref_pos_w[:, :-1] = self.ref_pos_w[:, :-1].roll(1, dims=1)
        self.ref_yaw_w[:, :-1] = self.ref_yaw_w[:, :-1].roll(1, dims=1)
        self.ref_yaw_vel_w[:, :-1] = self.ref_yaw_vel_w[:, :-1].roll(1, dims=1)

        # 更新当前状态到缓冲区首位 (Update current state to buffer front)
        if (hasattr(self, 'simulator') and
                hasattr(self.simulator, 'robot_root_states')):
            # 使用模拟器的当前状态 (Use current state from simulator)
            current_pos = self.simulator.robot_root_states[:, :3]
            current_vel = self.simulator.robot_root_states[:, 7:10]
            current_quat = self.simulator.robot_root_states[:, 3:7]
            
            # 计算当前偏航角 (Calculate current yaw angle)
            qw, qx, qy, qz = (current_quat[:, 0], current_quat[:, 1],
                              current_quat[:, 2], current_quat[:, 3])
            current_yaw = torch.atan2(2.0 * (qz * qy + qw * qx),
                                      1.0 - 2.0 * (qx**2 + qy**2))
            
            # 计算当前偏航角速度 (Calculate current yaw velocity)
            # 使用角速度的Z分量作为偏航角速度
            # Use Z component of angular velocity as yaw velocity
            current_ang_vel = self.simulator.robot_root_states[:, 10:13]
            current_yaw_vel = current_ang_vel[:, 2]  # Z component
            
            # 更新当前状态到缓冲区第一个位置 (Update current state to first position)
            self.ref_pos_w[:, 0] = current_pos
            self.ref_lin_vel_w[:, 0] = current_vel
            self.ref_yaw_w[:, 0] = current_yaw.unsqueeze(1)
            self.ref_yaw_vel_w[:, 0] = current_yaw_vel.unsqueeze(1)

        # 积分参考轨迹 (Integrate reference trajectory)
        self._integrate_reference_trajectory()

        # 更新代理位置目标 (Update surrogate position target)
        # 使用多个时间步的参考轨迹位置作为代理目标 (Use multi-time step reference positions)
        # 从参考轨迹中提取指定时间步的位置 (Extract positions at specified time steps)
        self.surrogate_pos_target = self.ref_pos_w[:, self.surr_steps]
        
        # 更新代理速度目标 (Update surrogate velocity target)
        # 从参考轨迹中提取指定时间步的速度 (Extract velocities at specified time steps)
        self.surrogate_lin_vel_target = self.ref_lin_vel_w[:, self.surr_steps]

        # print("self.surrogate_lin_vel_target:", self.surrogate_lin_vel_target)

        # 更新代理偏航角目标 (Update surrogate yaw target)
        # 从参考轨迹中提取指定时间步的偏航角 (Extract yaw angles at specified time steps)
        self.surrogate_yaw_target = self.ref_yaw_w[:, self.surr_steps]
        
        # 更新代理偏航角速度目标 (Update surrogate yaw velocity target)
        # 从参考轨迹中提取指定时间步的偏航角速度 (Extract yaw velocities at specified time steps)
        self.surrogate_yaw_vel_target = self.ref_yaw_vel_w[:, self.surr_steps]

        # 更新EMA滤波器 (Update EMA filters)
        # 使用机器人本体坐标系的速度 (Use velocities in robot body frame)
        # should be used in reward function for tracking
        if hasattr(self, 'base_lin_vel') and hasattr(self, 'base_ang_vel'):
            # print("EMA Update")
            self.lin_vel_ema.update(self.base_lin_vel)
            self.ang_vel_ema.update(self.base_ang_vel)

    def update_impedance_command(self):
        """更新阻抗控制指令 (Update impedance control commands)"""
        
        # 周期性采样新指令 (Periodically sample new commands)
        sample_command = ((self.impedance_command_time - 50) % 200 == 0) # every 150
        sample_command = sample_command & (
            torch.rand(self.num_envs, device=self.device) < 0.5)

        if sample_command.any():
            sample_ids = sample_command.nonzero().squeeze(-1)

            # 随机选择控制模式 (Randomly select control mode)
            # prob. distribution: pos., vel., comp., large force
            # hardcode now, should move in .yaml
            probs = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device) # [0.4, 0.5, 0.1, 0.0]
            mode = torch.multinomial(
                probs, num_samples=len(sample_ids), replacement=True)

            # 分配不同模式 (Assign different modes)
            self.sample_command_world(sample_ids[mode == 0])
            self.sample_command_setvel(sample_ids[mode == 1])
            self.sample_command_compliant(sample_ids[mode == 2])
            self.sample_command_large(sample_ids[mode == 3])

            # print("sample_ids:", sample_ids)

        self.impedance_command_time += 1

    def _integrate_reference_trajectory(self):
        """积分参考轨迹 (Integrate reference trajectory)"""
        dt = self.dt # 0.02s

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

        x_b = torch.cat([self.ref_yaw_w.cos(), self.ref_yaw_w.sin(), torch.zeros_like(self.ref_yaw_w)], dim=-1)
        y_b = torch.cat([-self.ref_yaw_w.sin(), self.ref_yaw_w.cos(), torch.zeros_like(self.ref_yaw_w)], dim=-1)
        # ref_acc_w = clamp_norm(ref_acc_w, 0., 80.)
        ref_acc_w = clamp_along(ref_acc_w, x_b, -self.max_acc_xyz[0], self.max_acc_xyz[0])
        ref_acc_w = clamp_along(ref_acc_w, y_b, -self.max_acc_xyz[1], self.max_acc_xyz[1])

        # 存储当前时间步的参考加速度 (Store current timestep reference acceleration)
        self.ref_lin_acc_w = ref_acc_w

        # 积分速度和位置 (Integrate velocity and position)
        ref_vel_w = self.ref_lin_vel_w + ref_acc_w * dt
        ref_vel_w = clamp_along(ref_vel_w, x_b, -self.max_vel_xyz[0], self.max_vel_xyz[0])
        ref_vel_w = clamp_along(ref_vel_w, y_b, -self.max_vel_xyz[1], self.max_vel_xyz[1])
        ref_vel_w[..., 2] = 0.0
        ref_vel_w = clamp_norm(ref_vel_w, 0., 1.5) # mannually set walking speed limit

        self.ref_lin_vel_w = ref_vel_w
        self.ref_pos_w.add_(self.ref_lin_vel_w * dt)
        
        # 保持Z位置稳定 - 使用当前机器人Z位置 (Keep Z position stable)
        if (hasattr(self, 'simulator') and
                hasattr(self.simulator, 'robot_root_states')):
            current_z = self.simulator.robot_root_states[:, 2:3]
            self.ref_pos_w[..., 2:3] = current_z.unsqueeze(1)

        # 积分偏航角轨迹 (Integrate yaw trajectory)
        # 计算期望偏航角 (Calculate desired yaw angle)
        setyaw_w = self.command_setrpy_w[:, 2:3].unsqueeze(1)
        
        # 计算偏航角加速度 (Calculate yaw angular acceleration)
        # 使用类似位置控制的PD控制器 (Use PD controller similar to position)
        ref_yaw_acc_w = (
            self.ang_kp.reshape(self.num_envs, 1, 1) *
            (setyaw_w - self.ref_yaw_w) +
            self.ang_kd.reshape(self.num_envs, 1, 1) *
            (0.0 - self.ref_yaw_vel_w)
        ) / self.virtual_mass_tensor.unsqueeze(1)
        
        # 限制偏航角加速度 (Limit yaw angular acceleration)
        max_yaw_acc = 5.0  # rad/s^2
        ref_yaw_acc_w = torch.clamp(ref_yaw_acc_w, -max_yaw_acc, max_yaw_acc)
        
        # 积分偏航角速度和偏航角 (Integrate yaw velocity and yaw angle)
        ref_yaw_vel_w = self.ref_yaw_vel_w + ref_yaw_acc_w * dt
        
        # 限制偏航角速度 (Limit yaw angular velocity)
        max_yaw_vel = 2.0  # rad/s
        ref_yaw_vel_w = torch.clamp(ref_yaw_vel_w, -max_yaw_vel, max_yaw_vel)
        
        self.ref_yaw_vel_w = ref_yaw_vel_w
        self.ref_yaw_w.add_(self.ref_yaw_vel_w * dt)

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
        # 临界阻尼：lin_kd = 2 * sqrt(kp * mass) (Critical damping)
        lin_kd = 2.0 * torch.sqrt(lin_kp * self.virtual_mass_tensor[env_ids])

        self.lin_kp[env_ids] = lin_kp
        self.lin_kd[env_ids] = lin_kd
        self.ang_kp[env_ids] = lin_kp
        self.ang_kd[env_ids] = lin_kd

        # 采样目标位置 (Sample target position)
        offset = torch.zeros(len(env_ids), 3, device=self.device)
        offset[:, 0].uniform_(-1.0, 1.0)  # X方向前进 (X direction forward)
        offset[:, 1].uniform_(-0.6, 0.6)  # Y方向左右 (Y direction left/right)
        # when stance or tapping, no offset
        self.tapping_in_place[env_ids, 0] = (
            torch.rand(len(env_ids), device=self.device) >
            self.tapping_in_place_prob).float()
        # Apply offset only in walking mode with tapping allowed
        offset[:, 0] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])
        offset[:, 1] *= (self.commands[env_ids, 4] * self.tapping_in_place[env_ids, 0])

        # 使用模拟器的正确根状态 (Use correct root states from simulator)
        if (hasattr(self, 'simulator') and
                hasattr(self.simulator, 'robot_root_states')):
            current_pos = self.simulator.robot_root_states[env_ids, :3]
            self.command_setpos_w[env_ids] = current_pos + offset

        # print(f"采样位置指令: {self.command_setpos_w[env_ids]}")  # 调试输出 (Debug output)

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
        # 临界阻尼：lin_kd = 2 * sqrt(kp * mass) (Critical damping)
        lin_kd = 2.0 * torch.sqrt(lin_kp * self.virtual_mass_tensor[env_ids])

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
        # 临界阻尼：lin_kd = 2 * sqrt(kp * mass) (Critical damping)
        lin_kd = 2.0 * torch.sqrt(lin_kp * self.virtual_mass_tensor[env_ids])

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
        # 临界阻尼：lin_kd = 2 * sqrt(kp * mass) (Critical damping)
        lin_kd = 2.0 * torch.sqrt(lin_kp * self.virtual_mass_tensor[env_ids])

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

    ########################### FACET REWARDS ###########################

    def _reward_impedance_pos_tracking(self):
        """
        阻抗位置跟踪奖励 (Impedance position tracking reward)
        
        使用代理位置目标而非直接指令位置，基于参考轨迹的积分结果
        Use surrogate position targets instead of direct command positions,
        based on integrated reference trajectory results
        
        Returns:
            位置跟踪奖励 (Position tracking reward)
        """
        if (not hasattr(self, 'surrogate_pos_target') or
                not hasattr(self, 'simulator') or
                not hasattr(self.simulator, 'robot_root_states')):
            return torch.zeros(self.num_envs, device=self.device)

        # 获取当前机器人位置 (Get current robot position)
        current_pos = self.simulator.robot_root_states[:, :3]
        
        # 方法1: 使用第一个代理时间步作为主要目标 (Method 1: Use first surrogate step)
        # 只考虑XY平面的误差，忽略Z方向 (Only consider XY plane error, ignore Z)
        # pos_diff = current_pos[:, :2] - self.surrogate_pos_target[:, 0, :2]
        # pos_error_l2 = pos_diff.square().sum(dim=-1)
        
        # 方法2: 可选的多时间步加权奖励 (Method 2: Optional multi-step weighted reward)
        # 对多个时间步进行加权平均 (Weighted average across multiple time steps)
        weights = torch.tensor([0.3, 0.3, 0.3], device=self.device)
        # weights = torch.tensor([0.7, 0.3], device=self.device)
        multi_step_errors = []
        for i in range(len(self.surr_steps)):
            diff = current_pos[:, :2] - self.surrogate_pos_target[:, i, :2]
            step_error = diff.square().sum(dim=-1)
            # Ensure we don't go out of bounds for weights
            weight = weights[i] if i < len(weights) else 0.01
            multi_step_errors.append(step_error * weight)
        pos_error_l2 = torch.stack(multi_step_errors, dim=1).sum(dim=1)
        
        # 使用指数衰减奖励函数 (Use exponential decay reward function)
        # 误差标准差设为0.5米，与原始impedance.py一致 (Error std = 0.5m)
        reward = torch.exp(-pos_error_l2 / 0.25)  # 0.25 = 0.5^2
        self.pos_err_r = reward.unsqueeze(1)  # Store actual error, not reward
        
        return reward

    def _reward_impedance_vel_tracking(self):
        """
        代理速度跟踪奖励 (Surrogate velocity tracking reward)
        
        使用多时间步代理速度目标与EMA滤波速度的跟踪奖励
        Multi-step surrogate velocity target tracking with EMA filtered vel
        
        Returns:
            代理速度跟踪奖励 (Surrogate velocity tracking reward)
        """
        if (not hasattr(self, 'surrogate_lin_vel_target') or
                not hasattr(self, 'lin_vel_ema') or
                not hasattr(self.lin_vel_ema, 'ema') or
                self.lin_vel_ema.ema is None):
            return torch.zeros(self.num_envs, device=self.device)

        # 计算多时间步速度误差 (Calculate multi-step velocity error)
        # surrogate_lin_vel_target: (num_envs, num_surr_steps, 3)
        # lin_vel_ema.ema: (num_envs, num_ema_gammas, 3)
        
        # 使用不同EMA gamma值与不同时间步进行比较
        # Compare different EMA gamma values with different time steps
        surr_vel = self.surrogate_lin_vel_target  # (n, t1, 3)
        ema_vel = self.lin_vel_ema.ema  # (n, t2, 3)
        
        # 扩展维度进行广播计算 (Expand dimensions for broadcast calculation)
        surr_vel_expanded = surr_vel.unsqueeze(2)  # (n, t1, 1, 3)
        ema_vel_expanded = ema_vel.unsqueeze(1)    # (n, 1, t2, 3)
        
        # 计算差值和L2误差 (Calculate difference and L2 error)
        diff = surr_vel_expanded - ema_vel_expanded  # (n, t1, t2, 3)
        error_l2 = diff.square().sum(dim=-1, keepdim=True)  # (n, t1, t2, 1)
        
        # 计算奖励 (Calculate reward)
        reward = torch.exp(-error_l2 / 0.25)  # (n, t1, t2, 1)
        # 对多个EMA和时间步取平均，然后取最佳匹配
        # Average across multiple EMA and time steps, then take best match
        reward = reward.mean(dim=[1, 2]).squeeze(-1)  # (n,)
        return reward
    
    def _reward_impedance_acc_tracking(self):
        """
        阻抗加速度跟踪奖励 (Impedance acceleration tracking reward)
        
        使用参考轨迹积分产生的加速度与当前机器人加速度进行比较
        Compare reference trajectory integrated acceleration with current robot acceleration
        
        Returns:
            加速度跟踪奖励 (Acceleration tracking reward)
        """
        if (not hasattr(self, 'ref_lin_acc_w') or
                not hasattr(self, 'simulator') or
                not hasattr(self.simulator, 'robot_root_states')):
            return torch.zeros(self.num_envs, device=self.device)

        # 计算当前加速度 (Calculate current acceleration)
        # 使用速度差分近似加速度 (Use velocity difference to approximate acceleration)
        current_vel = self.simulator.robot_root_states[:, 7:10]
        if not hasattr(self, 'prev_vel'):
            self.prev_vel = current_vel.clone()
            return torch.zeros(self.num_envs, device=self.device)
        
        current_acc = (current_vel - self.prev_vel) / self.dt
        self.prev_vel = current_vel.clone()
        
        # 使用存储的参考加速度 (Use stored reference acceleration)
        # ref_lin_acc_w: (num_envs, temporal_smoothing + 1, 3)
        # 使用当前时间步的参考加速度 (Use current timestep reference acceleration)
        ref_acc = self.ref_lin_acc_w[:, 0]  # Current timestep acceleration
        
        # 只考虑XY平面的加速度误差 (Only consider XY plane acceleration error)
        acc_diff = current_acc[:, :2] - ref_acc[:, :2]
        error_l2 = acc_diff.square().sum(dim=-1)

        # 使用指数衰减奖励函数 (Use exponential decay reward function)
        reward = torch.exp(-error_l2 / 2.0)
        
        return reward
    
    def _reward_impedance_yaw_vel_tracking(self):
        """
        代理偏航角速度跟踪奖励 (Surrogate yaw velocity tracking reward)
        
        使用多时间步代理偏航角速度目标与EMA滤波角速度的跟踪奖励
        Multi-step surrogate yaw velocity target tracking with EMA filtered angular vel
        
        Returns:
            代理偏航角速度跟踪奖励 (Surrogate yaw velocity tracking reward)
        """
        if (not hasattr(self, 'surrogate_yaw_vel_target') or
                not hasattr(self, 'ang_vel_ema') or
                not hasattr(self.ang_vel_ema, 'ema') or
                self.ang_vel_ema.ema is None):
            return torch.zeros(self.num_envs, device=self.device)

        # 计算多时间步偏航角速度误差 (Calculate multi-step yaw velocity error)
        # surrogate_yaw_vel_target: (num_envs, num_surr_steps, 1)
        # ang_vel_ema.ema: (num_envs, num_ema_gammas, 3)
        
        # 使用不同EMA gamma值与不同时间步进行比较
        # Compare different EMA gamma values with different time steps
        surr_yaw_vel = self.surrogate_yaw_vel_target  # (n, t1, 1)
        ema_yaw_vel = self.ang_vel_ema.ema[:, :, 2:3]  # (n, t2, 1) - Z component only
        
        # 扩展维度进行广播计算 (Expand dimensions for broadcast calculation)
        surr_yaw_vel_expanded = surr_yaw_vel.unsqueeze(2)  # (n, t1, 1, 1)
        ema_yaw_vel_expanded = ema_yaw_vel.unsqueeze(1)    # (n, 1, t2, 1)
        
        # 计算差值和L2误差 (Calculate difference and L2 error)
        diff = surr_yaw_vel_expanded - ema_yaw_vel_expanded  # (n, t1, t2, 1)
        error_l2 = diff.square().sum(dim=-1, keepdim=True)  # (n, t1, t2, 1)
        
        # 计算奖励 (Calculate reward)
        reward = torch.exp(-error_l2 / 0.25)  # (n, t1, t2, 1)
        # 对多个EMA和时间步取平均，然后取最佳匹配
        # Average across multiple EMA and time steps, then take best match
        reward = reward.mean(dim=[1, 2]).squeeze(-1)  # (n,)
        return reward

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

    ######################### Debug Visualization #########################

    def _draw_debug_vis(self):

        super()._draw_debug_vis()
        """
        调试可视化函数 (Debug visualization function)
        
        在位置指令模式下绘制目标位置球体
        Draw target position spheres in position command mode
        """
        if (not hasattr(self, 'simulator') or
                not hasattr(self.simulator, 'clear_lines')):
            return
            
        # 清除之前的可视化 (Clear previous visualization)
        # self.simulator.clear_lines()
        
        # 刷新仿真张量 (Refresh simulation tensors)
        if hasattr(self, '_refresh_sim_tensors'):
            self._refresh_sim_tensors()
        
        # 遍历所有环境 (Iterate through all environments)
        for env_id in range(self.num_envs):
            # 检查是否为位置指令模式 (Check if in position command mode)
            if self.impedance_command_mode[env_id, 0] == self.CMD_POSITION:
                # 获取目标位置 (Get target position)
                target_pos = self.command_setpos_w[env_id]
                
                # 绘制目标位置球体 (Draw target position sphere)
                # 使用绿色表示位置指令目标 (Use green color for position target)
                sphere_color = (0.0, 1.0, 0.0)  # 绿色 (Green)
                sphere_radius = 0.1  # 球体半径 (Sphere radius)

                # print("debug visualization")

                # 绘制球体 (Draw sphere)
                if hasattr(self.simulator, 'draw_sphere'):
                    self.simulator.draw_sphere(target_pos, sphere_radius,
                                               sphere_color, env_id)
                
                # 绘制目标偏航角箭头 (Draw target yaw arrow)
                if hasattr(self.simulator, 'draw_line'):
                    target_yaw = self.command_setrpy_w[env_id, 2]  # 获取偏航角
                    arrow_length = 0.5  # 箭头长度
                    
                    # 计算箭头终点位置 (Calculate arrow end position)
                    arrow_end = target_pos.clone()
                    arrow_end[0] += arrow_length * torch.cos(target_yaw)
                    arrow_end[1] += arrow_length * torch.sin(target_yaw)
                    # 保持与目标球体相同的高度 (Keep same height as target sphere)
                    
                    # 绘制主箭头线 (Draw main arrow line)
                    arrow_color = (1.0, 0.0, 1.0)  # 紫色 (Magenta)
                    arrow_start = target_pos.clone()  # 与球体相同高度
                    self.simulator.draw_line(
                        Point(arrow_start),
                        Point(arrow_end),
                        Point(arrow_color),
                        env_id
                    )
                    
                    # 绘制箭头头部 (Draw arrow head)
                    head_length = 0.1
                    head_angle = torch.pi / 6  # 30度
                    
                    # 左侧箭头线 (Left arrow head line)
                    left_head_end = arrow_end.clone()
                    left_head_end[0] -= (head_length *
                                         torch.cos(target_yaw - head_angle))
                    left_head_end[1] -= (head_length *
                                         torch.sin(target_yaw - head_angle))
                    self.simulator.draw_line(
                        Point(arrow_end),
                        Point(left_head_end),
                        Point(arrow_color),
                        env_id
                    )
                    
                    # 右侧箭头线 (Right arrow head line)
                    right_head_end = arrow_end.clone()
                    right_head_end[0] -= (head_length *
                                          torch.cos(target_yaw + head_angle))
                    right_head_end[1] -= (head_length *
                                          torch.sin(target_yaw + head_angle))
                    self.simulator.draw_line(
                        Point(arrow_end),
                        Point(right_head_end),
                        Point(arrow_color),
                        env_id
                    )
                
                # 绘制从当前位置到目标位置的连线 (Draw line from current to target)
                if (hasattr(self, 'simulator') and
                        hasattr(self.simulator, 'robot_root_states') and
                        hasattr(self.simulator, 'draw_line')):
                    
                    current_pos = self.simulator.robot_root_states[env_id, :3]
                    
                    # 绘制连接线 (Draw connection line)
                    line_color = (1.0, 1.0, 0.0)  # 黄色 (Yellow)
                    
                    # 绘制多条细线形成更明显的线条 (Draw multiple thin lines)
                    for _ in range(5):
                        # 小随机偏移 (Small random offset)
                        line_offset = torch.rand(3, device=self.device) * 0.01
                        self.simulator.draw_line(
                            Point(current_pos + line_offset),
                            Point(target_pos + line_offset),
                            Point(line_color),
                            env_id
                        )
                
                # 绘制代理位置目标球体 (Draw surrogate position target spheres)
                if hasattr(self, 'surrogate_pos_target'):
                    # 为每个代理时间步绘制不同颜色的球体 (Draw different colored spheres)
                    surrogate_colors = [
                        (0.8, 0.0, 0.0),  # 深红色 - 第一个时间步 (Dark red - first)
                        (1.0, 0.4, 0.4),  # 中红色 - 第二个时间步 (Medium red - second)
                        (1.0, 0.7, 0.7),  # 浅红色 - 第三个时间步 (Light red - third)
                        # (1.0, 0.9, 0.9),  # 浅红色 - 第四个时间步 (Light red - fourth)
                    ]
                    surrogate_radius = 0.08  # 稍小的半径 (Slightly smaller radius)
                    
                    for i, surr_step in enumerate(self.surr_steps):
                        if i < len(surrogate_colors):
                            surr_pos = self.surrogate_pos_target[env_id, i]
                            surr_color = surrogate_colors[i]
                            
                            # 绘制代理位置球体 (Draw surrogate position sphere)
                            if hasattr(self.simulator, 'draw_sphere'):
                                self.simulator.draw_sphere(surr_pos, surrogate_radius,
                                                          surr_color, env_id)
                            
                            # 绘制代理速度目标箭头 (Draw surrogate velocity target arrow)
                            if (hasattr(self, 'surrogate_lin_vel_target') and
                                    hasattr(self.simulator, 'draw_line')):
                                
                                surr_vel = self.surrogate_lin_vel_target[env_id, i]
                                vel_magnitude = surr_vel.norm()
                                
                                # 只在速度足够大时绘制箭头 (Only draw if significant)
                                if vel_magnitude > 0.01:
                                    # 箭头长度与速度大小成比例
                                    # Arrow length proportional to velocity magnitude
                                    base_len = 0.1  # Min arrow length
                                    scale = 0.2  # Velocity scale
                                    vel_comp = vel_magnitude * scale
                                    arrow_length = base_len + vel_comp
                                    # 限制最大箭头长度以避免过长
                                    arrow_length = min(arrow_length, 1.0)
                                    
                                    vel_normalized = (
                                        surr_vel /
                                        vel_magnitude.clamp_min(1e-6))
                                    arrow_end = (
                                        surr_pos +
                                        vel_normalized * arrow_length)
                                    
                                    # 使用与球体相同的颜色绘制速度箭头
                                    vel_color = surr_color
                                    
                                    # 绘制主箭头线 (Draw main arrow line)
                                    self.simulator.draw_line(
                                        Point(surr_pos),
                                        Point(arrow_end),
                                        Point(vel_color),
                                        env_id
                                    )
                                    
                                    # 绘制箭头头部 (Draw arrow head)
                                    head_length = 0.08
                                    vel_direction = torch.atan2(
                                        vel_normalized[1], vel_normalized[0])
                                    head_angle = torch.pi / 6  # 30度
                                    
                                    # 左侧箭头线 (Left arrow head line)
                                    left_head_end = arrow_end.clone()
                                    left_head_end[0] -= (
                                        head_length *
                                        torch.cos(vel_direction - head_angle))
                                    left_head_end[1] -= (
                                        head_length *
                                        torch.sin(vel_direction - head_angle))
                                    self.simulator.draw_line(
                                        Point(arrow_end),
                                        Point(left_head_end),
                                        Point(vel_color),
                                        env_id
                                    )
                                    
                                    # 右侧箭头线 (Right arrow head line)
                                    right_head_end = arrow_end.clone()
                                    right_head_end[0] -= (
                                        head_length *
                                        torch.cos(vel_direction + head_angle))
                                    right_head_end[1] -= (
                                        head_length *
                                        torch.sin(vel_direction + head_angle))
                                    self.simulator.draw_line(
                                        Point(arrow_end),
                                        Point(right_head_end),
                                        Point(vel_color),
                                        env_id
                                    )
                            
                            # 绘制从当前位置到代理位置的细线 (Draw thin line)
                            if hasattr(self.simulator, 'draw_line'):
                                # 使用半透明效果 (Use semi-transparent effect)
                                color_factor = 0.6
                                surr_line_color = tuple(c * color_factor
                                                        for c in surr_color)
                                self.simulator.draw_line(
                                    Point(current_pos),
                                    Point(surr_pos),
                                    Point(surr_line_color),
                                    env_id
                                )
            
            # # 为其他模式绘制不同颜色的指示器 (Draw colored indicators for other modes)
            # elif self.impedance_command_mode[env_id, 0] == self.CMD_LINVEL:
            #     # 速度模式 - 蓝色小球 (Velocity mode - blue small sphere)
            #     if (hasattr(self, 'simulator') and
            #             hasattr(self.simulator, 'robot_root_states') and
            #             hasattr(self.simulator, 'draw_sphere')):
                    
            #         current_pos = self.simulator.robot_root_states[env_id, :3]
            #         sphere_color = (0.0, 0.0, 1.0)  # 蓝色 (Blue)
            #         sphere_radius = 0.05
            #         offset_pos = current_pos + torch.tensor([0, 0, 0.3],
            #                                                 device=self.device)
            #         self.simulator.draw_sphere(offset_pos, sphere_radius,
            #                                    sphere_color, env_id)
                                               
            # elif self.impedance_command_mode[env_id, 0] == self.CMD_COMPLIANT:
            #     # 柔顺模式 - 紫色小球 (Compliant mode - purple small sphere)
            #     if (hasattr(self, 'simulator') and
            #             hasattr(self.simulator, 'robot_root_states') and
            #             hasattr(self.simulator, 'draw_sphere')):
                    
            #         current_pos = self.simulator.robot_root_states[env_id, :3]
            #         sphere_color = (1.0, 0.0, 1.0)  # 紫色 (Purple)
            #         sphere_radius = 0.05
            #         offset_pos = current_pos + torch.tensor([0, 0, 0.3],
            #                                                 device=self.device)
            #         self.simulator.draw_sphere(offset_pos, sphere_radius,
            #                                    sphere_color, env_id)

            # elif (self.impedance_command_mode[env_id, 0] ==
            #       self.CMD_LARGE_FORCE):
            #     # 大力模式 - 红色小球 (Large force mode - red small sphere)
            #     if (hasattr(self, 'simulator') and
            #             hasattr(self.simulator, 'robot_root_states') and
            #             hasattr(self.simulator, 'draw_sphere')):

            #         current_pos = self.simulator.robot_root_states[env_id, :3]
            #         sphere_color = (1.0, 0.0, 0.0)  # 红色 (Red)
            #         sphere_radius = 0.05
            #         offset_pos = current_pos + torch.tensor([0, 0, 0.3],
            #                                                 device=self.device)
            #         self.simulator.draw_sphere(offset_pos, sphere_radius,
            #                                    sphere_color, env_id)

    # ============= Observations =============

    