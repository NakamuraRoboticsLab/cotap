import torch
from humanoidverse.envs.decoupled_locomotion.decoupled_with_facet_rvc import LeggedRobotDecoupledLocomotionWithFACET
import numpy as np

class LeggedRobotDecoupledLocomotionWithFACETRVC(LeggedRobotDecoupledLocomotionWithFACET):
    def __init__(self, config, device):
        self.init_done = False
        # 首先调用父类初始化 (Initialize parent class first)
        super().__init__(config, device)


    def _compute_rvc_torques(self, actions):

        torques = actions

        return torques

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.
        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.config.robot.control.action_scale
        ref_actions_scaled = actions_scaled.clone()
        if self.residual_upper_body_action:
            actions_scaled[:, self.upper_dof_indices] += (self.ref_upper_dof_pos - self.default_dof_pos[:, self.upper_dof_indices])
        control_type = self.config.robot.control.control_type
        if control_type=="P":
            torques = self._kp_scale * self.p_gains*(actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
            ref_actions_scaled[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            self.ref_pd_torques = self._kp_scale * self.p_gains*(ref_actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self.p_gains*(actions_scaled - self.simulator.dof_vel) - self._kd_scale * self.d_gains*(self.simulator.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type=="C":
            torques = self._compute_rvc_torques(actions_scaled)
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques