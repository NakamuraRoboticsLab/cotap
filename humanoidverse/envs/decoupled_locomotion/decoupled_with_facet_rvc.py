import torch
from humanoidverse.envs.decoupled_locomotion.decoupled_with_facet import LeggedRobotDecoupledLocomotionWithFACET
import numpy as np
from isaacgym import gymtorch

class LeggedRobotDecoupledLocomotionWithFACETRVC(LeggedRobotDecoupledLocomotionWithFACET):
    def __init__(self, config, device):
        self.init_done = False
        # Initialize parent class first
        super().__init__(config, device)

        # Initialize contact state and masks
        self.cont_state = torch.zeros(self.num_envs, device=self.device)
        self.task_num = 6  # two hands position (3D each)
        # Initialize contact state masks for efficient computation
        self.no_contact_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.right_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.double_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) 

        self.upper_body_indices = torch.tensor([idx + 1 for idx in self.upper_dof_indices], device=self.device)
        self.grav_upper = torch.zeros(self.num_envs, self.config.robot.upper_body_actions_dim, device=self.device)

    # def _compute_rvc_torques(self, actions_scaled):
    #     # Initialize task stiffness and damping matrices
    #     task_stiffs = torch.zeros(self.num_envs, self.task_num, device=self.device)
        
    #     # Define stiffness and damping parameters for 6D task (two hands, 3D each)
    #     stiff_params = torch.tensor([100., 100., 100., 100., 100., 100.], device=self.device)
        
    #     # Repeat across all environments
    #     task_stiffs[:] = stiff_params.unsqueeze(0).repeat(self.num_envs, 1)

    #     # Joint control gains
    #     self.rev_p_gains = torch.ones(self.num_envs, self.num_dof, device=self.device) * 10.0 # null-space stiffness, hardcoding now

    #     # Compute Jacobians for hand positions (only position, not orientation)
    #     J_lelb_gen = self.compute_jacobian("left_elbow_link")[:, :3, :]  # Only position (3x(num_dof+6))
    #     J_relb_gen = self.compute_jacobian("right_elbow_link")[:, :3, :]  # Only position (3x(num_dof+6))

    #     # Determine contact states
    #     contact_states = torch.zeros(self.num_envs, device=self.device)
    #     contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.

    #     # Use proper boolean operations
    #     contact_states[contact[:, 0] & ~contact[:, 1]] = 0  # Left support only
    #     contact_states[~contact[:, 0] & contact[:, 1]] = 1  # Right support only
    #     contact_states[~contact[:, 0] & ~contact[:, 1]] = 2  # No contact
    #     contact_states[contact[:, 0] & contact[:, 1]] = 3   # Double contact

    #     self.cont_state = contact_states
    #     # Vectorized assignment based on cont_state
    #     self.left_mask = self.cont_state == 0  # Left support
    #     self.right_mask = self.cont_state == 1  # Right support
    #     self.no_contact_mask = self.cont_state == 2
    #     self.double_mask = self.cont_state == 3

    #     # Compute local Jacobians
    #     J_lelb = self._compute_local_jacobian(J_lelb_gen)  # Shape: (num_envs, 3, num_dof)
    #     J_relb = self._compute_local_jacobian(J_relb_gen)  # Shape: (num_envs, 3, num_dof)
        
    #     # Concatenate to form task Jacobian
    #     J_task_gen = torch.cat([J_lelb_gen, J_relb_gen], dim=1)  # Shape: (num_envs, 6, 6+num_dof)
    #     J_task = torch.cat([J_lelb, J_relb], dim=1)  # Shape: (num_envs, 6, num_dof)

    #     # Joint space stiffness and compliance matrices
    #     K_jnt = torch.diag_embed(self.rev_p_gains)  # Shape: (num_envs, num_dof, num_dof)
    #     C_jnt = torch.linalg.inv(K_jnt + torch.eye(self.num_dof, device=self.device) * 1e-6)  # Add regularization

    #     # Task space stiffness and compliance matrices  
    #     K_task = torch.diag_embed(task_stiffs)  # Shape: (num_envs, 6, 6)
    #     C_task = torch.linalg.inv(K_task + torch.eye(self.task_num, device=self.device) * 1e-6)  # Add regularization

    #     jnt_stiff_matrix = torch.zeros(self.num_envs, self.num_dof, self.num_dof, device=self.device)
        
    #     # Single-support situation (left or right foot contact)
    #     single_support_mask = self.left_mask | self.right_mask
    #     if torch.any(single_support_mask):
    #         jnt_comp_matrix = self._compute_jnt_compliance(J_task, C_task, C_jnt)
    #         # Apply regularization and invert for single support environments
    #         regularized_comp = jnt_comp_matrix + torch.eye(self.num_dof, device=self.device).unsqueeze(0) * 1e-6
    #         jnt_stiff_single = torch.linalg.inv(regularized_comp)
    #         jnt_stiff_matrix[single_support_mask] = jnt_stiff_single[single_support_mask]

    #     # Double-support situation - only apply to environments with double contact
    #     if torch.any(self.double_mask):
    #         G_2, J_2 = self._compute_double_support_jacobian(J_task_gen)
    #         jnt_stiffs_2 = self._compute_rvc_stiffness_close(C_task, K_jnt, J_2, G_2)
    #         jnt_stiff_matrix_double = self._compute_rvc_stiffness_double(jnt_stiffs_2, K_jnt, G_2)
    #         # Only apply to environments with double contact
    #         jnt_stiff_matrix[self.double_mask] = jnt_stiff_matrix_double[self.double_mask]

    #     # When no contact (cont_state == 2), use simple diagonal stiffness matrix
    #     if torch.any(self.no_contact_mask):
    #         diagonal_stiffness = torch.diag_embed(self._kp_scale * self.p_gains)  # Shape: (num_envs, num_dof, num_dof)
    #         jnt_stiff_matrix[self.no_contact_mask] = diagonal_stiffness[self.no_contact_mask]

    #     # Compute position error
    #     delta_pos = actions_scaled + self.default_dof_pos - self.simulator.dof_pos
    #     # Compute torques using stiffness control
    #     torques = torch.matmul(jnt_stiff_matrix, delta_pos.unsqueeze(-1)).squeeze(-1)
    #     # Add damping term
    #     torques = torques - self._kd_scale * self.d_gains * self.simulator.dof_vel

    #     # add gravity compensation
    #     grav_torques = self._gravity_compensation()
    #     torques += grav_torques

    #     return torques
    
    def _compute_upper_rvc_torques(self, actions_scaled):
        # Initialize task stiffness and damping matrices
        task_stiffs = torch.zeros(self.num_envs, self.task_num, device=self.device)
        stiff_torso = torch.zeros(self.num_envs, 6, device=self.device)

        # Define stiffness and damping parameters for 6D task (two hands, 3D each)
        stiff_params = torch.tensor([300., 300., 300., 300., 300., 300.], device=self.device) # should <= 500
        # torso_params = torch.tensor([2000., 2000., 2000., 500., 500., 500.], device=self.device)

        # Create [num_envs, 6] tensor: first 3 columns lin_kp, last 3 columns ang_kp
        # torso_params = torch.cat([self.lin_kp.repeat(1,3), self.ang_kp.repeat(1,3)], dim=1)

        # Repeat across all environments
        task_stiffs[:] = stiff_params.unsqueeze(0).repeat(self.num_envs, 1)
        stiff_torso = torch.cat([self.lin_kp.repeat(1,3), self.ang_kp.repeat(1,3)], dim=1)
        # stiff_torso[:] = torso_params.unsqueeze(0).repeat(self.num_envs, 1)

        # Joint control gains
        self.rev_p_gains = torch.ones(self.num_envs, self.config.robot.upper_body_actions_dim, device=self.device) * 30.0 # null-space stiffness, hardcoding now

        # Compute Jacobians for hand positions (only position, not orientation)
        J_lelb_gen = self.compute_jacobian("left_elbow_link")[:, :3, :]  # Only position (3x(num_dof+6))
        J_relb_gen = self.compute_jacobian("right_elbow_link")[:, :3, :]  # Only position (3x(num_dof+6))
        J_torso_gen = self.compute_jacobian("torso_link")  # position and rotation

        # Create selection matrix S_u to select upper body velocities from generalized velocity
        # Generalized velocity structure: [base_vel (6), joint_vel (num_dof)]
        # We want to select upper body joint velocities from the joint part
        S_u = torch.zeros(self.num_envs, self.config.robot.upper_body_actions_dim, self.num_dof + 6, device=self.device)
        # Assuming upper_dof_indices contains the indices of upper body DOFs in the joint space
        for i, dof_idx in enumerate(self.upper_dof_indices):
            S_u[:, i, 6 + dof_idx] = 1.0  # 6 offset for base DOFs
        
        # Create upper body constraint matrix
        B_upper = torch.cat([J_torso_gen, S_u], dim=1)  # Shape: (num_envs, 6 + upper_body_actions_dim, num_dof + 6)

        # Concatenate to form task Jacobian
        J_task_gen = torch.cat([J_lelb_gen, J_relb_gen], dim=1)  # Shape: (num_envs, 6, 6+num_dof)
        J_task_upper = J_task_gen @ torch.linalg.pinv(B_upper)

        # Joint space stiffness and compliance matrices
        K_jnt = torch.diag_embed(self.rev_p_gains)  # Shape: (num_envs, upper_body_actions_dim, upper_body_actions_dim)
        C_jnt = torch.linalg.inv(K_jnt + torch.eye(self.config.robot.upper_body_actions_dim, device=self.device) * 1e-6)  # Add regularization

        # Task space stiffness and compliance matrices  
        K_task = torch.diag_embed(task_stiffs)  # Shape: (num_envs, 6, 6)
        C_task = torch.linalg.inv(K_task + torch.eye(self.task_num, device=self.device) * 1e-6)  # Add regularization

        # Calculate block Jacobian
        J_eb = J_task_upper[:, :, :6]  # Extract torso part of the task Jacobian
        J_eu = J_task_upper[:, :, 6:]  # Extract upper body part of the task Jacobian

        K_torso = torch.diag_embed(stiff_torso)  # Shape: (num_envs, 6, 6)

        C_task -= J_eb @ torch.linalg.inv(K_torso) @ J_eb.transpose(-1, -2)  # Adjust task compliance matrix

        # Solution
        jnt_stiff_matrix = torch.zeros(self.num_envs, self.config.robot.upper_body_actions_dim, self.config.robot.upper_body_actions_dim, device=self.device)
        
        jnt_comp_matrix = self._compute_jnt_compliance(J_eu, C_task, C_jnt)
        # Apply regularization and invert for single support environments
        regularized_comp = jnt_comp_matrix + torch.eye(self.config.robot.upper_body_actions_dim, device=self.device).unsqueeze(0) * 1e-6
        jnt_stiff_matrix = torch.linalg.inv(regularized_comp)

        # Compute position error
        delta_pos = actions_scaled + self.default_dof_pos - self.simulator.dof_pos
        # Compute torques using stiffness control
        torques = self._kp_scale * self.p_gains*delta_pos
        # Compute upper body position error for RVC controller
        upper_body_pos_error = delta_pos[:, self.upper_dof_indices].unsqueeze(-1)  # Shape: (num_envs, upper_body_actions_dim, 1)
        # upper_body_pos_error = (self.ref_upper_dof_pos - self.simulator.dof_pos[:, self.upper_dof_indices]).unsqueeze(-1)
        upper_body_torques = torch.matmul(jnt_stiff_matrix, upper_body_pos_error).squeeze(-1) 
        torques[:, self.upper_dof_indices] = upper_body_torques
        # Add damping term
        torques = torques - self._kd_scale * self.d_gains * self.simulator.dof_vel

        # Gravity compensation calculation
        self.grav_upper = self._gravity_upper_compensation()
        torques[:, self.upper_dof_indices] += self.grav_upper

        return torques
    
    def _compute_jnt_compliance(self, J_task, C_task, C_jnt):
        """
        Computes joint space compliance matrix using task-space compliance.

        Formula: C_q = J_t^# * C_t * (J_t^#)^T + (C_jnt - J_t^# * J_t * C_jnt * J_t^T * (J_t^#)^T)
        where:
        - J_t^# is the pseudoinverse of the task Jacobian
        - C_t is the task-space compliance matrix
        - C_jnt is the joint-space compliance matrix
        - I is the identity matrix
        """
        # Compute pseudoinverse of task Jacobian
        J_pinv = torch.linalg.pinv(J_task)  # Shape: (num_envs, joint_dim, task_dim)
        J_pinv_T = J_pinv.transpose(-1, -2)  # Shape: (num_envs, task_dim, joint_dim)
        
        # Task-space contribution: J_t^# * C_t * (J_t^#)^T
        task_contribution = J_pinv @ C_task @ J_pinv_T
        # Null-space: 
        null_contribution = C_jnt - J_pinv @ J_task @ C_jnt @ J_task.transpose(-1, -2) @ J_pinv_T

        # Final joint compliance matrix
        jnt_comp_matrix = task_contribution + null_contribution

        return jnt_comp_matrix
    
    def compute_jacobian(self, link_name):
        # Get the index of the link
        link_index = self.simulator._body_list.index(link_name)

        jacobian = self.simulator.jacobian
        link_jacobian = jacobian[:, link_index, :, :].squeeze(1)  # Shape: (num_envs, 6, num_dof+6)

        return link_jacobian
    
    # def _compute_local_jacobian(self, J_gen):
    #     # Initialize J_foot_gen with zeros
    #     J_foot_gen = torch.zeros(self.num_envs, 6, self.num_dof + 6, device=self.device)

    #     # Compute Jacobians for all possible states
    #     J_left = self.compute_jacobian("left_ankle_link")  # Shape: (num_envs, 6, num_dof+6)
    #     J_right = self.compute_jacobian("right_ankle_link")  # Shape: (num_envs, 6, num_dof+6)

    #     J_foot_gen[self.left_mask] = J_left[self.left_mask]
    #     J_foot_gen[self.right_mask] = J_right[self.right_mask]
    #     # For no contact and double case, use left foot as default
    #     J_foot_gen[self.no_contact_mask] = J_left[self.no_contact_mask]
    #     J_foot_gen[self.double_mask] = J_left[self.double_mask]

    #     J_0 = J_foot_gen[:, :, :6]  # Shape: (num_envs, 6, 6)
    #     J_jnt = J_foot_gen[:, :, 6:]  # Shape: (num_envs, 6, num_dof)

    #     J_0_inv = torch.linalg.pinv(J_0)  # Shape: (num_envs, 6, 6)
    #     J_u = torch.matmul(J_0_inv, J_jnt)  # Shape: (num_envs, 6, num_dof)
    #     J_u = -J_u

    #     unit_matrix = torch.eye(self.num_dof, self.num_dof, device=self.device).unsqueeze(0).repeat(J_u.shape[0], 1, 1)
    #     J_u_extended = torch.cat((J_u, unit_matrix), dim=1)

    #     J = torch.matmul(J_gen, J_u_extended)  # Shape: (num_envs, 6, num_dof)

    #     return J

    def _compute_local_torso_jacobian(self, J_gen):
        # Compute Jacobians for all possible states
        J_torso_gen = self.compute_jacobian("torso_link")

        J_0 = J_torso_gen[:, :, :6]  # Shape: (num_envs, 6, 6)
        J_jnt = J_torso_gen[:, :, 6:]  # Shape: (num_envs, 6, num_dof)

        J_0_inv = torch.linalg.pinv(J_0)  # Shape: (num_envs, 6, 6)
        J_u = torch.matmul(J_0_inv, J_jnt)  # Shape: (num_envs, 6, num_dof)
        J_u = -J_u

        unit_matrix = torch.eye(self.num_dof, self.num_dof, device=self.device).unsqueeze(0).repeat(J_u.shape[0], 1, 1)
        J_u_extended = torch.cat((J_u, unit_matrix), dim=1)

        J = torch.matmul(J_gen, J_u_extended)  # Shape: (num_envs, 6, num_dof)

        return J
    
    def _gravity_upper_compensation(self):
        """
        Compute gravity compensation torques for all joints.
        
        Returns:
            torch.Tensor: Gravity compensation torques,
                         shape (num_envs, num_dof)
        """
        # Get COM Jacobian and total mass
        com_jacobian_gen, total_mass, com_position = self._compute_com_jacobian()

        # Convert to local coordinates
        com_jacobian = self._compute_local_torso_jacobian(com_jacobian_gen)

        upper_com_jacobian = com_jacobian[:, :, self.config.robot.lower_body_actions_dim:]  # Select upper body DOFs

        # Create gravity force vector (total mass * gravity acceleration)
        gravity_force = (
            total_mass.unsqueeze(-1) *
            torch.tensor([0., 0., -9.81],
                         device=self.device,
                         dtype=torch.float32).view(1, 3, 1)
        )
        
        # Compute gravity compensation torques: J_com^T * F_gravity
        gravity_torques = torch.bmm(
            upper_com_jacobian.transpose(-2, -1),
            gravity_force
        ).squeeze(-1)
        
        return gravity_torques
    
    def _compute_com_jacobian(self):
        # Get link masses for all rigid bodies
        body_masses = self.get_link_masses()

        # Compute total mass of all bodies
        total_mass = body_masses.sum(dim=1, keepdim=True)

        # Get body positions for all body links
        body_positions = self.simulator._rigid_body_pos
        com_position = (
            (body_positions * body_masses.unsqueeze(-1)).sum(dim=1) /
            total_mass
        )

        jacobian = self.simulator.jacobian[:, :, :3, :]

        # Mass-weighted Jacobian sum for all body links
        # Shape: (num_envs, num_body_links, 3, num_dof+6)
        weighted_jacobians = jacobian * body_masses.unsqueeze(-1).unsqueeze(-1)  # (num_envs, num_bodies, 3, num_dof+6)
        com_jacobian = weighted_jacobians.sum(dim=1) / total_mass.unsqueeze(-1)  # (num_envs, 3, num_dof+6)

        return com_jacobian, total_mass, com_position
    
    def get_link_masses(self):
        """
        Get masses for all rigid bodies (whole body).
        
        Returns:
            torch.Tensor: Masses of all rigid bodies,
                         shape (num_envs, num_bodies)
        """
        # Get rigid body properties for the first environment
        body_props = self.simulator.gym.get_actor_rigid_body_properties(
            self.simulator.envs[0],
            self.simulator.robot_handles[0]
        )
        # Extract masses for all rigid bodies
        all_body_masses = [body.mass for body in body_props]
        # Convert to tensor and repeat for all environments
        link_masses_single = torch.tensor(
            all_body_masses, device=self.device, dtype=torch.float32
        )
        link_masses = link_masses_single.unsqueeze(0).repeat(self.num_envs, 1)
        return link_masses

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
            # print("Residual upper body action applied")
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
            # print("Using RVC controller")
            # torques = self._compute_rvc_torques(actions_scaled)
            torques = self._compute_upper_rvc_torques(actions_scaled)
            ref_actions_scaled[:, self.upper_dof_indices] = self.ref_upper_dof_pos
            self.ref_pd_torques = self._kp_scale * self.p_gains*(ref_actions_scaled + self.default_dof_pos - self.simulator.dof_pos) - self._kd_scale * self.d_gains*self.simulator.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        if self.config.domain_rand.randomize_torque_rfi:
            torques = torques + (torch.rand_like(torques)*2.-1.) * self.config.domain_rand.rfi_lim * self._rfi_lim_scale * self.torque_limits
        
        if self.config.robot.control.clip_torques:
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            return torques
        
    ########################### RVC REWARDS ###########################

    def _reward_upper_ref_close(self):
        err = torch.sum(torch.square(self.config.robot.control.action_scale * \
                                      self.actions[:, self.upper_dof_indices] + \
                                        self.default_dof_pos[:, self.upper_dof_indices] - self.ref_upper_dof_pos), dim=1)
        reward = torch.exp(-err / 0.25)

        return reward
    
    ######################### Observations #########################
    
    def _get_obs_grav_upper(self):
        return self.grav_upper

    def _get_obs_lin_kp(self):
        return self.lin_kp

    def _get_obs_ang_kp(self):
        return self.ang_kp