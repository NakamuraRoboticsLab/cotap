import torch
from humanoidverse.envs.decoupled_locomotion.decoupled_with_facet import LeggedRobotDecoupledLocomotionWithFACET
import numpy as np
from isaacgym import gymtorch

class LeggedRobotDecoupledLocomotionWithFACETRVC(LeggedRobotDecoupledLocomotionWithFACET):
    def __init__(self, config, device):
        self.init_done = False
        # 首先调用父类初始化 (Initialize parent class first)
        super().__init__(config, device)

        # Initialize contact state and masks
        self.cont_state = torch.zeros(self.num_envs, device=self.device)
        self.task_num = 6  # two hands position (3D each)
        # Initialize contact state masks for efficient computation
        self.no_contact_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.right_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.double_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) 

    def _compute_rvc_torques(self, actions_scaled):
        # Initialize task stiffness and damping matrices
        task_stiffs = torch.zeros(self.num_envs, self.task_num, device=self.device)
        
        # Define stiffness and damping parameters for 6D task (two hands, 3D each)
        stiff_params = torch.tensor([100., 100., 100., 100., 100., 100.], device=self.device)
        
        # Repeat across all environments
        task_stiffs[:] = stiff_params.unsqueeze(0).repeat(self.num_envs, 1)

        # Joint control gains
        self.rev_p_gains = torch.ones(self.num_envs, self.num_dof, device=self.device) * 10.0 # null-space stiffness, hardcoding now

        # Compute Jacobians for hand positions (only position, not orientation)
        J_lelb_gen = self.compute_jacobian("left_elbow_link")[:, :3, :]  # Only position (3x(num_dof+6))
        J_relb_gen = self.compute_jacobian("right_elbow_link")[:, :3, :]  # Only position (3x(num_dof+6))

        # Determine contact states
        contact_states = torch.zeros(self.num_envs, device=self.device)
        contact = self.simulator.contact_forces[:, self.feet_indices, 2] > 1.

        # Use proper boolean operations
        contact_states[contact[:, 0] & ~contact[:, 1]] = 0  # Left support only
        contact_states[~contact[:, 0] & contact[:, 1]] = 1  # Right support only
        contact_states[~contact[:, 0] & ~contact[:, 1]] = 2  # No contact
        contact_states[contact[:, 0] & contact[:, 1]] = 3   # Double contact

        self.cont_state = contact_states
        # Vectorized assignment based on cont_state
        self.left_mask = self.cont_state == 0  # Left support
        self.right_mask = self.cont_state == 1  # Right support
        self.no_contact_mask = self.cont_state == 2
        self.double_mask = self.cont_state == 3

        # Compute local Jacobians
        J_lelb = self._compute_local_jacobian(J_lelb_gen)  # Shape: (num_envs, 3, num_dof)
        J_relb = self._compute_local_jacobian(J_relb_gen)  # Shape: (num_envs, 3, num_dof)
        
        # Concatenate to form task Jacobian
        J_task_gen = torch.cat([J_lelb_gen, J_relb_gen], dim=1)  # Shape: (num_envs, 6, 6+num_dof)
        J_task = torch.cat([J_lelb, J_relb], dim=1)  # Shape: (num_envs, 6, num_dof)

        # Joint space stiffness and compliance matrices
        K_jnt = torch.diag_embed(self.rev_p_gains)  # Shape: (num_envs, num_dof, num_dof)
        C_jnt = torch.linalg.inv(K_jnt + torch.eye(self.num_dof, device=self.device) * 1e-6)  # Add regularization

        # Task space stiffness and compliance matrices  
        K_task = torch.diag_embed(task_stiffs)  # Shape: (num_envs, 6, 6)
        C_task = torch.linalg.inv(K_task + torch.eye(self.task_num, device=self.device) * 1e-6)  # Add regularization

        jnt_stiff_matrix = torch.zeros(self.num_envs, self.num_dof, self.num_dof, device=self.device)
        
        # Single-support situation (left or right foot contact)
        single_support_mask = self.left_mask | self.right_mask
        if torch.any(single_support_mask):
            jnt_comp_matrix = self._compute_jnt_compliance(J_task, C_task, C_jnt)
            # Apply regularization and invert for single support environments
            regularized_comp = jnt_comp_matrix + torch.eye(self.num_dof, device=self.device).unsqueeze(0) * 1e-6
            jnt_stiff_single = torch.linalg.inv(regularized_comp)
            jnt_stiff_matrix[single_support_mask] = jnt_stiff_single[single_support_mask]

        # Double-support situation - only apply to environments with double contact
        if torch.any(self.double_mask):
            G_2, J_2 = self._compute_double_support_jacobian(J_task_gen)
            jnt_stiffs_2 = self._compute_rvc_stiffness_close(C_task, K_jnt, J_2, G_2)
            jnt_stiff_matrix_double = self._compute_rvc_stiffness_double(jnt_stiffs_2, K_jnt, G_2)
            # Only apply to environments with double contact
            jnt_stiff_matrix[self.double_mask] = jnt_stiff_matrix_double[self.double_mask]

        # When no contact (cont_state == 2), use simple diagonal stiffness matrix
        if torch.any(self.no_contact_mask):
            diagonal_stiffness = torch.diag_embed(self._kp_scale * self.p_gains)  # Shape: (num_envs, num_dof, num_dof)
            jnt_stiff_matrix[self.no_contact_mask] = diagonal_stiffness[self.no_contact_mask]

        # Compute position error
        delta_pos = actions_scaled + self.default_dof_pos - self.simulator.dof_pos
        # Compute torques using stiffness control
        torques = torch.matmul(jnt_stiff_matrix, delta_pos.unsqueeze(-1)).squeeze(-1)
        # Add damping term
        torques = torques - self._kd_scale * self.d_gains * self.simulator.dof_vel
        
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
        J_pinv = torch.linalg.pinv(J_task)  # Shape: (num_envs, num_dof, task_dim)
        J_pinv_T = J_pinv.transpose(-1, -2)  # Shape: (num_envs, task_dim, num_dof)
        
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
    
    def _compute_local_jacobian(self, J_gen):
        # Initialize J_foot_gen with zeros
        J_foot_gen = torch.zeros(self.num_envs, 6, self.num_dof + 6, device=self.device)

        # Compute Jacobians for all possible states
        J_left = self.compute_jacobian("left_ankle_link")  # Shape: (num_envs, 6, num_dof+6)
        J_right = self.compute_jacobian("right_ankle_link")  # Shape: (num_envs, 6, num_dof+6)

        J_foot_gen[self.left_mask] = J_left[self.left_mask]
        J_foot_gen[self.right_mask] = J_right[self.right_mask]
        # For no contact and double case, use left foot as default
        J_foot_gen[self.no_contact_mask] = J_left[self.no_contact_mask]
        J_foot_gen[self.double_mask] = J_left[self.double_mask]

        J_0 = J_foot_gen[:, :, :6]  # Shape: (num_envs, 6, 6)
        J_jnt = J_foot_gen[:, :, 6:]  # Shape: (num_envs, 6, num_dof)

        J_0_inv = torch.linalg.pinv(J_0)  # Shape: (num_envs, 6, 6)
        J_u = torch.matmul(J_0_inv, J_jnt)  # Shape: (num_envs, 6, num_dof)
        J_u = -J_u

        unit_matrix = torch.eye(self.num_dof, self.num_dof, device=self.device).unsqueeze(0).repeat(J_u.shape[0], 1, 1)
        J_u_extended = torch.cat((J_u, unit_matrix), dim=1)

        J = torch.matmul(J_gen, J_u_extended)  # Shape: (num_envs, 6, num_dof)

        return J
    
    def _compute_double_support_jacobian(self, J_gen):

        J_right_gen = self.compute_jacobian("right_ankle_link")  # Shape: (num_envs, 6, num_dof+6)
        J_right = self._compute_local_jacobian(J_right_gen)

        # SVD on J_right
        U, S, Vh = torch.linalg.svd(J_right, full_matrices=True) 
        # U: (num_envs, 6, 6), S: (num_envs, 6), Vh: (num_envs, num_dof, num_dof)
        V = Vh.transpose(-2, -1)  # Shape: (num_envs, num_dof, 6)
        G_2 = V[:, :, 6:]  # Shape: (num_envs, num_dof, num_dof-6)

        J_task = self._compute_local_jacobian(J_gen) # J_task in left support
        J_2 = torch.matmul(J_task, G_2)  # Shape: (num_envs, 6, 3)

        return G_2, J_2
    
    def _compute_rvc_stiffness_close(self, C_task, nullspace_mat, J_2, G_2):

        G_2T = G_2.transpose(-2, -1)  # Shape: (num_envs, num_dof, 3)
        nullspace_mat_2 = G_2T @ nullspace_mat @ G_2
        nullspace_mat_2 = torch.linalg.pinv(nullspace_mat_2) # Compliance matrix for nullspace
        
        J_pinv = torch.linalg.pinv(J_2)  # Shape: (num_envs, num_dof, 3)
        J_pinv_T = J_pinv.transpose(-2, -1)  # Shape: (num_envs, 3, num_dof)

        task_contribution = J_pinv @ C_task @ J_pinv_T
        null_contribution = nullspace_mat_2 - J_pinv @ J_2 @ nullspace_mat_2 @ J_2.transpose(-2, -1) @ J_pinv_T

        jnt_stiffs_mat = task_contribution + null_contribution
        jnt_stiffs_mat = torch.linalg.pinv(jnt_stiffs_mat)

        return jnt_stiffs_mat
    
    def _compute_rvc_stiffness_double(self, jnt_stiffs_2, nullspace_mat, G_2):

        G_pinv = torch.linalg.pinv(G_2)  # Shape: (num_envs, num_dof, 3)
        G_pinv_T = G_pinv.transpose(-2, -1)  # Shape: (num_envs, 3, num_dof)

        task_contribution = G_pinv_T @ jnt_stiffs_2 @ G_pinv
        null_contribution = nullspace_mat - G_pinv_T @ G_2.transpose(-2, -1) @ nullspace_mat @ G_2 @ G_pinv

        jnt_stiffs_mat_db = task_contribution + null_contribution

        return jnt_stiffs_mat_db
    
    # def _gravity_compensation(self, com_jacobian_gen, total_mass):
    #     """
    #     Computes gravity compensation torques using center of mass Jacobian.
        
    #     Args:
    #         com_jacobian_gen (torch.Tensor): Generalized COM Jacobian matrix of shape (num_envs, 3, num_dof+6)
    #         total_mass (float or torch.Tensor): Total mass of the robot (scalar or per-environment tensor)
            
    #     Returns:
    #         torch.Tensor: Gravity compensation torques of shape (num_envs, num_dof)
    #     """
    #     # Validate input shapes
    #     if com_jacobian_gen.dim() != 3 or com_jacobian_gen.shape[1] != 3:
    #         raise ValueError(f"Expected com_jacobian_gen shape (num_envs, 3, num_dof+6), got {com_jacobian_gen.shape}")
        
    #     # Compute local Jacobian for center of mass
    #     com_jacobian = self._compute_local_jacobian(com_jacobian_gen)  # Shape: (num_envs, 3, num_dof)
        
    #     # Validate computed Jacobian shape
    #     expected_shape = (self.num_envs, 3, self.num_dof)
    #     if com_jacobian.shape != expected_shape:
    #         raise ValueError(f"Computed COM Jacobian shape {com_jacobian.shape} doesn't match expected {expected_shape}")

    #     # Create gravity vector (pointing downward in z-direction)
    #     gravity_vec = torch.tensor([0., 0., -9.81], device=self.device, dtype=torch.float32)
    #     gravity_vec = gravity_vec.unsqueeze(0).repeat(self.num_envs, 1)  # Shape: (num_envs, 3)
        
    #     # Handle total_mass as scalar or tensor
    #     if isinstance(total_mass, (int, float)):
    #         # Scalar mass - same for all environments
    #         gravity_force = total_mass * gravity_vec  # Shape: (num_envs, 3)
    #     else:
    #         # Tensor mass - different per environment
    #         if total_mass.shape[0] != self.num_envs:
    #             raise ValueError(f"Mass tensor shape {total_mass.shape} doesn't match num_envs {self.num_envs}")
    #         gravity_force = total_mass.unsqueeze(-1) * gravity_vec  # Shape: (num_envs, 3)

    #     # Compute gravity compensation torques using transposed Jacobian
    #     # τ_gravity = J_com^T * F_gravity
    #     try:
    #         gravity_torques = torch.matmul(
    #             com_jacobian.transpose(-2, -1),  # Shape: (num_envs, num_dof, 3)
    #             gravity_force.unsqueeze(-1)      # Shape: (num_envs, 3, 1)
    #         ).squeeze(-1)  # Shape: (num_envs, num_dof)
    #     except RuntimeError as e:
    #         print(f"Error in gravity compensation matrix multiplication: {e}")
    #         print(f"COM Jacobian shape: {com_jacobian.shape}")
    #         print(f"Gravity force shape: {gravity_force.shape}")
    #         # Return zero torques as fallback
    #         gravity_torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)

    #     # Validate output shape
    #     expected_output_shape = (self.num_envs, self.num_dof)
    #     if gravity_torques.shape != expected_output_shape:
    #         print(f"Warning: Gravity torques shape {gravity_torques.shape} doesn't match expected {expected_output_shape}")

    #     return gravity_torques
    
    
    # def _gravity_compensation_double(self, com_jacobian_gen, total_mass):

    #     G_2, J_2 = self._compute_double_support_jacobian(com_jacobian_gen)
    #     G_2_pinv = torch.linalg.pinv(G_2)  # Shape: (num_envs, num_dof, num_dof-6)

    #     com_jacobian = torch.matmul(J_2, G_2_pinv).squeeze(-1)

    #     gravity_vec = torch.tensor([0., 0., -9.81], device=self.device).unsqueeze(0).repeat(com_jacobian_gen.shape[0], 1)  # Shape: (num_envs, 3)
    #     gravity_vec = total_mass * gravity_vec  # Shape: (num_envs, 3)

    #     gravity_torques = torch.zeros(com_jacobian_gen.shape[0], self.num_dof, device=self.device)  # Shape: (num_envs, num_dof)
    #     gravity_torques = torch.matmul(com_jacobian.transpose(1, 2), gravity_vec.unsqueeze(-1)).squeeze(-1)  # Shape: (num_envs, num_dof)

    #     return gravity_torques

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
            torques = self._compute_rvc_torques(actions_scaled)
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