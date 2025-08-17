import torch
from humanoidverse.envs.decoupled_locomotion.decoupled_with_facet import LeggedRobotDecoupledLocomotionWithFACET
import numpy as np
from isaacgym import gymtorch

class LeggedRobotDecoupledLocomotionWithFACETRVC(LeggedRobotDecoupledLocomotionWithFACET):
    def __init__(self, config, device):
        self.init_done = False
        # 首先调用父类初始化 (Initialize parent class first)
        super().__init__(config, device)


    def _compute_rvc_torques(self, actions_scaled):
        """
        Computes RVC (Resolved Velocity Control) torques using compliance control.
        
        Args:
            actions_scaled (torch.Tensor): Scaled action commands
            
        Returns:
            torch.Tensor: Computed torques for all DOFs
        """
        # Initialize contact state
        self.cont_state = torch.zeros(self.num_envs, device=self.device)
        self.task_num = 6  # two hands position (3D each)

        # Initialize task stiffness and damping matrices
        task_stiffs = torch.zeros(self.num_envs, self.task_num, device=self.device)
        task_viscos = torch.zeros(self.num_envs, self.task_num, device=self.device)
        
        # Define stiffness and damping parameters for 6D task (two hands, 3D each)
        stiff_params = torch.tensor([100., 100., 100., 100., 100., 100.], device=self.device)
        visco_params = torch.tensor([10., 10., 10., 10., 10., 10.], device=self.device)
        
        # Repeat across all environments
        task_stiffs[:] = stiff_params.unsqueeze(0).repeat(self.num_envs, 1)
        task_viscos[:] = visco_params.unsqueeze(0).repeat(self.num_envs, 1)

        # Joint control gains
        self.rev_p_gains = torch.ones(self.num_envs, self.num_dof, device=self.device) * 50.0
        self.rev_d_gains = torch.ones(self.num_envs, self.num_dof, device=self.device) * 5.0

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

        # Compute local Jacobians
        J_lelb = self._compute_local_jacobian(J_lelb_gen)  # Shape: (num_envs, 3, num_dof)
        J_relb = self._compute_local_jacobian(J_relb_gen)  # Shape: (num_envs, 3, num_dof)
        
        # Concatenate to form task Jacobian
        J_task = torch.cat([J_lelb, J_relb], dim=1)  # Shape: (num_envs, 6, num_dof)

        # Joint space stiffness and compliance matrices
        K_jnt = torch.diag_embed(self.rev_p_gains)  # Shape: (num_envs, num_dof, num_dof)
        C_jnt = torch.linalg.inv(K_jnt + torch.eye(self.num_dof, device=self.device) * 1e-6)  # Add regularization

        # Task space stiffness and compliance matrices  
        K_task = torch.diag_embed(task_stiffs)  # Shape: (num_envs, 6, 6)
        C_task = torch.linalg.inv(K_task + torch.eye(self.task_num, device=self.device) * 1e-6)  # Add regularization

        # Compute joint compliance matrix
        jnt_comp_matrix = self._compute_jnt_compliance(J_task, C_task, C_jnt)

        # Compute joint stiffness matrix (with regularization for stability)
        try:
            jnt_stiff_matrix = torch.linalg.inv(jnt_comp_matrix + torch.eye(self.num_dof, device=self.device) * 1e-6)
        except RuntimeError as e:
            print(f"Error inverting joint compliance matrix: {e}")
            print(f"J_task shape: {J_task.shape}, C_task shape: {C_task.shape}, C_jnt shape: {C_jnt.shape}")
            print(f"jnt_comp_matrix shape: {jnt_comp_matrix.shape}")
            # Fallback to pseudoinverse
            jnt_stiff_matrix = torch.linalg.pinv(jnt_comp_matrix)

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
        """
        Computes the Jacobian matrix for a specific link with respect to the generalized coordinates.

        Args:
            link_name (str): The name of the link for which the Jacobian is computed.

        Returns:
            torch.Tensor: The Jacobian matrix of shape (num_envs, 6, num_dof+6), where 6 represents the spatial Jacobian
                        (3 for linear velocity and 3 for angular velocity), and num_dof is the number of degrees of freedom.
        """
        # Get the index of the link
        link_index = self.simulator._body_list.index(link_name)

        jacobian = self.simulator.jacobian
        link_jacobian = jacobian[:, link_index, :, :].squeeze(1)  # Shape: (num_envs, 6, num_dof+6)

        return link_jacobian
    
    def _compute_local_jacobian(self, J_gen):
        """
        Computes the local Jacobian matrix by transforming from global to local coordinates.
        
        Args:
            J_gen (torch.Tensor): Generalized Jacobian matrix of shape (num_envs, 6, num_dof+6)
            
        Returns:
            torch.Tensor: Local Jacobian matrix of shape (num_envs, 6, num_dof)
        """
        # Initialize J_foot_gen with zeros
        J_foot_gen = torch.zeros(self.num_envs, 6, self.num_dof + 6, device=self.device)

        # Compute Jacobians for all possible states
        J_left = self.compute_jacobian("left_ankle_link")  # Shape: (num_envs, 6, num_dof+6)
        J_right = self.compute_jacobian("right_ankle_link")  # Shape: (num_envs, 6, num_dof+6)

        # Vectorized assignment based on cont_state
        left_mask = (self.cont_state == 0) | (self.cont_state == 3)  # Left support or double contact
        right_mask = self.cont_state == 1  # Right support
        no_contact_mask = self.cont_state == 2  # No contact

        J_foot_gen[left_mask] = J_left[left_mask]
        J_foot_gen[right_mask] = J_right[right_mask]
        # For no contact case, use left foot as default
        J_foot_gen[no_contact_mask] = J_left[no_contact_mask]

        J_0 = J_foot_gen[:, :, :6]  # Shape: (num_envs, 6, 6)
        J_jnt = J_foot_gen[:, :, 6:]  # Shape: (num_envs, 6, num_dof)

        J_0_inv = torch.linalg.pinv(J_0)  # Shape: (num_envs, 6, 6)
        J_u = torch.matmul(J_0_inv, J_jnt)  # Shape: (num_envs, 6, num_dof)
        J_u = -J_u

        unit_matrix = torch.eye(self.num_dof, self.num_dof, device=self.device).unsqueeze(0).repeat(J_u.shape[0], 1, 1)

        J_u_extended = torch.cat((J_u, unit_matrix), dim=1)

        J = torch.matmul(J_gen, J_u_extended)  # Shape: (num_envs, 6, num_dof)

        return J
    
    # def _compute_double_support_jacobian(self, J_gen):

    #     # J_left = self.compute_jacobian("left_ankle_link")  # Shape: (num_envs, 6, num_dof+6)
    #     J_right_gen = self.compute_jacobian("right_ankle_link")  # Shape: (num_envs, 6, num_dof+6)

    #     J_right = self._compute_local_jacobian(J_right_gen)

    #     # print("J_right: ", J_right)

    #     # SVD on J_right
    #     U, S, Vh = torch.linalg.svd(J_right, full_matrices=True) 
    #     # U: (num_envs, 6, 6), S: (num_envs, 6), Vh: (num_envs, num_dof, num_dof)
    #     V = Vh.transpose(1, 2)  # Shape: (num_envs, num_dof, 6)
    #     G_2 = V[:, :, 6:]  # Shape: (num_envs, num_dof, num_dof-6)

    #     J_task = self._compute_local_jacobian(J_gen) # J_task in left support
    #     J_2 = torch.matmul(J_task, G_2)  # Shape: (num_envs, 6, 3)

    #     return G_2, J_2
    
    # def _gravity_compensation(self, com_jacobian_gen, total_mass):

    #     com_jacobian = self._compute_local_jacobian(com_jacobian_gen)  # Shape: (num_envs, 3, num_dof)

    #     gravity_vec = torch.tensor([0., 0., -9.81], device=self.device).unsqueeze(0).repeat(com_jacobian_gen.shape[0], 1)  # Shape: (num_envs, 3)
    #     gravity_vec = total_mass * gravity_vec  # Shape: (num_envs, 3)

    #     gravity_torques = torch.zeros(com_jacobian_gen.shape[0], self.num_dof, device=self.device)  # Shape: (num_envs, num_dof)
    #     gravity_torques = torch.matmul(com_jacobian.transpose(1, 2), gravity_vec.unsqueeze(-1)).squeeze(-1)  # Shape: (num_envs, num_dof)

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