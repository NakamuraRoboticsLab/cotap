import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

from humanoidverse.agents.modules.ppo_modules import PPOActor, PPOCritic
from humanoidverse.agents.modules.data_utils import RolloutStorage
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.utils.average_meters import TensorAverageMeterDict
from humanoidverse.agents.ppo.ppo import PPO
from humanoidverse.utils.logger import Logger
from humanoidverse.utils.torch_utils import quat_rotate_inverse

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
import time
import os
import statistics
from collections import deque
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
console = Console()

class PPOMultiActorCritic(PPO):
    def __init__(self,
                 env: BaseTask,
                 config,
                 log_dir=None,
                 device='cpu'):
        self.device= device
        self.env = env
        self.config = config
        self.log_dir = log_dir
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.start_time = 0
        self.stop_time = 0
        self.collection_time = 0
        self.learn_time = 0

        self._init_config()

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.keys = self.env.config.robot.get('body_keys', ['lower_body', 'upper_body'])

        # Book keeping
        self.ep_infos = []
        # For reward tracking per key
        self.rewbuffer_decoupled = {key: deque(maxlen=100) for key in self.keys}
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum_decoupled = {key: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) for key in self.keys}
        self.cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        self.cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.eval_callbacks: list[RL_EvalCallback] = []
        self.episode_env_tensors = TensorAverageMeterDict()
        _ = self.env.reset_all()

        self.multi_transitions = RolloutStorage.MultiTransitions(keys=self.keys)

        self.mc_weight = 0.8 # TODO: hardcoded for now, can be made configurable later

        humanoidverse_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(humanoidverse_ROOT_DIR, \
                                       "../../logs/h1_19dof_falcon/new_whole_pd", "model_20000.pt") # 预训练模型路径
                                    #    "../../logs/h1_19dof_falcon/whole_body_pd", "model_20000.pt") # 预训练模型路径
                                    #    "../../logs/h1_19dof_falcon/whole_body_pd_extra_obs", "model_20000.pt") # 预训练模型路径

        self.external_actor = self.create_external_actor(checkpoint_path)

        self.kl_coef_start = self.env.config.regularization.kl_beta_init  # Initial KL coefficient
        self.kl_coef_end = self.env.config.regularization.kl_beta_final   # Final KL coefficient
        self.kl_coef_anneal_steps = self.env.config.regularization.kl_anneal_steps  # Number of iterations to anneal over
        self.kl_coef = self.kl_coef_start  # Current KL coefficient

    def _init_config(self):
        super()._init_config()
        self.algo_history_length_dict = self.env.config.obs.get('history_length', {})
        self.base_history_length_dict = self.env.config.obs.get('history_length', {})
        # Multi-critic reward related Config
        self.rw_groups = self.env.config.rewards.reward_groups
        self.rw_weights = self.env.config.rewards.reward_weights
        self.num_act_lower_body = self.env.config.robot.lower_body_actions_dim
        self.num_act_upper_body = self.env.config.robot.upper_body_actions_dim
        self.num_act = {'lower_body': self.num_act_lower_body, 'upper_body': self.num_act_upper_body}

    def _setup_models_and_optimizer(self):
        self.actors = {}
        self.critics = {}
        self.actor_learning_rates = {}
        self.critic_learning_rates = {}
        for key in self.keys:
            print(f"key: {key}")
            print("num_act[key]: ", self.num_act[key])
            self.actors[key] = PPOActor(
                obs_dim_dict=self.algo_obs_dim_dict,
                module_config_dict=getattr(self.config.module_dict, 'actor' + '_' + key),
                num_actions=self.num_act[key],
                init_noise_std=self.config.init_noise_std[key]
            ).to(self.device)
            self.critics[key] = PPOCritic(self.algo_obs_dim_dict,
                                          self.config.module_dict.critic).to(self.device)
            self.actor_learning_rates[key] = self.actor_learning_rate
            self.critic_learning_rates[key] = self.critic_learning_rate
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        for key in self.keys:
            self.actor_optimizers[key] = optim.AdamW(
                self.actors[key].parameters(), 
                lr=self.actor_learning_rates[key],
                weight_decay=self.config.get('weight_decay', 0.01),  # L2 regularization
            )
            self.critic_optimizers[key] = optim.AdamW(
                self.critics[key].parameters(), 
                lr=self.critic_learning_rates[key],
                weight_decay=self.config.get('weight_decay', 0.01),  # L2 regularization
            )

    def _setup_storage(self):
        self.storage = RolloutStorage(self.env.num_envs, self.num_steps_per_env, device=self.device)
        ## Register obs keys
        for obs_key, obs_dim in self.algo_obs_dim_dict.items():
            history_len = self.algo_history_length_dict.get(obs_key, 1)
            print(f"Registering key: {obs_key} with shape: {obs_dim * history_len}")
            self.storage.register_key(obs_key, shape=(obs_dim * history_len,), dtype=torch.float)
        
        ## Register others
        for key in self.keys:
            self.storage.register_key('actions' + '_' + key, shape=(self.num_act[key],), dtype=torch.float)
            self.storage.register_key('rewards' + '_' + key, shape=(1,), dtype=torch.float)
            self.storage.register_key('dones' + '_' + key, shape=(1,), dtype=torch.float)
            self.storage.register_key('returns' + '_' + key, shape=(1,), dtype=torch.float)
            self.storage.register_key('advantages' + '_' + key, shape=(1,), dtype=torch.float)
            self.storage.register_key('values' + '_' + key, shape=(1,), dtype=torch.float)
            self.storage.register_key('actions_log_prob' + '_' + key, shape=(1,), dtype=torch.float)
            self.storage.register_key('action_mean' + '_' + key, shape=(self.num_act[key],), dtype=torch.float)
            self.storage.register_key('action_sigma' + '_' + key, shape=(self.num_act[key],), dtype=torch.float)

    def _eval_mode(self):
        for actor in self.actors.values():
            actor.eval()
        for critic in self.critics.values():
            critic.eval()

    def _train_mode(self):
        for actor in self.actors.values():
            actor.train()
        for critic in self.critics.values():
            critic.train()

    def load(self, ckpt_path):
        if ckpt_path is not None:
            logger.info(f"Loading checkpoint from {ckpt_path}")
            loaded_dict = torch.load(ckpt_path, map_location=self.device)
            for key in self.keys:
                self.actors[key].load_state_dict(loaded_dict["actor_model_state_dict"][key])
                self.critics[key].load_state_dict(loaded_dict["critic_model_state_dict"][key])
            if self.load_optimizer:
                for key in self.keys:
                    self.actor_optimizers[key].load_state_dict(loaded_dict["actor_optimizer_state_dict"][key])
                    self.critic_optimizers[key].load_state_dict(loaded_dict["critic_optimizer_state_dict"][key])
                    self.actor_learning_rates[key] = loaded_dict['actor_optimizer_state_dict'][key]['param_groups'][0]['lr']
                    self.critic_learning_rates[key] = loaded_dict['critic_optimizer_state_dict'][key]['param_groups'][0]['lr']
                self.set_learning_rates(self.actor_learning_rates, self.critic_learning_rates)
                logger.info(f"Optimizer loaded from checkpoint")
                logger.info(f"Actor Learning rate: {self.actor_learning_rates}")
                logger.info(f"Critic Learning rate: {self.critic_learning_rates}")
            self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict["infos"]

    def set_learning_rates(self, actor_learning_rates, critic_learning_rates):
        self.actor_learning_rates = actor_learning_rates
        self.critic_learning_rates = critic_learning_rates

    def save(self, path, infos=None):
        logger.info(f"Saving checkpoint to {path}")
        torch.save({
            "actor_model_state_dict": {key: self.actors[key].state_dict() for key in self.keys},
            'critic_model_state_dict': {key: self.critics[key].state_dict() for key in self.keys},
            'actor_optimizer_state_dict': {key: self.actor_optimizers[key].state_dict() for key in self.keys},
            'critic_optimizer_state_dict': {key: self.critic_optimizers[key].state_dict() for key in self.keys},
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def critic_evaluation(self, critic_obs, key):
        return self.critics[key].evaluate(critic_obs).detach()

    def _multi_actor_critic_rollout_step(self, obs_dict):
        actions_dict = {}
        for key in self.keys:
            transition = getattr(self.multi_transitions, 'transition_' + key)
            actions = self.actors[key].act(obs_dict["actor_obs"])
            actions_dict[key] = actions
            transition.actions = actions
            transition.actions_log_prob = self.actors[key].get_actions_log_prob(actions).detach().unsqueeze(1)
            transition.action_mean = self.actors[key].action_mean.detach()
            transition.action_sigma = self.actors[key].action_std.detach()
            transition.values = self.critic_evaluation(obs_dict["critic_obs"], key)
            # need to record obs and critic_obs before env.step()
            transition.actor_obs = obs_dict["actor_obs"]
            transition.critic_obs = obs_dict["critic_obs"]

        return actions_dict

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for i in range(self.num_steps_per_env):
                self.env.current_iteration = self.current_learning_iteration
                # Compute the actions and values
                actions = self._multi_actor_critic_rollout_step(obs_dict)
                actor_state = {}
                actor_state["actions"] = torch.cat([actions[key] for key in self.keys], dim=1)
                obs_dict, rewards, dones, infos = self.env.step(actor_state)
                self.infos = infos

                for obs_key in obs_dict.keys():
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                
                dones = dones.to(self.device)
                dones_stored = dones.clone().unsqueeze(1)
                rewards_stored = {}
                for rw_key, rw_value in rewards.items():
                    rewards_stored[rw_key] = rw_value.to(self.device).clone().unsqueeze(1)

                self.episode_env_tensors.add(infos["to_log"])

                self._process_env_step(rewards_stored, dones_stored, infos)

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        self.ep_infos.append(infos['episode'])
                    for rw_key, rw_value in rewards.items():
                        if rw_key in self.keys:
                            self.cur_reward_sum_decoupled[rw_key] += rw_value
                            self.cur_reward_sum += rw_value
                    # self.cur_reward_sum += rewards
                    self.cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    for key in self.keys:
                        self.rewbuffer_decoupled[key].extend(self.cur_reward_sum_decoupled[key][new_ids][:, 0].cpu().numpy().tolist())
                        self.cur_reward_sum_decoupled[key][new_ids] = 0
                    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_reward_sum[new_ids] = 0
                    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    self.cur_episode_length[new_ids] = 0

            self.stop_time = time.time()
            self.collection_time = self.stop_time - self.start_time
            self.start_time = self.stop_time
            
            # prepare data for training
            for key in self.keys:
                returns, advantages = self._compute_returns(
                    last_obs_dict=obs_dict,
                    policy_state_dict=dict(values=self.storage.query_key('values' + '_' + key), 
                                           dones=self.storage.query_key('dones' + '_' + key), 
                                           rewards=self.storage.query_key('rewards' + '_' + key)),
                    key=key
                )
                self.storage.batch_update_data('returns' + '_' + key, returns)
                self.storage.batch_update_data('advantages' + '_' + key, advantages)
            # TODO: Hardcode
            # rt_lb = self.storage.query_key('returns' + '_' + 'lower_body')
            # rt_ub = self.storage.query_key('returns' + '_' + 'upper_body')
            # adv_lb = self.storage.query_key('advantages' + '_' + 'lower_body')
            # adv_ub = self.storage.query_key('advantages' + '_' + 'upper_body')
            # self.storage.batch_update_data('returns' + '_' + 'lower_body', rt_lb * self.mc_weight + rt_ub * (1-self.mc_weight))
            # self.storage.batch_update_data('returns' + '_' + 'upper_body', rt_ub * self.mc_weight + rt_lb * (1-self.mc_weight))
            # self.storage.batch_update_data('advantages' + '_' + 'lower_body', adv_lb * self.mc_weight + adv_ub * (1-self.mc_weight))
            # self.storage.batch_update_data('advantages' + '_' + 'upper_body', adv_ub * self.mc_weight + adv_lb * (1-self.mc_weight))

        return obs_dict

    def _process_env_step(self, rewards, dones, infos):
        for key in self.keys:
            transition = getattr(self.multi_transitions, 'transition_' + key)
            transition.rewards = rewards[key]
            transition.dones = dones
            # Bootstrapping on time outs
            if 'time_outs' in infos:
                transition.rewards += self.gamma * transition.values * infos['time_outs'].unsqueeze(1).to(self.device)
            assert len(transition.rewards.shape) == 2
        # Record the transition
        self.storage.add_multi_transitions(self.multi_transitions)
        
        self.multi_transitions.clear()
        for actor in self.actors.values():
            actor.reset(dones)
        for critic in self.critics.values():
            critic.reset(dones)

    def _compute_returns(self, last_obs_dict, policy_state_dict, key):
        """Compute the returns and advantages for the given policy state.
        This function calculates the returns and advantages for each step in the 
        environment based on the provided observations and policy state. It uses 
        Generalized Advantage Estimation (GAE) to compute the advantages, which 
        helps in reducing the variance of the policy gradient estimates.
        Args:
            last_obs_dict (dict): The last observation dictionary containing the 
                      final state of the environment.
            policy_state_dict (dict): A dictionary containing the policy state 
                          information, including 'values', 'dones', 
                          and 'rewards'.
        Returns:
            tuple: A tuple containing:
            - returns (torch.Tensor): The computed returns for each step.
            - advantages (torch.Tensor): The normalized advantages for each step.
        """
        last_values = self.critic_evaluation(last_obs_dict["critic_obs"], key)
        advantage = 0
        
        values = policy_state_dict['values']
        dones = policy_state_dict['dones']
        rewards = policy_state_dict['rewards']
        
        last_values = last_values.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        rewards = rewards.to(self.device)
        
        returns = torch.zeros_like(values)
        
        num_steps = returns.shape[0]
        
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * self.gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages
    
    def _init_loss_dict_at_training_step(self):
        loss_dict = {}
        for key in self.keys:
            loss_dict['Value' + '_' + key] = 0
            loss_dict['Surrogate' + '_' + key] = 0
            loss_dict['Entropy' + '_' + key] = 0
            loss_dict["KL" + '_' + key] = 0
        return loss_dict

    def _update_algo_step(self, policy_state_dict, loss_dict):
        for key in self.keys:
            loss_dict = self._update_ppo(policy_state_dict, loss_dict, key)
        return loss_dict

    def _compute_ppo_loss(self, policy_state_dict, key):
        actions_batch = policy_state_dict['actions' + '_' + key]
        # target_values_batch = policy_state_dict['returns'] # This is wrong
        target_values_batch = policy_state_dict['values' + '_' + key] # This is correct
        advantages_batch = policy_state_dict['advantages' + '_' + key]
        returns_batch = policy_state_dict['returns' + '_' + key]
        old_actions_log_prob_batch = policy_state_dict['actions_log_prob' + '_' + key]
        old_mu_batch = policy_state_dict['action_mean' + '_' + key]
        old_sigma_batch = policy_state_dict['action_sigma' + '_' + key]

        self.actors[key].act(policy_state_dict["actor_obs"])
        actions_log_prob_batch = self.actors[key].get_actions_log_prob(actions_batch)
        # value_batch = self.critic.evaluate(policy_state_dict["critic_obs"])
        value_batch = self.critics[key].evaluate(policy_state_dict["critic_obs"])
        mu_batch = self.actors[key].action_mean
        sigma_batch = self.actors[key].action_std
        entropy_batch = self.actors[key].entropy

        # KL
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch) + 1.e-5) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)
                
                if kl_mean > self.desired_kl * 2.0:
                    self.actor_learning_rates[key] = max(1e-5, self.actor_learning_rates[key] / 1.5)
                    self.critic_learning_rates[key] = max(1e-5, self.critic_learning_rates[key] / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.actor_learning_rates[key] = min(1e-2, self.actor_learning_rates[key] * 1.5)
                    self.critic_learning_rates[key] = min(1e-2, self.critic_learning_rates[key] * 1.5)

                for param_group in self.actor_optimizers[key].param_groups:
                    param_group['lr'] = self.actor_learning_rates[key]
                for param_group in self.critic_optimizers[key].param_groups:
                    param_group['lr'] = self.critic_learning_rates[key]

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                        1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                            self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - self.entropy_coef * entropy_loss
        
        critic_loss = self.value_loss_coef * value_loss

        # In _compute_ppo_loss, update kl_coef using linear annealing
        if key == 'lower_body':
            # Anneal kl_coef linearly from start to end over anneal_steps
            progress = min(self.current_learning_iteration / self.kl_coef_anneal_steps, 1.0)
            self.kl_coef = self.kl_coef_start - progress * (self.kl_coef_start - self.kl_coef_end)
            
            # 计算 KL 散度
            kl_div = self.compute_kl_divergence_lower_body(self.external_actor, policy_state_dict["actor_obs"])
            kl_loss = kl_div.mean() * self.kl_coef
            actor_loss = actor_loss + kl_loss

        return actor_loss, critic_loss, value_loss, surrogate_loss, entropy_loss, kl_mean

    def _update_ppo(self, policy_state_dict, loss_dict, key):
        actor_loss, critic_loss, value_loss, surrogate_loss, entropy_loss, kl_mean = self._compute_ppo_loss(policy_state_dict, key)

        self.actor_optimizers[key].zero_grad()
        self.critic_optimizers[key].zero_grad()
        
        actor_loss.backward()
        critic_loss.backward()

        # Gradient step
        nn.utils.clip_grad_norm_(self.actors[key].parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.critics[key].parameters(), self.max_grad_norm)

        self.actor_optimizers[key].step()
        self.critic_optimizers[key].step()

        loss_dict['Value' + '_' + key] += value_loss.item()
        loss_dict['Surrogate' + '_' + key] += surrogate_loss.item()
        loss_dict['Entropy' + '_' + key] += entropy_loss.item()
        loss_dict["KL" + '_' + key] += kl_mean.item()
        return loss_dict

    def _post_epoch_logging(self, log_dict, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += log_dict['collection_time'] + log_dict['learn_time']
        iteration_time = log_dict['collection_time'] + log_dict['learn_time']

        ep_string = f''
        if log_dict['ep_infos']:
            for key in log_dict['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in log_dict['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, log_dict['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        train_log_dict = {}
        fps = int(self.num_steps_per_env * self.env.num_envs / (log_dict['collection_time'] + log_dict['learn_time']))
        train_log_dict['fps'] = fps
        mean_std = {}
        for key, actor in self.actors.items():
            mean_std[key] = actor.std.mean().item()
        train_log_dict['mean_std'] = mean_std

        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"Env/{k}": v for k, v in env_log_dict.items()}

        self._logging_to_writer(log_dict, train_log_dict, env_log_dict)

        str = f" \033[1m Learning iteration {log_dict['it']}/{self.current_learning_iteration + log_dict['num_learning_iterations']} \033[0m "

        if len(log_dict['rewbuffer']) > 0:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                            f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (Collection: {log_dict[
                            'collection_time']:.3f}s, Learning {log_dict['learn_time']:.3f}s)\n"""
                        #   f"""{'Value function loss:':>{pad}} {log_dict['mean_value_loss']:.4f}\n"""
                        #   f"""{'Surrogate loss:':>{pad}} {log_dict['mean_surrogate_loss']:.4f}\n"""
                        #   f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(log_dict['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(log_dict['lenbuffer']):.2f}\n""")
            for key in mean_std.keys():
                log_string += f"""{f'Mean action noise std ({key}):':>{pad}} {mean_std[key]:.2f}\n"""
        else:
            log_string = (f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {train_log_dict['fps']:.0f} steps/s (collection: {log_dict[
                            'collection_time']:.3f}s, learning {log_dict['learn_time']:.3f}s)\n"""
                        #   f"""{'Value function loss:':>{pad}} {log_dict['mean_value_loss']:.4f}\n"""
                        #   f"""{'Surrogate loss:':>{pad}} {log_dict['mean_surrogate_loss']:.4f}\n"""
                        #   f"""{'Mean action noise std:':>{pad}} {train_log_dict['mean_std']:.2f}\n"""
                          )
            for key in mean_std.keys():
                log_string += f"""{f'Mean action noise std ({key}):':>{pad}} {mean_std[key]:.2f}\n"""

        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (log_dict['it'] + 1) * (
                               log_dict['num_learning_iterations'] - log_dict['it']):.1f}s\n""")
        log_string += f"Logging Directory: {self.log_dir}"

        # Use rich Live to update a specific section of the console
        with Live(Panel(log_string, title="Training Log"), refresh_per_second=4, console=console):
            # Your training loop or other operations
            pass
    
    def _logging_to_writer(self, log_dict, train_log_dict, env_log_dict):
        # Logging Loss Dict
        for loss_key, loss_value in log_dict['loss_dict'].items():
            self.writer.add_scalar(f'Loss/{loss_key}', loss_value, log_dict['it'])
        # self.writer.add_scalar('Loss/actor_learning_rate', self.actor_learning_rate, log_dict['it'])
        # self.writer.add_scalar('Loss/critic_learning_rate', self.critic_learning_rate, log_dict['it'])
        # self.writer.add_scalar('Policy/mean_noise_std', train_log_dict['mean_std'], log_dict['it'])
        for key in self.keys:
            self.writer.add_scalar(f'Loss/actor_learning_rate_{key}', self.actor_learning_rates[key], log_dict['it'])
            self.writer.add_scalar(f'Loss/critic_learning_rate_{key}', self.critic_learning_rates[key], log_dict['it'])
            self.writer.add_scalar(f'Policy/mean_noise_std_{key}', train_log_dict['mean_std'][key], log_dict['it'])
            if self.config.get('log_all_action_std', False):
                for i, std_val in enumerate(self.actors[key].std.tolist()):
                    self.writer.add_scalar(f'Policy/action_std_{key}/dim_{i}', std_val, log_dict['it'])
            # Log min std dim and value
            self.writer.add_scalar(f'Policy/min_noise_std_val_{key}', self.actors[key].std.min().item(), log_dict['it'])
            self.writer.add_scalar(f'Policy/min_noise_std_dim_{key}', torch.argmin(self.actors[key].std).item(), log_dict['it'])

        self.writer.add_scalar('Perf/total_fps', train_log_dict['fps'], log_dict['it'])
        self.writer.add_scalar('Perf/collection time', log_dict['collection_time'], log_dict['it'])
        self.writer.add_scalar('Perf/learning_time', log_dict['learn_time'], log_dict['it'])
        if len(log_dict['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(log_dict['rewbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(log_dict['lenbuffer']), log_dict['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(log_dict['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(log_dict['lenbuffer']), self.tot_time)
        for key in self.keys:
            if len(self.rewbuffer_decoupled[key]) > 0:
                mean_reward = statistics.mean(self.rewbuffer_decoupled[key])
                self.writer.add_scalar(f'Train/mean_reward_{key}', mean_reward, log_dict['it'])
                self.writer.add_scalar(f'Train/mean_reward_{key}/time', mean_reward, self.tot_time)

        if len(env_log_dict) > 0:
            for k, v in env_log_dict.items():
                self.writer.add_scalar(k, v, log_dict['it'])
        if hasattr(self.env, 'action_scale_upper_body'):
            self.writer.add_scalar('Env/action_scale_upper_body', torch.mean(self.env.action_scale_upper_body).item(), log_dict['it'])
        if hasattr(self.env, 'apply_force_scale'):
            self.writer.add_scalar('Env/apply_force_scale', torch.mean(self.env.apply_force_scale).item(), log_dict['it'])
        if hasattr(self.env, 'command_height_scale'):
            self.writer.add_scalar('Env/command_height_scale', torch.mean(self.env.command_height_scale).item(), log_dict['it'])
        if hasattr(self.env, 'upper_body_tracking_sigma'):
            self.writer.add_scalar('Env/upper_body_tracking_sigma', torch.mean(self.env.upper_body_tracking_sigma).item(), log_dict['it'])
    
    @property
    def inference_model(self):
        return {
            "actors": self.actors,
            "critics": self.critics
        }

    ##########################################################################################
    # Code for Evaluation
    ##########################################################################################
    @torch.no_grad()
    def evaluate_policy(self):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()
        actor_state = self._create_actor_state()
        step = 0
        self.eval_policy = self._get_inference_policy()
        obs_dict = self.env.reset_all()
        init_actions = torch.zeros(self.env.num_envs, self.num_act_lower_body + self.num_act_upper_body, device=self.device)
        actor_state.update({"obs": obs_dict, "actions": init_actions})
        actor_state = self._pre_eval_env_step(actor_state)

        logger = Logger(self.env.dt)
        robot_index = 0 # which robot is used for logging
        load_motion_log = int(1 / self.env.dt) # step to start loading motion
        start_state_log = int(9 / self.env.dt) # step to start plotting states
        stop_state_log = int(13 / self.env.dt) # number of steps before plotting states

        base_quat = torch.zeros(self.env.num_envs, 4, device=self.device)

        integrated_torso_vel_error = np.zeros(self.env.num_envs)
        integrated_hand_pos_error = np.zeros(self.env.num_envs)
        integrated_torque = np.zeros(self.env.num_envs)
        integrated_upper_body_error = np.zeros(self.env.num_envs)

        while True:
            actor_state["step"] = step
            actor_state = self._pre_eval_env_step(actor_state)
            actor_state = self.env_step(actor_state)
            actor_state = self._post_eval_env_step(actor_state)

            upper_body_pos = self.env.simulator.dof_pos[:, self.env.upper_dof_indices]
            upper_body_dofs_error =  torch.sum(torch.square(upper_body_pos - self.env.ref_upper_dof_pos), dim=1)

            # only log the first hand (left hand)
            act_ext = self.env.marker_coords[:, -4, :3]  # (num_envs, 4, 3)
            ref_ext = self.env.ref_body_pos_extend[:, -4, :3]  # (num_envs, 4, 3)
            hand_pos_error = ref_ext - act_ext

            # 获取 torso link 的四元数 (num_envs, 4)
            torso_quat = self.env.simulator._rigid_body_rot[:, self.env.torso_index, :]

            # 先将 hand_pos_error 从世界系平移到 torso 原点
            # 假设 hand_pos_error 是 (num_envs, 4, 3)，不需要补齐
            hand_pos_error_3d = hand_pos_error

            # 变换到 torso-link 坐标系
            hand_pos_error_torso = quat_rotate_inverse(
                torso_quat,  # shape (1, 4)
                hand_pos_error_3d  # shape (1, 3)
            )

            if step <= start_state_log:
                base_quat = self.env.simulator.robot_root_states[:, 3:7]  # (num_envs, 4)
            base_lin_vel = self.env.simulator.robot_root_states[:, 7:10]  # (num_envs, 3)
            # 变换到base坐标系
            base_lin_vel_base_frame = quat_rotate_inverse(base_quat, base_lin_vel)

            torso_lin_vel = self.env.simulator._rigid_body_vel[:, self.env.torso_index, :]  # (num_envs, 3)
            torso_ang_vel = self.env.simulator._rigid_body_ang_vel[:, self.env.torso_index, :]  # (num_envs, 3)
            torso_quat = self.env.simulator._rigid_body_rot[:, self.env.torso_index, :]  # (num_envs, 4)
            torso_lin_vel_base = quat_rotate_inverse(torso_quat, torso_lin_vel)
            torso_ang_vel_base = quat_rotate_inverse(torso_quat, torso_ang_vel)

            arm_torques = self.env.simulator.dof_forces[:, self.env.upper_dof_indices]
            arm_torques_abs = torch.abs(arm_torques)
            arm_torques_left = arm_torques_abs[:, :4]
            arm_torques_right = arm_torques_abs[:, 4:]
            arm_torques_sum_left = torch.sum(arm_torques_left, dim=1)  # shape: (num_envs,)
            arm_torques_sum_right = torch.sum(arm_torques_right, dim=1)  # shape: (num_envs,)
            # print("current arm torques:", arm_torques_sum)

            if step == load_motion_log:
                self.env.simulator.next_task()

            if step < stop_state_log and step > start_state_log:
                torso_vel_error = np.abs(torso_lin_vel_base[:, 0].cpu().numpy() - self.env.commands[:, 0].cpu().numpy()) \
                                + np.abs(torso_lin_vel_base[:, 1].cpu().numpy() - self.env.commands[:, 1].cpu().numpy()) \
                                + np.abs(torso_lin_vel_base[:, 2].cpu().numpy()) \
                                + np.abs(torso_ang_vel_base[:, 2].cpu().numpy()) # only consider x, y, z velocity and yaw angular velocity
                integrated_torso_vel_error += torso_vel_error

                hand_pos_error_np = np.linalg.norm(hand_pos_error_torso.cpu().numpy(), axis=1)  # shape: (num_envs, 4)
                integrated_hand_pos_error += hand_pos_error_np  # only consider the first hand
                
                integrated_torque += (arm_torques_sum_left + arm_torques_sum_right).cpu().numpy()

                integrated_upper_body_error += upper_body_dofs_error.cpu().numpy() 

                logger.log_states(
                    {
                        # 'dof_pos_target': actions[robot_index, joint_index].item() * self.env.simulator.cfg.control.action_scale + self.env.simulator.default_dof_pos[robot_index, joint_index].item(),
                        # 'dof_pos_target': self.env.simulator.actions[robot_index, joint_index].item() * self.env.simulator.cfg.control.action_scale + self.env.simulator.default_dof_pos[robot_index, joint_index].item(),
                        'dof_pos_err': upper_body_dofs_error[robot_index].item(),
                        # 'dof_vel': self.env.simulator.dof_vel[robot_index, joint_index].item(),
                        'dof_torque_left': arm_torques_sum_left[robot_index].item(),
                        'dof_torque_right': arm_torques_sum_right[robot_index].item(),
                        'base_vel_x': base_lin_vel_base_frame[robot_index, 0].item(),
                        'torso_vel_x': torso_lin_vel_base[robot_index, 0].item(),
                        'base_vel_target_x': self.env.commands[robot_index, 0].item(),
                        'base_vel_y': base_lin_vel_base_frame[robot_index, 1].item(),
                        'torso_vel_y': torso_lin_vel_base[robot_index, 1].item(),
                        'base_vel_target_y': self.env.commands[robot_index, 1].item(),
                        'hand_pos_x_err': hand_pos_error_torso[robot_index, 0].item(),
                        'hand_pos_y_err': hand_pos_error_torso[robot_index, 1].item(),
                        'hand_pos_z_err': hand_pos_error_torso[robot_index, 2].item(),
                        # 'base_vel_y': simulator.base_lin_vel[robot_index, 1].item(),
                        # 'base_vel_z': simulator.base_lin_vel[robot_index, 2].item(),
                        # 'base_vel_yaw': simulator.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': self.env.simulator.contact_forces[robot_index, self.env.feet_indices, 2].cpu().numpy()
                    }
                )
                print("Logging states...", step)
            elif step == start_state_log:
                logger.state_log.clear()
            elif step == stop_state_log:
                logger.plot_states()
                # integrated torso velocity error
                integrated_torso_vel_error /= (stop_state_log - start_state_log)
                avg_error = integrated_torso_vel_error.mean()
                std_error = integrated_torso_vel_error.std()
                print(f"Average integrated torso error over all envs: {avg_error:.3f}")
                print(f"Std integrated torso error over all envs: {std_error:.3f}")

                # integrated hand position error
                integrated_hand_pos_error /= (stop_state_log - start_state_log)
                avg_hand_error = integrated_hand_pos_error.mean()
                std_hand_error = integrated_hand_pos_error.std()
                print(f"Average integrated hand position error over all envs: {avg_hand_error:.3f}")
                print(f"Std integrated hand position error over all envs: {std_hand_error:.3f}")

                # integrated torque
                integrated_torque /= (stop_state_log - start_state_log)
                avg_torque = integrated_torque.mean()
                std_torque = integrated_torque.std()
                print(f"Average integrated arm torque over all envs: {avg_torque:.3f}")
                print(f"Std integrated arm torque over all envs: {std_torque:.3f}")

                # integrated upper error
                integrated_upper_body_error /= (stop_state_log - start_state_log)
                avg_upper_error = integrated_upper_body_error.mean()
                std_upper_error = integrated_upper_body_error.std()
                print(f"Average integrated upper body error over all envs: {avg_upper_error:.3f}")
                print(f"Std integrated upper body error over all envs: {std_upper_error:.3f}")

                # # === Log state_log to CSV ===
                # log_path = os.path.join(self.log_dir if self.log_dir else ".", "state_log.csv")
                # with open(log_path, "w", newline="") as csvfile:
                #     if len(logger.state_log) > 0:
                #         writer = csv.DictWriter(csvfile, fieldnames=logger.state_log[0].keys())
                #         writer.writeheader()
                #         for row in logger.state_log:
                #             writer.writerow(row)
                # print(f"State log saved to {log_path}")

            step += 1
        self._post_evaluate_policy()
    
    def _pre_eval_env_step(self, actor_state: dict):
        actions = self.eval_policy(actor_state["obs"]['actor_obs'])
        actor_state.update({"actions": actions})
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update(
            {"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras}
        )

        # 计算 KL 散度
        # kl_div = self.compute_kl_divergence_lower_body(self.external_actor)
        # print("KL:", kl_div)

        return actor_state

    def act_inference(self, actor_obs):
        actions = {}
        for key in self.keys:
            actions[key] = self.actors[key].act_inference(actor_obs)
        return torch.cat([actions[key] for key in self.keys], dim=1)

    def _get_inference_policy(self, device=None):
        for key in self.keys:
            self.actors[key].eval() # switch to evaluation mode (dropout for example)
            if device is not None:
                self.actors[key].to(device)
        return self.act_inference
    
    ##########################################################################################
    # KL divergence
    ##########################################################################################
    def compute_kl_divergence_upper_body(self, external_actor, obs=None):
        """
        计算当前 upper-body policy 与给定 external_actor 的 KL 散度。
        Args:
            external_actor: PPOActor 实例，与 self.actors['upper_body'] 结构一致
            obs: 用于推理的观测张量（如 actor_obs），如果为 None，则用最近一次 rollout 的观测
        Returns:
            kl_div: [num_envs] 张量，每个环境的 KL 散度
        """
        from torch.distributions import kl_divergence, Normal
        if obs is None:
            obs = self.storage.query_key('actor_obs')
        # 当前 policy
        self.actors['upper_body'].eval()
        self.actors['upper_body'].update_distribution(obs)
        mu1 = self.actors['upper_body'].action_mean
        std1 = self.actors['upper_body'].action_std
        dist1 = Normal(mu1, std1)
        # 给定 policy
        external_actor.eval()
        external_actor.update_distribution(obs)
        mu2 = external_actor.action_mean
        std2 = external_actor.action_std
        dist2 = Normal(mu2, std2)
        # 计算 KL 散度（逐元素，最后按动作维度求和）
        kl = kl_divergence(dist1, dist2)
        kl_div = kl.sum(dim=-1)  # [num_envs]

        return kl_div
    
    def compute_kl_divergence_lower_body(self, external_actor, obs=None):
        """
        计算当前 lower-body policy 与给定 external_actor 的 KL 散度。
        支持 obs 维度裁剪：将当前 obs 按 base policy 的 obs 名称和维度裁剪后输入 external_actor。
        Args:
            external_actor: PPOActor 实例，与 self.actors['lower_body'] 结构一致
            obs: 用于推理的观测张量（如 actor_obs），如果为 None，则用最近一次 rollout 的观测
        Returns:
            kl_div: [num_envs] 张量，每个环境的 KL 散度
        """
        from torch.distributions import kl_divergence, Normal
        if obs is None:
            obs = self.storage.query_key('actor_obs')

        # 1. 获取 base policy 需要的 obs 名称和维度
        if hasattr(external_actor, 'obs_dim_dict'):
            base_obs_dim_dict = external_actor.obs_dim_dict
        else:
            # fallback: 使用 self.base_obs_dim_dict
            base_obs_dim_dict = self.base_obs_dim_dict

        # base_obs_keys = list(base_obs_dim_dict.keys())
        # print("base_obs_keys:", base_obs_keys)  # 调试：打印 obs 名称

        history_len = self.base_history_length_dict.get("actor_obs", 1)
        base_obs_dim = base_obs_dim_dict["actor_obs"]
        # print("base_obs_dim:", base_obs_dim)

        # 2. 按照 base policy 的 obs 名称裁剪当前 obs
        # 选择前 base_obs_dim 大小作为 actor dim（假设 base obs 在存储结构中是前面的部分）
        obs_clipped = obs[..., :base_obs_dim * history_len]

        # print("obs_clipped:", obs_clipped.shape)
        # print("obs:", obs.shape)

        # 当前 policy
        self.actors['lower_body'].eval()
        self.actors['lower_body'].update_distribution(obs)
        mu1 = self.actors['lower_body'].action_mean
        std1 = self.actors['lower_body'].action_std
        dist1 = Normal(mu1, std1)
        # 给定 policy
        external_actor.eval()
        external_actor.update_distribution(obs_clipped)
        mu2 = external_actor.action_mean
        std2 = external_actor.action_std
        dist2 = Normal(mu2, std2)

        # 计算 KL 散度（逐元素，最后按动作维度求和）
        kl = kl_divergence(dist1, dist2)
        kl_div = kl.sum(dim=-1)  # [num_envs]

        return kl_div

    def load_base_distribution(self, checkpoint_path):
        """
        只从 checkpoint 加载 lower_body actor 的 state_dict，避免重复创建仿真环境。
        参数：
            checkpoint_path (str): checkpoint 文件路径 (.pt)
        返回：
            state_dict: lower_body actor 的参数字典
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        lower_body_state_dict = checkpoint['actor_model_state_dict']['lower_body']

        return lower_body_state_dict
    
    def create_external_actor(self, checkpoint_path):
        # 1. 加载 state_dict
        lower_body_state_dict = self.load_base_distribution(checkpoint_path)

        # 2. 新建 PPOActor 实例（参数需与你训练时一致）
        external_actor = PPOActor(
            obs_dim_dict=self.base_obs_dim_dict,
            module_config_dict=self.config.module_dict.actor_lower_body,
            num_actions=self.num_act_lower_body,
            init_noise_std=self.config.init_noise_std['lower_body']
        ).to(self.device)

        # print("obs_dim_dict:", self.base_obs_dim_dict)
        # print("algo_obs_dim_dict:", self.algo_obs_dim_dict)
        # input()

        # 3. 加载权重
        external_actor.load_state_dict(lower_body_state_dict)

        return external_actor