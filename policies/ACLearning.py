from CuriousSamplePlanner.policies.policy import EnvPolicy
import torch 
import numpy as np

from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr import algo, utils
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.storage import RolloutStorage
# DDPG imports 

from CuriousSamplePlanner.scripts.utils import *


class ACLearningPolicy(EnvPolicy):
	def __init__(self, *args):
		super(ACLearningPolicy, self).__init__(*args)
		# Build models
		self.actor_critic = Policy([self.environment.config_size], self.environment.action_space, base_kwargs={'recurrent': self.experiment_dict["recurrent_policy"]})
		if self.experiment_dict["algo"] == 'a2c':
			self.agent = algo.A2C_ACKTR(
				self.actor_critic,
				self.experiment_dict["value_loss_coef"],
				self.experiment_dict["entropy_coef"],
				lr=self.experiment_dict["learning_rate"],
				eps=self.experiment_dict["eps"],
				alpha=self.experiment_dict["alpha"],
				max_grad_norm=self.experiment_dict["max_grad_norm"])
		elif self.experiment_dict['algo'] == 'ppo':
			self.agent = algo.PPO(
				self.actor_critic,
				self.experiment_dict["clip_param"],
				self.experiment_dict["ppo_epoch"],
				self.experiment_dict["num_mini_batch"],
				self.experiment_dict["value_loss_coef"],
				self.experiment_dict["entropy_coef"],
				lr=self.experiment_dict["learning_rate"],
				eps=self.experiment_dict["eps"],
				max_grad_norm=self.experiment_dict["max_grad_norm"])

		self.rollouts = RolloutStorage(self.experiment_dict["num_steps"], 1,
								  [self.environment.config_size], self.environment.action_space,
								  self.actor_critic.recurrent_hidden_state_size)
		self.stepi = 0

	def reset(self):
		obs = self.environment.reset()
		self.rollouts.obs[0].copy_(obs)
		return obs

	def select_action(self, obs):
		with torch.no_grad():
			value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
				self.rollouts.obs[self.stepi], self.rollouts.recurrent_hidden_states[self.stepi],
				self.rollouts.masks[self.stepi])

		action = torch.squeeze(action)
		info = {'value': value, 'action_log_prob': action_log_prob, 'recurrent_hidden_states': recurrent_hidden_states}
		self.stepi+=1
		return action, info


	def store_results(self, *args):
		self.rollouts.insert(*args)

	def update(self, total_numsteps):
		num_updates = int(
			self.experiment_dict["num_env_steps"]) // self.experiment_dict["num_steps"]

		self.stepi = 0;
		with torch.no_grad():
			next_value = self.actor_critic.get_value(
				self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
				self.rollouts.masks[-1]).detach()


		self.rollouts.compute_returns(next_value, self.experiment_dict["use_gae"], self.experiment_dict["gamma"],
								 0, self.experiment_dict["use_proper_time_limits"])
		value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)

		self.rollouts.after_update()

		if self.experiment_dict['use_linear_lr_decay']:
			# decrease learning rate linearly
			utils.update_linear_schedule(
				self.agent.optimizer, total_numsteps//self.experiment_dict["num_steps"], num_updates,
				self.agent.optimizer.lr if self.experiment_dict["algo"] == "acktr" else self.experiment_dict["learning_rate"])
			for param_group in self.agent.optimizer.param_groups:
				print(param_group['lr'])

		return value_loss, action_loss, dist_entropy
