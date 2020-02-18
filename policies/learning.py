from CuriousSamplePlanner.policies.policy import EnvPolicy
import torch 
import numpy as np
# DDPG imports 
from CuriousSamplePlanner.ddpg.ddpg import DDPG
from CuriousSamplePlanner.ddpg.naf import NAF
from CuriousSamplePlanner.ddpg.normalized_actions import NormalizedActions
from CuriousSamplePlanner.ddpg.ounoise import OUNoise
from CuriousSamplePlanner.ddpg.param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from CuriousSamplePlanner.ddpg.balanced_replay_memory import BalancedReplayMemory, Transition
from CuriousSamplePlanner.ddpg.replay_memory import ReplayMemory
from CuriousSamplePlanner.scripts.utils import *


class LearningPolicy(EnvPolicy):
	def __init__(self, *args):
		super(LearningPolicy, self).__init__(*args)

		self.agent = DDPG(self.experiment_dict['gamma'], self.experiment_dict['tau'], self.experiment_dict['hidden_size'], self.environment.config_size, self.environment.action_space, actor_lr = self.experiment_dict['actor_lr'], critic_lr = self.experiment_dict["critic_lr"])
		self.agent.cuda()
		if(self.experiment_dict['use_splitter']):
			self.memory = BalancedReplayMemory(self.experiment_dict['replay_size'], split=self.experiment_dict["split"])
		else:
			self.memory = ReplayMemory(self.experiment_dict['replay_size'])

		self.ounoise = OUNoise(self.environment.action_space.shape[0]) if self.experiment_dict['ou_noise'] else None
		self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=20, desired_action_stddev=self.experiment_dict['noise_scale'], adaptation_coefficient=1.05) if self.experiment_dict['param_noise'] else None

		if self.experiment_dict['ou_noise']: 
			self.ounoise.scale = (self.experiment_dict['noise_scale'] - self.experiment_dict['final_noise_scale']) * max(0, self.experiment_dict['exploration_end'] - self.i_episode) / self.experiment_dict['exploration_end'] + self.experiment_dict['final_noise_scale']
			self.ounoise.reset()

	def got_reward(self):
		if self.experiment_dict['enable_asm'] and self.experiment_dict['param_noise'] and len(self.memory) >= self.experiment_dict['batch_size']:
			episode_transitions = self.memory.memory[self.memory.position-self.experiment_dict['batch_size']:self.memory.position]
			states = torch.cat([transition[0] for transition in episode_transitions], 0)
			unperturbed_actions = self.agent.select_action(states, None, None)
			perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)
			ddpg_dist = ddpg_distance_metric(perturbed_actions.detach().cpu().numpy(), unperturbed_actions.detach().cpu().numpy())
			self.param_noise.adapt(ddpg_dist)

		if self.experiment_dict['enable_asm'] and self.experiment_dict['ou_noise']: 
			self.ounoise.scale = (self.experiment_dict['noise_scale'] - self.experiment_dict['final_noise_scale']) * max(0, self.experiment_dict['exploration_end'] - self.i_episode) / self.experiment_dict['exploration_end'] + self.experiment_dict['final_noise_scale']
			self.ounoise.reset()

	def select_action(self, obs):
		action = self.agent.select_action(obs, self.ounoise, self.param_noise)
		return action

	def store_results(self, *args):
		next_state, reward, done, infos, parent, action = args
		mask = opt_cuda(torch.Tensor([not done]))
		reward = opt_cuda(torch.Tensor([reward]))
		parent_config = torch.unsqueeze(opt_cuda(torch.Tensor(parent.config).type(torch.FloatTensor)), dim=0)

		if(self.experiment_dict['use_splitter']):
			self.memory.push(int(reward.item() == 1), parent_config.detach().cpu(), action.detach().cpu(), mask.detach().cpu(), next_state.detach().cpu(), reward.detach().cpu())
		else:
			self.memory.push(parent_config.detach().cpu(), action.detach().cpu(), mask, next_state.detach().cpu(), reward.detach().cpu())

	def update(self, total_numsteps):
		if len(self.memory) > self.experiment_dict['batch_size'] and total_numsteps%self.experiment_dict['update_interval']==0:
			for _ in range(self.experiment_dict['updates_per_step']):
				transitions = self.memory.sample(self.experiment_dict['batch_size'])
				transitions = [[opt_cuda(i) for i in r] for r in transitions]
				batch = Transition(*zip(*transitions))
				value_loss, policy_loss = self.agent.update_parameters(batch)
				transitions = None
		
