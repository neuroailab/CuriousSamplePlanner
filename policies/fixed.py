from CuriousSamplePlanner.policies.policy import EnvPolicy
import torch 
import numpy as np
from CuriousSamplePlanner.scripts.utils import *

from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.envs import make_vec_envs
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr import algo, utils
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.algo import gail
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.storage import RolloutStorage


class FixedPolicy(EnvPolicy):
	def __init__(self, *args):
		super(FixedPolicy, self).__init__(*args)

		# Load the policy from the given dataset
		self.device = torch.device(opt_cuda_str() if torch.cuda.is_available() else "cpu")
		self.full_path = self.experiment_dict['policy_path']+self.experiment_dict['task']+".pt"
		if(torch.cuda.is_available()):
			self.agent, self.rm_mean = torch.load(self.full_path)
		else:			
			self.agent, self.rm_mean = torch.load(self.full_path, map_location='cpu')

		self.recurrent_hidden_states = torch.zeros(1, self.agent.recurrent_hidden_state_size)
		self.masks = torch.ones(1, 1)
		self.envs = make_vec_envs(self.experiment_dict['task'], 0, 1, self.experiment_dict['gamma'], "", self.device, True, experiment_dict = self.experiment_dict, made_env = self.environment)
		setattr(utils.get_vec_normalize(self.envs), 'ob_rms', self.rm_mean)

	def reset(self):
		return self.envs.reset()

	def step(self, action):
		normalized_state, reward, done, infos = self.envs.step(action)
		next_state = opt_cuda(torch.tensor(self.environment.get_current_config()))
		return next_state, reward, done, infos[0]

	def select_action(self, obs):



		value, action, action_log_prob, recurrent_hidden_states = self.agent.act(
                    obs, self.recurrent_hidden_states,
                    self.masks)
		info = {}
		return action, info


