from CuriousSamplePlanner.policies.policy import EnvPolicy
import torch 
import numpy as np
from CuriousSamplePlanner.scripts.utils import *


class RandomPolicy(EnvPolicy):
	def __init__(self, *args):
		super(RandomPolicy, self).__init__(*args)

	def select_action(self, obs):
		action = opt_cuda(torch.unsqueeze(torch.tensor(np.random.uniform(low=-1, high=1, size=self.environment.action_space_size)), dim=0).type(torch.FloatTensor))
		info = {}
		return action, info