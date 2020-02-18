import torch 
import numpy as np
from CuriousSamplePlanner.scripts.utils import *


class EnvPolicy:
	def __init__(self, experiment_dict, environment):
		self.environment = environment
		self.experiment_dict = experiment_dict
		self.i_episode = 0

	def step(self, action):
		return self.environment.step(action.cpu().detach().numpy())

	def reset(self):
		return self.environment.reset()
	def got_reward(self):
		pass
		
	def select_action(self, obs):
		pass

	def update(self, total_numsteps):
		pass
		
	def store_results(self, *args):
		pass