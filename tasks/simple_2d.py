#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from CuriousSamplePlanner.scripts.utils import *
from gym import spaces

class Simple2d():
	def __init__(self, *args):
		self.current_state = torch.tensor(np.array([0, 0]))
		self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
		self.config_size =  2
		self.predict_mask = list(range(2))
		self.max_timesteps = 32
		self.num_fails = 0
		self.render = False
		self.trajectories = []
		self.trajectory = []


	@property
	def fixed(self):
		return []

	def set_state(self, conf):
		self.current_state = conf

	def check_goal_state(self, config):
		if(config[0]>0.5 and config[1]>0.5):
			return True
		return False

	def get_reward(self, config):
		if(self.check_goal_state(config)):
			return 40
		else:
			return -1


	def compute_reward(self, obs, goal, uk):
		# We want to test an observation in the current environment
		return np.array([self.get_reward(obs[i]) for i in range(obs.shape[0])])

	def get_current_config(self):
		return self.current_state


	def step(self, action, terminate_unreachable=False, state_estimation=False):

		action = action

		# Action is just the rotation
		action = action * math.pi
		action_mag = 0.1
		next_state = self.current_state+torch.tensor(np.array([action_mag*math.sin(action[0]), action_mag*math.cos(action[0])]))
	
		clipped_next_state = torch.clamp(next_state, -1, 1)
		self.current_state = clipped_next_state
		self.trajectory.append(self.current_state)
		done = self.check_goal_state(clipped_next_state)
		reward = self.get_reward(clipped_next_state)

		self.num_fails+=1
		goal_state = clipped_next_state
		if(self.num_fails >= self.max_timesteps or done):
			if(done):
				goal_state = self.current_state
			done = True
			clipped_next_state = self.reset()
			self.num_fails = 0
			self.trajectories.append(self.trajectory)
			# self.trajectory = []
			# if(len(self.trajectories)>10):
			# 	for ti, traj in enumerate(self.trajectories):
			# 		plt.plot([t[0].item() for t in traj],[t[1].item() for t in traj], 'r', alpha = 0.2)
			# 		if(ti>=10):
			# 			break
			# 	plt.ylim([-1, 1])
			# 	plt.xlim([-1, 1])
			# 	plt.show()
			# 	self.trajectories = []

		return clipped_next_state, reward, done, {"episode": {"r": reward} , "inputs": None, "prestable": None, "feasible":None, "command":None, "goal_state":goal_state}

	def reset(self):
		self.current_state = self.get_start_state()
		return torch.tensor(self.current_state)

	def get_start_state(self):
		return torch.tensor(np.array([0, 0]))


