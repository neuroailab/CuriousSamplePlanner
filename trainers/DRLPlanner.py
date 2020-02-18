#!/usr/bin/env python
from __future__ import print_function
import pybullet as p
import numpy as np
import time
import random
import math
import imageio
import matplotlib.pyplot as plt
import os
import shutil
import h5py

import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from motion_planners.discrete import astar
import sys
from collections import deque

from CuriousSamplePlanner.tasks.two_block_stack import TwoBlocks
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks

from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.plan_graph import PlanGraph
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer

from CuriousSamplePlanner.policies.fixed import FixedPolicy
from CuriousSamplePlanner.policies.random import RandomPolicy
from CuriousSamplePlanner.policies.learning import LearningPolicy
from CuriousSamplePlanner.policies.ACLearning import ACLearningPolicy


from CuriousSamplePlanner.trainers.architectures import WorldModel, SkinnyWorldModel

class DRLPlanner():
	def __init__(self, experiment_dict):
		self.experiment_dict = experiment_dict

		# Create the replay buffer for training world models
		self.experience_replay = ExperienceReplayBuffer()

		# Create the environment
		EC = getattr(sys.modules[__name__], self.experiment_dict["task"])
		self.environment = EC(experiment_dict)

		# Create the policy
		PC = getattr(sys.modules[__name__], self.experiment_dict["policy"])
		self.policy = PC(experiment_dict, self.environment)

		# Init the world model
		self.worldModel = opt_cuda(WorldModel(config_size=self.environment.config_size))
		self.transform = list(self.environment.predict_mask)
		random.shuffle(self.transform)

		self.criterion = nn.MSELoss()
		self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["wm_learning_rate"])

		# Init the plan graph 
		self.graph = PlanGraph(environment=self.environment, node_sampling = self.experiment_dict['node_sampling'])
		super(DRLPlanner, self).__init__()

	def save_params(self):
		with open(self.experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
			pickle.dump(self.experiment_dict, fa)
			fa.close()

		graph_filehandler = open(self.experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
		pickle.dump(self.graph, graph_filehandler)

	def plan(self):

		obs = self.policy.reset()
		reset_obs = obs
		# rollouts.to(device)

		episode_rewards = deque(maxlen=1000)
		value_losses = deque(maxlen=10)
		action_losses = deque(maxlen=10)
		# Create the replay buffer for training world models
		experience_replay = ExperienceReplayBuffer()

		start = time.time()
		num_updates = int(
			self.experiment_dict["num_env_steps"]) // self.experiment_dict["num_steps"]

		for j in range(num_updates):
			for step in range(self.experiment_dict['num_steps']):
				self.experiment_dict['num_sampled_nodes']+=1
				# Sample actions
				action, ainfos = self.policy.select_action(None)
				value, action_log_prob, recurrent_hidden_states = ainfos['value'], ainfos['action_log_prob'], ainfos['recurrent_hidden_states']
				# Step in environment
				next_state, reward, done, infos = self.policy.step(action/2.0)
				inputs, prestate, feasible, command = infos['inputs'], infos['prestable'], infos['feasible'], infos['command']


				# Insert into buffer for world model training
				if(self.experiment_dict['reward_alpha'] != 1):
					obs_tensor = opt_cuda(next_state.type(torch.FloatTensor))
					inputs = opt_cuda(inputs.type(torch.FloatTensor))
					intrinsic_reward = torch.nn.functional.mse_loss(obs_tensor[:, self.transform], self.worldModel(obs_tensor)[:, self.environment.predict_mask]).detach()

					targets = opt_cuda(next_state)
					filler = opt_cuda(torch.tensor([0]))
					experience_replay.bufferadd_single(torch.squeeze(inputs), torch.squeeze(targets), torch.squeeze(prestate), torch.squeeze(action), filler, filler, filler)
					rew_e = self.experiment_dict['reward_alpha']*reward
					rew_i  = (1-self.experiment_dict['reward_alpha'])*intrinsic_reward.item()
					rew_t = rew_e+rew_i
				else:
					rew_t = reward

				episode_rewards.append(rew_t)

				# print(reward)

				# Inserted for single-task experimentation					
				if random.uniform(0,1) < self.experiment_dict['reset_frequency']:
					next_state = self.policy.reset()

				# If done then clean the history of observations.
				masks = torch.FloatTensor(
					[[0.0] if done_ else [1.0] for done_ in [done]])
				bad_masks = torch.FloatTensor(
					[[1.0]])
				self.policy.store_results(next_state, recurrent_hidden_states, action,
								action_log_prob, value, torch.tensor(rew_t), masks, bad_masks)
			value_loss, action_loss, dist_entropy = self.policy.update(self.experiment_dict['num_sampled_nodes'])
			value_losses.append(value_loss)
			action_losses.append(action_loss)

			# Update world model
			if(self.experiment_dict['reward_alpha'] != 1):
				print("Updating world model")
				for epoch in range(self.experiment_dict['num_training_epochs']):
					for next_loaded in enumerate(DataLoader(experience_replay, batch_size=self.experiment_dict['batch_size'], shuffle=True, num_workers=0)):
						
						_, batch = next_loaded
						inputs, labels, prestates, actions, _, _, index = batch
						#outputs_target = opt_cuda(self.worldModelTarget(labels).type(torch.FloatTensor))
						labels = opt_cuda(labels.type(torch.FloatTensor))
						actions = opt_cuda(actions.type(torch.FloatTensor))
						prestates = opt_cuda(prestates.type(torch.FloatTensor))

						# TODO: Fix
						self.optimizer_world.zero_grad()
						outputs = opt_cuda(self.worldModel(opt_cuda(labels.type(torch.FloatTensor))))
						loss = self.criterion(outputs[:, self.environment.predict_mask], labels[:, self.transform])
						loss.backward()
						self.optimizer_world.step()
						self.experiment_dict["world_model_losses"].append(loss.item())


				del experience_replay
				experience_replay = ExperienceReplayBuffer()

			if j % self.experiment_dict['log_interval'] == 0 and len(episode_rewards) > 1:
				total_num_steps = (j + 1) * self.experiment_dict['num_steps']
				end = time.time()
				self.experiment_dict['rewards'].append(np.mean(episode_rewards))
				print(
					"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.1f}/{:.1f}, Value Loss {:.3f}, Action Loss {:.3f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards),
							np.mean(value_losses),
							np.mean(action_loss)))
				stats_filehandler = open(self.experiment_dict['exp_path'] + "/stats.pkl", 'wb')
				pickle.dump(self.experiment_dict, stats_filehandler)


		return None, None, self.experiment_dict



