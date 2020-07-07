#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import time
import random
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from collections import deque

from CuriousSamplePlanner.tasks.two_block_stack import TwoBlocks
from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
from CuriousSamplePlanner.tasks.simple_2d import Simple2d

from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.plan_graph import PlanGraph
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer

from CuriousSamplePlanner.policies.fixed import FixedPolicy
from CuriousSamplePlanner.policies.random import RandomPolicy
from CuriousSamplePlanner.policies.DDPGLearning import DDPGLearningPolicy
from CuriousSamplePlanner.policies.PPOLearning import PPOLearningPolicy
from CuriousSamplePlanner.policies.HERLearning import HERLearningPolicy

from CuriousSamplePlanner.trainers.architectures import WorldModel, SkinnyWorldModel
from CuriousSamplePlanner.trainers.planner import Planner


class DRLPlanner(Planner):
	def __init__(self, experiment_dict):
		# Init the plan graph 
		super(DRLPlanner, self).__init__(experiment_dict)

		# Create the replay buffer for training world models
		self.experience_replay = ExperienceReplayBuffer()

		# Init the world model
		self.worldModel = opt_cuda(WorldModel(config_size=self.environment.config_size))
		self.transform = list(self.environment.predict_mask)
		random.shuffle(self.transform)

		self.criterion = nn.MSELoss()
		self.optimizer_world = optim.Adam(self.worldModel.parameters(), lr=self.experiment_dict["wm_learning_rate"])


	def save_params(self):
		with open(self.experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
			pickle.dump(self.experiment_dict, fa)
			fa.close()

		graph_filehandler = open(self.experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
		pickle.dump(self.graph, graph_filehandler)

	def plan(self):

		obs = self.policy.reset()
		reset_obs = obs
		next_state = obs
		# rollouts.to(device)

		episode_rewards = deque(maxlen=1000)
		value_losses = deque(maxlen=10)
		action_losses = deque(maxlen=10)
		wm_losses = deque(maxlen=1000)
		# Create the replay buffer for training world models
		experience_replay = ExperienceReplayBuffer()

		start = time.time()
		num_updates = int(
			self.experiment_dict["num_env_steps"]) // self.experiment_dict["num_steps"]

		for j in range(num_updates):
			for step in range(self.experiment_dict['num_steps']):
				self.experiment_dict['num_sampled_nodes']+=1

				# Sample actions, obs is none because it is taken from rollouts inside the policy object
				start_time = time.time()
				action, ainfos = self.policy.select_action(next_state)
				if(self.experiment_dict['debug_timing']):
					self.experiment_dict['action_selection_timings'].append(time.time()-start_time)

				# Step in environment
				prev_state = next_state
				next_state, reward, done, infos = self.policy.step(torch.squeeze(action))
				if(reward>0 and self.experiment_dict['return_on_solution']):
					return None, None, self.experiment_dict

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

				episode_rewards.append(reward)
				
				# # Inserted for single-task experimentation					
				# if random.uniform(0,1) < self.experiment_dict['reset_frequency']:
				# 	next_state = self.policy.reset()
				self.policy.store_results(next_state, action, torch.tensor(rew_t), ainfos, prev_state, done or (step==self.experiment_dict['num_steps']-1), infos['goal_state'])
					
				# Update every step if ddpg (off policy) or once per trajectory if a2c/ppo (on policy)
				if(self.experiment_dict["policy"] == "DDPGLearningPolicy"):
					start_time = time.time()
					value_loss, action_loss, dist_entropy = self.policy.update(self.experiment_dict['num_sampled_nodes'])
					self.experiment_dict['policy_update_timings'].append(time.time()-start_time)

				if(done):
					break

			if(self.experiment_dict['policy'] == "HERLearningPolicy" or self.experiment_dict['policy'] == "PPOLearningPolicy" ):
				start_time = time.time()
				value_loss, action_loss, dist_entropy = self.policy.update(self.experiment_dict['num_sampled_nodes'])
				self.experiment_dict['policy_update_timings'].append(time.time()-start_time)
			print(self.experiment_dict['policy'] )
			value_losses.append(value_loss)
			action_losses.append(action_loss)

			# Update world model
			if(self.experiment_dict['reward_alpha'] != 1):
				start_time = time.time()
				for epoch in range(self.experiment_dict['num_training_epochs']):
					for next_loaded in enumerate(DataLoader(experience_replay, batch_size=self.experiment_dict['batch_size'], shuffle=True, num_workers=0)):
						
						_, batch = next_loaded
						inputs, labels, prestates, actions, _, _, index = batch
						#outputs_target = opt_cuda(self.worldModelTarget(labels).type(torch.FloatTensor))
						labels = opt_cuda(labels.type(torch.FloatTensor))
						actions = opt_cuda(actions.type(torch.FloatTensor))
						prestates = opt_cuda(prestates.type(torch.FloatTensor))

						self.optimizer_world.zero_grad()
						outputs = opt_cuda(self.worldModel(opt_cuda(labels.type(torch.FloatTensor))))
						loss = self.criterion(outputs[:, self.environment.predict_mask], labels[:, self.transform])
						loss.backward()
						self.optimizer_world.step()
						wm_losses.append(loss.item())

				del experience_replay
				experience_replay = ExperienceReplayBuffer()

				if(self.experiment_dict['debug_timing']):
					self.experiment_dict['wm_batch_timings'].append(time.time()-start_time)


			if j % self.experiment_dict['log_interval'] == 0 and len(episode_rewards) > 1:
				total_num_steps = (j + 1) * self.experiment_dict['num_steps']
				end = time.time()
				self.experiment_dict['rewards'].append(np.mean(episode_rewards))
				self.experiment_dict["world_model_losses"].append(np.mean(wm_losses))
				print(
					"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.3f}/{:.3f}, min/max reward {:.1f}/{:.1f}, Value Loss {:.3f}, Action Loss {:.3f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards),
							np.mean(value_losses),
							np.mean(action_loss)))
			if j % self.experiment_dict['save_interval'] == 0 and len(episode_rewards) > 1:
				stats_filehandler = open(self.experiment_dict['exp_path'] + "/stats.pkl", 'wb')
				pickle.dump(self.experiment_dict, stats_filehandler)


		return None, None, self.experiment_dict



