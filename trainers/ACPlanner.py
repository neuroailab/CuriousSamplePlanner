#!/usr/bin/env python
from __future__ import print_function
import pybullet as p
import numpy as np
import random
import math
import imageio
import matplotlib.pyplot as plt
import os
import shutil
import imageio
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys

from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.model import Policy
from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr.storage import RolloutStorage
from CuriousSamplePlanner.rl_ppo_rnd.a2c_ppo_acktr import algo, utils
from CuriousSamplePlanner.scripts.utils import *
from CuriousSamplePlanner.trainers.planner import Planner
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer


class ACPlanner(Planner):
	def __init__(self, *args):
		super(ACPlanner, self).__init__(*args)
		self.value_loss_coef = 0.5
		self.entropy_coef = 0
		self.lr = 1e-3
		self.eps = 1e-5
		self.alpha = 0.99
		self.max_grad_norm = 0.5
		self.agent = algo.PPO(
			self.environment.actor_critic,
			value_loss_coef=self.value_loss_coef,
			ppo_epoch=4,
			clip_param=0.2,
			entropy_coef=self.entropy_coef,
			lr=self.lr,
			eps=self.eps,
			max_grad_norm=self.max_grad_norm,
			num_mini_batch=32)

	def init_rollouts(self):
		self.rollouts = RolloutStorage(self.experiment_dict["nsamples_per_update"], 1, [self.environment.config_size], self.environment.action_space, 0)
		self.rollouts.opt_cuda()

	def expand_graph(self, run_index):

		# Create rollouts structure for actor-critic
		self.init_rollouts()
		self.train_world_model(run_index)

		# If asm enabled, train the actor-critic networks
		if(self.experiment_dict["enable_asm"]):
			with torch.no_grad():
				next_value = self.environment.actor_critic.get_value(
					rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
					rollouts.masks[-1]).detach()

			rollouts.compute_returns(next_value, False, 0, 0, False)
			value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
			rollouts.after_update()

		# Get the losses from all observations
		whole_losses, whole_indices, whole_feasibles = self.calc_novelty()
		sort_args = np.array(whole_losses).argsort()[::-1]
		high_loss_indices = [whole_indices[p] for p in sort_args]
		average_loss = sum(whole_losses) / len(whole_losses)

		# Update stats
		self.experiment_dict['feasibility'].append((sum(whole_feasibles)/len(whole_feasibles)))
		self.experiment_dict['world_model_losses'].append(average_loss)
		self.print_exp_dict(verbose=False)

		# Adaptive batch
		# if (average_loss <= self.loss_threshold):
		added_base_count = 0
		for en_index, hl_index in enumerate(high_loss_indices):
			input, target, pretarget, action, _, _, _, parent_index, _ = self.experience_replay.__getitem__(hl_index)
			command = self.experience_replay.get_command(hl_index)
			ntarget = target.cpu().numpy()
			npretarget = pretarget.cpu().numpy()
			if (not self.graph.is_node(ntarget)):
				self.environment.set_state(ntarget)
				for perspective in self.environment.perspectives:
					imageio.imwrite(self.exp_path
									+ '/run_index=' + str(run_index)
									+ ',index=' + str(en_index)
									+ ',parent_index=' + str(int(parent_index.item()))
									+ ',node_index=' + str(self.graph.node_key) + '.jpg',
									take_picture(perspective[0], perspective[1], 0, size=512))
				self.graph.add_node(ntarget, npretarget, action.cpu().numpy(), torch.squeeze(parent_index).item(), command = command)
				added_base_count += 1
			if (added_base_count == self.growth_factor):
				break
		del self.experience_replay
		self.experience_replay = ExperienceReplayBuffer()
		self.experiment_dict["num_graph_nodes"] += self.growth_factor

		# Update novelty scores for tree nodes
		self.update_novelty_scores()

		self.save_params()




