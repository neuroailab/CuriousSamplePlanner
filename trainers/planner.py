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
import imageio
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
	get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import WorldSaver, enable_gravity, connect, dump_world, set_pose, \
	Pose, Point, Euler, set_default_camera, stable_z, \
	BLOCK_URDF, load_model, wait_for_user, disconnect, DRAKE_IIWA_URDF, user_input, update_state, disable_real_time,inverse_kinematics,end_effector_from_body,approach_from_grasp, get_joints, get_joint_positions


from CuriousSamplePlanner.planning_pybullet.pybullet_tools.utils import get_pose, set_pose, get_movable_joints, \
	set_joint_positions, add_fixed_constraint, enable_real_time, disable_real_time, joint_controller, \
	enable_gravity, get_refine_fn, user_input, wait_for_duration, link_from_name, get_body_name, sample_placement, \
	end_effector_from_body, approach_from_grasp, plan_joint_motion, GraspInfo, Pose, INF, Point, \
	inverse_kinematics, pairwise_collision, remove_fixed_constraint, Attachment, get_sample_fn, \
	step_simulation, refine_path, plan_direct_joint_motion, get_joint_positions, dump_world, get_link_pose,control_joints

from CuriousSamplePlanner.planning_pybullet.pybullet_tools.kuka_primitives import BodyPath, Attach, Detach
import pickle
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from motion_planners.discrete import astar
import sys

from CuriousSamplePlanner.tasks.three_block_stack import ThreeBlocks

from CuriousSamplePlanner.tasks.ball_ramp import BallRamp
from CuriousSamplePlanner.tasks.pulley import PulleySeesaw
from CuriousSamplePlanner.tasks.bookshelf import BookShelf

from CuriousSamplePlanner.tasks.five_block_stack import FiveBlocks
from CuriousSamplePlanner.scripts.utils import *

from CuriousSamplePlanner.trainers.plan_graph import PlanGraph
from CuriousSamplePlanner.trainers.dataset import ExperienceReplayBuffer

class Planner():
	def __init__(self, experiment_dict):
		self.experiment_dict = experiment_dict

		# Transfer dict properties to class properties
		self.loss_threshold = self.experiment_dict["loss_threshold"]
		self.growth_factor = self.experiment_dict["growth_factor"]
		self.num_training_epochs = self.experiment_dict["num_training_epochs"]
		self.batch_size = self.experiment_dict["batch_size"]
		self.mode = self.experiment_dict['mode']
		self.task = self.experiment_dict['task']
		self.exp_path = self.experiment_dict['exp_path']
		self.load_path = self.experiment_dict['load_path']
		self.sample_cap = self.experiment_dict['sample_cap']
		self.node_sampling = self.experiment_dict['node_sampling']

		# Create the replay buffer for training world models
		self.experience_replay = ExperienceReplayBuffer()

		# Create the environment
		EC = getattr(sys.modules[__name__], self.experiment_dict["task"])
		self.environment = EC(experiment_dict)

		self.agent = DDPG(experiment_dict['gamma'], experiment_dict['tau'], experiment_dict['hidden_size'], self.environment.config_size, self.environment.action_space, actor_lr = experiment_dict['actor_lr'], critic_lr = experiment_dict["critic_lr"])
		self.agent.cuda()
		if(experiment_dict['use_splitter']):
			self.memory = BalancedReplayMemory(experiment_dict['replay_size'], split=experiment_dict["split"])
		else:
			self.memory = ReplayMemory(experiment_dict['replay_size'])

		self.ounoise = OUNoise(self.environment.action_space.shape[0]) if experiment_dict['ou_noise'] else None
		self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=experiment_dict['noise_scale'], adaptation_coefficient=1.05) if experiment_dict['param_noise'] else None

		# Init the plan graph 
		self.graph = PlanGraph(environment=self.environment, node_sampling = self.node_sampling)
		super(Planner, self).__init__()

	def print_exp_dict(self, verbose=False):
		print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1])+"\t Feasibility: "+str(self.experiment_dict['feasibility'][-1]))
		# print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1]))

	def plan(self):


		# Set up the starting position
		start_state = self.environment.get_start_state()
		start_node = self.graph.add_node(start_state, None, None, None)
		run_index = 0
		total_numsteps = 0
		feature = torch.zeros((1, 3, 84, 84))
		while (True):
			total_numsteps += 1
			# State selection
			parent = self.graph.expand_node(1)[0]
			self.environment.set_state(parent.config)
			# Get an action from the agent given the current state
			action = self.agent.select_action(state, self.ounoise, self.param_noise)
			next_state, reward, done, infos, inputs, prestate, feasible = self.environment.step(action)

			mask = opt_cuda(torch.Tensor([not done]))
			reward = opt_cuda(torch.Tensor([reward]))
			experiment_dict['rewards'].append(reward.item())

			if(experiment_dict['use_splitter']):
				self.memory.push(int(reward.item() == 1), state, action, mask, next_state, reward)
			else:
				self.memory.push(state, action, mask, next_state, reward)

			state = opt_cuda(next_state.type(torch.FloatTensor))

			if len(self.memory) > experiment_dict['batch_size']:
				for _ in range(experiment_dict['updates_per_step']):
					transitions = self.memory.sample(experiment_dict['batch_size'])
					batch = Transition(*zip(*transitions))
					value_loss, policy_loss = self.agent.update_parameters(batch)
					updates += 1

			target = opt_cuda(next_state)
			pretarget = opt_cuda(prestate)
			action = opt_cuda(action)
			parent_nodes = opt_cuda(parents)
			feasible = opt_cuda(feasible)
			combined_perspective = opt_cuda(torch.cat(feature))

			self.experience_replay.bufferadd_single(combined_perspective, target, pretarget,\
				actions, feasible, parent_nodes, commands)

			# Adaptive batch
			if (average_loss <= self.loss_threshold or not self.experiment_dict['adaptive_batch']):
				added_base_count = 0
				for en_index, hl_index in enumerate(high_loss_indices):
					input, target, pretarget, action, _, _, _, parent_index, _ = self.experience_replay.__getitem__(hl_index)
					command = self.experience_replay.get_command(hl_index)
					ntarget = target.cpu().numpy()
					npretarget = pretarget.cpu().numpy()
					if (not self.graph.is_node(ntarget)):
						self.environment.set_state(ntarget)
						for perspective in self.environment.perspectives:
							picture, _, _ = take_picture(perspective[0], perspective[1], 0, size=512)
							imageio.imwrite(self.exp_path
											+ '/run_index=' + str(run_index)
											+ ',index=' + str(en_index)
											+ ',parent_index=' + str(int(parent_index.item()))
											+ ',node_index=' + str(self.graph.node_key) + '.jpg',
											picture)
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


			self.experiment_dict["num_sampled_nodes"] += self.experiment_dict["nsamples_per_update"]
			if(self.experiment_dict["num_sampled_nodes"]>self.sample_cap):
				return None, None, self.experiment_dict
			run_index+=1

		return self.graph, plan, self.experiment_dict



