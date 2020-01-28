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

# DDPG imports 
from CuriousSamplePlanner.ddpg.ddpg import DDPG
from CuriousSamplePlanner.ddpg.naf import NAF
from CuriousSamplePlanner.ddpg.normalized_actions import NormalizedActions
from CuriousSamplePlanner.ddpg.ounoise import OUNoise
from CuriousSamplePlanner.ddpg.param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from CuriousSamplePlanner.ddpg.balanced_replay_memory import BalancedReplayMemory, Transition
from CuriousSamplePlanner.ddpg.replay_memory import ReplayMemory


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
		self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=20, desired_action_stddev=experiment_dict['noise_scale'], adaptation_coefficient=1.05) if experiment_dict['param_noise'] else None

		# Init the plan graph 
		self.graph = PlanGraph(environment=self.environment, node_sampling = self.node_sampling)
		super(Planner, self).__init__()

	def print_exp_dict(self, verbose=False):
		mean_reward = sum(self.experiment_dict['rewards'][-128:])/len(self.experiment_dict['rewards'][-128:])
		print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1])+"\t Feasibility: "+str(self.experiment_dict['feasibility'][-1])+"\t Reward: "+str(mean_reward))
		# print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1]))

	def plan(self):


		# Set up the starting position
		start_state = torch.tensor(self.environment.get_start_state())

		start_node = self.graph.add_node(start_state, None, None, None)

		start_state = torch.unsqueeze(start_state, dim=0).type(torch.FloatTensor)
		run_index = 0
		total_numsteps = 0
		feature = torch.zeros((1, 3, 84, 84))
		i_episode = 0
		if self.experiment_dict['ou_noise']: 
			self.ounoise.scale = (self.experiment_dict['noise_scale'] - self.experiment_dict['final_noise_scale']) * max(0, self.experiment_dict['exploration_end'] - i_episode) / self.experiment_dict['exploration_end'] + self.experiment_dict['final_noise_scale']
			self.ounoise.reset()
		last_reward = 0
		while (True):
			total_numsteps += 1
			last_reward += 1
			# Parent State selection
			parent = self.graph.expand_node(1)[0]
			self.environment.set_state(parent.config)
			parent_state = torch.unsqueeze(opt_cuda(torch.tensor(parent.config)), dim=0)
			if(self.experiment_dict["enable_asm"]):
				action = self.agent.select_action(parent_state, self.ounoise, self.param_noise)
			else:	
				action = torch.unsqueeze(torch.tensor(np.random.uniform(low=-1, high=1, size=self.environment.action_space_size)), dim=0).type(torch.FloatTensor)

			next_state, reward, done, infos = self.environment.step(torch.squeeze(action))
			# Extract extra info from intos
			inputs, prestate, feasible, command = infos['inputs'], infos['prestable'], infos['feasible'], infos['command']

			# Current State selection
			# parent = self.graph.expand_node(1)[0]
			# # self.environment.set_state(parent.config)
			# # Get an action from the agent given the current state
			# # parent_state = torch.unsqueeze(start_state, dim=0)
			# action = self.agent.select_action(torch.unsqueeze(opt_cuda(torch.tensor(parent.config)), dim=0), self.ounoise, self.param_noise)
			# next_state, reward, done, infos, inputs, prestate, feasible, command = self.environment.step(action)

			start_state = next_state
			mask = opt_cuda(torch.Tensor([not done]))
			reward = opt_cuda(torch.Tensor([reward]))
			parent_config = torch.unsqueeze(opt_cuda(torch.Tensor(parent.config).type(torch.FloatTensor)), dim=0)
			self.experiment_dict['rewards'].append(reward.item())

			if(self.experiment_dict['use_splitter']):
				self.memory.push(int(reward.item() == 1), parent_config.detach().cpu(), action.detach().cpu(), mask.detach().cpu(), next_state.detach().cpu(), reward.detach().cpu())
			else:
				self.memory.push(parent_config.detach().cpu(), action.detach().cpu(), mask, next_state.detach().cpu(), reward.detach().cpu())

			if (reward.item() == 1):
				i_episode  += 1
				last_reward = 0

				ntarget = torch.squeeze(next_state).numpy()
				npretarget = prestate.cpu().numpy()

				goal_node = self.graph.add_node(ntarget, npretarget, action.numpy(), parent.node_key, command = command)
				if(self.experiment_dict['return_on_solution']):
					return self.graph, self.graph.get_optimal_plan(start_node, goal_node), self.experiment_dict
				# Found a reward, creating a new graph
				self.graph = PlanGraph(environment=self.environment, node_sampling = self.node_sampling)
				ss = self.environment.reset().cpu().numpy()[0]
				start_node = self.graph.add_node(ss, None, None, None)
				self.reset_world_model()

				if self.experiment_dict['param_noise'] and len(self.memory) >= self.experiment_dict['batch_size']:
					episode_transitions = self.memory.memory[self.memory.position-self.experiment_dict['batch_size']:self.memory.position]
					states = torch.cat([transition[0] for transition in episode_transitions], 0)
					unperturbed_actions = self.agent.select_action(states, None, None)
					perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)
					ddpg_dist = ddpg_distance_metric(perturbed_actions.detach().cpu().numpy(), unperturbed_actions.detach().cpu().numpy())
					self.param_noise.adapt(ddpg_dist)

				if self.experiment_dict['ou_noise']: 
					self.ounoise.scale = (self.experiment_dict['noise_scale'] - self.experiment_dict['final_noise_scale']) * max(0, self.experiment_dict['exploration_end'] - i_episode) / self.experiment_dict['exploration_end'] + self.experiment_dict['final_noise_scale']
					self.ounoise.reset()

			if len(self.memory) > self.experiment_dict['batch_size'] and total_numsteps%self.experiment_dict['update_interval']==0:
				for _ in range(self.experiment_dict['updates_per_step']):
					transitions = self.memory.sample(self.experiment_dict['batch_size'])
					transitions = [[opt_cuda(i) for i in r] for r in transitions]
					batch = Transition(*zip(*transitions))
					value_loss, policy_loss = self.agent.update_parameters(batch)
					transitions = None

			target = next_state.type(torch.FloatTensor)
			pretarget = prestate.type(torch.FloatTensor)
			action = action.type(torch.FloatTensor)
			parent_nodes = parent.node_key
			feasible = feasible
			combined_perspective = feature.type(torch.FloatTensor)

			self.experience_replay.bufferadd_single(combined_perspective, target, pretarget, action, feasible, parent_nodes, command)

			if(total_numsteps%self.batch_size == 0):
				self.train_world_model(0)
			
				# # Get the losses from all observations
				whole_losses, whole_indices, whole_feasibles = self.calc_novelty()
				sort_args = np.array(whole_losses).argsort()[::-1]
				high_loss_indices = [whole_indices[p] for p in sort_args]
				average_loss = sum(whole_losses) / len(whole_losses)

				# Update stats
				self.experiment_dict['feasibility'].append((sum(whole_feasibles)/len(whole_feasibles)))
				self.experiment_dict['world_model_losses'].append(average_loss)
				self.print_exp_dict(verbose=False)

				# # Adaptive batch
				if (average_loss <= self.loss_threshold or not self.experiment_dict['adaptive_batch']):
					added_base_count = 0
					for en_index, hl_index in enumerate(high_loss_indices):
						input, target, pretarget, action, _, parent_index, _ = self.experience_replay.__getitem__(hl_index)
						command = self.experience_replay.get_command(hl_index)
						ntarget = torch.squeeze(target).numpy()
						npretarget = pretarget.cpu().numpy()
						self.environment.set_state(ntarget)
						# for perspective in self.environment.perspectives:
						# 	picture, _, _ = take_picture(perspective[0], perspective[1], 0, size=512)
						# 	imageio.imwrite(self.exp_path
						# 					+ '/run_index=' + str(run_index)
						# 					+ ',index=' + str(en_index)
						# 					+ ',parent_index=' + str(int(parent_index))
						# 					+ ',node_index=' + str(self.graph.node_key) + '.jpg',
						# 					picture)


						self.graph.add_node(ntarget, npretarget, action.numpy(), parent_index, command = command)
						added_base_count += 1
						if (added_base_count == self.growth_factor):
							break

					del self.experience_replay
					self.experience_replay = ExperienceReplayBuffer()
					self.experiment_dict["num_graph_nodes"] += self.growth_factor

					# Update novelty scores for tree nodes
					self.update_novelty_scores()

				# Save Params
				self.save_params()


			self.experiment_dict["num_sampled_nodes"] += 1
			if(self.experiment_dict["num_sampled_nodes"] > self.sample_cap):
				return None, None, self.experiment_dict
			run_index+=1

		return self.graph, plan, self.experiment_dict



