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

		# Init the plan graph 
		self.graph = PlanGraph(environment=self.environment, node_sampling = self.node_sampling)
		super(Planner, self).__init__()

	def print_exp_dict(self, verbose=False):
		print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1])+"\t Feasibility: "+str(self.experiment_dict['feasibility'][-1]))
		# print("Sampled: "+str(self.experiment_dict['num_sampled_nodes'])+"\t Graph Size: "+str(self.experiment_dict['num_graph_nodes'])+"\t WM Loss: "+str(self.experiment_dict['world_model_losses'][-1]))

	def expand_graph(self, run_index):
		raise NotImplementedError

	def plan(self):
		# Set up the starting position
		start_state = self.environment.get_start_state()
		start_node = self.graph.add_node(start_state, None, None, None)
		run_index=0

		while (True):
			features, states, prestates, actions, action_log_probs, \
			values, feasible, parents, goal, goal_prestate, goal_parent, \
			goal_action, goal_command, commands = self.environment.collect_samples(self.graph)
			if(goal is not None):
				self.environment.set_state(goal)
				for perspective in self.environment.perspectives:
					picture, _, _ = take_picture(perspective[0], perspective[1], 0, size=512)
					imageio.imwrite(self.exp_path
									+ '/GOAL'
									+ ',run_index=' + str(run_index)
									+ ',parent_index=' + str(goal_parent)
									+ ',node_key=' + str(self.graph.node_key) + '.jpg',
									picture)

				
				goal_node = self.graph.add_node(goal, goal_prestate, goal_action, goal_parent, command=goal_command)
				plan = self.graph.get_optimal_plan(start_node, goal_node)
				break

			targets = opt_cuda(states)
			pretargets = opt_cuda(prestates)
			actions = opt_cuda(actions)
			action_log_probs = opt_cuda(action_log_probs)
			values = opt_cuda(values)
			parent_nodes = opt_cuda(parents)
			feasible = opt_cuda(feasible)
			combined_perspectives = opt_cuda(torch.cat(features))

			self.experience_replay.bufferadd(combined_perspectives, targets, pretargets,\
				actions, action_log_probs, values, feasible, parent_nodes, commands)
			self.expand_graph(run_index)
			self.experiment_dict["num_sampled_nodes"] += self.experiment_dict["nsamples_per_update"]
			if(self.experiment_dict["num_sampled_nodes"]>self.sample_cap):
				return None, None, self.experiment_dict
			run_index+=1

		return self.graph, plan, self.experiment_dict



