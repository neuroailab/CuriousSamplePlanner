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

class RRTPlanner():
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

		# Init the plan graph 
		self.graph = PlanGraph(environment=self.environment, node_sampling = self.experiment_dict['node_sampling'])
		super(RRTPlanner, self).__init__()

	def save_params(self):
		# Save the updated models
		# with open(self.exp_path + "/worldModel.pkl", 'wb') as fw:
		# 	pickle.dump(self.worldModel, fw)
		# with open(self.exp_path + "/actor_critic.pkl", "wb") as fa:
		# 	pickle.dump(self.environment.actor_critic, fa)
		
		with open(self.experiment_dict['exp_path'] + "/exp_dict.pkl", "wb") as fa:
			pickle.dump(self.experiment_dict, fa)
			fa.close()

		graph_filehandler = open(self.experiment_dict['exp_path'] + "/found_graph.pkl", 'wb')
		pickle.dump(self.graph, graph_filehandler)

	def plan(self):
		# Add starting node to the graph
		self.policy.reset()
		start_state = torch.tensor(self.environment.get_start_state())
		print(start_state.shape)
		start_node = self.graph.add_node(start_state, None, None, None, run_index=0)
		run_index = 1

		# Begin RRT loop
		nodes_added = 0
		while (True):
			print("Run_index: "+str(run_index))
			# Select a random point within the configuration space for the objects
			sample_config = self.environment.get_random_config()

			# Find the node that is closest to the sample location
			nearest_node = self.graph.nn(sample_config)

			# Sample a bunch of actions from that node
			results = []
			state_action_dict = {}
			# for _ in range(int(self.experiment_dict["batch_size"]/self.experiment_dict["growth_factor"])):
			for _ in range(1):
				self.environment.set_state(nearest_node.config)
				action = self.policy.select_action(sample_config)
				result = self.policy.step(torch.squeeze(action))
				state_action_dict[result[0]] = action
				results.append(result)

			# Sort the results based on proximity to the sample
			sorted(results, key=lambda result: self.graph.l2dist(result[0], sample_config), reverse=True)

			# Select the actions that takes you closest to the selected point
			for result in results:
				(next_state, reward, done, infos) = result
				action = state_action_dict[result[0]]
				ntarget = torch.squeeze(next_state.detach().cpu()).numpy()
				naction = torch.squeeze(action.detach().cpu()).numpy()

				added_node = self.graph.add_node(ntarget, ntarget, naction, nearest_node.node_key, run_index = run_index)
				if(done):
					return self.graph, self.graph.get_optimal_plan(start_node, added_node), self.experiment_dict

				nodes_added+=1
				break

			if(nodes_added>=self.experiment_dict["growth_factor"]):
				nodes_added=0
				self.save_params()
				run_index+=1





