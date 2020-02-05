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


from CuriousSamplePlanner.policies.fixed import FixedPolicy
from CuriousSamplePlanner.policies.random import RandomPolicy
from CuriousSamplePlanner.policies.learning import LearningPolicy

class RRTPlanner():
	def __init__(self, experiment_dict):
		print("RRT planning")
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


	def plan(self):
		print("plan")
		# Add starting node to the graph
		self.policy.reset()
		run_index = 0
		start_state = torch.tensor(self.environment.get_start_state())
		start_node = self.graph.add_node(start_state, None, None, None, run_index=run_index)

		# Begin RRT loop
		while (True):
			print("Run_index: "+str(run_index))
			# Select a random point within the configuration space for the objects (10:30)
			sample_config = []

			for obj in self.environment.objects:
				random_vector=opt_cuda(torch.tensor(np.random.uniform(low=-1, high=1, size=self.environment.action_space_size))).type(torch.FloatTensor)
				random_state=self.environment.macroaction.reparameterize(obj, random_vector)
				pos, quat=random_state
				euler=p.getEulerFromQuaternion(quat)
				sample_config+=list(pos)+list(euler)

			# Find the node that is closest to the sample location (11:00)
			nearest_node = self.graph.nn(sample_config)

			# Sample a bunch of actions from that node (11:30)
			results = []
			state_action_dict = {}
			for _ in range(self.experiment_dict["batch_size"]):
				action = self.policy.select_action(sample_config)
				result = self.policy.step(torch.squeeze(action))
				state_action_dict[result[0]] = action
				results.append(result)

			# Sort the results based on proximity to the sample
			sorted(results, key=lambda result: self.graph.l2dist(result[0], sample_config), reverse=True)
			# Select the actions that takes you closest to the selected point (12:00)
			nodes_added = 0
			for result in results:
				(next_state, reward, done, infos) = result
				action = state_action_dict[result[0]]
				self.graph.add_node(next_state.detach().cpu().numpy(), next_state.detach().cpu().numpy(), action.detach().cpu().numpy(), sample_config, run_index = run_index)
				
				nodes_added+=1
				if(nodes_added>=self.experiment_dict["growth_factor"]):
					break


			# Add the resulting nodes to the graph (12:30)


			# Fix all the bugs to get it to work(1:00)


		return self.graph, plan, self.experiment_dict


